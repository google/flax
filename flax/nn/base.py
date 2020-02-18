# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NN base modules for JAX."""

import abc
import contextlib
import functools
import inspect
from typing import Any

from . import utils
from flax import jax_utils
from flax import serialization
from flax import struct

import jax
from jax import random


_module_stack = utils.CallStack()
_module_output_trackers = utils.CallStack()
_state_stack = utils.CallStack()


def _track_outputs(x):
  for module_output_tracker in _module_output_trackers:
    module_output_tracker.store(x)


class ModuleFrame:
  """A ModuleFrame contains all the information needed to apply a Module."""

  def __init__(self, module, mode, params=None, rng=None,  # pylint: disable=redefined-outer-name
               name=None, index=None, transparent=False):
    if params is None:
      params = {}
    self.module = module
    self.mode = mode
    self.rng = rng
    self.params = params
    self.shared_names = set()
    self.name = name
    self.name_counter = 0
    self.index = index
    self.index_counter = {}
    self.transparent = transparent

  @property
  def identifier(self):
    if self.name is None or self.index is None:
      return None
    return '{}:{}'.format(self.name, self.index)


def _path_from_frames(frames, include_index=True):
  """Determine the path of a module based on the call stack."""
  parts = []
  prev_frame = None
  for frame in frames:
    if prev_frame and prev_frame.transparent:
      prev_frame = frame
      continue
    prev_frame = frame
    if frame.name is None:
      continue
    if include_index:
      parts.append(frame.identifier)
    else:
      parts.append(frame.name)
  return '/' + '/'.join(parts)


def _populate_name(kwargs):
  """Determine the name of a Module."""
  name = kwargs.get('name', None)
  top_level = kwargs.get('_top_level', False)
  if _module_stack and not top_level:
    parent = _module_stack[-1]
    if name is None:
      name = str(parent.name_counter)
      parent.name_counter += 1
  elif _module_stack and name is None:
    name = '__nested_model__'
  else:
    assert top_level, 'Expected a top-level Module'

  if name is not None:
    if not isinstance(name, str):
      raise ValueError('Name must be a string.')
    if '/' in name or ':' in name:
      raise ValueError('Name should not contain slashes or colons.')

  kwargs['name'] = name
  kwargs['_top_level'] = top_level
  return name


def _populate_index(kwargs):
  """Determine the index of a Module.

  The index is used to track how many times a module with shared parameters is
  invoked. This is used to detect name collisions and to make sure that
  collections can be track data for each invocation seperatly.

  Args:
    kwargs: the keyword arguments of the module.
  Returns:
    The index of this module invocation.
  """
  name = kwargs['name']
  top_level = kwargs['_top_level']
  shared = kwargs.get('_shared', False)
  index = kwargs.get('_index', None)
  if _module_stack:
    parent = _module_stack[-1]
    if index is None:
      if name not in parent.index_counter:
        parent.index_counter[name] = 0
      index = parent.index_counter[name]
      parent.index_counter[name] += 1
    if name in parent.shared_names and not shared:
      raise ValueError(
          f'a shared module named "{name}" already exists.')
  else:
    assert top_level, 'Expected a top-level Module'
    index = 0
  if index >= 1 and not shared and not top_level:
    raise ValueError(
        'use `Module.shared()` when parameter sharing is intended.')
  kwargs['_index'] = index
  kwargs['_shared'] = shared
  return index


def _pop_identifiers(kwargs):
  name = _populate_name(kwargs)
  index = _populate_index(kwargs)
  kwargs.pop('name')
  kwargs.pop('_index')
  kwargs.pop('_shared')
  kwargs.pop('_top_level')
  return name, index


def module_method(fn):
  """Decorates a function as a module method.

  The `module_method` allows modules to have multiple methods that make use of
  the modules parameters.

  Example::

    class MyLinearModule(nn.Module):
      def apply(self, x, features, kernel_init):
        kernel = self.param('kernel', (x.shape[-1], features), kernel_init)
        return jnp.dot(x, kernel)

      @nn.module_method
      def apply_transpose(self, x, **kwargs):
        kernel = self.get_param('kernel')
        return jnp.dot(x, kernel.transpose((1, 0)))

  A module method can be called on A Model instance directly::

    y, model = MyLinearModule.create(rng, x)
    z = model.apply_transpose(y)

  Module methods can also be called on shared modules::
  
    class AutoEncoder(nn.module):
      def apply(self, x, features):
        linear_fn = MyLinearModule.shared(features=features)
        h = linear_fn(x)
        y = linear_fn.apply_transpose(h)
        return y
  

  Args:
    fn: the function to be decorated
  Returns:
    the decorated function
  """
  def wrapper(cls, *args, **kwargs):
    """wraps fn such that it behaves as a module method."""
    # TODO(flax-dev): dedup some of the logic here and in the Module constructor
    if not _module_stack:
      raise ValueError('A module method only be called inside another module.'
                       ' It is also available as a method on a Model instance.')
    new_module = cls.new_instance()
    extended_kwargs = new_module._extend_kwargs(kwargs)  # pylint: disable=protected-access
    if not extended_kwargs.get('_shared', False):
      raise ValueError('A module method only be used on a shared module.')
    name = _populate_name(extended_kwargs)
    kwargs['name'] = name
    parent = _module_stack[-1]
    if parent.mode == 'init' and name not in parent.params:
      parent.rng, rng = random.split(parent.rng)
      y, params = new_module._init_module_method(  # pylint: disable=protected-access
          functools.partial(fn, new_module), rng, args, kwargs)
      parent.params[name] = params
      return y
    else:  # apply
      if name not in parent.params:
        raise ValueError(f'No module named {name} was created during'
                         ' initialization.')
      params = parent.params[name]
      return new_module._apply_module_method(  # pylint: disable=protected-access
          functools.partial(fn, new_module), params, args, kwargs)

  wrapper.module_method = fn
  return classmethod(wrapper)


def _fn_parameters(fn):
  return tuple(inspect.signature(fn).parameters.values())


MODULE_CLASSMETHODS = [
    'create', 'create_by_shape', 'init', 'init_by_shape', 'call', 'partial'
]


class _ModuleMeta(abc.ABCMeta):
  """Meta class for automatically setting the doc of Modules."""

  def __init__(cls, name, bases, attrs):
    super(_ModuleMeta, cls).__init__(name, bases, attrs)
    apply_fn = cls.apply
    apply_doc = apply_fn.__doc__
    cls.__doc__ = apply_doc
    apply_params = _fn_parameters(apply_fn)
    cls.__signature__ = inspect.signature(cls).replace(
        parameters=apply_params[1:])

    if not bases:
      return  # skip method signature overides for Module class.

    def wrap_special_method(name):
      """override the signature and docstring for one of Module's classmethods."""
      orig_fn = getattr(Module, name)

      @functools.wraps(orig_fn)
      def wrapper(class_, *args, **kwargs):
        super_fn = getattr(super(cls, class_), name)
        return super_fn(*args, **kwargs)
      wrapper.__doc__ = f'''{orig_fn.__doc__}

      Apply docstring:

      {apply_doc}
      '''
      base_params = tuple(x for x in _fn_parameters(orig_fn)
                          if x.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)
      new_params = base_params + apply_params[1:]
      wrapper.__signature__ = inspect.signature(orig_fn).replace(
          parameters=new_params)
      setattr(cls, name, classmethod(wrapper))

    for name in MODULE_CLASSMETHODS:
      wrap_special_method(name)


class Module(metaclass=_ModuleMeta):
  """Functional modules."""

  def __new__(cls, *args, **kwargs):
    if not _module_stack:
      raise ValueError('A Module should only be instantiated directly inside'
                       ' another module.')
    new_module = super(Module, cls).__new__(cls)
    extended_kwargs = new_module._extend_kwargs(kwargs)  # pylint: disable=protected-access
    name = _populate_name(extended_kwargs)
    kwargs['name'] = name
    parent = _module_stack[-1]
    if parent.mode == 'init' and name not in parent.params:
      parent.rng, rng = random.split(parent.rng)
      y, params = new_module._init(rng, *args, **kwargs)  # pylint: disable=protected-access
      parent.params[name] = params
    else:  # apply
      if name not in parent.params:
        raise ValueError(f'No module named {name} was created during'
                         ' initialization.')
      params = parent.params[name]
      y = new_module(params, *args, **kwargs)
    return y

  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  @classmethod
  def shared(class_, **kwargs):
    """Partially applies a module and shared parameters for each call."""
    name = _populate_name(kwargs)
    if _module_stack:
      frame = _module_stack[-1]
      if name in frame.shared_names:
        raise ValueError(
            f'Another shared module named "{name}" already exists.')
      if name in frame.index_counter:
        raise ValueError(
            f'Another module named "{name}" already exists.')
      frame.shared_names.add(name)

    kwargs['_shared'] = True
    return class_.partial(**kwargs)

  @classmethod
  def partial(class_, **kwargs):
    """Partially applies a module with the given arguments."""

    shared = kwargs.get('_shared', False)
    name = kwargs.get('name', None)

    class PartialModule(class_):
      """Wraps a module with partial application."""

      def _extend_kwargs(self, invoke_kwargs):
        if shared and 'name' in invoke_kwargs and invoke_kwargs['name'] != name:
          raise ValueError('Cannot override the name of a shared module.')

        extended_kwargs = kwargs.copy()
        extended_kwargs.update(invoke_kwargs)
        return super()._extend_kwargs(extended_kwargs)
    # __doc__ is handled by the Module meta class
    PartialModule.__name__ = class_.__name__

    return PartialModule

  @classmethod
  def create(cls, rng, *args, name=None, **kwargs):
    """Create a module instance by evaluating the model.

    Use create_by_shape instead to initialize without doing computation.
    Initializer functions can depend both on the shape and the value of inputs.

    Args:
      rng: the random number generator used to initialize parameters.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and an instance of Model
    """
    instance = cls.new_instance()
    y, params = instance._init(rng, *args, name=name, _top_level=True, **kwargs)  # pylint: disable=protected-access
    model = Model(instance, params)
    return y, model

  @classmethod
  def create_by_shape(cls, rng, input_specs, *args, name=None, **kwargs):
    """Create a module instance using only shape and dtype information.

    This method will initialize the model without computation.
    Initializer functions can depend on the shape but not the value of inputs.

    Args:
      rng: the random number generator used to initialize parameters.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs
      *args: other arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and an instance of Model
    """
    def lazy_create(*inputs):
      return cls.create(rng, *(inputs + args), name=name, **kwargs)
    return jax_utils.partial_eval_by_shape(lazy_create, input_specs)

  @classmethod
  def init(cls, rng, *args, name=None, **kwargs):
    """Initialize the module parameters.

    Args:
      rng: the random number generator used to initialize parameters.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    """
    instance = cls.new_instance()
    return instance._init(rng, *args, name=name, _top_level=True, **kwargs)  # pylint: disable=protected-access

  @classmethod
  def init_by_shape(cls, rng, input_specs, *args, name=None, **kwargs):
    """Initialize the module parameters.

    This method will initialize the module parameters without computation.
    Initializer functions can depend on the shape but not the value of inputs.

    Args:
      rng: the random number generator used to initialize parameters.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    """
    def lazy_init(*inputs):
      instance = cls.new_instance()
      return instance._init(  # pylint: disable=protected-access
          rng, *(inputs + args), name=name, _top_level=True, **kwargs)
    return jax_utils.partial_eval_by_shape(lazy_init, input_specs)

  @classmethod
  def new_instance(cls):
    return object.__new__(cls)

  @classmethod
  def call(cls, params, *args, **kwargs):
    """Evaluate the module with the given parameters.

    Args:
      params: the parameters of the module. Typically, inital parameter values
        are constructed using `Module.init` or `Module.init_by_shape`.
      *args: arguments passed to the module's apply function
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      The output of the module's apply function.
    """
    instance = cls.new_instance()
    kwargs['_top_level'] = True
    return instance(params, *args, **kwargs)

  def __call__(self, params, *args, **kwargs):
    return self._apply_module_method(self.apply, params, args, kwargs)

  def param(self, name, shape, initializer):
    """Defines a parameter within the module's apply function.

    Args:
      name: The name of the parameter.
      shape: The shape of the parameter.
      initializer: An initializer function
                   taking an RNG and the shape as arguments.
    Returns:
      The value of the parameter.
    """
    frame = _top_frame('param')
    if frame.mode == 'init':
      if name in frame.params:
        raise ValueError(
            "Name '%s' was already used for another parameter." % name)
      frame.rng, key = random.split(frame.rng)
      frame.params[name] = initializer(key, shape)
    if name not in frame.params:
      raise ValueError("Parameter with name '%s' does not exist." % name)
    param = frame.params[name]
    if shape is not None and param.shape != shape:
      raise ValueError(
          'Existing shape {} differs from requested shape {}'.format(
              param.shape, shape))
    return param

  def get_param(self, name):
    """Retrieves a parameter within the module's apply function.

    Args:
      name: The name of the parameter.
    Returns:
      The value of the parameter.
    """
    frame = _top_frame('param')
    if name not in frame.params:
      raise ValueError("Parameter with name '%s' does not exist." % name)
    return frame.params[name]

  def state(self, name, shape=None, initializer=None, collection=None):
    """Declare a state variable within the module's apply function.

    A state variable has an attribute value which can be updated by simply
    assigning a value to it. For example::

      class Example(nn.Module):
        def apply(self, inputs, decay=0.9):
          ema = self.state('ema', inputs.shape, initializers.zeros)
          ema.value = decay * ema.value + (1 - decay) * inputs
          return inputs

    By default Modules are stateless. See `flax.nn.stateful` to enable stateful
    computations.

    Args:
      name: the name of the state variable.
      shape: optional shape passed to the initializer (default: None)
      initializer: optional initializer function
        taking an RNG and the shape as arguments.
      collection: optional `flax.nn.Collection` used to store the state.
        By default the state collection passed to the `nn.stateful` context is
        used.
    Returns:
      An instance of ModuleState.
    """
    _top_frame('state')
    if collection is None:
      collection = get_state()
    state = ModuleState(collection, name)
    # find the frames that are in init mode
    init_frames = [f for f in _module_stack if f.mode == 'init']
    if initializer is not None and init_frames:
      # use the closest frame that is initializing to get an rng
      init_frame = init_frames[-1]
      init_frame.rng, key = random.split(init_frame.rng)
      init_value = initializer(key, shape)
      state.value = init_value
    return state

  def is_stateful(self):
    return is_stateful()

  def is_initializing(self):
    _top_frame('is_initializing')
    return _module_stack[0].mode == 'init'

  def _extend_kwargs(self, kwargs):
    return kwargs

  def _pre_process_params(self, params):
    return params

  def _post_process_params(self, params):
    return params

  def _is_transparent(self):
    return False

  def _init(self, rng, *args, **kwargs):
    return self._init_module_method(self.apply, rng, args, kwargs)

  def _init_module_method(self, fn, rng, args, kwargs):
    """Apply a function in a module initialization scope."""
    kwargs = self._extend_kwargs(kwargs)
    name, index = _pop_identifiers(kwargs)
    frame = ModuleFrame(self, 'init', rng=rng, name=name, index=index,
                        transparent=self._is_transparent())
    with _module_stack.frame(frame):
      y = fn(*args, **kwargs)
      _track_outputs(y)
    params = self._post_process_params(frame.params)
    return y, params

  def _apply_module_method(self, fn, params, args, kwargs):
    """Apply a function in a module scope."""
    params = self._pre_process_params(params)
    kwargs = self._extend_kwargs(kwargs)
    name, index = _pop_identifiers(kwargs)
    frame = ModuleFrame(self, 'apply', params=params, name=name, index=index,
                        transparent=self._is_transparent())
    with _module_stack.frame(frame):
      y = fn(*args, **kwargs)
      _track_outputs(y)
    return y


def module(fun):
  """Convert a function into the apply method of a new Module.

  This is convenient shortcut for writing higher level modules that don't need
  access to `self` for creating parameters directly.

  Example usage::

    @nn.module
    def DenseLayer(x, features):
      x = flax.nn.Dense(x, features)
      x = flax.nn.relu(x)
      return x

  This is exactly equivalent to defining the following `nn.Module` subclass::

    class DenseLayer(nn.Module):
      def apply(self, x, features):
        x = flax.nn.Dense(x, features)
        x = flax.nn.relu(x)
        return x

  Args:
    fun: the function to convert.
  Returns:
    New Module subclass.
  """
  @functools.wraps(fun)
  def apply(self, *args, **kwargs):
    del self  # unused
    return fun(*args, **kwargs)
  return type(fun.__name__, (Module,), dict(apply=apply))


class TransparentModule(Module):
  """Transparent module."""

  def _pre_process_params(self, params):
    return {'0': params}

  def _post_process_params(self, params):
    entries = list(params.items())
    if len(entries) != 1:
      raise ValueError('Transparent modules should have exactly one child.')
    key, value = entries[0]
    if key != '0':
      raise ValueError('Transparent module should contain an unnamed child.')
    return value

  def _is_transparent(self):
    return True


class TruncatedModule(TransparentModule):
  """Wraps a Module and returns the requested intermediate outputs instead.

  See `Model.truncate_at` for a simple api to get the intermediate outputs of
  an existing Model.
  """

  def apply(self, *args, wrapped_module=None, truncate_path=None, **kwargs):
    """Apply the wrapped module and return some of its intermediate outputs.

    Args:
      *args: the positional arguments for the wrapped module.
      wrapped_module: The module class to be wrapped.
      truncate_path: the full name of the module (eg. '/module/sub_module').
        A list or dict of module paths can be provided to obtain the
        intermediate outputs of multiple modules.
      **kwargs: the keyword arguments for the wrapped module.
    Returns:
      The intermediate outputs specified by truncate_path.
    """
    if wrapped_module is None or truncate_path is None:
      raise ValueError(
          '`wrapped_module` and `trucate_path` are required keyword arguments')
    with capture_module_outputs() as module_outputs:
      wrapped_module(*args, **kwargs)

    def lookup_output(path):
      return module_outputs.lookup(path)
    return jax.tree_map(lookup_output, truncate_path)


@contextlib.contextmanager
def capture_module_outputs():
  """A context manager that captures all model outputs.

  Yields:
    A `flax.nn.Collection` containing all module outputs.
  """
  with Collection().mutate() as module_outputs:
    with _module_output_trackers.frame(module_outputs):
      yield module_outputs


class ModuleState():
  """Tracks a state variable.

  ModuleState instances should not be created directly. See `Module.state` on
  how to create state variables inside modules.
  """

  def __init__(self, collection, name):
    self._collection = collection
    self._name = name

  def _get_state_dict(self):
    state_dict = self._collection.retrieve(default={})
    assert isinstance(state_dict, dict)
    return state_dict

  @property
  def name(self):
    return self._name

  @property
  def value(self):
    state_dict = self._get_state_dict()
    if self._name not in state_dict:
      raise ValueError(f'No state variable named `{self._name}` exists.')
    return state_dict[self._name]

  @value.setter
  def value(self, v):
    state_dict = self._get_state_dict()
    state_dict[self._name] = v
    self._collection.store(state_dict)


@contextlib.contextmanager
def stateful(state=None, mutable=True):
  """A context manager for stateful computations.

  Module's that use the `Module.state` by default store state inside the
  `Collection` specified by the (innermost) `nn.stateful` context manager.

  Typically stateful is used in 3 different modes:

  1. During init no existing state is available and the stateful context creates
     a new state collection.
  2. During training the state is passed to `nn.stateful` and the new state
     is returned which will contain the updated state.
  3. During evaluation the state is passed with `mutable=False` such that the
     model can retrieve the state but is not allowed to mutate it.

  Example::

    class MyModel(nn.Module):
      def apply(self, x):
        x = nn.Dense(x, 12)
        x = nn.BatchNorm(x)
        return x

    with nn.stateful() as state:
      _, model = MyModel.create(rng, x)

    with nn.stateful(state) as new_state:
      model(x2)

    with nn.stateful(new_state, mutable=False):
      evaluate_model(model)

  Args:
    state: a `flax.nn.Collection` containing the current state.
      By default a new collection will be created.
    mutable: If true the state will be mutable otherwise it will be frozen.
  Yields:
    A `flax.nn.Collection` containing the new state.
  """
  if state is None:
    state = Collection()
  if mutable:
    with state.mutate() as new_state:
      with _state_stack.frame(new_state):
        yield new_state
  else:
    with _state_stack.frame(state):
      yield state


def is_stateful():
  """Returns true if a stateful scope is currently active (see `flax.nn.stateful`)."""
  return bool(_state_stack)


def get_state():
  if not _state_stack:
    raise ValueError('Use the flax.nn.stateful context manager to enable'
                     ' stateful computations.')
  return _state_stack[-1]


def _top_frame(call_name):
  if not _module_stack:
    raise ValueError('%s should only be used inside a '
                     'module\'s apply function.' % call_name)
  return _module_stack[-1]


@struct.dataclass
class Model:
  """A Model contains the model paramaters, state and definition."""

  module: Module = struct.field(pytree_node=False)
  params: Any

  def __call__(self, *args, **kwargs):
    kwargs['_top_level'] = True
    return self.module(self.params, *args, **kwargs)

  def truncate_at(self, module_path):
    """Truncate the model by returning the outputs of the given sub-module.

    Args:
      module_path: the full name of the module (eg. '/module/sub_module').
        A list or dict of module paths can be provided to obtain the
        intermediate outputs of multiple modules.
    Returns:
      A new model with the truncated outputs. If module_path is a pytree of
      paths the outputs will be have the same structure where each path is
      replaced by the corresponding intermediate output.
    """
    truncated_module_cls = TruncatedModule.partial(
        wrapped_module=type(self.module), truncate_path=module_path)
    module_instance = truncated_module_cls.new_instance()
    return self.replace(module=module_instance)

  def __getattr__(self, name):
    value = getattr(self.module, name)
    if callable(value) and hasattr(value, '__func__'):
      # class methods are bound methods so we must check the underlying function
      # to verify that it defines a module method.
      fn = value.__func__
      if hasattr(fn, 'module_method'):
        def wrapper(*args, **kwargs):
          kwargs['_top_level'] = True
          return self.module._apply_module_method(  # pylint: disable=protected-access
              functools.partial(fn.module_method, self.module),
              self.params, args, kwargs)
        return wrapper
    raise AttributeError(f'No attribute named "{name}".')


class Collection:
  """A collection of tensors useful for tracking state.

  A Collection can be used to associate data with the application of a Module.
  For example a collection can be used to collect activations across modules.
  Another common use case for collections is to track internal state.
  For example, The running averages in BatchNorm can be stored in a collection.

  Attributes:
    state: the initial state by default an empty collection is created.
    shared: If true, a module with shared parameters
            will also share its slot in the collection (default: False).
  """

  def __init__(self, state=None, shared=False):
    if state is None:
      state = {}
    self.state = state
    self.shared = shared
    self._mutable = False

  def _path_prefix(self, relative):
    if relative and len(_module_stack) > 1:
      return _path_from_frames(_module_stack, include_index=not self.shared)
    else:
      return ''

  def as_dict(self, relative=True):
    """Returns a dictionary with module paths as keys and the stored values.

    Args:
      relative: whether the path should be relative to the module that is
          currently applied (default: True). This argument only has an effect
          within the apply function of a Module.
    Returns:
      The stored values as a dictionary.
    """
    prefix = self._path_prefix(relative)
    result = {}
    for key, value in self.state.items():
      if key.startswith(prefix):
        relative_key = key[len(prefix):]
        result[relative_key] = value
    return result

  def lookup(self, path, relative=True):
    """Lookup a single value stored in the collection.

    Args:
      path: the queried path (eg. '/module/sub_module/conv'). When a module is
        called multiple times the call can be specified using a prefix colon
        (eg. '/module/shared_module:1/conv').
      relative: whether the path should be relative to the module that is
        currently applied (default: True). This argument only has an effect
        within the apply function of a Module.
    Returns:
      The value stored in the collection if any, otherwise None.
    """
    def colonize(part):
      if not part or ':' in part:
        return part
      else:
        return part + ':0'
    prefix = self._path_prefix(relative)
    path = prefix + '/'.join(map(colonize, path.split('/')))
    return self.state.get(path, None)

  @contextlib.contextmanager
  def mutate(self):
    # pylint: disable=protected-access
    new_col = jax.tree_map(lambda x: x, self)  # clone the collection
    new_col._mutable = True
    try:
      yield new_col
    finally:
      new_col._mutable = False

  def retrieve(self, default=None):
    """Retrieves a value from the Collection.

    This functions should only be called with the apply function of a module.
    Args:
      default: The default returned when nothing is stored (defualt: None)
    Returns:
      The value previously stored in the collection.
    """
    _top_frame('retrieve')
    path = _path_from_frames(_module_stack, include_index=not self.shared)
    return self.state.get(path, default)

  def store(self, value):
    """Stores a value in the Collection.

    This functions should only be called with the apply function of a module.
    Args:
      value: The value to be stored in the collection
    Returns:
      The previous value stored in the collection or None.
    """
    _top_frame('store')
    if not self._mutable:
      raise ValueError('Collection is not mutable. Use the `mutate` method to'
                       'create a mutable copy.')
    path = _path_from_frames(_module_stack, include_index=not self.shared)
    old_value = self.state.get(path, None)
    self.state[path] = value
    return old_value


def _iterate_collection(collection):
  if collection._mutable:  # pylint: disable=protected-access
    raise ValueError('A mutable collection should not be transformed by Jax.')
  return (collection.state,), collection.shared


def _collection_from_iterable(shared, state):
  return Collection(state[0], shared=shared)

# make sure a collection is traced.
jax.tree_util.register_pytree_node(Collection,
                                   _iterate_collection,
                                   _collection_from_iterable)


def _collection_state_dict(collection):
  return collection.as_dict()


def _collection_from_state_dict(collection, state):
  return Collection(state, shared=collection.shared)


serialization.register_serialization_state(
    Collection, _collection_state_dict, _collection_from_state_dict)

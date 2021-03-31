# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  NN base modules for JAX."""

from typing import Type

import abc
import contextlib
import functools
import hashlib
import inspect
from typing import Any
import warnings

from . import utils
from . import stochastic
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
    xs = module_output_tracker.retrieve(default=[])
    xs.append(x)
    module_output_tracker.store(xs)


class _ModuleFrame:
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A ModuleFrame the context needed to initialize (init) or apply a Module.

  In particular, `self.params` is a dictionary where parameters are
  stored (during module init) and read from (during module application).

  When `module.init()` is first called, a new ModuleFrame is created with
  an empty `params` dictionary. When `self.param` is called within that
  module, a new key is added to track that parameter, with the computed
  parameter's initial value.

  When a module calls into a submodule, a new key is added, with a value
  being an empty dictionary. Then that new dictionary is passed in as `params`
  on a new sub-ModuleFrame. That new sub-ModuleFrame keeps track of its parent
  with the `parent` attribute.

  When the whole init process is complete, the top-level ModuleFrame'
  `params` are returned, which contain a nested dictionary of parameters.

  During module application, a similar process happens but this time
  the parameters are only read from.

  Additional attributes on ModuleFrame track context needed to assist error
  handling, shared parameters and transparent modules that are wrapped without
  creating additional sub-parameters. TODO: Consider elaborating on this
  last paragraph.
  """

  def __init__(self, name,
               parent=None, params=None, rng=None,
               transparent=False):
    if params is None:
      params = {}
    self.parent = parent
    self.rng = rng
    self.params = params
    self.shared = {}
    self.shared_names = set()
    self.name = name
    self.transparent = transparent

    self._name_counter = 0

  @property
  def is_init(self):
    return self.rng is not None

  @property
  def path(self):
    """Returns the path of the Module scope.

    Paths are similar to Unix file names (e.g. '/module/nested/dense')

    Returns:
      The path of this Module scope.
    """
    if self.parent is None:
      if self.name is None:
        return '/'
      else:
        return '/' + self.name

    path = self.parent.path
    if not self.parent.transparent:
      if path[-1] != '/':
        path += '/'
      path += self.name
    return path

  def is_descendent_of(self, frame):
    """Checks whether this frame is a descendent of the given frame."""
    if frame is self.parent:
      return True
    elif self.parent:
      return self.parent.is_descendent_of(frame)
    else:
      return False

  def create_name(self):
    name = str(self._name_counter)
    self._name_counter += 1
    return name


def module_method(fn):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Decorates a function as a module method.

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

    y, initial_params = MyLinearModule.init(rng, x)
    model = nn.Model(MyLinearModule, initial_params)
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

  cache = {}

  # Module methods are just Module class instances.
  # But we want it to inherit from the class such that we can call other methods
  # of the module. We need a class property to find out which class the method
  # is defined on.
  def wrapper(cls):
    if cls not in cache:
      class ModuleMethod(cls):
        apply = fn
      ModuleMethod.__name__ = fn.__name__
      ModuleMethod.__qualname__ = f'{cls.__qualname__}.{fn.__name__}'
      cache[cls] = ModuleMethod
    return cache[cls]

  return utils.classproperty(wrapper)


def _fn_parameters(fn):
  return tuple(inspect.signature(fn).parameters.values())


MODULE_CLASSMETHODS = [
    'create', 'create_by_shape', 'init', 'init_by_shape', 'call', 'partial'
]


class _ModuleMeta(abc.ABCMeta):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A meta class for automatically setting the doc of Modules."""

  def __init__(cls, name, bases, attrs):
    super(_ModuleMeta, cls).__init__(name, bases, attrs)
    apply_fn = cls.apply  # pytype: disable=attribute-error
    apply_doc = apply_fn.__doc__
    cls.__doc__ = apply_doc
    apply_params = _fn_parameters(apply_fn)
    cls.__signature__ = inspect.signature(cls).replace(
        parameters=apply_params[1:])

    if not bases:
      return  # skip method signature overrides for Module class.

    def wrap_special_method(name):
      """Overrides the signature and docstring for one of Module's class methods."""
      orig_fn = getattr(Module, name)  # pytype: disable=name-error

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


def _fold_in_str(rng, data):
  """Folds a string into a jax.random.PRNGKey using its SHA-1 hash."""
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big', signed=True)
  return random.fold_in(rng, hash_int)


class Module(metaclass=_ModuleMeta):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Functional modules."""

  def __new__(cls, *args, name=None, **kwargs):
    warnings.warn("The `flax.nn` module is Deprecated, use `flax.linen` instead. Learn more and find an upgrade guide at https://github.com/google/flax/blob/master/flax/linen/README.md", DeprecationWarning)
    # DO NOT REMOVE - Marker for internal logging.
    if not _module_stack:
      raise ValueError('A Module should only be instantiated directly inside'
                       ' another module.')
    parent = cls._get_construction_frame()
    apply_kwargs = cls._extend_kwargs(kwargs)
    if name is None:
      name = cls._default_name()
    elif cls._is_shared():
      raise ValueError('Cannot override the name of a shared module')
    if name is None:  # also no default name
      name = cls.__name__ + '_' + parent.create_name()
    cls._check_name(name, parent)
    if parent.is_init and name not in parent.params:
      with jax.core.eval_context():
        rng = _fold_in_str(parent.rng, name)
      params = {}
      parent.params[name] = params
    else:  # apply
      if name not in parent.params:
        raise ValueError(f'No module named {name} was created during'
                         ' initialization.')
      params = parent.params[name]
      rng = None
    frame = _ModuleFrame(name, parent=parent, rng=rng, params=params,
                         transparent=cls._is_transparent())
    with cls._with_instance(frame) as instance:
      y = instance.apply(*args, **apply_kwargs)
      _track_outputs(y)
    return y

  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  @classmethod
  def shared(class_, *, name=None, **kwargs):
    """Partially applies a module and shared parameters for each call.

    Args:
      name: name of this module.
      **kwargs: keyword arguments that should be partially applied.
    Returns:
      A subclass of Module that shares parameters when called multiple times.
    """
    if not _module_stack:
      raise ValueError(
          'The shared module should be used during Module application')

    parent = _module_stack[-1]
    if name is None:
      name = parent.create_name()
    if name in parent.shared_names:
      raise ValueError(f'Shared module named "{name}" already exists.')
    parent.shared_names.add(name)

    partial_module = class_.partial(**kwargs)

    class SharedModule(partial_module):
      """Wraps a module to enable shared parameters."""

      @classmethod
      def _default_name(cls):
        return name

      @classmethod
      def _is_shared(cls):
        return True

      @classmethod
      def _get_construction_frame(cls):
        return parent

    SharedModule.__name__ = class_.__name__
    SharedModule.__qualname__ = class_.__qualname__

    return SharedModule

  @classmethod
  def _get_construction_frame(cls):
    """Returns the ModuleFrame where this module was constructed.

    Modules can be shared across different parts of a parameter tree.
    We need to ensure that the parameter object is the same in every instance
    of the same shared module. We resolve this by deciding on a canonical
    ModuleFrame (corresponding to a particular part of the top-level parameter
    tree) where parameters are stored. Concretely, it is the
    "construction frame" -- that is, the frame in which the module is first
    defined. For non-shared modules, that's where it's called. For shared
    modules, it's where `submodule.shared(...)` is called (which may or may
    not be the frame in which it is used.)

    Returns:
      The ModuleFrame instance where this module was constructed.
    """
    return _module_stack[-1]

  @classmethod
  def partial(class_, *, name=None, **kwargs):
    """Partially applies a module with the given arguments.

    Unlike `functools.partial` this will return a subclass of Module.

    Args:
      name: the name used the module
      **kwargs: the argument to be applied.
    Returns:
      A subclass of Module which partially applies the given keyword arguments.
    """

    class PartialModule(class_):
      """Wraps a module with partial application."""

      @classmethod
      def _default_name(cls):
        if name is not None:
          return name
        else:
          return super()._default_name()

      @classmethod
      def _extend_kwargs(cls, invoke_kwargs):
        extended_kwargs = kwargs.copy()
        extended_kwargs.update(invoke_kwargs)
        return super()._extend_kwargs(extended_kwargs)
    # __doc__ is handled by the Module meta class
    PartialModule.__name__ = class_.__name__
    PartialModule.__qualname__ = class_.__qualname__

    return PartialModule

  @classmethod
  def create(cls, _rng, *args, name=None, **kwargs):
    """Creates a module instance by evaluating the model.

    DEPRECATION WARNING:
    `create()` is deprecated use `init()` to initialize parameters and
    then explicitly create a `nn.Model` given the module and initialized
    parameters.

    Use create_by_shape instead to initialize without doing computation.
    Initializer functions can depend both on the shape and the value of inputs.

    Args:
      _rng: the random number generator used to initialize parameters.
      *args: arguments passed to the module's apply function
      name: name of this module
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and an instance of Model
    """
    warnings.warn("`create()` will be removed soon."
                  " Use `init()` to initialize parameters and then explicitly"
                  " create a `nn.Model` given the module and initialized"
                  " parameters.",
                  DeprecationWarning)
    y, params = cls.init(_rng, *args, name=name, **kwargs)
    model = Model(cls, params)
    return y, model

  @classmethod
  def create_by_shape(cls, _rng, input_specs, *args, name=None, **kwargs):
    """Creates a module instance using only shape and dtype information.

    DEPRECATION WARNING:
    `create_by_shape()` is deprecated use `init_by_shape()` to initialize
    parameters and then explicitly create a `nn.Model` given the module and
    initialized parameters.


    This method will initialize the model without computation.
    Initializer functions can depend on the shape but not the value of inputs.

    Args:
      _rng: the random number generator used to initialize parameters.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs
      *args: other arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and an instance of Model
    """
    warnings.warn("`create_by_shape()` will be removed soon."
                  " Use `init_by_shape()` to initialize parameters and then"
                  " explicitly create a `nn.Model` given the module and "
                  " initialized parameters.",
                  DeprecationWarning)

    y, params = cls.init_by_shape(_rng, input_specs, *args, name=name, **kwargs)
    model = Model(cls, params)
    return y, model

  @classmethod
  def init(cls, _rng, *args, name=None, **kwargs):
    """Initializes the module parameters.

    Args:
      _rng: the random number generator used to initialize parameters.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    """
    kwargs = cls._extend_kwargs(kwargs)
    if _module_stack:
      parent = _module_stack[-1]
    else:
      parent = None
    if name is None:
      name = cls._default_name()

    frame = _ModuleFrame(name, rng=_rng, parent=parent,
                         transparent=cls._is_transparent())
    with cls._with_instance(frame) as instance:
      y = instance.apply(*args, **kwargs)
      _track_outputs(y)
    return y, cls._post_process_params(frame.params)

  @classmethod
  def init_by_shape(cls, _rng, input_specs, *args, name=None, **kwargs):
    """Initialize the module parameters.

    This method will initialize the module parameters without computation.
    Initializer functions can depend on the shape but not the value of inputs.
    
    Example::

      input_shape = (batch_size, image_size, image_size, 3)
      model_output, initial_params = model.init_by_shape(jax.random.PRNGKey(0),
                                      input_specs=[(input_shape, jnp.float32)])

    Args:
      _rng: the random number generator used to initialize parameters.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      A pair consisting of the model output and the initialized parameters
    """
    stochastic_rng = None
    try:
      stochastic_rng = stochastic.make_rng()
    except ValueError:
      # Either there is no stochastic scope or the current
      # scope is invalid due to another jax transformation.
      # In both cases we should not try to lift the stochastic
      # scope into the lazy evaluation
      pass

    def lazy_init(*inputs):
      def init_fn():
        return cls.init(_rng, *(inputs + args), name=name, **kwargs)
      if stochastic_rng is not None:
        # Create a new stochastic scope inside the lazy evaluation
        # this way we can use a stochastic scope in combination
        # with init_by_shape.
        with stochastic.stochastic(stochastic_rng):
          return init_fn()
      else:
        return init_fn()
    return jax_utils.partial_eval_by_shape(lazy_init, input_specs)

  @classmethod
  def call(cls, params, *args, name=None, **kwargs):
    """Evaluate the module with the given parameters.

    Args:
      params: the parameters of the module. Typically, initial parameter values
        are constructed using `Module.init` or `Module.init_by_shape`.
      *args: arguments passed to the module's apply function
      name: name of this module.
      **kwargs: keyword arguments passed to the module's apply function
    Returns:
      The output of the module's apply function.
    """
    params = cls._pre_process_params(params)
    kwargs = cls._extend_kwargs(kwargs)
    if _module_stack:
      parent = _module_stack[-1]
    else:
      parent = None
    if name is None:
      name = cls._default_name()
    frame = _ModuleFrame(name, params=params, parent=parent,
                         transparent=cls._is_transparent())
    with cls._with_instance(frame) as instance:
      y = instance.apply(*args, **kwargs)
      _track_outputs(y)
    return y

  def param(self, name, shape, initializer):
    """Defines a parameter within the module's apply function.

    Args:
      name: The name of the parameter.
      shape: The shape of the parameter. If None the param be any type.
      initializer: An initializer function
                   taking an RNG and the shape as arguments.
    Returns:
      The value of the parameter.
    """
    frame = self._frame
    if frame.is_init:
      if name in frame.params:
        raise ValueError(
            "Name '%s' was already used for another parameter." % name)
      with jax.core.eval_context():
        key = _fold_in_str(frame.rng, name)
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
    frame = self._frame
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

    By default, Modules are stateless. See `flax.nn.stateful` to enable stateful
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
    init_frames = [f for f in _module_stack if f.is_init]
    if initializer is not None and init_frames:
      # use the closest frame that is initializing to get an rng
      init_frame = init_frames[-1]
      with jax.core.eval_context():
        init_frame.rng, key = random.split(init_frame.rng)
        init_value = initializer(key, shape)
        state.value = init_value
    return state

  def is_stateful(self):
    return is_stateful()

  def is_initializing(self):
    _top_frame('is_initializing')
    return self._frame.is_init

  @classmethod
  @contextlib.contextmanager
  def _with_instance(cls, frame):
    """A private constructor for Module.

    A module instance is constructed using a scope and is tied to a _ModuleFrame
    This way the methods on the Module instance can rely on the _ModuleFrame
    being available.

    Args:
      frame: an instance of _ModuleFrame
    Yields:
      An instance of Module
    """
    instance = object.__new__(cls)
    instance._frame = frame  # pylint: disable=protected-access
    with _module_stack.frame(frame):
      yield instance

  @classmethod
  def _check_name(cls, name, parent):
    """Check whether the name of the module is valid within the parent scope."""
    if name is not None:
      if not isinstance(name, str):
        raise ValueError('Name must be a string.')
      if '/' in name or ':' in name:
        raise ValueError('Name should not contain slashes or colons.')
    shared = cls._is_shared()
    if name in parent.shared:
      # a module with this name already exists. Check validity of sharing
      if shared != parent.shared[name]:
        raise ValueError(f'The name "{name}" is used for both a shared'
                         ' and unshared module.')
      if not parent.shared[name]:
        raise ValueError(f'A module with named "{name}" already exists.')
    parent.shared[name] = shared

  @classmethod
  def _extend_kwargs(cls, kwargs):
    return kwargs

  @classmethod
  def _pre_process_params(cls, params):
    return params

  @classmethod
  def _post_process_params(cls, params):
    return params

  @classmethod
  def _is_transparent(cls):
    return False

  @classmethod
  def _is_shared(cls):
    return False

  @classmethod
  def _default_name(cls):
    return None


def module(fun):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Convert a function into the apply method of a new Module.

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


# TODO(flax-dev) consider removing this...
class TransparentModule(Module):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A transparent module.

  A transparent module can only have one parameter named '0'.
  """

  @classmethod
  def _pre_process_params(cls, params):
    return {'0': params}

  @classmethod
  def _post_process_params(cls, params):
    entries = list(params.items())
    if len(entries) != 1:
      raise ValueError('Transparent modules should have exactly one child.')
    key, value = entries[0]
    if key != '0':
      raise ValueError('Transparent modules should contain an unnamed child.')
    return value

  @classmethod
  def _is_transparent(cls):
    return True


class TruncatedModule(TransparentModule):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Wraps a Module and returns the requested intermediate outputs instead.

  Check `Model.truncate_at` for a simple API to get the intermediate outputs of
  an existing Model.
  """

  def apply(self, *args, wrapped_module=None, truncate_path=None, **kwargs):
    """Applies the wrapped module and return some of its intermediate outputs.

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
          '`wrapped_module` and `truncate_path` are required keyword arguments')
    with capture_module_outputs() as module_outputs:
      wrapped_module(*args, **kwargs, name='0')

    def lookup_output(path):
      return module_outputs[path]
    return jax.tree_map(lookup_output, truncate_path)


@contextlib.contextmanager
def capture_module_outputs():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A context manager that captures all model outputs.

  Yields:
    A `flax.nn.Collection` containing all module outputs.
  """
  with Collection().mutate() as module_outputs:
    with _module_output_trackers.frame(module_outputs):
      yield module_outputs


class ModuleState():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Tracks a state variable.

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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A context manager for stateful computations.

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
      _, initial_params = MyModel.init(rng, x)
      model = nn.Model(MyModel, initial_params)

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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns true if a stateful scope is currently active (see `flax.nn.stateful`)."""
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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md

  A Model contains the model parameters, state and definition."""

  module: Type[Module] = struct.field(pytree_node=False)
  params: Any = struct.field(pytree_node=True)

  def __call__(self, *args, **kwargs):
    return self.module.call(self.params, *args, **kwargs)

  def truncate_at(self, module_path):
    """Truncates the model by returning the outputs of the given sub-module.

    Args:
      module_path: the full name of the module (e.g. '/module/sub_module').
        A list or dict of module paths can be provided to obtain the
        intermediate outputs of multiple modules.
    Returns:
      A new model with the truncated outputs. If module_path is a pytree of
      paths the outputs will be have the same structure where each path is
      replaced by the corresponding intermediate output.
    """
    truncated_module_cls = TruncatedModule.partial(
        wrapped_module=self.module, truncate_path=module_path)
    return self.replace(module=truncated_module_cls)

  def __getattr__(self, name):
    value = getattr(self.module, name)
    if inspect.isclass(value) and issubclass(value, Module):
      def wrapper(*args, **kwargs):
        return value.call(self.params, *args, **kwargs)
      return wrapper
    raise AttributeError(f'No attribute named "{name}".')

  def __hash__(self):
    # Jax will call hash when the model is passed to a function transform.
    # The compiled function should not be shared among model instances because
    # it closes over the specific parameters of this model instance.
    return id(self)


class Collection:
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A collection of tensors useful for tracking state.

  A Collection can be used to associate data with the application of a Module.
  For example, a collection can be used to collect activations across modules.
  Another common use case for collections is to track internal state.
  For example, the running averages in BatchNorm can be stored in a collection.

  Attributes:
    state: the initial state by default an empty collection is created.
  """

  def __init__(self, state=None):
    if state is None:
      state = {}
    self.state = state
    # The anchor is used to determine the prefix of the collection.
    # This way we can create/nest collections inside modules.
    self._anchor = _module_stack[-1] if _module_stack else None

    self._mutable = False
    self._master_level = None
    self._root = None

  def as_dict(self):
    """Returns a dictionary with module paths as keys and the stored values.

    Returns:
      The stored values as a dictionary.
    """
    return self.state.copy()

  def __getitem__(self, key):
    return self.state[key]

  @contextlib.contextmanager
  def mutate(self):
    # pylint: disable=protected-access
    new_col = jax.tree_map(lambda x: x, self)  # clone the collection
    new_col._mutable = True
    new_col._master_level = utils._trace_level(utils._current_trace())
    try:
      yield new_col
    finally:
      new_col._mutable = False

  def retrieve(self, default=None):
    """Retrieves a value from the Collection.

    This function should only be called with the apply function of a module.
    
    Args:
      default: The default returned when nothing is stored (default: None)
    Returns:
      The value previously stored in the collection.
    """
    _top_frame('retrieve')
    path = self._current_path()
    return self.state.get(path, default)

  def store(self, value):
    """Stores a value in the Collection.

    This function should only be called with the apply function of a module.
    
    Args:
      value: The value to be stored in the collection
    Returns:
      The previous value stored in the collection or None.
    """
    frame = _top_frame('store')
    if not self._mutable:
      raise ValueError('Collection is not mutable. Use the `mutate` method to'
                       ' create a mutable copy.')
    # Use the Jax TraceMaster to determine if a Collection is modified from
    # inside a nested jax transformation.
    # In this case, we throw an error because transforming a stateful function
    # is ill-defined (eg. what does vmap of BatchNorm do?).
    # TODO(jheek): Add doc guide on combining jax transforms and state.
    # TODO(jheek): Should some transformations be exempt from this error?
    value_level = utils._level_of_value(value)
    if value_level > self._master_level:
      raise ValueError('Stateful operations are not allowed when the Collection'
                       ' is created outside of the current Jax transformation')

    # The root of a Collection is the first module scope that gets created
    # inside the mutate scope of the Collection. By allowing only one unique
    # root scope, we guarantee that state is not accidentally shared
    # between different models. When a user specifies an explicit name we can
    # distinguish models and a collection can have multiple roots.
    if frame == self._anchor:
      # Example:
      # with nn.Collection.mutate() as coll:
      #   coll.store(1)
      raise ValueError('State should be stored from within a module.'
                       ' Consider using the value directly instead of'
                       ' storing it in a Collection.')
    if not frame.is_descendent_of(self._anchor):
      # edge case where the Collection cannot capture the scope of a shared Module
      # See test_collection_store_fails_if_out_of_scope in nn_test.py
      raise ValueError('Trying to capture state outside the scope of this Collection.'
                       ' Most likely due to passing around a shared Module.')
    root = self._find_root(frame)
    if self._root is None:
      self._root = root
    elif self._root != root:
      if self._root.name is None or root.name is None:
        # In the following examples, should the two calls to `StatefulModule` share state or not?
        # Because it's ambiguous, we throw an error and require the user to explicitly separate state
        # by giving each instance a separate name, or to explicitly pass the same name
        # in order to share state.
        # with nn.statefull(state) as new_state:
        #   StatefulModule.call(params)
        #   StatefulModule.call(params2)
        raise ValueError('When multiple top-level module calls use a Collection'
                         ' each top-level module should have a name.')
    path = self._current_path()
    old_value = self.state.get(path, None)
    self.state[path] = value
    return old_value

  def _find_root(self, frame):
    """Finds the root frame with respect to the anchor.

    The root frame is defined as the child of anchor
    that is an ancestor of a frame.
    The root is used to verify that a Collection does not
    have multiple unnamed roots.

    Args:
      - frame: the frame of which we want to know the root
    Returns:
      The root of the given frame.
    """
    assert frame.is_descendent_of(self._anchor)
    root = frame
    while root.parent != self._anchor:
      root = root.parent
    return root

  def _current_path(self):
    """"The relative path from the currently active module scope to the root of the collection.

    For example: If a collection is created in the path '/module/nested' and
    something is stored by a module with the path '/module/nested/block/conv'
    the key in the collection dict will be '/block/conv'.

    Returns:
      the relative path of the active module scope.
    """
    frame = _module_stack[-1]
    assert frame.is_descendent_of(self._anchor)
    path = _module_stack[-1].path
    if self._anchor is not None and self._anchor.path != '/':
      prefix = self._anchor.path
      assert prefix == path[:len(prefix)]
      return path[len(prefix):]
    else:
      return path

def iterate_collection(collection):
  # jax iterates through pytrees for each argument/return value of a functional
  # transformations. When the collection is mutable we throw an error this way
  # we avoid silent errors due to impurity of a traced function.
  if collection._mutable:  # pylint: disable=protected-access
    raise ValueError('A mutable collection should not be transformed by Jax.')
  meta = (type(collection), collection._anchor)  # pylint: disable=protected-access
  return (collection.state,), meta


def collection_from_iterable(meta, state):
  ty, anchor = meta
  coll = ty(state[0])
  coll._anchor = anchor  # pylint: disable=protected-access
  return coll

# make sure a collection is traced.
jax.tree_util.register_pytree_node(Collection,
                                   iterate_collection,
                                   collection_from_iterable)


def _collection_state_dict(collection):
  return serialization._dict_state_dict(collection.as_dict())  # pylint: disable=protected-access


def _collection_from_state_dict(xs, state):
  restored_state = serialization._restore_dict(xs.as_dict(), state)  # pylint: disable=protected-access

  return Collection(restored_state)


serialization.register_serialization_state(
    Collection, _collection_state_dict, _collection_from_state_dict)

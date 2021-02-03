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

"""Flax Modules."""
from contextlib import contextmanager
import dataclasses
import functools
import inspect
import os
import threading
from typing import (Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)

from flax.errors import CallCompactUnboundModuleError, InitModuleInvalidRngError, JaxOmnistagingError
import jax
from jax import tree_util
import numpy as np

import flax
from flax import traverse_util
from flax import serialization
from flax.core import Scope, apply
from flax.core.scope import CollectionFilter, Variable, VariableDict
from flax.core.frozen_dict import FrozenDict, freeze

# from .dotgetter import DotGetter

PRNGKey = Any  # pylint: disable=invalid-name
RNGSequences = Dict[str, PRNGKey]
Array = Any    # pylint: disable=invalid-name
T = TypeVar('T')

# pylint: disable=protected-access,attribute-defined-outside-init

def _check_omnistaging():
  if not jax.config.omnistaging_enabled:
    raise JaxOmnistagingError()


def _indent(x: str, num_spaces: int):
  indent_str = ' ' * num_spaces
  lines = x.split('\n')
  # skip last line because it is always empty and should not be indented.
  assert lines[-1] == ''
  return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


def _attr_repr(value: Any):
  if isinstance(value, Callable) and getattr(value, '__name__', None):
    value_rep = value.__name__
  else:
    value_rep = repr(value)
  return value_rep


def _module_repr(module: 'Module', num_spaces: int = 4):
  """Returns a pretty printed representation of the module"""
  cls = type(module)
  cls_name = cls.__name__
  rep = ''
  attributes = {k: v for k, v in cls.__annotations__.items()
                if k not in ('parent', 'name')}
  child_modules = {k: v for k, v in module._state.children.items()  # pytype: disable=attribute-error
                   if isinstance(v, Module)}
  if attributes:
    rep += '# attributes\n'
    for attr in attributes.keys():
      # TODO(jheek): can we get a nice string representation of attribute types?
      value = getattr(module, attr, None)
      value_rep = _attr_repr(value)
      rep += f'{attr} = {value_rep}\n'
  if child_modules:
    rep += '# children\n'
    for name, child in child_modules.items():
      child_rep = _module_repr(child, num_spaces)
      rep += f'{name} = {child_rep}\n'
  if rep:
    return f'{cls_name}(\n{_indent(rep, num_spaces)})'
  else:
    return f'{cls_name}()'


# Track parent relationship across Modules.
# -----------------------------------------------------------------------------
class _DynamicContext:
  # TODO: switch to using contextvars once minimum python version is 3.7
  def __init__(self):
    self._thread_data = threading.local()
  @property
  def module_stack(self):
    if not hasattr(self._thread_data, 'module_stack'):
      self._thread_data.module_stack = [None,]
    return self._thread_data.module_stack

# The global context 
_context = _DynamicContext()

class _Sentinel:
  pass
_unspecified_parent = _Sentinel()


# Enable automatic named_call wrapping for labelling profile traces.
# -----------------------------------------------------------------------------
_use_named_call = True if os.getenv('FLAX_PROFILE', '') else False

def enable_named_call():
  """Enables named call wrapping for labelling profile traces."""
  global _use_named_call
  _use_named_call = True

def disable_named_call():
  """Disables named call wrapping."""
  global _use_named_call
  _use_named_call = False


# Utilities for pytrees of Modules defined inside setup()
# -----------------------------------------------------------------------------
def _get_suffix_value_pairs(
    tree_or_leaf: Any) -> List[Tuple[str, Type["Module"]]]:
  """Helper for naming pytrees of submodules."""
  dict_or_leaf = serialization.to_state_dict(tree_or_leaf)
  if dict_or_leaf == {} or not isinstance(dict_or_leaf, dict):
    return [('', tree_or_leaf)]
  else:
    flat_dict = traverse_util.flatten_dict(dict_or_leaf)
    return [('_' + '_'.join(k), v) for k, v in flat_dict.items()]


def _all_names_on_object(obj: Any) -> Set[str]:
  """Gets all names of attributes on `obj` and its classes throughout MRO.
  
  Args:
    obj: The object to get names for.
  Returns:
    A set of names of attributes of `obj` and its classes.
  """
  nameset = set(obj.__dict__.keys())
  for cls in obj.__class__.__mro__:
    nameset = nameset.union(set(cls.__dict__.keys()))
  return nameset


def _freeze_attr(val: Any) -> Any:
  if isinstance(val, (dict, FrozenDict)):
    return FrozenDict({k: _freeze_attr(v) for k, v in val.items()})
  elif isinstance(val, (list, tuple)):
    return tuple(_freeze_attr(v) for v in val)
  else:
    return val


# Method wrapping of "compact methods" and setup()
# -----------------------------------------------------------------------------
def compact(fun: Callable) -> Callable:
  """Marks the given module method allowing inlined submodules. 
  
  Methods wrapped in @compact can define submodules directly within the method.

  For instance::

    @compact
    __call__(self, x, features):
      x = nn.Dense(features)(x)
      ...
  
  At most one method in each Module may be wrapped with @compact.

  Args:
    fun: The Module method to mark as compact.
  Returns:
    The given function `fun` marked as compact.
  """
  fun.compact = True
  return fun


def _get_local_method_names(cls: Any, exclude: Iterable[str] = ()) -> Tuple[str]:
  """Gets method names of a class, excluding class and static methods.
  
  Args:
    cls: The class to get method names for.
    excludes: Names to exclude from output.
  Returns:
    A list of method names.
  """
  true_methods = set()
  for m in cls.__dict__:
    if callable(cls.__dict__[m]):
      mtype = type(cls.__dict__[m])
      if mtype != staticmethod and mtype != classmethod:
        true_methods.add(m)
  return tuple(true_methods.difference(set(exclude)))


def wrap_method_once(fun: Callable[..., Any]) -> Callable[..., Any]:
  """Manages Module state for a given user-defined method.
  
  Args:
    fun: User-defined Module method to manage state for.
  Returns:
    Wrapped method.
  """
  # Don't rewrap methods that have already had the state management wrapper
  # applied in the decorator stack.  This wrapper should always be applied
  # before transformation wrappers.
  if hasattr(fun, 'method_handler_wrapped'):
    return fun

  @functools.wraps(fun)
  def wrapped_module_method(self, *args, **kwargs):
    is_compact_method = hasattr(fun, 'compact')
    is_setup_method = fun.__name__ == 'setup'
    # We lazily call setup() only when needed.
    if not is_setup_method:
      self._try_setup()

    if is_compact_method:
      if self.scope is None:
        raise CallCompactUnboundModuleError()
      self._state.in_compact_method = True
    elif is_setup_method:
      self._state.in_setup = True
    _context.module_stack.append(self)
    try:
      return fun(self, *args, **kwargs)
    finally:
      _context.module_stack.pop()
      if is_compact_method:
        object.__setattr__(self, 'scope', self.scope.rewound())
      if is_compact_method or is_setup_method:
        self._state.reset()
  wrapped_module_method.method_handler_wrapped = True
  return wrapped_module_method


def _wrap_hash(hash_fn: Callable[..., Any]) -> Callable[..., Any]:
  @functools.wraps(hash_fn)
  def wrapped(self):
    assert self.scope == None, 'Can\'t call __hash__ on modules that hold variables.'
    return hash_fn(self)
  return wrapped


def _get_unbound_fn(method_or_fn: Callable[..., Any]) -> Callable[..., Any]:
  """Returns an unbound function from a method that is possibly bound.
  
  This means that the returned function does no longer depend on the instance
  of the class, which is passed as it first argument. 

  Args:
    method_or_fn: A class method or function.
  Returns:
    An unbound version of input function.
  """
  if inspect.ismethod(method_or_fn):
    return method_or_fn.__func__  # pytype: disable=attribute-error
  elif callable(method_or_fn):
    return method_or_fn
  assert True, 'Expected a function or method.'


@dataclasses.dataclass
class _ModuleInternalState:
  """Ephemeral Module Evaluation State.

  For clarity, we collect all of the temporary flags and ephemeral state used by
  Modules for autonaming and error messages here, alongside the rules used
  to pass this ephemeral state across transform boundaries.
  """
  in_compact_method: bool = False
  in_setup: bool = False
  setup_called: bool = False
  last_varname: Optional[str] = None
  autoname_cursor: Optional[dict] = dataclasses.field(default_factory=dict)
  children: Dict[str, Union[str, 'Module']] = dataclasses.field(default_factory=dict)

  def reset(self):
    """Resets transient state."""
    self.in_compact_method = False
    self.in_setup = False
    self.last_varname = None
    self.autoname_cursor = dict()

  def export(self):
    """Exports transform-preserved state across transform boundary."""
    cloned = _ModuleInternalState(
      in_compact_method=self.in_compact_method,
      in_setup=self.in_setup,
      setup_called=False,  # setup_called is object local, not shared.
      last_varname=self.last_varname,
      autoname_cursor=dict(self.autoname_cursor))
    return cloned

  def reimport(self, other):
    """Re-imports transform-preserved state from across transform boundary."""
    self.in_compact_method = other.in_compact_method
    self.in_setup = other.in_setup
    self.last_varname = other.last_varname
    self.autoname_cursor = dict(other.autoname_cursor)

_uninitialized_module_internal_state = _ModuleInternalState()


_UNDEFINED_COPY_PICKLE_METHODS = (
    '__getstate__', '__setstate__', '__getnewargs_ex__',
    '__reduce__', '__reduce_ex__', '__copy__', '__deepcopy__')

# Base Module definition.
# -----------------------------------------------------------------------------
class Module:
  """Base class for all neural network modules. Layers and models should subclass this class.

  All Flax Modules are Python 3.7 
  `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_. Since
  dataclasses take over ``__init__``, you should instead override :meth:`setup`,
  which is automatically called to initialize the module.

  Modules can contain submodules, and in this way can be nested in a tree
  structure. Submodels can be assigned as regular attributes inside the
  :meth:`setup` method.

  You can define arbitrary "forward pass" methods on your Module subclass.
  While no methods are special-cased, ``__call__`` is a popular choice because
  it allows you to use module instances as if they are functions::

    from flax import linen as nn

    class Module(nn.Module):
      features: Tuple[int] = [16, 4]

      def setup(self):
        self.dense1 = Dense(self.features[0])
        self.dense2 = Dense(self.features[1])

      def __call__(self, x):
        return self.dense2(nn.relu(self.dense1(x)))

  Optionally, for more concise module implementaions where submodules 
  definitions are co-located with their usage, you can use the 
  :meth:`compact` wrapper.
  """

  def __init__(*args, **kwargs):
    # this stub makes sure pytype accepts constructor arguments.
    pass

  @classmethod
  def __init_subclass__(cls):
    """Automatically initializes all subclasses as custom dataclasses."""
    # All Flax Modules are dataclasses.  We force this convention since
    # it encourages the stateless behavior needed to clone module instances for
    # functional transformation.  Instead of using a python metaclass, we
    # automatically transform Modules into dataclasses at subclass creation
    # time, and we set the last dataclass arguments to `parent` and `name`.
    cls._customized_dataclass_transform()
    # We wrap user-defined methods including setup and __call__ to enforce
    # a number of different checks and to provide clear error messages.
    cls._verify_single_or_no_compact()
    cls._wrap_module_methods()
    # Set empty class defaults.
    cls._state = _uninitialized_module_internal_state
    cls.scope = None

  @classmethod
  def _customized_dataclass_transform(cls):
    """Handles final optional dataclass attributes: `parent` and `name`."""
    # Use cls.__dict__ to get annotations of cls itself (no parent class).
    annotations = dict(cls.__dict__.get('__annotations__', {}))
    if 'parent' in annotations or 'name' in annotations:
      raise ValueError(
          f'properties `parent` and `name` are reserved: {annotations}')
    # Add `parent` and `name` default fields at end.
    # We temporarily modify base class __dataclass_fields__ to force desired
    # argument behavior and ordering from dataclass class-transform.
    parent_dataclass_fields = dict(getattr(cls, '__dataclass_fields__', {}))
    # Remove 'parent' and 'name' from parents because we always want parent and
    # name to show up last in the dataclass args.
    if 'parent' in parent_dataclass_fields:
      cls.__dataclass_fields__.pop('parent')  # pytype: disable=attribute-error
    if 'name' in parent_dataclass_fields:
      cls.__dataclass_fields__.pop('name')  # pytype: disable=attribute-error
    annotations['parent'] = Union[Type["Module"], Type["Scope"],
                                  Type["_Sentinel"], None]
    cls.parent = dataclasses.field(repr=False, default=_unspecified_parent)
    annotations['name'] = str
    cls.name = None  # default value of name is None.
    cls.__annotations__ = annotations
    # Now apply dataclass transform (which operates in-place).
    dataclasses.dataclass(cls, unsafe_hash=True, repr=False)  # pytype: disable=wrong-keyword-args
    cls.__hash__ = _wrap_hash(cls.__hash__)
    # Restore original base class __dataclass_fields__.
    if dataclasses.is_dataclass(cls.__bases__[0]):
      cls.__bases__[0].__dataclass_fields__ = parent_dataclass_fields

  @classmethod
  def _verify_single_or_no_compact(cls):
    """Statically verifies that at most a single method is labelled compact."""
    methods = [m[0] for m in inspect.getmembers(cls, predicate=callable)]
    n_compact_fns = len([method_name for method_name in methods
                         if hasattr(getattr(cls, method_name), 'compact')])
    if n_compact_fns > 1:
      raise RuntimeError(
          'Only one method per class can be @compact. You can remove @compact '
          'and define submodules and variables in setup(), or use two '
          'separate modules.')

  @classmethod
  def _wrap_module_methods(cls):
    """Wraps user-defined non-inherited methods with state management functions."""
    exclusions = ([f.name for f in dataclasses.fields(cls)] +
                  ['__eq__', '__repr__', '__init__', '__hash__'])
    for key in _get_local_method_names(cls, exclude=exclusions):
      method = getattr(cls, key)
      wrapped_method = wrap_method_once(method)
      if _use_named_call and key != 'setup':
        # We import named_call at runtime to avoid a circular import issue.
        from flax.linen.transforms import named_call  # pylint: disable=g-import-not-at-top
        wrapped_method = named_call(wrapped_method)
      setattr(cls, key, wrapped_method)
    return cls

  def __setattr__(self, name: str, val: Any):
    """Sets an attribute on this Module.
    
    We overload setattr solely to support pythonic naming via assignment of 
    submodules in the special setup() function::

      self.submodule_name = MyModule(...)

    We also support lists and other general pytrees, e.g.::

      self.submodules = [MyModule0(..), MyModule1(..), ...]

    Args:
      name: Attribute to set.
      val: Value of the attribute.
    """
    if name != '_state' and self._state.setup_called:
      # Raises a TypeError just like frozen python dataclasses.
      raise TypeError("Module instance is frozen outside of setup method.")

    # We don't mess with the parent module.
    if name == 'parent':
      pass
    # Modules have been passed in as dataclass args and set in __init__.
    elif name in self.__dataclass_fields__ and self.__dataclass_fields__[name].init:  # pytype: disable=attribute-error
      pass
    # Submodules are being defined and attached in setup()
    else:
      val = _freeze_attr(val)
      for suffix, subvalue in _get_suffix_value_pairs(val):
        if isinstance(subvalue, Module):
          if not self._state.in_setup:
            raise ValueError(
                "You can only assign submodules to self in setup().")
          if subvalue.parent is _unspecified_parent:
            subvalue.parent = self
          elif subvalue.parent is not self:
            raise ValueError("Can't attach to remote parent in setup, pass in "
                             "bound Modules from outside as an argument.")
          if subvalue.name is not None:
            raise ValueError(
                "In setup, assign names of Modules via self.<name> and not "
                "using keyword argument name=\"<name>\"")
          subvalue.name = f'{name}{suffix}'
          subvalue.__post_init__()
        # val is a parameter array or a Variable reference class.
        elif isinstance(subvalue, (np.ndarray, jax.interpreters.xla.DeviceArray,
                                   Variable)) and self._state.in_setup:
          var_name = f'{name}{suffix}'
          # namecheck to ensure named variable matches self attribute name.
          if (suffix == '' and  # not when assigning lists or dicts
              self._state.last_varname and self._state.last_varname != var_name):
            raise ValueError(f'Variable name {self._state.last_varname} must '
                             f'equal attribute name {var_name}.')
          self._state.last_varname = None
    # Finally, always run default __setattr__ to attach to self.__dict__.
    object.__setattr__(self, name, val)

  def __getattr__(self, name: str) -> Any:
    """Call setup() before getting any setup-defined attributes."""
    # We don't want to return anything for python copy / pickle methods.
    if name in _UNDEFINED_COPY_PICKLE_METHODS:
      raise AttributeError()
    self._try_setup()
    if name in self.__dict__:
      return self.__dict__[name]
    else:
      raise AttributeError(
          f"'{self.__class__.__name__}' object has no attribute '{name}'")

  def __dir__(self) -> List[str]:
    """Call setup() before listing attributes."""
    self._try_setup()
    return object.__dir__(self)  # pytype: disable=attribute-error

  def __post_init__(self):
    _check_omnistaging()
    # In dataclasses, __init__ is overridden to process dataclass arguments,
    # and __post_init__ is called immediately afterwards. Here, depending on the
    # type of `parent` passed to initialize the Module, we either defer 
    # initialization, attach this Module as a submodule of a parent, or bind
    # this Module at the top-level to variables and rngs.

    self._state = _ModuleInternalState()

    # Typically we set the parent based on the dynamic module context.
    if self.parent is _unspecified_parent:  # pytype: disable=attribute-error
      self.parent = _context.module_stack[-1]

    # Initialization is deferred for top level Modules or any other "orphan"
    # Modules until attachment by __setattr__ i.e. MyModule(..., parent=None)
    if self.parent is None:
      return

    # Register submodule on parent Module.
    if isinstance(self.parent, Module):
      # When initializing an unnamed Module inside setup()
      # initialization is deferred until attachment by __setattr__
      # i.e. self.mymodule = MyModule(...)
      if self.parent._state.in_setup and self.name is None:  # pytype: disable=attribute-error
        return
      if not self.parent._initialization_allowed:
        raise ValueError(
            'Submodules must be defined in `setup()` or in a method wrapped '
            'in `@compact`')
      # Autonaming of submodules.
      if self.name is None:  # pytype: disable=attribute-error
        prefix = f"{self.__class__.__name__}"
        cursor = self.parent._state.autoname_cursor.get(prefix, 0)
        self.name = f"{prefix}_{cursor}"
        self.parent._state.autoname_cursor[prefix] = cursor + 1
      if self.parent._name_taken(self.name):
        raise ValueError(
            f"A variable of name {self.name} exists already, or "
            f"trying to share submodule {self.__class__.__name__} by name "
            f"{self.name}. To share submodules, store module instances as a"
            f" Python object or as an attribute on self and reuse.")
      self.parent._state.children[self.name] = self
      self.scope = self.parent.scope.push(self.name)

    # Top-level invocation with a functional Scope.
    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be None, Module or Scope")

  def __repr__(self):
    return _module_repr(self)

  def setup(self):
    """Initializes a Module lazily (similar to a lazy ``__init__``).

    ``setup`` is called once lazily on a module instance when a module
    is bound, immediately before any other methods like ``__call__`` are
    invoked, or before a ``setup``-defined attribute on `self` is accessed.

    This can happen in three cases:

      1. Immediately when invoking :meth:`apply`, :meth:`init` or
         :meth:`init_and_output`.

      2. Once the module is given a name by being assigned to an attribute of
         another module inside the other module's ``setup`` method
         (see :meth:`__setattr__`)::

           class MyModule(nn.Module):
             def setup(self):
               submodule = Conv(...)

               # Accessing `submodule` attributes does not yet work here.

               # The following line invokes `self.__setattr__`, which gives
               # `submodule` the name "conv1".
               self.conv1 = submodule

               # Accessing `submodule` attributes or methods is now safe and
               # either causes setup() to be called once.

      3. Once a module is constructed inside a method wrapped with
         :meth:`compact`, immediately before another method is called or
         ``setup`` defined attribute is accessed.
    """
    pass

  def _try_setup(self):
    """Tries to setup module if scope is available and setup has not been called yet."""
    if self.scope and not self._state.setup_called and not self._state.in_setup:
      try:
        self.setup()
      finally:
        self._state.setup_called = True

  def _name_taken(self, name: str) -> bool:
    return (name in self.scope.reservations or
            name in _all_names_on_object(self))

  @property
  def _initialization_allowed(self):
    return self._state.in_setup or self._state.in_compact_method

  def clone(self, *,
            parent: Optional[Union[Scope, 'Module']] = None,
            **updates) -> 'Module':
    """Creates a clone of this Module, with optionally updated arguments.
    
    Args:
      parent: The parent of the clone. The clone will have no parent if no 
        explicit parent is specified.
      **updates: Attribute updates.
    Returns:
      A clone of the this Module with the updated attributes and parent.
    """
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}
    attrs.update(parent=parent, **updates)
    return self.__class__(**attrs)

  def variable(self, col: str, name: str, init_fn, *init_args) -> Variable:
    """Declares and returns a variable in this Module.

    See :mod:`flax.core.variables` for more information. See also :meth:`param`
    for a shorthand way to define read-only variables in the "params"
    collection.

    Args:
      col: The variable collection name.
      name: The variable name.
      init_fn: The function that will be called to compute the initial value
        of this variable. This function will only be called the first time
        this variable is used in this module.
      *init_args: The arguments to pass to init_fn.

    Returns:
      A :class:`flax.core.variables.Variable` that can be read or set via 
      ".value" attribute. Throws an error if the variable exists already.
    """
    if not self._initialization_allowed:
      raise ValueError(
          'Variables must be initialized in `setup()` or in a method '
          'wrapped in `@compact`')
    if self._name_taken(name):
      raise ValueError(
          f'Name {name} already in use in {self.__class__.__name__}.')
    # ephemeral state for setattr name-equality-check
    self._state.last_varname = name
    v = self.scope.variable(col, name, init_fn, *init_args)
    self._state.children[name] = col
    return v

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    """Declares and returns a parameter in this Module.

    Parameters are read-only variables in the collection named "params". See
    :mod:`flax.core.variables` for more details on variables.

    Args:
      name: The parameter name.
      init_fn: The function that will be called to compute the initial value
        of this variable. This function will only be called the first time
        this parameter is used in this module.
      *init_args: The arguments to pass to init_fn.

    Returns:
      The value of the initialized parameter.
    """
    if not self._initialization_allowed:
      raise ValueError(
          'Parameters must be initialized in `setup()` or in a method '
          'wrapped in `@compact`')
    if self._name_taken(name):
      raise ValueError(
          f'Name {name} already in use in {self.__class__.__name__}.')
    # ephemeral state for setattr name-equality-check
    self._state.last_varname = name
    v = self.scope.param(name, init_fn, *init_args)
    self._state.children[name] = 'params'
    return v

  def has_variable(self, col: str, name: str) -> bool:
    """Checks if a variable of given collection and name exists in this Module.

    See :mod:`flax.core.variables` for more explanation on variables and
    collections.
    
    Args:
      col: The variable collection name.
      name: The name of the variable.
    Returns:
      True if the variable exists.
    """
    if self.scope is None:
      raise ValueError("Can't access variables on unbound modules")
    return self.scope.has_variable(col, name)

  def make_rng(self, name: str) -> PRNGKey:
    """Returns a new RNG key from a given RNG sequence for this Module.
    
    The new RNG key is split from the previous one. Thus, every call to 
    `make_rng` returns a new RNG key, while still guaranteeing full
    reproducibility.

    TODO: Link to Flax RNG design note.

    Args:
      name: The RNG sequence name.
    Returns:
      The newly generated RNG key.
    """
    if self.scope is None:
      raise ValueError("Can't use RNGs on unbound modules")
    return self.scope.make_rng(name)

  def apply(self, variables: VariableDict, *args, rngs: RNGSequences = None,
            method: Callable[..., Any] = None, 
            mutable: Union[bool, str, Sequence[str]] = False,
            **kwargs) -> Union[Any, Tuple[Any, VariableDict]]:
    """Applies a module method to variables and returns output and modified variables.

    Note that `method` should be set if one would like to call `apply` on a 
    different class method than `_call__`. For instance, suppose a Transformer
    modules has a method called `encode`, then the following calls `apply` on
    that method::

      model = models.Transformer(config)
      encoded = model.apply({'params': params}, inputs, method=model.encode) 

    Args:
      variables: A dictionary containing variables keyed by variable
        collections. See :mod:`flax.core.variables` for more details
        about variables.
      rngs: The rngs for the variable collections.
      method: The literal name of a method in this class. If provided, applies
        this method. If not provided, applies the ``__call__`` method.
      mutable: Can be bool, str, or list. Specifies which collections should be
               treated as mutable: ``bool``: all/no collections are mutable.
               ``str``: The name of a single mutable collection. ``list``: A
               list of names of mutable collections.
    Returns:
      If ``mutable`` is False, returns output. If any collections are
      mutable, returns ``(output, vars)``, where ``vars`` are is a dict
      of the modified collections.
    """
    if method is None:
      method = self.__class__.__call__
    else:
      if not inspect.ismethod(method):
        raise ApplyModuleInvalidMethodError(type(self).__name__, method)
      method = _get_unbound_fn(method)
    fn = lambda scope: method(self.clone(parent=scope), *args, **kwargs)
    return apply(fn, mutable=mutable)(variables, rngs=rngs)

  def init_with_output(self, rngs: Union[PRNGKey, RNGSequences], *args,
                       method: Optional[Callable[..., Any]] = None, 
                       **kwargs) -> Tuple[Any, VariableDict]:
    """Initializes a module method with variables and returns output and modified variables.

    Args:
      rngs: The rngs for the variable collections.
      method: An optional method. If provided, applies this method. If not
              provided, applies the ``__call__`` method.
    Returns:
      `(output, vars)``, where ``vars`` are is a dict of the modified
      collections.
    """
    if not isinstance(rngs, dict):
      if rngs.shape != (2,):
        raise InitModuleInvalidRngError(type(self).__name, rngs)
      rngs = {'params': rngs}
    return self.apply(
        {}, *args, rngs=rngs, method=method, mutable=True, **kwargs)

  def init(self, rngs: Union[PRNGKey, RNGSequences], *args,
           method: Optional[Callable[..., Any]] = None, 
           **kwargs) -> VariableDict:
    """Initializes a module method with variables and returns modified variables.

    Jitting `init` initializes a model lazily using only the shapes of the 
    provided arguments, and avoids computing the forward pass with actual 
    values. Example::

      jit_init = jax.jit(SomeModule.init)
      jit_init(rng, jnp.ones(input_shape, jnp.float32))      

    Args:
      rngs: The rngs for the variable collections.
      method: An optional method. If provided, applies this method. If not
              provided, applies the ``__call__`` method.
    Returns:
      The initialized variable dict.
    """
    _, v_out = self.init_with_output(rngs, *args, method=method, **kwargs)
    return v_out

  @property
  def variables(self) -> VariableDict:
    """Returns the variables in this module."""
    if self.scope is None:
      raise ValueError("Can't access variables on unbound modules")
    return self.scope.variables()


  # @contextmanager
  # def mutate(self, mutable=True, **updates):
  #   cloned = self.clone(**updates)
  #   try:
  #     cloned.scope._variables = _unfreeze_variables(
  #         cloned.scope._variables, mutable)
  #     yield cloned
  #   finally:
  #     cloned.scope._variables = freeze(cloned.scope._variables)

  # def initialized(self, rngs, *args, method='__call__', **kwargs):
  #   if self.parent is not None:
  #     raise ValueError("Pattern for initialized is "
  #                      "`Module(parent=None, ...attrs...).initialized(...)`")
  #   scope = Scope(variables={}, rngs=rngs)
  #   with self.mutate(parent=scope) as initialized:
  #     if method is not None:
  #       getattr(initialized, method)(*args, **kwargs)
  #   return initialized

  # @property
  # def variables(self):
  #   """Get a view of Module variables with easy dot-syntax navigation."""
  #   return DotGetter(self.scope.variables())

  # def __getattr__(self, name):
  #   # Used for easy colab/jupyter introspection, and to provide a
  #   # consistent top-level interface to self.<attr> for both simple
  #   # and multi-method modules.
  #   if name in self.children:
  #     val = self.children[name]
  #     if isinstance(val, str):  # variable
  #       return self.variables[val][name]
  #     else:  # submodule
  #       val.scope = self.scope.push(name)
  #       self.scope.reservations.remove(name)
  #       return val
  #   else:
  #     raise AttributeError(
  #         f"'{self.__class__.__name__}' object has no attribute '{name}'")

  # def __dir__(self):
  #   return list(self.children.keys()) + object.__dir__(self)

  # TODO: Should this be what `clone` always does if you don't pass in an explicit
  # parent?
  # def detached(self):
  #   return self.clone(parent=None)

  # # TODO: Consider whether this is a helpful abstraction, and think about naming.
  # # See its use in design_test/linen/weight_std.py
  # def materialized(self, variables={}, rngs={}):
  #   assert self.scope is None, ("Can't attach a module twice."
  #                               " Maybe you want to clone first?")
  #   return self.clone(parent=Scope(variables, rngs))


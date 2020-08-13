"""Linen: a refined Flax."""
import sys
import os
import functools
from importlib import reload
from pprint import pprint
import inspect
from contextlib import contextmanager
import dataclasses
import threading

from typing import (Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar)
import jax
from jax import numpy as jnp, random, lax
from jax import tree_util
import numpy as np

import flax
from flax import nn
from flax.nn import initializers
from flax import traverse_util
from flax import serialization

from flax.core import Scope, init, apply, lift, Array
from flax.core.scope import _unfreeze_variables, Variable, _fold_in_str
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

# Require JAX omnistaging mode.
from jax.config import config
config.enable_omnistaging()

# from .dotgetter import DotGetter

PRNGKey = Any  # pylint: disable=invalid-name
Array = Any    # pylint: disable=invalid-name
T = TypeVar('T')

# pylint: disable=protected-access,attribute-defined-outside-init

# Track parent relationship across Modules.
# -----------------------------------------------------------------------------
class _Context:
  def __init__(self):
    self._thread_data = threading.local()
  @property
  def module_stack(self):
    if not hasattr(self._thread_data, 'module_stack'):
      self._thread_data.module_stack = [None,]
    return self._thread_data.module_stack
_context = _Context()

class _Sentinel:
  pass
_unspecified_parent = _Sentinel()

# Utilities for autonaming pytrees of Modules defined inside setup()
# -----------------------------------------------------------------------------
def is_module_tree(in_tree: Any) -> bool:
  """Determine if in_tree is a pytree of subclasses of Module.

  Args:
    in_tree: python object, typically a python tree.

  Returns:
    False in_tree is empty or if any leaf is not a Module, True otherwise.
  """
  # reject trivial pytrees, {}, [], (), etc.
  if not tree_util.tree_leaves(in_tree):
    return False
  reduce_fn = lambda prev, cur: prev and isinstance(cur, Module)
  return jax.tree_util.tree_reduce(reduce_fn, in_tree, True)

def get_suffix_module_pairs(module_tree) -> List[Tuple[str, "Module"]]:
  """Helper for naming pytrees of submodules."""
  if isinstance(module_tree, Module):
    return [('', module_tree)]
  else:
    flat_tree = traverse_util.flatten_dict(
        serialization.to_state_dict(module_tree))
    return [('_' + '_'.join(k), v) for k, v in flat_tree.items()]

# Gathers all names on object.
# -----------------------------------------------------------------------------
def all_names_on_object(obj: Any) -> Set[str]:
  """Get all names of attributes on self and its classes throughout MRO."""
  nameset = set(obj.__dict__.keys())
  for cls in obj.__class__.__mro__:
    nameset = nameset.union(set(cls.__dict__.keys()))
  return nameset

# Method wrapping of "compact methods" and setup()
# -----------------------------------------------------------------------------
def compact(fun: Callable) -> Callable:
  """Decorator to mark a single Module method as compact."""
  fun.compact = True
  return fun

def get_method_names(cls: Any, exclude: Tuple[str] = ()) -> Tuple[str]:
  """Get method names of a class, allow exclusions."""
  methods = {m[0] for m in inspect.getmembers(cls, predicate=callable)}
  return tuple(methods.difference(set(exclude)))

def wrap_method(fun: Callable) -> Callable:
  """Manages Module state for user-defined methods."""
  @functools.wraps(fun)
  def wrapped_module_method(self, *args, **kwargs):
    is_compact_method = hasattr(fun, 'compact')
    is_setup_method = fun.__name__ == 'setup'

    if self.scope is None:
      raise ValueError("Can't call methods on orphaned modules")

    if is_compact_method:
      object.__setattr__(self, '_in_compact_method', True)
    elif is_setup_method:
      object.__setattr__(self, '_in_setup', True)
    _context.module_stack.append(self)
    try:
      return fun(self, *args, **kwargs)
    finally:
      _context.module_stack.pop()
      if is_compact_method:
        object.__setattr__(self, '_in_compact_method', False)
        object.__setattr__(self, '_reservations', set())
        object.__setattr__(self, '_autoname_cursor', dict())
        object.__setattr__(self, 'scope', self.scope.rewound())
      elif is_setup_method:
        object.__setattr__(self, '_in_setup', False)
  return wrapped_module_method


# Base Module definition.
# -----------------------------------------------------------------------------
class Module:
  """Base Module Class"""
  # Class defaults at init. NB: these __must not__ be made annotations,
  # as we don't want them to become dataclass fields.
  _in_compact_method = False
  _in_setup = False
  _last_varname = None
  scope = None

  @classmethod
  def __init_subclass__(cls):
    """Automatically initialize all subclasses as custom dataclasses."""
    # All Flax Modules are dataclasses.  We force this convention since
    # it encourages the stateless behavior needed to clone module instances for
    # functional transformation.  Instead of using a python metaclass, we
    # automatically transform Modules into dataclasses at subclass creation
    # time, and we set the last dataclass arguments to `parent` and `name`.
    cls._add_parent_and_name_attrs()
    dataclasses.dataclass(cls)
    # We wrap user-defined methods including setup and __call__ to enforce
    # a number of different checks and to provide clear error messages.
    cls._verify_single_or_no_compact()
    cls._wrap_module_methods()

  @classmethod
  def _add_parent_and_name_attrs(cls):
    """Add final optional dataclass attributes: `parent` and `name`."""
    annotations = cls.__dict__.get('__annotations__', {})
    if 'parent' in annotations or 'name' in annotations:
      raise ValueError(
          f'properties `parent` and `name` are reserved: {annotations}')
    # Add `parent` and `name` default fields at end.
    new_annotations = {}
    new_annotations.update(annotations)
    if 'parent' in getattr(cls, '__dataclass_fields__', {}):
      cls.__dataclass_fields__.pop('parent')
    new_annotations['parent'] = Union[Type["Module"], Type["Scope"],
                                      Type["Sentinel"], None]
    cls.parent = dataclasses.field(repr=False, default=_unspecified_parent)
    if 'name' in getattr(cls, '__dataclass_fields__', {}):
      cls.__dataclass_fields__.pop('name')
    new_annotations['name'] = str
    cls.__annotations__ = new_annotations
    cls.name = None  # default value of name is None.

  @classmethod
  def _verify_single_or_no_compact(cls):
    """Statically verify that at most a single method is labelled compact."""
    n_compact_fns = len([method_name for method_name in get_method_names(cls)
                         if hasattr(getattr(cls, method_name), 'compact')])
    if n_compact_fns > 1:
      raise RuntimeError(
          'Only one method per class can be @compact. You can remove @compact '
          'and define submodules and variables in setup(), or use two '
          'separate modules.')

  @classmethod
  def _wrap_module_methods(cls):
    # We only want to wrap user-defined methods.
    ignore_fns = get_method_names(cls, exclude=('__call__', 'setup'))
    dataclass_fieldnames = set([f.name for f in dataclasses.fields(cls)])
    for key in get_method_names(cls):
      if (key not in dataclass_fieldnames  # don't touch passed-in Modules
          and key not in ignore_fns):      # ignore base functions
        setattr(cls, key, wrap_method(getattr(cls, key)))
    return cls

  def __setattr__(self, name: str, val: Any):
    """We overload setattr solely to support pythonic naming via
    assignment of submodules in the special setup() function:
      self.submodule_name = MyModule(...)
    we also support lists and other general pytrees, e.g.:
      self.submodules = [MyModule0(..), MyModule1(..), ...]
    """
    # val is a Module or pytree whose leaves are all Modules.
    if is_module_tree(val):
      # We don't mess with the parent module.
      if name == 'parent':
        pass
      # Modules have been passed in as dataclass args.
      # if they're orphaned, i.e. Module(None, ...) then attach them.
      # TODO: should we not do this immediately, and give the user more
      #       control over when such "module defs" are attached?
      elif name in self.__dataclass_fields__.keys():
        for suffix, submodule in get_suffix_module_pairs(val):
          if submodule.parent is _unspecified_parent:
            submodule.parent = self
            if submodule.name is not None:
              raise ValueError(
                  "In setup assign names via self.<name> assignment.")
            submodule.name = f'{name}{suffix}'
            submodule.__post_init__()
      # Submodules are being defined and attached in setup()
      else:
        if not self._in_setup:
          raise ValueError("You can only assign submodules to self in setup().")
        for suffix, submodule in get_suffix_module_pairs(val):
          if submodule.parent is _unspecified_parent:
            submodule.parent = self
          elif submodule.parent != self:
            raise ValueError("Can't attach to remote parent in setup, pass in "
                             "bound Modules from outside as an argument.")
          if submodule.name is not None:
            raise ValueError(
                "In setup assign names via self.<name> assignment.")
          submodule.name = f'{name}{suffix}'
          submodule.__post_init__()
    # val is a parameter array or a Variable reference class.
    elif isinstance(val, (np.ndarray, jax.interpreters.xla.DeviceArray,
                          Variable)) and self._in_setup:
      # namecheck to ensure named variable matches self attribute name.
      if self._last_varname and self._last_varname != name:
        raise ValueError(f'Variable name {self._last_varname} must equal'
                         f' attribute name {name}.')
      self._last_varname = None
    # Finally, always run default __setattr__ to attach to self.__dict__.
    object.__setattr__(self, name, val)

  def __post_init__(self):
    # In dataclasses, __init__ is overridden to process dataclass arguments,
    # and __post_init__ is called immediately afterwards.  Here, depending
    # on the type of `parent` passed to initialize the Module, we either
    # defer initialization, attach this Module as a submodule of a parent,
    # or bind this Module at the top-level to variables and rngs.

    self._autoname_cursor = dict()  # autonaming state
    self._reservations = set()  # autonaming state
    self.children = dict()  # tracks child modules

    # Typically we set the parent based on the dynamic module context.
    if self.parent is _unspecified_parent:
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
      if self.parent._in_setup and self.name is None:
        return
      if not self.parent._initialization_allowed:
        raise ValueError(
            'Submodules must be defined in `setup()` or in a method wrapped '
            'in `@compact`')
      # Autonaming of submodules.
      if self.name is None:
        prefix = f"{self.__class__.__name__}"
        cursor = self.parent._autoname_cursor.get(prefix, 0)
        self.name = f"{prefix}_{cursor}"
        self.parent._autoname_cursor[prefix] = cursor + 1
      if self.parent._name_taken(self.name):
        raise ValueError(
            f"A variable of name {self.name} exists already, or "
            f"trying to share submodule {self.__class__.__name__} by name "
            f"{self.name}. To share submodules, store module instances as a"
            f" Python object or as an attribute on self and reuse.")
      self.parent._reservations.add(self.name)
      self.parent.children[self.name] = self
      self.scope = self.parent.scope.push(self.name)
      # TODO: find cleaner way to opt-out of core name collision check
      self.parent.scope.reservations.remove(self.name)

    # Top-level invocation with a functional Scope.
    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be None, Module or Scope")

    # Call the user-defined initialization function.
    self.setup()

  def setup(self):
    """Called when Module instance receives variables and PRNGs.

    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically
    called from the dataclass `__post_init__`.
    """
    pass

  def _name_taken(self, name):
    return name in self._reservations or name in all_names_on_object(self)

  @property
  def _initialization_allowed(self):
    return self._in_setup or self._in_compact_method

  def clone(self, **updates):
    """Create a clone of this Module, with optionally updated arguments."""
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs.update(**updates)
    return self.__class__(**attrs)

  def variable(self, kind: str, name: str, init_fn, *init_args):
    """Declare a variable in this Module.

    Args:
      kind: the variable kind.
      name: the variable name.
      init_fn: a function taking any number of positiona arguments.
      *init_args: the arguments to evaluate init_fn on lazily.

    Returns:
      A flax.core state Variable that can be read or set via ".value"
      attribute.
    """
    if not self._initialization_allowed:
      raise ValueError(
          'Variables must be initialized in `setup()` or in a method '
          'wrapped in `@compact`')
    if self._name_taken(name):
      raise ValueError(
          f'Name {name} already in use in {self.__class__.__name__}.')
    self._reservations.add(name)
    # ephemeral state for setattr name-equality-check
    self._last_varname = name
    v = self.scope.variable(kind, name, init_fn, *init_args)
    # TODO: find cleaner way to opt-out of core name collision check
    self.scope.reservations.remove(name)
    self.children[name] = kind
    return v

  def param(self, name: str, init_fn: Callable[..., T], *init_args,
            kind='param') -> T:
    """Declare a parameter in this Module.

    Args:
      name: the parameter name.
      init_fn: a function taking a PRNGKey plus any other number of
        positional arguments.
      *init_args: the arguments to evaluate init_fn on lazily.
      kind: an optional kind for the parameter.

    Returns:
      An initialized array.
    """
    p_init_fn = lambda *args: init_fn(self.make_rng(kind), *args)
    return self.variable(kind, name, p_init_fn, *init_args).value

  def get_variable(self, kind: str, name: str, default: T = None) -> T:
    """Get raw value of variable on this Module."""
    return self.scope.get_variable(kind, name, default)

  def has_variable(self, kind: str, name: str):
    """Check if a variable of given kind and name exists in this Module."""
    return self.scope.has_variable(kind, name)

  def put_variable(self, kind: str, name: str, value: Any):
    """Direectly set value of a variable on this Module."""
    return self.scope.put_variable(kind, name, value)

  def make_rng(self, kind: str) -> PRNGKey:
    """Get a new rng key of a given kind from this Module."""
    return self.scope.make_rng(kind)

  def apply(self, variables, *args, rngs=None,
            method='__call__', mutable=False, **kwargs):
    """Apply module to variables and return output and modified variables."""
    new_variables = _unfreeze_variables(variables, mutable)
    with Scope(new_variables, rngs=rngs).temporary() as root:
      clone = self.clone(parent=root)
      y = getattr(clone, method)(*args, **kwargs)
    if mutable:
      return y, freeze(new_variables)
    else:
      return y

  def init_with_output(self, rngs, *args, method='__call__', **kwargs):
    """Create initialized data for module and return it with output."""
    if not isinstance(rngs, dict):
      assert rngs.shape == (2,)
      rngs = {'param': rngs}
    return self.apply(
        {}, *args, rngs=rngs, method=method, mutable=True, **kwargs)

  def init(self, rngs, *args, method='__call__', **kwargs):
    """Create and return initialized data for module with rngs."""
    y, v_out = self.init_with_output(rngs, *args, method=method, **kwargs)
    return v_out

  @property
  def variables(self):
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

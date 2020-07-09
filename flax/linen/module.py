"""liNeN: lit neural-nets"""
import sys
import os
import functools
from importlib import reload
from pprint import pprint
import inspect
from contextlib import contextmanager
import dataclasses

from typing import Any, Callable, Sequence, Iterable, List, Optional, Tuple, Type, Union

import jax
from jax import numpy as jnp, random, lax
import numpy as np

import flax
from flax import nn
from flax.nn import initializers
from flax import traverse_util
from flax import serialization

from flax.core import Scope, init, apply, lift, Array
from flax.core.scope import _unfreeze_variables
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

from .dotgetter import DotGetter

from jax import tree_util

# pylint: disable=protected-access,attribute-defined-outside-init

SPECIAL_METHODS = ('__call__', 'setup', 'setup_variables')

# Utilities for autonaming pytrees of Modules defined inside setup()
# -----------------------------------------------------------------------------
def is_module_tree(in_tree):
  """Determine if in_tree is a pytree of subclasses of Module."""
  if not in_tree:  # reject trivial pytrees, {}, [], (), etc.
    return False
  reduce_fn = lambda prev, cur: prev and isinstance(cur, Module)
  return jax.tree_util.tree_reduce(reduce_fn, in_tree, True)

def get_suffix_module_pairs(module_tree):
  """Helper for naming pytrees of submodules."""
  if isinstance(module_tree, Module):
    return [('', module_tree)]
  else:
    flat_tree = traverse_util.flatten_dict(
        serialization.to_state_dict(module_tree))
    return [('_' + '_'.join(k), v) for k, v in flat_tree.items()]


# Method wrapping
# -----------------------------------------------------------------------------
#
# We use wrappers on Module methods to do 3 things:
# 1. Prevent calling private internal methods by accident.
# 2. Record current invoked methods for safeguards against misuse of submodules
#    and variable initialization in the multi-method context.
# 3. Reset the Module autonaming context on method exit.
#
# The below functionality might be better written as separate function wrappers,
# but each additional wrapper pollutes the stack trace.  In fact, we might wish
# to use the gnarly exec() trick from dataclasses.py to collapse this wrappers'
# traceback entry to a single line.

def manage_name_scopes(fun):
  @functools.wraps(fun)
  def wrapped_module_method(self, *args, **kwargs):
    if fun.__name__ not in self._public and self._cur_method is None:
      raise ValueError("Can't call private methods from outside Module.")
    # could simplify this by just recording a proper call stack:
    if fun.__name__ in self._public:
      prev_method = self._cur_method
      object.__setattr__(self, '_cur_method', fun.__name__)
    if fun.__name__ == 'setup':
      object.__setattr__(self, '_in_setup', True)
    if fun.__name__ == 'setup_variables':
      object.__setattr__(self, '_in_setup_variables', True)
    try:
      return fun(self, *args, **kwargs)
    finally:
      if fun.__name__ in self._public:
        self._cur_method = prev_method
      if fun.__name__ == 'setup':
        object.__setattr__(self, '_in_setup', False)
      if fun.__name__ == 'setup_variables':
        object.__setattr__(self, '_in_setup_variables', False)
      if fun.__name__ in self._public:
        object.__setattr__(self, '_reservations', set())
        object.__setattr__(self, '_autoname_cursor', dict())
        object.__setattr__(self, 'scope', self.scope.rewound())
  return wrapped_module_method

def get_method_names(cls, exclude=()):
  """Get method names of a class, allow exclusions."""
  methods = {m[0] for m in inspect.getmembers(cls, predicate=callable)}
  return tuple(methods.difference(set(exclude)))

def wrap_module_methods(cls):
  # We only want to wrap user-defined and special methods.
  ignore_fns = get_method_names(Module, exclude=SPECIAL_METHODS)
  dataclass_fieldnames = set([f.name for f in dataclasses.fields(cls)])
  for key in get_method_names(cls):
    if (key not in dataclass_fieldnames  # don't touch passed-in Modules
        and key not in ignore_fns):      # ignore base functions
      setattr(cls, key, manage_name_scopes(getattr(cls, key)))
  return cls


# Base Module definition.
# -----------------------------------------------------------------------------
def module_method(fn):
  """Marks Module method as public."""
  fn.public = True
  return fn

class Module:
  """Base Module Class"""
  _multimethod_module = False  # statically determined on subclass init
  _public = set()              #

  @classmethod
  def __init_subclass__(cls):
    """Automatically initialize all subclasses as custom dataclasses."""
    # Record public functions and determine if this is a multimethod Module
    cls._public = set(SPECIAL_METHODS).union(cls.__base__._public)
    cls._multimethod_module = cls.__base__._multimethod_module
    for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
      if hasattr(fn, 'public'):
        cls._public.add(name)
        cls._multimethod_module = True

    # Setup dataclass with defaults.
    if not hasattr(cls.__base__, '__annotations__'):
      annotations = cls.__dict__.get('__annotations__', {})
      if 'parent' in annotations or 'name' in annotations:
        raise ValueError(
          f'properties `parent` and `name` are reserved: {annotations}')
      # Add `parent` and `name` default fields at beginning and end, resp.
      new_annotations = {'parent': Union[Type["Module"], Type["Scope"], None]}
      new_annotations.update(annotations)
      new_annotations['name'] = str
      cls.__annotations__ = new_annotations
      cls.name = None
    dataclasses.dataclass(cls)

    # Apply module wrappers.
    wrap_module_methods(cls)

  def __setattr__(self, name, val):
    """We overload setattr solely to support pythonic naming via
    assignment of submodules in the special setup() function:
      self.submodule_name = MyModule(...)
    we also support lists and other general pytrees, e.g.:
      self.submodules = [MyModule0(..), MyModule1(..), ...]
    """
    if is_module_tree(val):
      if name == 'parent':
        pass
      elif not self._in_setup:
        raise ValueError("You can only assign submodules to self in setup().")
      else:
        # If we're attaching a pytree of submodules (list, dict, etc),
        # give each submodule in tree a suffix corresponding to tree location.
        for suffix, submodule in get_suffix_module_pairs(val):
          if submodule.parent is None:
            submodule.parent = self
          elif not name in self.__dataclass_fields__.keys():
            if submodule.parent != self:
              raise ValueError("Can't attach to remote parent in setup, pass in "
                               "bound Modules from outside as an argument.")
          if submodule.name is not None:
            raise ValueError("In setup assign names via self.<name> assignment.")
          submodule.name = f'{name}{suffix}'
          submodule.__post_init__()
    super().__setattr__(name, val)

  def __post_init__(self):
    """Register self as a child of self.parent."""
    self._cur_method = None
    self._autoname_cursor = dict()
    self._reservations = set()
    self.submodules = dict()
    self.scope = Scope({})

    # Initialization is deferred until attachment by __setattr__ in 2 cases:
    # For orphan Modules, we defer setup until attachment to a parent Module.
    # i.e. MyModule(None, ...)
    if self.parent is None:
      return
    # When initializing an unnamed Module inside setup()
    # i.e. self.mymodule = MyModule(self, ...)
    if (isinstance(self.parent, Module)
        and self.parent._in_setup
        and self.name is None):
      return

    # Register submodule on parent Module.
    if isinstance(self.parent, Module):
      if self.parent._multimethod_module and not self.parent._in_setup:
        raise ValueError("For multimethod Modules you must initialize any "
                         "submodules inside the setup() function.")
      # Autonaming of submodules.
      if self.name is None:
        prefix = f"{self.__class__.__name__}"
        cursor = self.parent._autoname_cursor.get(prefix, 0)
        self.name = f"{prefix}_{cursor}"
        self.parent._autoname_cursor[prefix] = cursor + 1
      if self.name in self.parent._reservations:
       raise ValueError(
           f"Trying to share submodule {self.__class__.__name__} by name "
           f"{self.name} To share submodules, store module instances as a"
           f" Python object or as an attribute on self and reuse.")
      self.parent._reservations.add(self.name)

      # Register submodule on parent.
      self.parent.submodules[self.name] = self
      self.scope = self.parent.scope.push(self.name)

    # Top-level invocation with a functional Scope.
    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be a Module or Scope")

    # Call the user-defined initialization function.
    self.setup()


  def setup(self):
    """Called when module instance receives variables and PRNGs.

    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically
    called indirectly from `__post_init__`.
    """
    pass

  def setup_variables(self, *args, **kwargs):
    """Call to initialize variables for multimethod Modules.

    If you want to use shared parameters in a multimethod, override this
    function, accepting whatever inputs are needed for shape-inference,
    and assign the parameters to self for use in the module_methods.
    """
    pass

  def clone(self, **updates):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs.update(**updates)
    return self.__class__(**attrs)

  def param(self, name, init_fn, *init_args):
    if (self._multimethod_module
        and not (self._in_setup_variables or self._in_setup)):
      raise ValueError(
        'For multi-method Modules, you must initialize parameters'
        ' in the `setup` or `setup_variables` function.')
    return self.scope.param(name, init_fn, *init_args)

  def variable(self, kind, name, init_fn, *init_args):
    if (self._multimethod_module
        and not (self._in_setup_variables or self._in_setup)):
      raise ValueError(
        'For multi-method Modules, you must initialize variables'
        ' in the `setup` or `setup_variables` function.')
    return self.scope.variable(kind, name, init_fn, *init_args)

  def get_variable(self, kind, name, default=None):
    return self.scope.get_variable(kind, name, default)

  @property
  def variables(self):
    return self.scope.variables()

  # Simple dot-syntax, tab-completed navigation in jupyter/colab:
  @property
  def vars(self):
    return DotGetter(self.scope._variables)

  @classmethod
  def toplevel(cls, *args, rngs=None, variables=None, **kwargs):
    if rngs is None:
      rngs = {}
    if variables is None:
      variables = {'param': {}}
    variables = unfreeze(variables)
    scope = Scope(variables, rngs=rngs)
    module = cls(scope, *args, **kwargs)
    scope.variables = freeze(scope.variables)
    return module

  @contextmanager
  def mutate(self, mutable=True):
    cloned = self.clone()
    try:
      cloned.scope._variables = _unfreeze_variables(
          cloned.scope._variables, mutable)
      yield cloned
    finally:
      cloned.scope._variables = freeze(cloned.scope._variables)

  def initialized(self, *args, method=lambda self: self.__call__, **kwargs):
    with self.mutate() as initialized:
      method(initialized)(*args, **kwargs)
    return initialized

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

SPECIAL_METHODS = ('__call__', 'setup')

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
def wrap_call(fun):
  @functools.wraps(fun)
  def wrapped_call_method(self, *args, **kwargs):
    object.__setattr__(self, '_in_call', True)
    try:
      return fun(self, *args, **kwargs)
    finally:
      object.__setattr__(self, '_in_call', False)
      object.__setattr__(self, '_reservations', set())
      object.__setattr__(self, '_autoname_cursor', dict())
      object.__setattr__(self, 'scope', self.scope.rewound())
  return wrapped_call_method

def wrap_setup(fun):
  @functools.wraps(fun)
  def wrapped_setup_method(self, *args, **kwargs):
      object.__setattr__(self, '_in_setup', True)
    try:
      return fun(self, *args, **kwargs)
    finally:
        object.__setattr__(self, '_in_setup', False)
  return wrapped_module_method

# Base Module definition.
# -----------------------------------------------------------------------------
class Module:
  """Base Module Class"""
  _multimethod_module = False
  _public = set(SPECIAL_METHODS)

  @classmethod
  def __init_subclass__(cls):
    """Automatically initialize all subclasses as custom dataclasses."""
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
      cls.name = None  # default value of name is None.
    dataclasses.dataclass(cls)
    # wrap setup and call methods
    cls.__call__ = wrap_call(cls.__call__)
    cls.setup = wrap_setup(cls.setup)

  def __setattr__(self, name, val):
    """We overload setattr solely to support pythonic naming via
    assignment of submodules in the special setup() function:
      self.submodule_name = MyModule(...)
    we also support lists and other general pytrees, e.g.:
      self.submodules = [MyModule0(..), MyModule1(..), ...]
    """
    if is_module_tree(val):
      # Ignore parent and other Modules passed in as dataclass args.
      if name == 'parent':
        pass
      # Special setattr assignment of modules to self is only allowed in setup()
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
    self._autoname_cursor = dict()
    self._reservations = set()
    self._in_call = False
    self._in_setup = False
    self.scope = Scope({})
    self.submodules = dict()

    # Initialization is deferred until attachment by __setattr__ for orphan
    # Modules i.e. MyModule(None, ...)
    if self.parent is None:
      return

    # Register submodule on parent Module.
    if isinstance(self.parent, Module):
      # When initializing an unnamed Module inside setup()
      # initialization is deferred until attachment by __setattr__
      # i.e. self.mymodule = MyModule(self, ...)
      if self.parent._in_setup and self.name is None:
        return
      if not self._initialization_allowed:
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

  @property
  def _initialization_allowed(self):
    return self._in_setup or (self._in_call and not self._multimethod_module)

  def clone(self, **updates):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs.update(**updates)
    return self.__class__(**attrs)

  def param(self, name, init_fn, *init_args):
    if not self._initialization_allowed:
      raise ValueError(
        'For multi-method Modules, you must initialize parameters'
        ' in the `setup` or `setup_variables` function.')
    return self.scope.param(name, init_fn, *init_args)

  def variable(self, kind, name, init_fn, *init_args):
    if not self._initialization_allowed:
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


class MultiModule(Module):
  _multimethod_module = True

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

from flax.core import scope
from flax.core import Scope, init, apply, lift, Array
from flax.core.scope import _unfreeze_variables
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

from jax import tree_util

# pylint: disable=protected-access,attribute-defined-outside-init


# Utilities
# -----------------------------------------------------------------------------

is_leaf = lambda x: tree_util.treedef_is_leaf(tree_util.tree_flatten(x)[1])

class DotGetter:
  """Dot-notation helper for interactive access of variable trees."""
  def __init__(self, data):
    object.__setattr__(self, '_data', data)

  def __getattr__(self, key):
    """Returns leaves unwrapped."""
    if is_leaf(self._data[key]):
      return self._data[key]
    else:
      return DotGetter(self._data[key])

  def __getitem__(self, key):
    if is_leaf(self._data[key]):
      return self._data[key]
    else:
      return DotGetter(self._data[key])

  def __setitem__(self, key, val):
    self._data[key] = val

  def __setattr__(self, key, val):
    self._data[key] = val

  def __dir__(self):
    if isinstance(self._data, dict):
      return list(self._data.keys())
    elif isinstance(self._data, FrozenDict):
      return list(self._data._dict.keys())
    else:
      return []

  def __repr__(self):
    return f'{self._data}'


def is_module_tree(in_tree):
  """Determine if in_tree is a pytree of subclasses of Module."""
  if not in_tree:
    return False
  reduce_fn = lambda prev, cur: prev and isinstance(cur, Module)
  return jax.tree_util.tree_reduce(reduce_fn, in_tree, True)


def get_suffix_module_pairs(module_tree):
  """Helper for naming pytrees of submodules."""
  flat_tree = traverse_util.flatten_dict(
      serialization.to_state_dict(module_tree))
  return [('_'.join(k), v) for k, v in flat_tree.items()]


# Dataclass utils
# -----------------------------------------------------------------------------

# The below functionality might be better written as separate function wrappers,
# but each additional wrapper pollutes the stack trace.
def manage_name_scopes(fun, manage_names=False):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    # Grab local call stack at entry.
    call_stack_at_entry = getattr(self, '_call_stack', [])
    # Reset debugging information for preventing share-by-name.
    if manage_names:
      object.__setattr__(self, '_method_by_name',
          {k: v for k, v in self._method_by_name.items() if v != fun.__name__})
    # Maintain call stack, update by overwriting.
    object.__setattr__(self, '_call_stack',
                       getattr(self, '_call_stack', []) + [fun.__name__,])
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._call_stack.pop()
      # Rewind autonaming when called from outside self
      if manage_names and not call_stack_at_entry:
        self._autoname_cursor = {}
        self.scope = self.scope.rewound()
  return wrapped

SPECIAL = {'__setattr__', '__post_init__', 'setup'}

def wrap_module_methods(cls):
  base_fns = set(dict(inspect.getmembers(Module, predicate=callable)).keys())
  ignore_fns = base_fns.difference(SPECIAL)
  dataclass_fieldnames = set([f.name for f in
                              dataclasses.fields(dataclasses.dataclass(cls))])
  for key, val in inspect.getmembers(cls, predicate=callable):
    if (key not in dataclass_fieldnames  # don't touch passed-in Modules
        and key not in ignore_fns):      # ignore base functions
      manage_names = key not in ('__setattr__', '__post_init__')
      val = manage_name_scopes(val, manage_names)
      setattr(cls, key, val)
  return cls


# Base Module definition.
# -----------------------------------------------------------------------------

class Module:
  """Base Module Class"""

  @classmethod
  def __init_subclass__(cls):
    """Automatically initialize all subclasses as custom dataclasses."""
    annotations = getattr(cls, '__annotations__', {})
    assert 'parent' not in annotations, 'property `parent` is reserved.'
    assert 'name' not in annotations, 'property `name` is reserved.'
    new_annotations = {'parent': Union[Type["Module"], Type["Scope"]]}
    new_annotations.update(annotations)
    new_annotations['name'] = str
    cls.__annotations__ = new_annotations
    cls.name = None
    wrap_module_methods(cls)
    dataclasses.dataclass(cls)

  def __setattr__(self, name, val):
    # We overload setattr solely to support pythonic naming of submodules
    # via self.submodule_name = SubModule(...) syntax in the setup() function.
    if is_module_tree(val): #isinstance(val, Module):
      # Ignore parent and other Modules passed in as dataclass args.
      if name == 'parent' or name in self.__dataclass_fields__.keys():
        pass
      # Special setattr naming is only allowed in setup()
      elif self._call_stack[-2] != 'setup':
        raise ValueError("You can only assign submodules to self in setup().")
      # Assign name then finish deferred initialization of this class.
      else:
        val.name = name
        val.__post_init__()
    super().__setattr__(name, val)

  def __post_init__(self):
    """Register self as a child of self.parent."""
    parent_call_stack = getattr(self.parent, '_call_stack', [])
    self.submodules = {}
    self._autoname_cursor = {}
    self._method_by_name = {}
    #self._public_methods = set()
    # In setup() defer naming and registration on parent until __setattr__.
    if parent_call_stack[-2:] == ['__post_init__', 'setup']:
      if self.name is None:
        return
      else:
        raise ValueError("In setup assign names via self.<name> assignment.")

    if isinstance(self.parent, Module):
      method_name = parent_call_stack[-1]
      if self.name is None:
        prefix_method = '' if method_name == '__call__' else f'_{method_name}'
        prefix = f"{self.__class__.__name__}{prefix_method}"
        cursor = self.parent._autoname_cursor.get(prefix, 0)
        self.name = f"{prefix}_{cursor}"
        self.parent._autoname_cursor[prefix] = cursor + 1

      if self.name in self.parent._method_by_name:
       raise ValueError(
           f"Trying to share submodule {self.__class__.__name__} by name "
           f"{self.name} in methods {method_name} and "
           f"{self.parent._method_by_name[self.name]}. To share submodules, "
           f"store module instances as a Python object and reuse. You can "
           f"store module instances on `self` to share across methods.")
      self.parent.submodules[self.name] = self
      self.parent._method_by_name[self.name] = method_name
      self.scope = self.parent.scope.push(self.name)

    elif isinstance(self.parent, Scope):
      self.scope = self.parent
    else:
      raise ValueError("parent must be a Module or Scope")

    # call user-defined "post post init" function
    self.setup()

  def setup(self):
    """Called when module instance receives variables and PRNGs.

    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically
    called indirectly from `__post_init__`.
    """
    pass

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

  def clone(self, **updates):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs.update(**updates)
    return self.__class__(**attrs)

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

  @property
  def variables(self):
    return self.scope.variables()

  def param(self, name, init_fn, *init_args):
    return self.scope.param(name, init_fn, *init_args)

  def variable(self, kind, name, init_fn, *init_args):
    return self.scope.variable(kind, name, init_fn, *init_args)

  def get_variable(self, kind, name, default=None):
    return self.scope.get_variable(kind, name, default)

  @property
  def vars(self):
    return DotGetter(self.scope._variables)

  @property
  def subs(self):
    return DotGetter(self.submodules)

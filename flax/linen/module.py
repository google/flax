"""liNeN: lit neural-nets"""
import sys
import os
import functools
from importlib import reload
from pprint import pprint
import inspect
from contextlib import contextmanager
import dataclasses

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

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


# Dot-notation helper for interactive access of variable trees.
# -----------------------------------------------------------------------------

is_leaf = lambda x: tree_util.treedef_is_leaf(tree_util.tree_flatten(x)[1])

def get_suffix_module_pairs(module_tree):
  flat_tree = traverse_util.flatten_dict(
      serialization.to_state_dict(module_tree))
  return [('_'.join(k), v) for k,v in flat_tree.items()]

class DotGetter:
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


BASE_METHODS = (
    '__init__', '__post_init__', '__repr__','__eq__', '__setattr__',
    'mutate', 'clone', 'param', 'variables', 'variable', 'initialized')


# Dataclass utils
# -----------------------------------------------------------------------------


def track_calls(fun):
  """Maintain per-module local call stack."""
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    object.__setattr__(self, '_call_stack',
                       getattr(self, '_call_stack', []) + [fun.__name__,])
    try:
      return fun(self, *args, **kwargs)
    finally:
      self._call_stack.pop()
  return wrapped

def manage_name_scopes(fun):
  @functools.wraps(fun)
  def wrapped(self, *args, **kwargs):
    call_stack = self._call_stack
    # Reset debugging information for preventing share-by-name.
    self._method_by_name = {k: v for k, v in self._method_by_name.items()
                            if v != fun.__name__}
    try:
      return fun(self, *args, **kwargs)
    finally:
      # Rewind autonaming when called from outside self
      if not call_stack:
        #print('----------REWIND----------', self.__class__)
        self._autoname_cursor = {}
        self.scope = self.scope.rewound()
  return wrapped

def wrap_module_methods(cls):
  dataclass_fieldnames = set([f.name for f in
                              dataclasses.fields(dataclasses.dataclass(cls))])
  for key, val in cls.__dict__.items():
    if (callable(val) and
        key not in dataclass_fieldnames  # don't touch passed-in Modules
        and not inspect.ismethod(val)):  # don't touch classmethods
      val = track_calls(val)
      if key not in BASE_METHODS:
        val = manage_name_scopes(val)
      setattr(cls, key, val)
  return cls

def dataclass(cls):
  if not issubclass(cls, Module):
    raise ValueError("Must extend from Module to use @module.dataclass")
  cls = wrap_module_methods(cls)
  cls = dataclasses.dataclass(cls)
  return cls


# Base Module definition.
# -----------------------------------------------------------------------------


class Module:
  """Base Module Class"""
  parent: Union[Type["Module"], Type["Scope"]]

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

  def __setattr__(self, name, val):
    if name != 'parent' and isinstance(val, Module):
      # don't mess with Modules passed in as dataclass args.
      if name in self.__dataclass_fields__.keys():
        pass
      # special setattr naming is only allowed in setup()
      elif self._call_stack[-2] != 'setup':
        raise ValueError("You can only assign submodules to self in setup().")
      else:
        if val.name is None:  # raise error if not None?
          val.name = name
        val.__post_init__()
    super().__setattr__(name, val)

  def __post_init__(self):
    """Register self as a child of self.parent."""
    parent_call_stack = getattr(self.parent, '_call_stack', [])
    # defer naming and registration on parent until __setattr__.
    if parent_call_stack[-2:] == ['__post_init__', 'setup']:
      if self.name is None:
        return
      else:
        raise ValueError("In setup assign names via self.<name> assignment.")
    self.submodules = {}
    self._autoname_cursor = {}
    self._method_by_name = {}
    self._public_methods = set()

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

  def clone(self):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs = {**attrs}
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

# wrap after definitions to avoid "circular defs" issue
Module = dataclass(Module)




def vmap_bound_method(method, *vmap_args, **vmap_kwargs):
  if inspect.ismethod(method):
    instance, unbound_fn = method.__self__, method.__func__
  else: # transform __call__ by default
    instance, unbound_fn = method, method.__call__.__func__

  attrs = {f.name: getattr(instance, f.name) for f in dataclasses.fields(instance)}
  attrs['name'] = 'Vmap' + instance.name
  instance = instance.__class__(**attrs)

  def call_fn(scope, *args, **kwargs):
    attrs = {f.name: getattr(instance, f.name) 
             for f in dataclasses.fields(instance)
             if f.name != 'parent'}
    cloned = instance.__class__(parent=scope, **attrs)
    ret = unbound_fn(cloned, *args, **kwargs)
    return ret

  vmap_fn = lift.vmap(call_fn, *vmap_args, **vmap_kwargs)
  @functools.wraps(unbound_fn)
  def wrap_fn(self, *args, **kwargs):
    ret = vmap_fn(self.scope, *args, **kwargs)
    return ret

  # return function bound to instance again
  wrap_fn = wrap_fn.__get__(instance, instance.__class__)
  return wrap_fn
"""
@dataclass
class MLP(Module):
  name: str = None

  # use inside makes new instances of Dense
  def __call__(self, x):
    #y = Dense(self, 3)(x)
    dense = Dense(self, 3)
    new_fn = vmap(dense,
             in_axes=0, out_axes=0,
             variable_in_axes={'param': None},
             variable_out_axes={'param': None},
             split_rngs={'param': False})
    new_fn2 = vmap(new_fn,
             in_axes=0, out_axes=0,
             variable_in_axes={'param': 0},
             variable_out_axes={'param': 0},
             split_rngs={'param': True})
    y = dense(x)
    y += new_fn(x)
    y += new_fn2(x)
    return y

"""

def vmap_module_instance(instance, *vmap_args, methods=None, **vmap_kwargs):
  if methods is None:
    methods = ['__call__']
  elif isinstance(methods, str):
    methods = [method,]
  for fn_name in methods:
    unbound_fn = getattr(instance, fn_name).__func__

    def call_fn(scope, *args, **kwargs):
      attrs = {f.name: getattr(instance, f.name)
               for f in dataclasses.fields(instance) if f.name != 'parent'}
      cloned = instance.__class__(parent=scope, **attrs)
      return getattr(cloned, fn_name)(*args, **kwargs)

    vmap_fn = lift.vmap(call_fn, *vmap_args, **vmap_kwargs)

    @functools.wraps(unbound_fn)
    def wrap_fn(self, *args, **kwargs):
      return vmap_fn(self.scope, *args, **kwargs)

    # return function bound to instance again
    new_name = 'vmap' if fn_name == '__call__' else 'vmap_' + fn_name
    setattr(instance, new_name, wrap_fn.__get__(instance, instance.__class__))
  return instance

"""
@dataclass
class MLP(Module):
  name: str = None

  def __call__(self, x):
    new_inst = vmap(Dense(self, 3), 
                    in_axes=0, out_axes=0,
                    variable_in_axes={'param': 0},
                    variable_out_axes={'param': 0},
                    split_rngs={'param': False})
    #print(inspect.signature(new_inst.__call__))
    z = new_inst.vmap(x)
    return z
"""

# class transform
def vmap_class(module_class, *vmap_args, methods=None, **vmap_kwargs):
  if methods is None:
    # Default case, just transform __call__
    class_vmap_args = {'__call__': (vmap_args, vmap_kwargs)}
  elif isinstance(methods, str):
    # Transform every method in methods with given args, kwargs.
    class_vmap_args = {methods: (vmap_args, vmap_kwargs)}
  elif isinstance(methods, dict):
    # Pass different vmap args per each method.
    assert vmap_args == () and vmap_kwargs == {}, (
    """When passing different vmap args per method, all args must be 
    passed via methods kwarg.""")
    class_vmap_args = {k: ((),v) for k,v in methods.items()}

  transformed_fns = {}
  for fn_name, fn_vmap_args in class_vmap_args.items():
    fn = getattr(module_class, fn_name)
    vmap_args, vmap_kwargs = fn_vmap_args
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
      def call_fn(scope, *args, **kwargs):
        attrs = {f.name: getattr(self, f.name) 
                 for f in dataclasses.fields(self) if f.name != 'parent'}
        cloned = module_class(parent=scope, **attrs)
        return getattr(cloned, fn_name)(*args, **kwargs)        
      vmap_fn = lift.vmap(call_fn, *vmap_args, **vmap_kwargs)    
      return vmap_fn(self.scope, *args, **kwargs)
    transformed_fns[fn_name] = wrapped_fn
  return type('Vmap' + module_class.__name__, (module_class,), transformed_fns)

"""
@dataclass
class MLP(Module):
  name: str = None

  # use inside makes new instances of Dense
  def __call__(self, x):
    d = Dense
    vd = vmap(d,
             in_axes=0, out_axes=0,
             variable_in_axes={'param': 0},
             variable_out_axes={'param': 0},
             split_rngs={'param': False})
    vvd = vmap(vd,
             in_axes=0, out_axes=0,
             variable_in_axes={'param': 0},
             variable_out_axes={'param': 0},
             split_rngs={'param': False})
    return d(self, 3)(x) + vd(self, 3)(x) + vvd(self, 3)(x)
"""

# class transforms
def module_class_lift_transform(
    transform,
    module_class,
    *trafo_args,
    methods=None,
    **trafo_kwargs):
  transform_name = transform.__name__
  # TODO (levskaya): find nicer argument convention for multi-method case?
  # prepare per-method transform args, kwargs
  if methods is None:
    # Default case, just transform __call__
    class_trafo_args = {'__call__': (trafo_args, trafo_kwargs)}
  elif isinstance(methods, str):
    # Transform every method in methods with given args, kwargs.
    class_trafo_args = {methods: (trafo_args, trafo_kwargs)}
  elif isinstance(methods, dict):
    # Pass different trafo args per each method.
    assert trafo_args == () and trafo_kwargs == {}, (
        f"""When passing different {transform_name} args per method,
        all args must be passed via methods kwarg.""")
    class_trafo_args = {k: ((),v) for k,v in methods.items()}

  transformed_fns = {}
  # for each of the specified methods:
  for fn_name, fn_trafo_args in class_trafo_args.items():
    # get existing unbound method from class
    fn = getattr(module_class, fn_name)
    trafo_args, trafo_kwargs = fn_trafo_args
    # we need to create a scope-function from our class for the given method
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
      # make a scope-function to transform
      def call_fn(scope, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self) if f.name != 'parent'}
        cloned = module_class(parent=scope, **attrs)
        return getattr(cloned, fn_name)(*args, **kwargs)
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(call_fn, *trafo_args, **trafo_kwargs)
      ret = trafo_fn(self.scope, *args, **kwargs)
      return ret
    transformed_fns[fn_name] = wrapped_fn
  # construct new dynamic class w. transformed methods
  return type(transform_name.capitalize() + module_class.__name__,
              (module_class,),
              transformed_fns)

vmap = functools.partial(module_class_lift_transform, lift.vmap)
scan = functools.partial(module_class_lift_transform, lift.scan)
jit = functools.partial(module_class_lift_transform, lift.jit)
remat = functools.partial(module_class_lift_transform, lift.remat)

# if we want to expose the class on the module for pickling, add at end:
#   transformed_class = type(
#     transform_name.capitalize() + module_class.__name__,
#     (module_class,),
#     transformed_fns)
# setattr(sys.modules[transformed_class.__module__], transformed_class.__name__, transformed_class)
# return getattr(sys.modules[transformed_class.__module__], transformed_class.__name__)

"""
@dataclass
class MLP(Module):
  name: str = None

  # use inside makes new instances of Dense
  def __call__(self, x):
    JitDense = jit(Dense)
    VmapDense = vmap(
      Dense,
      in_axes=0, out_axes=0,
      variable_in_axes={'param': 0},
      variable_out_axes={'param': 0},
      split_rngs={'param': False})
    VmapVmapDense = vmap(
      VmapDense,
      in_axes=0, out_axes=0,
      variable_in_axes={'param': 0},
      variable_out_axes={'param': 0},
      split_rngs={'param': False})
    return Dense(self, 3)(x) + JitDense(self, 3)(x) + VmapDense(self, 3)(x) + VmapVmapDense(self, 3)(x)
"""


def class_method_lift_transform(transform, class_fn, *trafo_args, **trafo_kwargs):
    @functools.wraps(class_fn)
    def wrapped_fn(self, *args, **kwargs):
      # make a scope-function to transform
      def call_fn(scope, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name) 
                 for f in dataclasses.fields(self)
                 if f.name != 'parent'}
        cloned = self.__class__(parent=scope, **attrs)
        return class_fn(cloned, *args, **kwargs)
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(call_fn, *trafo_args, **trafo_kwargs)
      return trafo_fn(self.scope, *args, **kwargs)
    return wrapped_fn

vmap_method = functools.partial(class_method_lift_transform, lift.vmap)
scan_method = functools.partial(class_method_lift_transform, lift.scan)
jit_method = functools.partial(class_method_lift_transform, lift.jit)
remat_method = functools.partial(class_method_lift_transform, lift.remat)

"""
@dataclass
class Dense2(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  name: str = None

  @functools.partial(vmap_method, in_axes=0, out_axes=0,
                     variable_in_axes={'param': 0},
                     variable_out_axes={'param': 0},
                     split_rngs={'param': False})
  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    y = jnp.dot(x, kernel)
    return y
"""
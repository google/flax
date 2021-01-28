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

"""Lifting / Transforms of Modules."""
import copy
import dataclasses
import functools
import inspect
from flax.core import lift, Scope
from flax.linen.module import Module
from flax.linen.module import wrap_method_once
import jax

# Utils
# -----------------------------------------------------------------------------
def clean_clone(x):
  """Remove scopes and tracers from children."""
  if isinstance(x, Module):
    object.__setattr__(
        x, 'children',
        {k: clean_clone(v) for k, v in x.children.items()})
    object.__setattr__(x, 'scope', None)
  return x


def get_module_scopes(module):
  """Get all scopes on module, including constructor Module arguments.

  To properly functionalize a Module that has other bound Modules passed in
  "from the outside" as dataclass attributes, we need to traverse all dataclass
  fields to find the Scopes associated with the Module.  Additionally, because
  we allow Modules to be passed inside pytrees on the dataclass attributes, we
  must traverse all dataclass attributes as pytrees to find all Modules.

  Args:
    module: a bound flax Module.

  Returns:
    A list of all functional-core Scopes bound on self and inside dataclass
    fields.
  """
  module._try_setup(shallow=True)
  outer_scopes = []
  def get_scope(x):
    nonlocal outer_scopes
    if isinstance(x, Module) and isinstance(x.scope, Scope):
      outer_scopes.extend(get_module_scopes(x))
    return x
  attrs = {f.name: getattr(module, f.name)
           for f in dataclasses.fields(module) if f.name != 'parent' and f.init}
  jax.tree_map(get_scope, attrs)
  return outer_scopes + [module.scope,]


def set_module_scopes(module, scopes):
  """Set all scopes on module, including those on Modules in dataclass fields.

  To properly functionalize a Module we must also "rehydrate" it with Scopes
  from `get_module_scopes`.  We need to set scopes not just on the Module but
  also on any Module living inside dataclass attributes or even pytrees in its
  dataclass attributes.  The order of traversal through both methods is the
  same, guaranteeing the correct Scopes are applied to each Module.

  Args:
    module: a flax Module.
    scopes: a list of Scopes corresponding to this Module and its arguments that
      was created by the `get_module_scopes` function.

  Returns:
    A copy of the module with it and its attributes bound to the scopes passed
    to this function.
  """
  idx = 0
  def set_scopes(module):
    nonlocal idx
    def set_scopes_inner(x):
      if isinstance(x, Module) and isinstance(x.scope, Scope):
        return set_scopes(x)
      else:
        return x
    attrs = {f.name: getattr(module, f.name)
             for f in dataclasses.fields(module) if f.name != 'parent' and f.init}
    new_attrs = jax.tree_map(set_scopes_inner, attrs)
    new_module = module.clone(parent=scopes[idx], **new_attrs)
    idx += 1
    return new_module
  new_module = set_scopes(module)
  assert len(scopes) == idx, f'scope list mismatch {len(scopes)} != {idx}'
  return new_module


# Class lifting
# -----------------------------------------------------------------------------
def module_class_lift_transform(
    transform,
    module_class,
    *trafo_args,
    methods=None,
    **trafo_kwargs):
  # TODO(levskaya): find nicer argument convention for multi-method case?

  # Prepare per-method transform args, kwargs.
  if methods is None:
    # Default case, just transform __call__
    class_trafo_args = {'__call__': (trafo_args, trafo_kwargs)}
  elif isinstance(methods, (list, tuple)):
    # Transform every method in methods with given args, kwargs.
    class_trafo_args = {m: (trafo_args, trafo_kwargs) for m in methods}
  elif isinstance(methods, dict):
    # Pass different trafo args per each method.
    assert trafo_args == () and trafo_kwargs == {}, (
        f"""When passing different {transform.__name__} args per method,
        all args must be passed via methods kwarg.""")
    class_trafo_args = {k: ((), v) for k, v in methods.items()}

  def create_trans_fn(fn_name, fn_trafo_args):
    # get existing unbound method from class
    fn = getattr(module_class, fn_name)
    trafo_args, trafo_kwargs = fn_trafo_args
    # we need to create a scope-function from our class for the given method
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
      # make a scope-function to transform
      def core_fn(scopes, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self) if f.name != 'parent' and f.init}
        # we reference module_class, not self.__class__ to avoid infinite loop
        cloned = module_class(parent=None, **attrs)
        cloned = set_module_scopes(cloned, scopes)
        object.__setattr__(cloned, '_state', self._state.export())  # pylint: disable=protected-access
        res = fn(cloned, *args, **kwargs)
        self._state.reimport(cloned._state)  # pylint: disable=protected-access
        return res
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(core_fn, *trafo_args, **trafo_kwargs)
      ret = trafo_fn(get_module_scopes(self), *args, **kwargs)
      return ret
    return wrapped_fn
  transformed_fns = {fn_name: create_trans_fn(fn_name, fn_trafo_args)
                     for fn_name, fn_trafo_args in class_trafo_args.items()}
  # construct new dynamic class w. transformed methods
  transformed_cls = type(transform.__name__.capitalize() + module_class.__name__,
                         (module_class,),
                         transformed_fns)
  return transformed_cls


# Function lifting as decorator on methods __inside__ class definition.
# -----------------------------------------------------------------------------
def decorator_lift_transform(transform, class_fn, *trafo_args, **trafo_kwargs):
  # Due to the ordering of method decorators, we must wrap the class_fn
  # with the module state management wrapper first to maintain Module state correctly.
  prewrapped_fn = wrap_method_once(class_fn)
  @functools.wraps(prewrapped_fn)
  def wrapped_fn(self, *args, **kwargs):
    # make a scope-function to transform
    def core_fn(scopes, *args, **kwargs):
      cloned = set_module_scopes(self, scopes)
      object.__setattr__(cloned, '_state', self._state.export())  # pylint: disable=protected-access
      res = prewrapped_fn(cloned, *args, **kwargs)
      self._state.reimport(cloned._state)  # pylint: disable=protected-access
      return res
    # here we apply the given lifting transform to the scope-ingesting fn
    trafo_fn = transform(core_fn, *trafo_args, **trafo_kwargs)
    return trafo_fn(get_module_scopes(self), *args, **kwargs)
  return wrapped_fn


# Utility to wrap a class or to use as decorator in def of class method.
# -----------------------------------------------------------------------------
def lift_transform(transform, target, *trafo_args, methods=None, **trafo_kwargs):
  """Applies to class or as a decorator on class fns."""
  if inspect.isclass(target) and issubclass(target, Module):
    return module_class_lift_transform(
        transform, target, *trafo_args, methods=methods, **trafo_kwargs)
  # we presume this is being used as a function decorator in class definition
  elif inspect.isfunction(target):
    return decorator_lift_transform(
        transform, target, *trafo_args, **trafo_kwargs)
  else:
    raise ValueError(
        'Can only transform a Module subclass or decorate a function'
        ' in class definition.')


# TODO: provide wrappers with annotated args/kwargs and docstrings.
vmap = functools.partial(lift_transform, lift.vmap)
jit = functools.partial(lift_transform, lift.jit)
remat = functools.partial(lift_transform, lift.remat)
scan = functools.partial(lift_transform, lift.scan)


# Special case of decorator_lift_transform to handle named calls for profiling.
def named_call(class_fn):
  """Labels a method for labelled traces in profiles."""
  # Due to the ordering of method decorators, we must wrap the class_fn
  # with the module state management wrapper first to maintain Module state correctly.
  prewrapped_fn = wrap_method_once(class_fn)
  @functools.wraps(prewrapped_fn)
  def wrapped_fn(self, *args, **kwargs):
    fn_name = class_fn.__name__
    method_suffix = f'.{fn_name}' if fn_name != '__call__' else ''
    module_name = self.name or self.__class__.__name__
    full_name = f'{module_name}{method_suffix}'
    # make a scope-function to transform
    def core_fn(scopes, *args, **kwargs):
      cloned = set_module_scopes(self, scopes)
      object.__setattr__(cloned, '_state', self._state.export())  # pylint: disable=protected-access
      res = prewrapped_fn(cloned, *args, **kwargs)
      self._state.reimport(cloned._state)  # pylint: disable=protected-access
      return res
    # here we apply the given lifting transform to the scope-ingesting fn
    trafo_fn = lift.named_call(core_fn, full_name)
    return trafo_fn(get_module_scopes(self), *args, **kwargs)
  return wrapped_fn

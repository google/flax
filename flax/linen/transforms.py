# Copyright 2020 The Flax Authors.
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
import dataclasses
import functools
import inspect
from flax.core import lift
from flax.linen import Module
from flax.linen.module import wrap_method


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


# Class lifting
# -----------------------------------------------------------------------------
def module_class_lift_transform(
    transform,
    module_class,
    *trafo_args,
    methods=None,
    **trafo_kwargs):
  # TODO (levskaya): find nicer argument convention for multi-method case?

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

  # Build the actual transformed class.
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
      def scope_fn(scope, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self) if f.name != 'parent'}
        # we reference module_class, not self.__class__ to avoid infinite loop
        cloned = module_class(parent=scope, **attrs)
        res = getattr(cloned, fn_name)(*args, **kwargs)
        # preserve submodule-tree stripped of scopes/tracers for introspection
        object.__setattr__(self, 'children', clean_clone(cloned).children)
        return res
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(scope_fn, *trafo_args, **trafo_kwargs)
      ret = trafo_fn(self.scope, *args, **kwargs)
      return ret
    transformed_fns[fn_name] = wrapped_fn
  # construct new dynamic class w. transformed methods
  return type(transform.__name__.capitalize() + module_class.__name__,
              (module_class,),
              transformed_fns)

# Function lifting as decorator on methods __inside__ class definition.
# -----------------------------------------------------------------------------
def decorator_lift_transform(transform, class_fn, *trafo_args, **trafo_kwargs):
  # NB: due to the ordering of method decorators, we must re-wrap the class_fn
  # to maintain Module state correctly for multiple invocations.  If we want to
  # save another stacktrace entry we could instead replicate its logic below.
  rewrapped_fn = wrap_method(class_fn)
  @functools.wraps(class_fn)
  def wrapped_fn(self, *args, **kwargs):
    # make a scope-function to transform
    def scope_fn(scope, *args, **kwargs):
      cloned = self.clone(parent=scope)
      res = rewrapped_fn(cloned, *args, **kwargs)
      # preserve submodule-tree stripped of scopes/tracers for introspection
      object.__setattr__(self, 'children', clean_clone(cloned).children)
      return res
    # here we apply the given lifting transform to the scope-ingesting fn
    trafo_fn = transform(scope_fn, *trafo_args, **trafo_kwargs)
    return trafo_fn(self.scope, *args, **kwargs)
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

# Used for annotating profiles.
named_call = functools.partial(decorator_lift_transform, lift.named_call)

# TODO: provide wrappers with annotated args/kwargs and docstrings.
vmap = functools.partial(lift_transform, lift.vmap)
jit = functools.partial(lift_transform, lift.jit)
remat = functools.partial(lift_transform, lift.remat)


# Scan specific class lifting
# -----------------------------------------------------------------------------
def module_class_scan_transform(
    module_class,
    *trafo_args,
    methods=None,
    **trafo_kwargs):
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
        f"""When passing different scan args per method,
        all args must be passed via methods kwarg.""")
    class_trafo_args = {k: ((), v) for k, v in methods.items()}

  # Build the actual transformed class.
  transformed_fns = {}
  # for each of the specified methods:
  for fn_name, fn_trafo_args in class_trafo_args.items():
    # get existing unbound method from class
    fn = getattr(module_class, fn_name)
    trafo_args, trafo_kwargs = fn_trafo_args
    # we need to create a scope-function from our class for the given method
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
      if len(args) != 2:
        raise ValueError('scan requires a Module taking two arguments, '
                         'a carry and an input.')
      # make a scope-function to transform
      def scope_fn(scope, *args_inner):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self) if f.name != 'parent'}
        # we reference module_class, not self.__class__ to avoid infinite loop
        cloned = module_class(parent=scope, **attrs)
        res = getattr(cloned, fn_name)(*args_inner, **kwargs)
        # preserve submodule-tree stripped of scopes/tracers for introspection
        object.__setattr__(self, 'children', clean_clone(cloned).children)
        return res
      # here we apply the given lifting transform to the scope-ingesting fn
      return lift.scan(scope_fn, self.scope, *args, *trafo_args, **trafo_kwargs)
    transformed_fns[fn_name] = wrapped_fn
  # construct new dynamic class w. transformed methods
  return type('Scan' + module_class.__name__,
              (module_class,),
              transformed_fns)


# Scan as decorator on methods __inside__ class definition.
# -----------------------------------------------------------------------------
def decorator_scan_transform(class_fn, *trafo_args, **trafo_kwargs):
  # NB: due to the ordering of method decorators, we must re-wrap the class_fn
  # to maintain Module state correctly for multiple invocations.  If we want to
  # save another stacktrace entry we could instead replicate its logic below.
  rewrapped_fn = wrap_method(class_fn)
  @functools.wraps(class_fn)
  def wrapped_fn(self, *args, **kwargs):
    if len(args) != 2:
      raise ValueError('scan requires a Module taking two arguments, '
                       'a carry and an input.')
    # make a scope-function to transform
    def scope_fn(scope, *args_inner):
      cloned = self.clone(parent=scope)
      res = rewrapped_fn(cloned, *args_inner, **kwargs)
      # preserve submodule-tree stripped of scopes/tracers for introspection
      object.__setattr__(self, 'children', clean_clone(cloned).children)
      return res
    # here we apply the given lifting transform to the scope-ingesting fn
    return lift.scan(scope_fn, self.scope, *args, *trafo_args, **trafo_kwargs)
  return wrapped_fn


# Utility to wrap a class or to use as decorator in def of class method.
# -----------------------------------------------------------------------------
def scan(target, *trafo_args, methods=None, **trafo_kwargs):
  """Applies scan to a Module class or as a decorator on class functions."""
  if inspect.isclass(target) and issubclass(target, Module):
    return module_class_scan_transform(
        target, *trafo_args, methods=methods, **trafo_kwargs)
  # we presume this is being used as a function decorator in class definition
  elif inspect.isfunction(target):
    return decorator_scan_transform(
        target, *trafo_args, **trafo_kwargs)
  else:
    raise ValueError(
        'Can only scan a Module subclass or decorate a function'
        ' with scan in class definition.')

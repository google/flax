"""Lifting / Transforms of Modules."""

import functools
import inspect

from flax.core import scope
from flax.core import Scope, init, apply, lift, Array
from flax.core.scope import _unfreeze_variables
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict


# Module Lifting utils
# -----------------------------------------------------------------------------

def vmap_bound_method(method, *vmap_args, **vmap_kwargs):
  if inspect.ismethod(method):
    instance, unbound_fn = method.__self__, method.__func__
  else:
    # when module itself passed in, transform __call__
    instance, unbound_fn = method, method.__call__.__func__
  def scope_fn(scope, *args, **kwargs):
    return unbound_fn(instance.clone(parent=scope), *args, **kwargs)
  vmap_fn = lift.vmap(scope_fn, *vmap_args, **vmap_kwargs)
  @functools.wraps(unbound_fn)
  def wrap_fn(self, *args, **kwargs):
    return vmap_fn(self.scope, *args, **kwargs)
  # return function bound to instance again
  return wrap_fn.__get__(instance.clone(name=f'Vmap{instance.name}'),
                         instance.__class__)


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
      def scope_fn(scope, *args, **kwargs):
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self)
                 if f.name != 'parent'}
        # we reference module_class, not self.__class__ to avoid infinite loop
        cloned = module_class(parent=scope, **attrs)
        return getattr(cloned, fn_name)(*args, **kwargs)
      vmap_fn = lift.vmap(scope_fn, *vmap_args, **vmap_kwargs)
      return vmap_fn(self.scope, *args, **kwargs)
    transformed_fns[fn_name] = wrapped_fn
  return type('Vmap' + module_class.__name__, (module_class,), transformed_fns)


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
      def scope_fn(scope, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self) if f.name != 'parent'}
        # we reference module_class, not self.__class__ to avoid infinite loop
        cloned = module_class(parent=scope, **attrs)
        return getattr(cloned, fn_name)(*args, **kwargs)
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(scope_fn, *trafo_args, **trafo_kwargs)
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


def decorator_lift_transform(transform, class_fn, *trafo_args, **trafo_kwargs):
    @functools.wraps(class_fn)
    def wrapped_fn(self, *args, **kwargs):
      # make a scope-function to transform
      def scope_fn(scope, *args, **kwargs):
        # make a clone of self using its arguments
        attrs = {f.name: getattr(self, f.name)
                 for f in dataclasses.fields(self)
                 if f.name != 'parent'}
        cloned = self.__class__(parent=scope, **attrs)
        return class_fn(cloned, *args, **kwargs)
      # here we apply the given lifting transform to the scope-ingesting fn
      trafo_fn = transform(scope_fn, *trafo_args, **trafo_kwargs)
      return trafo_fn(self.scope, *args, **kwargs)
    return wrapped_fn


vmap_decorator = functools.partial(decorator_lift_transform, lift.vmap)
scan_decorator = functools.partial(decorator_lift_transform, lift.scan)
jit_decorator = functools.partial(decorator_lift_transform, lift.jit)
remat_decorator = functools.partial(decorator_lift_transform, lift.remat)

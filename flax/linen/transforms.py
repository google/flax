"""Lifting / Transforms of Modules."""
import functools
import inspect
from flax.core import lift
from flax.linen import Module

# Module method lifting
# -----------------------------------------------------------------------------

def lift_method(transform, target, *trafo_args, **trafo_kwargs):
  """Applies to existing instance methods or as a decorator on class fns."""
  if isinstance(target, Module):
    # when Module instance passed in, transform __call__
    instance, unbound_fn = target, target.__call__.__func__
  elif inspect.ismethod(target):
    instance, unbound_fn = target.__self__, target.__func__
  elif inspect.isfunction(target):
    # we presume this is being used as a function decorator
    instance, unbound_fn = None, target
  else:
    raise ValueError('Can only transform a method or class function.')

  @functools.wraps(unbound_fn)
  def wrapped_fn(self, *args, **kwargs):
    # make a scope-function to transform
    def scope_fn(scope, *args, **kwargs):
      return unbound_fn(self.clone(parent=scope), *args, **kwargs)
    # here we apply the given lifting transform to the scope-ingesting fn
    trafo_fn = transform(scope_fn, *trafo_args, **trafo_kwargs)
    return trafo_fn(self.scope, *args, **kwargs)

  if instance:
    # return function bound to clone of instance again
    return wrapped_fn.__get__(
        instance,  #  TODO(levskaya): double-check is this always safe?
        instance.__class__)
  else:
    return wrapped_fn

vmap = functools.partial(lift_method, lift.vmap)
scan = functools.partial(lift_method, lift.scan)
jit = functools.partial(lift_method, lift.jit)
remat = functools.partial(lift_method, lift.remat)

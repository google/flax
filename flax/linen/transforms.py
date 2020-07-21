"""Lifting / Transforms of Modules."""
import dataclasses
import functools
import inspect
from flax.core import lift
from flax.linen import Module

# Utils
# -----------------------------------------------------------------------------
def clean_clone(x):
  """Remove scopes and tracers from children."""
  if isinstance(x, Module):
    object.__setattr__(x, 'children',
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

# Function listing as Decorator on methods __inside__ class definition.
# -----------------------------------------------------------------------------
def decorator_lift_transform(transform, class_fn, *trafo_args, **trafo_kwargs):
  @functools.wraps(class_fn)
  def wrapped_fn(self, *args, **kwargs):
    # make a scope-function to transform
    def scope_fn(scope, *args, **kwargs):
      cloned = self.clone(parent=scope)
      res = class_fn(cloned, *args, **kwargs)
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

named_call = functools.partial(decorator_lift_transform, lift.named_call)

vmap = functools.partial(lift_transform, lift.vmap)
jit = functools.partial(lift_transform, lift.jit)
remat = functools.partial(lift_transform, lift.remat)


# Module method lifting
# -----------------------------------------------------------------------------
# Concise, but problematic for reconstruction of submodules....

def lift_instance_method(transform, target, *trafo_args, **trafo_kwargs):
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
      cloned = self.clone(parent=scope)
      res = unbound_fn(cloned, *args, **kwargs)
      # preserve submodule-tree stripped of scopes/tracers for introspection
      object.__setattr__(self, 'children', clean_clone(cloned).children)
      return res
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

vmap_instance_method = functools.partial(lift_instance_method, lift.vmap)
jit_instance_method = functools.partial(lift_instance_method, lift.jit)
remat_instance_method = functools.partial(lift_instance_method, lift.remat)


# Scan specific lifting
# TODO(levskaya): when using params in scan mode, init works, but apply
#                 then seems busted. investigate and fix...
def scan(
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
      ret = lift.scan(scope_fn, self.scope, *args, **trafo_kwargs)
      # ret = trafo_fn(self.scope, *args, **kwargs)
      return ret
    transformed_fns[fn_name] = wrapped_fn
  # construct new dynamic class w. transformed methods
  return type('Scan' + module_class.__name__,
              (module_class,),
              transformed_fns)

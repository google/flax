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

"""JAX transformations on Modules.

Jax functional transformations operate on pure functions.
Flax extends these transformations to also operate on Module's which
have stateful variables and PRNG sequences. We refer to these extended
versions as "lifted transformations".

A lifted transformation can be applied to a ``Module`` class or a
function that takes a ``Module`` instance as its first argument.
"""
from typing import Any, Type, Callable, Union, Mapping, Optional, TypeVar, Iterable

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

  # Handle partially initialized module class constructors.
  if (isinstance(module_class, functools.partial) and
      issubclass(module_class.func, Module)):
    partial_object = module_class
    module_class = module_class.func
  else:
    partial_object = None

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
  transformed_cls = type(
      transform.__name__.capitalize() + module_class.__name__,
      (module_class,),
      transformed_fns)
  # Handle partially initialized module class constructors.
  if partial_object is not None:
    transformed_cls = functools.partial(transformed_cls,
                                        *partial_object.args,
                                        **partial_object.keywords)
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
  if (inspect.isclass(target) and issubclass(target, Module)
      or isinstance(target, functools.partial)):
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


TransformTarget = Union[Type[Module], Callable[..., Any]]

Target = TypeVar('Target', bound=TransformTarget)

def vmap(target: Target,
         variable_axes: Mapping[lift.CollectionFilter, lift.InOutAxis],
         split_rngs: Mapping[lift.PRNGSequenceFilter, bool],
         in_axes=0, out_axes=0,
         axis_size: Optional[int] = None,
         axis_name: Optional[str] = None,
         methods=None) -> Target:
  """A lifted version of ``jax.vmap``.

  See ``jax.vmap`` for the unlifted batch transform in Jax.

  ``vmap`` can be used to add a batch axis to a ``Module``.
  For example we could create a version of ``Dense`` with
  a batch axis that does not share parameters::
  
    BatchDense = nn.vmap(
        nn.Dense,
        in_axes=0, out_axes=0,
        variable_axes={'params': 0},
        split_rngs={'params': True})

  By using ``variable_axes={'params': 0}``, we indicate that the
  parameters themselves are mapped over and therefore not shared along
  the mapped axis. Consequently, we also split the 'params' RNG,
  otherwise the parameters would be initialized identically along
  the mapped axis.

  Similarly, ``vmap`` could be use to add a batch axis with parameter
  sharing::

    BatchFoo = nn.vmap(
        Foo,
        in_axes=0, out_axes=0,
        variable_axes={'params': None},
        split_rngs={'params': False})

  Here we use ``variable_axes={'params': None}`` to indicate the parameter
  variables are shared along the mapped axis. Consequently, the 'params'
  RNG must also be shared.

  Args:
    target: a ``Module`` or a function taking a ``Module``
      as its first argument.
    variable_axes: the variable collections that are lifted into the
      batching transformation. Use `None` to indicate a broadcasted
      collection or an integer to map over an axis.
    split_rngs: Split PRNG sequences will be different for each index
      of the batch dimension. Unsplit PRNGs will be broadcasted.
    in_axes: Specifies the mapping of the input arguments (see `jax.vmap).
    out_axes: Specifies the mapping of the return value (see `jax.vmap).
    axis_size: Specifies the size of the batch axis. This only needs
      to be specified if it cannot be derived from the input arguments.
    axis_name: Specifies a name for the batch axis. Can be used together
      with parallel reduction primitives (e.g. `jax.lax.pmean`,
      `jax.lax.ppermute`, etc.)
  """
  return lift_transform(
      lift.vmap, target, variable_axes, split_rngs,
      methods=methods,
      in_axes=in_axes, out_axes=out_axes,
      axis_size=axis_size, axis_name=axis_name)


def jit(target: Target,
        variables: lift.CollectionFilter = True,
        rngs: lift.PRNGSequenceFilter = True,
        static_argnums: Union[int, Iterable[int]] = (),
        donate_argnums: Union[int, Iterable[int]] = (),
        device=None,
        backend: Union[str, None] = None,
        methods=None) -> Target:
  """Lifted version of ``jax.jit``.
  
  Args:
    target: a ``Module`` or a function taking a ``Module``
      as its first argument.
    variables: The variable collections that are lifted. By default all
      collections are lifted.
    rngs: The PRNG sequences that are lifted. By default all PRNG sequences
      are lifted.
    static_argnums: An int or collection of ints specifying which positional
      arguments to treat as static (compile-time constant). Operations that only
      depend on static arguments will be constant-folded in Python (during
      tracing), and so the corresponding argument values can be any Python
      object. Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation. If
      the jitted function is called with fewer positional arguments than
      indicated by ``static_argnums`` then an error is raised. Arguments that
      are not arrays or containers thereof must be marked as static.
      Defaults to ().
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited from
      XLA's DeviceAssignment logic and is usually to use ``jax.devices()[0]``.
    backend: a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

  Returns:
    A wrapped version of target, set up for just-in-time compilation.
  """
  return lift_transform(
      lift.jit, target,
      variables=variables, rngs=rngs,
      static_argnums=static_argnums,
      donate_argnums=donate_argnums,
      device=device,
      backend=backend,
      methods=methods)


def checkpoint(target: Target,
        variables: lift.CollectionFilter = True,
        rngs: lift.PRNGSequenceFilter = True,
        concrete: bool = False,
        methods=None) -> Target:
  """Lifted version of ``jax.checkpoint``.
  
  This function is aliased to ``lift.remat`` just like ``jax.remat``.

  Args:
    target: a ``Module`` or a function taking a ``Module``
      as its first argument. intermediate computations will be
      re-computed when computing gradients for the target.
    variables: The variable collections that are lifted. By default all
      collections are lifted.
    rngs: The PRNG sequences that are lifted. By default all PRNG sequences
      are lifted.
    concrete: Optional, boolean indicating whether ``fun`` may involve
      value-dependent Python control flow (default False). Support for such
      control flow is optional, and disabled by default, because in some
      edge-case compositions with :func:`jax.jit` it can lead to some extra
      computation.
  Returns:
    A wrapped version of ``target``. When computing gradients intermediate
    computations will be re-computed on the backward pass.
  """
  return lift_transform(
      lift.checkpoint, target,
      variables=variables, rngs=rngs,
      concrete=concrete, methods=methods)


remat = checkpoint


def scan(target: Target,
         variable_axes: Mapping[lift.CollectionFilter, lift.InOutScanAxis] = {},
         variable_broadcast: lift.CollectionFilter = False,
         variable_carry: lift.CollectionFilter = False,
         split_rngs: Mapping[lift.PRNGSequenceFilter, bool] = {},
         in_axes=0, out_axes=0,
         length: Optional[int] = None,
         reverse: bool = False,
         methods=None) -> Target:
  """A lifted version of ``jax.lax.scan``.

  See ``jax.lax.scan`` for the unlifted scan in Jax.

  To improve consistency with ``vmap``, this version of scan
  uses ``in_axes`` and ``out_axes`` to determine which arguments
  are scanned over and along which axis.

  ``scan`` distinguishes between 3 different types of values inside the loop:

  1. **scan**: a value that is iterated over in a loop. All scan values must
    have the same size in the axis they are scanned over. Scanned outputs
    will be stacked along the scan axis.
  2. **carry**: A carried value is updated at each loop iteration. It must
    have the same shape and dtype throughout the loop.
  3. **broadcast**: a value that is closed over by the loop. When a variable
    is broadcasted they are typically initialized inside the loop body but
    independent of the loop variables.

  The loop body should have the signature
  ``(scope, body, carry, *xs) -> (carry, ys)``, where ``xs`` and ``ys``
  are the scan values that go in and out of the loop.

  Example::

    import flax
    import flax.linen as nn
    from jax import random

    class SimpleScan(nn.Module):
      @nn.compact
      def __call__(self, c, xs):
        LSTM = nn.scan(nn.LSTMCell,
                       variable_broadcast="params",
                       split_rngs={"params": False},
                       in_axes=1,
                       out_axes=1)
        return LSTM()(c, xs)

    seq_len, batch_size, in_feat, out_feat = 20, 16, 3, 5
    key_1, key_2, key_3 = random.split(random.PRNGKey(0), 3)

    xs = random.uniform(key_1, (batch_size, seq_len, in_feat))
    init_carry = nn.LSTMCell.initialize_carry(key_2, (batch_size,), out_feat)

    model = SimpleScan()
    variables = model.init(key_3, init_carry, xs)
    out_carry, out_val = model.apply(variables, init_carry, xs)

    assert out_val.shape == (batch_size, seq_len, out_feat)


  Args:
    target: a ``Module`` or a function taking a ``Module``
      as its first argument.
    variable_axes: the variable collections that are scanned over.
    variable_broadcast: Specifies the broadcasted variable collections.
      A broadcasted variable should not depend on any computation that cannot be lifted out of the loop.
      This is typically used to define shared parameters inside the fn.
    variable_carry: Specifies the variable collections that are carried through the loop.
      Mutations to these variables are carried to the next iteration and will be preserved
      when the scan finishes.
    split_rngs: Split PRNG sequences will be different for each loop iterations.
      If split is False the PRNGs will be the same across iterations.
    in_axes: Specifies the axis to scan over for the arguments. Should be a prefix
      tree of the arguments. Use `flax.core.broadcast` to feed an entire input
      to each iteration of the scan body.
    out_axes: Specifies the axis to scan over for the return value. Should be a prefix
      tree of the return value.
    length: Specifies the number of loop iterations. This only needs
      to be specified if it cannot be derivied from the scan arguments.
    reverse: If true, scan from end to start in reverse order.
  Returns:
    The scan function with the signature ``(scope, carry, *xxs) -> (carry, yys)``,
    where ``xxs`` and ``yys`` are the scan values that go in and out of the loop.
  """
  return lift_transform(
      lift.scan, target,
      variable_axes=variable_axes,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      split_rngs=split_rngs,
      in_axes=in_axes, out_axes=out_axes,
      length=length,
      reverse=reverse,
      methods=methods)


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

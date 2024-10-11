# Copyright 2024 The Flax Authors.
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

from collections import deque
import dataclasses
import functools
import typing as tp


from flax import struct
from flax.nnx import (
  extract,
  filterlib,
  graph,
  variablelib,
)
from flax.nnx.statelib import State
import jax
import jax.core
import jax.stages

from flax.nnx.transforms import general
from flax.nnx.transforms.transforms import resolve_kwargs
from flax.typing import MISSING, Missing


A = tp.TypeVar('A')
# C = tp.TypeVar('C')
# B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
# G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
# M = tp.TypeVar('M', bound=Module)
# MA = tp.TypeVar('MA', bound=Module)
# N = tp.TypeVar('N', bound=Module)
# StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
# Leaves = tp.List[Leaf]
# Index = int


# -------------------------------
# grad
# -------------------------------


@dataclasses.dataclass(frozen=True)
class DiffState:
  argnum: int
  filter: filterlib.Filter


@dataclasses.dataclass(eq=False)
class GradFn:
  f: tp.Callable[..., tp.Any]
  has_aux: bool

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args):
    # rebuild diff_state from substates in args
    nondiff_states: deque[State | None] = extract.get_broadcast_state('grad')

    def _grad_merge_fn(
      ctx: graph.MergeContext, path, prefix, value: extract.NodeStates
    ):
      nondiff = nondiff_states.popleft()
      if nondiff is None:
        return ctx.merge(value.graphdef, value.state)
      else:
        return ctx.merge(value.graphdef, value.state, nondiff)

    args = extract.from_tree(pure_args, merge_fn=_grad_merge_fn, ctxtag='grad')

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree((args_out, out), ctxtag='grad')

    if self.has_aux:
      loss, pure_aux = pure_out
      fn_out = (loss, (pure_args_out, pure_aux))
    else:
      loss = pure_out
      fn_out = (loss, pure_args_out)

    return fn_out


def _grad_general(
  f: tp.Callable[..., tp.Any],
  argnums: int | DiffState | tp.Sequence[int | DiffState],
  has_aux: bool,
  holomorphic: bool,
  allow_int: bool,
  reduce_axes: tp.Sequence[AxisName],
  return_value: bool,
) -> tp.Callable[..., tp.Any]:
  transform = jax.value_and_grad if return_value else jax.grad

  jax_argnums: int | tuple[int, ...]
  if isinstance(argnums, (int, DiffState)):
    jax_argnums = argnums.argnum if isinstance(argnums, DiffState) else argnums
  else:
    jax_argnums = tuple(
      x.argnum if isinstance(x, DiffState) else x for x in argnums
    )

  _argnums = (argnums,) if isinstance(argnums, (int, DiffState)) else argnums
  index_filter: dict[int, DiffState] = {}
  for argnum in _argnums:
    index = argnum.argnum if isinstance(argnum, DiffState) else argnum
    if index in index_filter:
      raise ValueError(f'argnum {index} is repeated in argnums')
    index_filter[index] = (
      dataclasses.replace(argnum, argnum=-1)
      if isinstance(argnum, DiffState)
      else DiffState(-1, variablelib.Param)
    )

  gradded_fn = transform(
    GradFn(f, has_aux),
    argnums=jax_argnums,
    has_aux=True,
    holomorphic=holomorphic,
    allow_int=allow_int,
    reduce_axes=reduce_axes,
  )

  @graph.update_context('grad')
  def grad_wrapper(*args, **kwargs):
    args = resolve_kwargs(f, args, kwargs)
    del kwargs
    nondiff_states: deque[State | None] = deque()

    def _grad_split_fn(
      ctx: graph.SplitContext, path, prefix: DiffState | None, value
    ):
      if prefix is None:
        nondiff_states.append(None)
        return extract.NodeStates.from_split(*ctx.split(value))
      else:
        graphdef, diff, nondiff = ctx.split(value, prefix.filter, ...)  # type: ignore[misc]
        nondiff_states.append(nondiff)
        return extract.NodeStates.from_split(graphdef, diff)

    arg_filters = tuple(index_filter.get(i) for i in range(len(args)))
    pure_args = extract.to_tree(
      args, prefix=arg_filters, split_fn=_grad_split_fn, ctxtag='grad'
    )

    with extract.broadcast_state('grad', nondiff_states):
      fn_out = gradded_fn(*pure_args)

    def process_grads(grads):
      return jax.tree.map(
        lambda x: x.state if isinstance(x, extract.NodeStates) else x,
        grads,
        is_leaf=lambda x: isinstance(x, extract.NodeStates),
      )

    def process_out(pure_out: A, /) -> A:
      return extract.from_tree(pure_out, ctxtag='grad')

    if return_value:
      # unpack value_and_grad output
      if has_aux:
        (loss, (pure_args_out, pure_aux)), grads = fn_out
        grads = process_grads(grads)
        _args_out, aux = process_out((pure_args_out, pure_aux))
        return (loss, aux), grads
      else:
        (loss, pure_args_out), grads = fn_out
        grads = process_grads(grads)
        _args_out = process_out(pure_args_out)
        return loss, grads
    else:
      # unpack grad output
      if has_aux:
        grads, (pure_args_out, pure_aux) = fn_out
        grads = process_grads(grads)
        _args_out, aux = process_out((pure_args_out, pure_aux))
        return grads, aux
      else:
        grads, pure_args_out = fn_out
        grads = process_grads(grads)
        _args_out = process_out(pure_args_out)
        return grads

  return grad_wrapper


@tp.overload
def grad(
  f: tp.Callable[..., tp.Any],
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Any]: ...
@tp.overload
def grad(
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...
def grad(
  f: tp.Callable[..., tp.Any] | Missing = MISSING,
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> (
  tp.Callable[..., tp.Any]
  | tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]
):
  """Lifted version of ``jax.grad`` that can handle Modules / graph nodes as
  arguments.

  The differentiable state of each graph node is defined by the `wrt` filter,
  which by default is set to `nnx.Param`. Internally the ``State`` of
  graph nodes is extracted, filtered according to `wrt` filter, and
  passed to the underlying ``jax.grad`` function. The gradients
  of graph nodes are of type ``State``.

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 3))
    ...
    >>> loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    >>> grad_fn = nnx.grad(loss_fn)
    ...
    >>> grads = grad_fn(m, x, y)
    >>> jax.tree.map(jnp.shape, grads)
    State({
      'bias': VariableState(
        type=Param,
        value=(3,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(2, 3)
      )
    })

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, graph nodes or standard Python
      containers. Argument arrays in the positions specified by ``argnums`` must
      be of inexact (i.e., floating-point or complex) type. It should return a
      scalar (which includes arrays with shape ``()`` but not arrays with shape
      ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``grad(f, reduce_axes=('batch',))`` will create a
      function that computes the total gradient while ``grad(f)`` will create
      one that computes the per-example gradient.
  """

  if isinstance(f, Missing):
    return functools.partial(
      grad,
      argnums=argnums,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
    )
  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    return_value=False,
  )


@tp.overload
def value_and_grad(
  f: tp.Callable[..., tp.Any],
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Any]: ...
@tp.overload
def value_and_grad(
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]: ...
def value_and_grad(
  f: tp.Callable[..., tp.Any] | type[Missing] = Missing,
  *,
  argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> (
  tp.Callable[..., tp.Any]
  | tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]]
):
  if f is Missing:
    return functools.partial(
      value_and_grad,
      argnums=argnums,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
    )
  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    return_value=True,
  )


def _custom_vjp_merge_fn(
  ctx: graph.MergeContext,
  path,
  prefix: bool | DiffState,
  value: extract.NodeStates,
  *,
  nondiff_states: deque[extract.GraphDefState],
):
  nondiff = nondiff_states.popleft()
  return ctx.merge(nondiff.graphdef, value.state, nondiff.state)


def _custom_vjp_split_fn(
  ctx: graph.SplitContext,
  path,
  prefix: bool | DiffState,
  value,
  *,
  nondiff_states: deque[extract.GraphDefState],
):
  if prefix is False:
    # pure non-differentiable arg, we pass all the state through
    # but we return TreeNode.from_split with a graphdef to we can call from_tree
    # on the nondiff args during the backward pass
    graphdef, passed = ctx.split(value)
    broadcast = State({})  # type: ignore[var-annotated]
    nondiff_states.append(extract.GraphDefState(graphdef, broadcast))
    return extract.NodeStates.from_split(graphdef, passed)
  elif prefix is True:
    # pure differentiable arg, we pass all the state through
    # but we return a TreeNode.from_states which doesn't have a graphdef
    # in order to keep the gradients clean from any metadata
    graphdef, passed = ctx.split(value)
    broadcast = State({})
    nondiff_states.append(extract.GraphDefState(graphdef, broadcast))
    return extract.NodeStates.from_states(passed)
  else:
    # differentiable arg with DiffState filter, we use the filter to split the state
    # as before we return a TreeNode.from_states to keep the gradients clean
    # from any metadata, the non-differentiable state is stored in a deque
    # which is broadcasted during the forward pass
    graphdef, passed, broadcast = ctx.split(value, prefix.filter, ...)  # type: ignore[misc]
    nondiff_states.append(extract.GraphDefState(graphdef, broadcast))
    return extract.NodeStates.from_states(passed)


class CustomVjpMetadata(struct.PyTreeNode):
  tangent_tree_node_args: tuple[tp.Any, ...] = struct.field(pytree_node=False)


@dataclasses.dataclass(eq=False)
class CustomVjpFnWrapper:
  f: tp.Callable[..., tp.Any]
  ctxtag: str

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args):
    broadcast: tuple[CustomVjpMetadata, deque[extract.GraphDefState]] = (
      extract.get_broadcast_state(self.ctxtag)
    )
    metadata, nondiff_states = broadcast
    args = extract.from_tree(
      pure_args,
      merge_fn=functools.partial(
        _custom_vjp_merge_fn, nondiff_states=nondiff_states
      ),
      ctxtag=self.ctxtag,
    )

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
      (args_out, out), ctxtag=self.ctxtag
    )

    return pure_args_out, pure_out


@dataclasses.dataclass(eq=False)
class FwdFn:
  fwd: tp.Callable[..., tp.Any]
  ctxtag: str

  def __post_init__(self):
    functools.update_wrapper(self, self.fwd)

  def __call__(self, *pure_args):
    broadcast: tuple[CustomVjpMetadata, deque[extract.GraphDefState]] = (
      extract.get_broadcast_state(self.ctxtag)
    )
    metadata, nondiff_states = broadcast
    args = extract.from_tree(
      pure_args,
      merge_fn=functools.partial(
        _custom_vjp_merge_fn, nondiff_states=nondiff_states
      ),
      ctxtag=self.ctxtag,
    )

    out, residual = self.fwd(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
      (args_out, out), ctxtag=self.ctxtag
    )
    pure_residual = extract.to_tree(residual)

    return (pure_args_out, pure_out), (metadata, pure_residual)


@dataclasses.dataclass(eq=False)
class BwdFn:
  bwd: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.bwd)

  def __call__(self, *args):
    res: tuple[CustomVjpMetadata, tp.Any]
    pure_g: tuple[tp.Any, tp.Any]
    *nondiff, res, pure_g = args
    metadata, pure_residual = res
    nondiff = extract.from_tree(nondiff)
    residual = extract.from_tree(pure_residual)
    pure_g = jax.tree.map(
      lambda x: x.state if isinstance(x, extract.NodeStates) else x,
      pure_g,
      is_leaf=lambda x: isinstance(x, extract.NodeStates),
    )

    tangent = self.bwd(*nondiff, residual, pure_g)

    def state_to_tree_node(is_tree_node: bool, x):
      if is_tree_node:
        if not isinstance(x, State):
          raise ValueError(f'Expected State, got {type(x)}')
        return extract.NodeStates.from_states(x)
      return x

    pure_tangent = jax.tree.map(
      state_to_tree_node,
      metadata.tangent_tree_node_args,
      tangent,
      is_leaf=lambda x: isinstance(x, State),
    )
    return pure_tangent


class CustomVjp(tp.Generic[A]):
  def __init__(
    self,
    fun: tp.Callable[..., A],
    nondiff_argnums: tuple[int | DiffState, ...],
  ):
    functools.update_wrapper(self, fun)
    jax_nondiff_argnums = tuple(
      x.argnum if isinstance(x, DiffState) else x for x in nondiff_argnums
    )
    self.ctxtag = f'custom_vjp_{fun.__name__}_{id(fun)}'
    self.custom_vjp_fn = jax.custom_vjp(
      CustomVjpFnWrapper(fun, self.ctxtag),
      nondiff_argnums=jax_nondiff_argnums,
    )
    self.nondiff_argnums = nondiff_argnums
    self.diff_filter: dict[int, tp.Literal[False] | DiffState] = {}
    for argnum in self.nondiff_argnums:
      index = argnum.argnum if isinstance(argnum, DiffState) else argnum
      if index in self.diff_filter:
        raise ValueError(f'argnum {index} is repeated in nondiff_argnums')
      self.diff_filter[index] = (
        dataclasses.replace(argnum, argnum=-1)
        if isinstance(argnum, DiffState)
        else False
      )

  def __getattr__(self, name: str) -> tp.Any:
    return getattr(self.custom_vjp_fn, name)

  def __call__(
    self, *args: tp.Any, **kwargs: tp.Any
  ) -> A:  # pytype: disable=invalid-annotation
    with graph.update_context(self.ctxtag):
      args = resolve_kwargs(self.custom_vjp_fn, args, kwargs)
      del kwargs
      nondiff_states: deque[extract.GraphDefState] = deque()
      arg_filters = tuple(
        self.diff_filter.get(i, True) for i in range(len(args))
      )
      pure_args = extract.to_tree(
        args,
        prefix=arg_filters,
        split_fn=functools.partial(
          _custom_vjp_split_fn, nondiff_states=nondiff_states
        ),
        ctxtag=self.ctxtag,
      )
      tangent_args = tp.cast(
        tuple[tp.Literal[True] | DiffState, ...],
        tuple(x for x in arg_filters if x is not False),
      )
      tree_node_args = jax.tree.map(
        lambda x: isinstance(x, extract.NodeStates),
        pure_args,
        is_leaf=lambda x: isinstance(x, extract.NodeStates),
      )
      tangent_tree_node_args = tuple(
        arg
        for arg, is_tree_node in zip(args, tree_node_args)
        if is_tree_node is not False
      )
      metadata = CustomVjpMetadata(tangent_args)

      with extract.broadcast_state(self.ctxtag, (metadata, nondiff_states)):
        pure_args_out, pure_out = self.custom_vjp_fn(*pure_args)

      args_out, out = extract.from_tree(
        (pure_args_out, pure_out), ctxtag=self.ctxtag
      )

      return out

  def defvjp(
    self,
    fwd: tp.Callable[..., tuple[A, tp.Any]],
    bwd: tp.Callable[..., tuple[tp.Any, ...]],
    symbolic_zeros: bool = False,
  ) -> None:
    """Define a custom VJP rule for the function represented by this instance.

    Args:
      fwd: a Python callable representing the forward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``fwd`` function has
        the same input signature as the underlying primal function. It should
        return as output a pair, where the first element represents the primal
        output and the second element represents any "residual" values to store
        from the forward pass for use on the backward pass by the function
        ``bwd``. Input arguments and elements of the output pair may be arrays
        or nested tuples/lists/dicts thereof.
      bwd: a Python callable representing the backward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``bwd`` function takes
        two arguments, where the first is the "residual" values produced on the
        forward pass by ``fwd``, and the second is the output cotangent with the
        same structure as the primal function output. The output of ``bwd`` must
        be a tuple of length equal to the number of arguments of the primal
        function, and the tuple elements may be arrays or nested
        tuples/lists/dicts thereof so as to match the structure of the primal
        input arguments.
      symbolic_zeros: boolean, determining whether to indicate symbolic zeros
        to the ``fwd`` and ``bwd`` rules. Enabling this option allows custom
        derivative rules to detect when certain inputs, and when certain
        output cotangents, are not involved in differentiation. If ``True``:

        * ``fwd`` must accept, in place of each leaf value ``x`` in
          the pytree comprising an argument to the original function,
          an object (of type
          ``jax.custom_derivatives.CustomVJPPrimal``) with two
          attributes instead: ``value`` and ``perturbed``. The
          ``value`` field is the original primal argument, and
          ``perturbed`` is a boolean.  The ``perturbed`` bit indicates
          whether the argument is involved in differentiation (i.e.,
          if it is ``False``, then the corresponding Jacobian "column"
          is zero).

        * ``bwd`` will be passed objects representing static symbolic zeros in
          its cotangent argument in correspondence with unperturbed values;
          otherwise, only standard JAX types (e.g. array-likes) are passed.

        Setting this option to ``True`` allows these rules to detect whether
        certain inputs and outputs are not involved in differentiation, but at
        the cost of special handling. For instance:

        * The signature of ``fwd`` changes, and the objects it is passed cannot
          be output from the rule directly.

        * The ``bwd`` rule is passed objects that are not entirely array-like,
          and that cannot be passed to most ``jax.numpy`` functions.

        * Any custom pytree nodes involved in the primal function's arguments
          must accept, in their unflattening functions, the two-field record
          objects that are given as input leaves to the ``fwd`` rule.

        Default ``False``.

    Returns:
      None.

    Examples:

      @jax.custom_vjp
      def f(x, y):
        return jnp.sin(x) * y

      def f_fwd(x, y):
        return f(x, y), (jnp.cos(x), jnp.sin(x), y)

      def f_bwd(res, g):
        cos_x, sin_x, y = res
        return (cos_x * g * y, sin_x * g)

      f.defvjp(f_fwd, f_bwd)
    """

    self.custom_vjp_fn.defvjp(
      FwdFn(fwd, self.ctxtag),
      BwdFn(bwd),
      symbolic_zeros=symbolic_zeros,
    )


@tp.overload
def custom_vjp(
  fun: tp.Callable[..., A],
  *,
  nondiff_argnums: tuple[int | DiffState, ...] = (),
) -> CustomVjp[A]: ...
@tp.overload
def custom_vjp(
  *,
  nondiff_argnums: tuple[int | DiffState, ...] = (),
) -> tp.Callable[[tp.Callable[..., A]], CustomVjp[A]]: ...
def custom_vjp(
  fun: tp.Callable[..., A] | Missing = MISSING,
  *,
  nondiff_argnums: tuple[int | DiffState, ...] = (),
) -> CustomVjp[A] | tp.Callable[[tp.Callable[..., A]], CustomVjp[A]]:
  """Reference aware version of
  `jax.custom_vjp <https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_vjp.html>`__.

  Example::

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, x, y):
    ...     self.x = nnx.Param(x)
    ...     self.y = nnx.Param(y)
    ...
    >>> @nnx.custom_vjp
    ... def f(m: Foo):
    ...   return jnp.sin(m.x) * m.y
    ...
    >>> def f_fwd(m: Foo):
    ...   return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)
    ...
    >>> def f_bwd(res, g):
    ...   inputs_g, out_g = g
    ...   cos_x, sin_x, m = res
    ...   tangent_m = nnx.State(dict(x=cos_x * out_g * m.y, y=sin_x * out_g))
    ...   return (tangent_m,)
    ...
    >>> f.defvjp(f_fwd, f_bwd)
    ...
    >>> m = Foo(x=jnp.array(1.), y=jnp.array(2.))
    >>> grads = nnx.grad(f)(m)
    ...
    >>> jax.tree.map(jnp.shape, grads)
    State({
      'x': VariableState(
        type=Param,
        value=()
      ),
      'y': VariableState(
        type=Param,
        value=()
      )
    })

  """
  if isinstance(fun, Missing):
    return functools.partial(custom_vjp, nondiff_argnums=nondiff_argnums)
  return CustomVjp(fun, nondiff_argnums)


# -------------------------------
# remat
# -------------------------------


@tp.overload
def remat(
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def remat(
  f: F,
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> F: ...
def remat(
  f: F | Missing = MISSING,
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
      remat,
      prevent_cse=prevent_cse,
      static_argnums=static_argnums,
      policy=policy,
    )  # type: ignore[return-value]

  return resolve_kwargs()(
    graph.update_context('remat')(
      general.split_inputs(
        jax.checkpoint(
          general.merge_inputs(f, ctxtag='remat'),
          prevent_cse=prevent_cse,
          static_argnums=static_argnums,
          policy=policy,
        ),
        ctxtag='remat',
      ),
    )
  )

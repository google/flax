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
  nondiff_states: deque[State | None]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args):
    # rebuild diff_state from substates in args

    def _grad_merge_fn(
      ctx: graph.MergeContext, path, prefix, value: extract.NodeStates
    ):
      nondiff = self.nondiff_states.popleft()
      if nondiff is None:
        return ctx.merge(value.graphdef, value.state)
      else:
        return ctx.merge(value.graphdef, value.state, nondiff)

    args = extract.from_tree(
      pure_args, merge_fn=_grad_merge_fn, ctxtag='grad', is_inner=True
    )

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
        nondiff_states.append(nondiff)  # type: ignore[container-type-mismatch]
        return extract.NodeStates.from_split(graphdef, diff)

    arg_filters = tuple(index_filter.get(i) for i in range(len(args)))
    pure_args = extract.to_tree(
      args, prefix=arg_filters, split_fn=_grad_split_fn, ctxtag='grad'
    )

    gradded_fn = transform(
      GradFn(f, has_aux, nondiff_states),
      argnums=jax_argnums,
      has_aux=True,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
    )

    fn_out = gradded_fn(*pure_args)

    def process_grads(grads):
      return jax.tree.map(
        lambda x: x.state if isinstance(x, extract.NodeStates) else x,
        grads,
        is_leaf=lambda x: isinstance(x, extract.NodeStates),
      )

    def process_out(pure_out: A, /) -> A:
      return extract.from_tree(pure_out, ctxtag='grad', is_inner=False)

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

# -----------------------------------------------
# custom_vjp
# -----------------------------------------------
# custom_vjp is one of the most complicated transforms as it requires
# to handle 4 different functions:
# 1. CustomVJP: the main object that runs the outer logic, converts input graph nodes
#    to pytrees and output pytrees to graph nodes.
# 2. CustomVjpFnWrapper: function that wraps the user's function, it converts
#    its input pytrees to graph nodes and output graph nodes to pytrees.
# 3. FwdFn: wraps the user's fwd function, it converts its input pytrees to graph nodes
#    and output graph nodes to pytrees. Since it might run by itself in a separate context,
#    it needs to be aware if the update_context is active or not in order to update the outer
#    referenes.
# 4. BwdFn: wraps the user's bwd function, it converts its input pytrees to graph nodes
#    and output graph nodes to pytrees. It doesn't need to be aware of the outer context
#    since it will never update the outer references as it runs during the backward pass.

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
  nondiff_states: list[extract.GraphDefState],
):
  broadcast: graph.GraphState
  if prefix is False:
    # pure non-differentiable arg, not supported
    raise TypeError(
      'Passing integers to nondiff_argnums for graph nodes arguments in custom_vjp is not supported. '
      f'Got {prefix} at path {jax.tree_util.keystr(path)} for value {value}'
    )
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


  nondiff_argnums: tuple[int, ...] = struct.field(pytree_node=False)
  tangent_tree_node_args: tuple[tp.Any, ...] = struct.field(pytree_node=False)

def _extract_nodedefs(x, *, nodedefs: deque[graph.NodeDef]):
  if isinstance(x, graph.NodeDef):
    assert x.outer_index is not None
    nodedefs.append(x)
    return x.with_no_outer_index()
  return x

@dataclasses.dataclass(eq=False)
class CustomVjpFnWrapper:
  f: tp.Callable[..., tp.Any]
  jax_nondiff_argnums: tuple[int, ...]
  ctxtag: str
  nondiff_states: list[extract.GraphDefState]
  nodedefs: deque[graph.NodeDef]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args):
    nondiff_states = deque(self.nondiff_states)
    args = extract.from_tree(
      pure_args,
      merge_fn=functools.partial(
        _custom_vjp_merge_fn, nondiff_states=nondiff_states
      ),
      ctxtag=self.ctxtag,
      is_inner=True,
    )

    out = self.f(*args)

    # remove nondiff from pure_args_out_g
    args_out = tuple(
      x for i, x in enumerate(args) if i not in self.jax_nondiff_argnums
    )
    args_out = extract.clear_non_graph_nodes(args_out)
    pure_args_out, pure_out = extract.to_tree(
      (args_out, out), ctxtag=self.ctxtag
    )
    # remove outer_index from NodeDef's but store them in global context

    pure_args_out, pure_out = jax.tree.map(
      functools.partial(_extract_nodedefs, nodedefs=self.nodedefs),
      (pure_args_out, pure_out),
      is_leaf=lambda x: isinstance(x, graph.NodeDef),
    )

    return pure_args_out, pure_out


@dataclasses.dataclass(eq=False)
class FwdFn:
  fwd: tp.Callable[..., tp.Any]
  nondiff_argnums: tuple[int, ...]
  ctxtag: str
  nondiff_states: list[extract.GraphDefState]
  nodedefs: deque[graph.NodeDef]

  def __post_init__(self):
    functools.update_wrapper(self, self.fwd)

  def __call__(self, *pure_args):
    # here we need to be aware if the update_context is active or not
    # when its not active, index_mappings will be None
    # when its active, we will remove the index_mappings from the NodeDef's and store them
    # in the index_mappings deque created by CustomVjp
    update_context_active = (
      self.ctxtag in graph.GRAPH_CONTEXT.update_context_stacks
    )
    nondiff_states = deque(self.nondiff_states)
    args = extract.from_tree(
      pure_args,
      merge_fn=functools.partial(
        _custom_vjp_merge_fn, nondiff_states=nondiff_states
      ),
      ctxtag=self.ctxtag if update_context_active else None,
      is_inner=True,
    )

    out, residual = self.fwd(*args)

    # remove nondiff from pure_args_out_g
    args_out = tuple(
      x for i, x in enumerate(args) if i not in self.nondiff_argnums
    )
    args_out = extract.clear_non_graph_nodes(args_out)
    pure_args_out, pure_out = extract.to_tree(
      (args_out, out),
      ctxtag=self.ctxtag if update_context_active else None,
    )
    pure_residual = extract.to_tree(residual)

    if update_context_active:
      # remove outer_index from NodeDef's but store them in global context
      pure_args_out, pure_out = jax.tree.map(
        functools.partial(_extract_nodedefs, nodedefs=self.nodedefs),
        (pure_args_out, pure_out),
        is_leaf=lambda x: isinstance(x, graph.NodeDef),
      )

    return (pure_args_out, pure_out), pure_residual


@dataclasses.dataclass(eq=False)
class BwdFn:
  bwd: tp.Callable[..., tp.Any]
  jax_nondiff_argnums: tuple[int, ...]
  tree_node_args: tuple[tp.Any, ...]

  def __post_init__(self):
    functools.update_wrapper(self, self.bwd)

  def __call__(self, *args):
    *nondiff, pure_residual, (pure_args_out_g, pure_out_g) = args
    residual = extract.from_tree(pure_residual, is_inner=True)
    (pure_args_out_g, pure_out_g) = jax.tree.map(
      lambda x: x.state if isinstance(x, extract.NodeStates) else x,
      (pure_args_out_g, pure_out_g),
      is_leaf=lambda x: isinstance(x, extract.NodeStates),
    )
    iter_nondiff = iter(nondiff)
    iter_pure_args_out_g = iter(pure_args_out_g)
    input_args = tuple(
      next(iter_nondiff)
      if i in self.jax_nondiff_argnums
      else next(iter_pure_args_out_g)
      for i in range(len(nondiff) + len(pure_args_out_g))
    )

    tangent = self.bwd(*input_args, residual, pure_out_g)

    def state_to_node_states(is_differentiable: bool, x):
      if is_differentiable:
        if isinstance(x, jax.Array):
          return x
        elif not isinstance(x, State):
          raise ValueError(f'Expected State, got {type(x)}')
        return extract.NodeStates.from_states(x)
      return x

    pure_tangent = jax.tree.map(
      state_to_node_states,
      self.tree_node_args,
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
    # first argument is metadata
    self.jax_nondiff_argnums = tuple(
      x for x in nondiff_argnums if isinstance(x, int)
    )
    self.ctxtag = f'custom_vjp_{fun.__name__}_{id(fun)}'
    self.fun = fun
    self.fwd: tp.Callable | None = None
    self.bwd: tp.Callable | None = None
    self.symbolic_zeros: bool | None = None
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

  def __call__(
    self, *args: tp.Any, **kwargs: tp.Any
  ) -> A:  # pytype: disable=invalid-annotation
    with graph.update_context(self.ctxtag):
      args = resolve_kwargs(self.fun, args, kwargs)
      del kwargs
      nondiff_states: list[extract.GraphDefState] = []
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
      tree_node_args = jax.tree.map(
        lambda x: isinstance(x, extract.NodeStates),
        pure_args,
        is_leaf=lambda x: isinstance(x, extract.NodeStates),
      )
      tree_node_args = tuple(
        x
        for i, x in enumerate(tree_node_args)
        if i not in self.jax_nondiff_argnums
      )
      nodedefs: deque[graph.NodeDef] = deque()
      if self.fwd is None or self.bwd is None or self.symbolic_zeros is None:
        raise ValueError()

      custom_vjp_fn = jax.custom_vjp(
        fun=CustomVjpFnWrapper(
          f=self.fun,
          jax_nondiff_argnums=self.jax_nondiff_argnums,
          ctxtag=self.ctxtag,
          nondiff_states=nondiff_states,
          nodedefs=nodedefs,
        ),
        nondiff_argnums=self.jax_nondiff_argnums,
      )
      custom_vjp_fn.defvjp(
        fwd=FwdFn(
          fwd=self.fwd,
          nondiff_argnums=self.jax_nondiff_argnums,
          ctxtag=self.ctxtag,
          nondiff_states=nondiff_states,
          nodedefs=nodedefs,
        ),
        bwd=BwdFn(
          bwd=self.bwd,
          jax_nondiff_argnums=self.jax_nondiff_argnums,
          tree_node_args=tree_node_args,
        ),
        symbolic_zeros=self.symbolic_zeros,
      )
      pure_args_out, pure_out = custom_vjp_fn(*pure_args)

      # insert index_mappings
      def _insert_index_mappings(x):
        if isinstance(x, graph.NodeDef):
          nodedef: graph.NodeDef = nodedefs.popleft()
          return nodedef
        return x

      pure_args_out, pure_out = jax.tree_util.tree_map(
        _insert_index_mappings,
        (pure_args_out, pure_out),
        is_leaf=lambda x: isinstance(x, graph.NodeDef),
      )

      args_out, out = extract.from_tree(
        (pure_args_out, pure_out), ctxtag=self.ctxtag, is_inner=False
      )

      return out

  def defvjp(
    self,
    fwd: tp.Callable[..., tuple[A, tp.Any]],
    bwd: tp.Callable[..., tuple[tp.Any, ...]],
    symbolic_zeros: bool = False,
  ) -> None:
    self.fwd = fwd
    self.bwd = bwd
    self.symbolic_zeros = symbolic_zeros


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

  ``nnx.custom_vjp`` accepts Modules and other Flax NNX objects as arguments. The main difference
  with the JAX version is that, because Modules follow reference semantics, they propagate the State
  updates for the inputs as auxiliary outputs. For convenience the signature of the ``bwd`` function
  is modified to have the form::

    (*inputs, residual, output_gradient) -> inputs_tangent

  Where each element in `*inputs` is either:
  * The exact input value if it was declared as non differentiable in `nondiff_argnums`.
  * A ``State`` object representing the gradient of state updates if the input is
    a graph node.
  * ``None`` for all input Arrays.

  The shape of the tanget must be a tuple corresponding to the differentiable
  inputs but with ``State`` terms in place of the corresponding Module terms.

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
    >>> def f_bwd(m_update_g, res, out_g):
    ...   cos_x, sin_x, m = res
    ...   m_g = m_update_g # use as template
    ...   m_g['x'].value = cos_x * out_g * m.y
    ...   m_g['y'].value = sin_x * out_g
    ...   return (m_g,)
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

  Note that the gradient terms on the backward's ``*inputs`` usually have the
  same shape as the objects expected in the output tanget, this means that you can
  just update them to construct the tangent values.

  You can select which substates are differentiable (have a tangent) for Modules and other
  graph nodes by passing a ``DiffState`` with a `Filter <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__
  to ``nondiff_argnums``. For example, here's how to differentiate only the ``x``
  attribute of ``m``::

    >>> x_attr = nnx.PathContains('x')
    ...
    >>> @nnx.custom_vjp(nondiff_argnums=nnx.DiffState(0, x_attr))
    ... def f(m: Foo):
    ...   return jnp.sin(m.x) * m.y  # type: ignore

    >>> def f_fwd(m: Foo):
    ...   y = f(m)
    ...   res = (jnp.cos(m.x), m)  # type: ignore
    ...   return y, res
    ...
    >>> def f_bwd(m_out_g, res, out_g):
    ...   cos_x, m = res
    ...   m_g = m_out_g # use as template
    ...   m_g.x.value = cos_x * out_g * m.y
    ...   del m_g['y'] # y is not differentiable
    ...   return (m_g,)

    >>> f.defvjp(f_fwd, f_bwd)
    ...
    >>> m = Foo(x=jnp.array(1.), y=jnp.array(2.))
    >>> grad = nnx.grad(f, argnums=nnx.DiffState(0, x_attr))(m)
    ...
    >>> jax.tree.map(jnp.shape, grad)
    State({
      'x': VariableState(
        type=Param,
        value=()
      )
    })

  Note that ``grad`` cannot calculate gradients for states that don't have a tangent
  defined by ``custom_vjp``, so in the example above we reuse the same ``x_attr``
  filter to keep ``custom_vjp`` and ``grad`` in sync.

  Args:
    fun: Callable base function.
    nondiff_argnums: Tuple of integers or DiffState objects specifying the
      argument indices that are not differentiated. By default all arguments are
      differentiated. Integers cannot be used to mark graph nodes such as Modules
      as non-differentiable, in this case use a DiffState object. DiffState objects
      define the set of differentiable substates, contrary to what the name of this
      argument suggests, this is done for compatibility with ``grad``.

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
  """A 'lifted' version of the
  `jax.checkpoint <https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html>`__
  (a.k.a. ``jax.remat``).

  ``flax.nnx.remat``, similar to ``jax.checkpoint`` can provide control over, for
    example, how ``flax.nnx.grad`` values are computed and saved during the forward pass versus
    how they are recomputed during the backward pass, trading off memory and FLOPs.

  Learn more in `Flax NNX vs JAX Transformations <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`_.

  To learn about ``jax.remat``, go to JAX's
    `fundamentals of jax.checkpoint <https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#fundamentals-of-jax-checkpoint>`_
    and `practical notes <https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#practical-notes>`_.
  """


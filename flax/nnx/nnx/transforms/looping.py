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

# Copyright 2023 The Flax Authors.
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
# pytype: skip-file
from __future__ import annotations

import dataclasses
import functools
import typing as tp

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx.nnx import extract, filterlib, graph, rnglib, spmd
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.proxy_caller import DelayedAccessor
from flax.nnx.nnx.state import State
from flax.nnx.nnx.transforms.parallelization import Vmap
from flax.nnx.nnx.transforms.transforms import LiftedModule
from flax.typing import Leaf
import jax
from jax._src.tree_util import broadcast_prefix
import jax.core
import jax.numpy as jnp
import jax.stages

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
MA = tp.TypeVar('MA', bound=Module)
N = tp.TypeVar('N', bound=Module)
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int

class Missing:
  pass


MISSING = Missing()

# -------------------------------
# scan
# -------------------------------


@dataclasses.dataclass(frozen=True)
class FlatDef(tp.Generic[A]):
  type: type[A]
  treedef: jax.tree_util.PyTreeDef
  flat_axes: list[int | None]


jax.tree_util.register_static(FlatDef)


def _transpose_tree(tree: A, axes, /, *, move_front: bool) -> A:
  flatdef, flat_transposes, _ = _transpose_and_split(
    tree, axes, allow_none=False, move_front=move_front
  )
  return flatdef.treedef.unflatten(flat_transposes)


def _transpose_and_split(
  tree: A, axes, /, *, allow_none: bool = True, move_front: bool = True
) -> tuple[
  FlatDef[A],
  list[jax.Array | None],
  list[tp.Any],
]:
  flat_axes: list[int | None] = broadcast_prefix(
    axes, tree, is_leaf=lambda x: x is None
  )
  flat_tree, treedef = jax.tree.flatten(tree)

  flat_broadcasts: list[tp.Any] = []
  flat_transposes: list[jax.Array | None] = []

  for i, (axis, node) in enumerate(zip(flat_axes, flat_tree)):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')

      flat_broadcasts.append(node)
      flat_transposes.append(None)
    else:
      if not isinstance(node, jax.Array):
        raise TypeError(
          f'Expected a jax.Array, got {type(node).__name__} for axis {axis}'
        )
      # normalize axis
      if axis < 0:
        if axis < -len(node.shape):
          raise ValueError(
            f'Axis {axis} out of bounds for array with shape {node.shape}'
          )
        axis = len(node.shape) + axis
        flat_axes[i] = axis

      if node.shape == ():
        raise ValueError(f'Cannot map over a scalar array, got {node}')
      elif axis >= len(node.shape):
        raise ValueError(
          f'Axis {axis} out of bounds for array with shape {node.shape}'
        )

      if move_front:
        node = jnp.moveaxis(node, axis, 0)
      else:
        node = jnp.moveaxis(node, 0, axis)
      flat_broadcasts.append(None)
      flat_transposes.append(node)

  flatdef = FlatDef(type(tree), treedef, flat_axes)

  return flatdef, flat_transposes, flat_broadcasts


def _unflatten_splits(
  flatdef: FlatDef[A],
  flat_transposes: list[jax.Array | None],
  flat_broadcasts: list[tp.Any] | None = None,
  /,
  *,
  allow_none: bool = True,
) -> A:
  flat_axes = flatdef.flat_axes
  treedef = flatdef.treedef
  if flat_broadcasts is None:
    if allow_none:
      raise ValueError('flat_broadcasts must be provided if allow_none is True')
    flat_broadcasts = [None] * len(flat_axes)

  flat_tree = []
  for axis, transpose, broadcast in zip(
    flat_axes, flat_transposes, flat_broadcasts
  ):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')
      flat_tree.append(broadcast)
    else:
      if transpose is None:
        raise ValueError('None transpose not allowed')
      flat_tree.append(transpose)

  tree = treedef.unflatten(flat_tree)
  return tree


def _extract_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, /
) -> tuple[tp.Any, tuple[tp.Any, ...]]:
  # extract carry arg
  if len(args) < carry_argnum + 1:
    raise TypeError(
      f'Expected at least {carry_argnum + 1} positional arguments, '
      f'got {len(args)}'
    )

  args_ = list(args)
  carry_arg = args_[carry_argnum]
  args_[carry_argnum] = None
  args = tuple(args_)

  return carry_arg, args


def _insert_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, carry_arg: tp.Any, /
) -> tuple[tp.Any, ...]:
  args_ = list(args)
  args_[carry_argnum] = carry_arg
  args = tuple(args_)

  return args


@struct.dataclass
class ScanBroadcasts(tp.Generic[C, B]):
  flatdef: FlatDef[
    tuple[tuple[tp.Any, ...], dict[str, tp.Any], list[State]]
  ] = struct.field(pytree_node=False)
  flat_carry: list[tp.Any] = struct.field(pytree_node=True)
  graphdef: GraphDef[tuple[tp.Any, ...]] = struct.field(pytree_node=False)
  filters: tuple[filterlib.Filter, ...] = struct.field(pytree_node=False)
  f: tp.Callable[..., tuple[C, B] | C] = struct.field(pytree_node=False)
  # options
  carry_argnum: int = struct.field(pytree_node=False)
  state_axes: tp.Mapping[filterlib.Filter, int] = struct.field(
    pytree_node=False
  )
  split_rngs: filterlib.Filter = struct.field(pytree_node=False)
  transform_metadata: tp.Mapping[str, tp.Any] = struct.field(pytree_node=False)
  scan_output: bool = struct.field(pytree_node=False)


def scan_fn(
  carry: tuple[
    State,  # split_rng_state
    State,  # broadcast_rng_state
    State,  # carry_state
    tp.Any,  # carry_arg
    ScanBroadcasts[C, B],  # broadcasts
  ],
  scan: tuple[
    list[jax.Array | None],  # flat_scan
  ],
):
  split_rng_state, broadcast_rng_state, carry_state, carry_arg, broadcasts = (
    carry
  )
  (flat_scan,) = scan
  flatdef = broadcasts.flatdef
  flat_carry = broadcasts.flat_carry
  graphdef, filters = broadcasts.graphdef, broadcasts.filters
  f = broadcasts.f
  ctx = graph.current_update_context('scan')

  # merge args and kwargs
  args, kwargs, scan_states = _unflatten_splits(flatdef, flat_scan, flat_carry)
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states = [
      spmd.remove_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states, broadcasts.state_axes.values())
    ]

  # insert carry arg
  args = _insert_carry_arg(args, broadcasts.carry_argnum, carry_arg)

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef, *scan_states, carry_state, split_rng_state, broadcast_rng_state
  )
  (args, kwargs) = extract.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  if broadcasts.scan_output:
    if not isinstance(out, tuple) or len(out) != 2:
      raise ValueError(
        'Expected a tuple of length 2 as the output of the scan function, '
        f'got {out}'
      )
    out = tp.cast(tuple[C, B], out)  # type: ignore[invalid-annotation]
    carry_arg_out, scan_args_out = out
  else:
    out = tp.cast(C, out)  # type: ignore[invalid-annotation]
    carry_arg_out = out
    scan_args_out = None

  ((carry_arg_out, scan_args_out), output_graph_nodes) = (
    extract.extract_graph_nodes((carry_arg_out, scan_args_out))
  )

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *scan_states_out,
    carry_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  split_rng_state_out, broadcast_rng_state_out = rng_state_out.split(
    broadcasts.split_rngs, ...
  )

  def _extract_carry_state(state: State, /):
    if 1 in state:
      raise ValueError(
        f'Cannot add new carry state during scan, got {state[1]}'
      )
    if 0 in state:
      _state = state[0]
      assert isinstance(_state, State)
      state = _state

    return state

  carry_state_out = _extract_carry_state(carry_state_out)
  split_rng_state_out = _extract_carry_state(split_rng_state_out)
  broadcast_rng_state_out = _extract_carry_state(broadcast_rng_state_out)

  # override  broadcast_rng_state_out to keep the same state
  # for the next iteration
  broadcast_rng_state_out = broadcast_rng_state

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states_out = [
      spmd.add_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states_out, broadcasts.state_axes.values())
    ]

  carry_out = (
    split_rng_state_out,
    broadcast_rng_state_out,
    carry_state_out,
    carry_arg_out,
    broadcasts,
  )
  scan_out = (graphdef_out, scan_args_out, scan_states_out)

  return carry_out, scan_out

@tp.overload
def scan(
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> p.Callable[[F], F]: ...
@tp.overload
def scan(
  f: F,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> F: ...
def scan(
  f: F | Missing = MISSING,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
      scan, length=length, reverse=reverse, unroll=unroll
    )

  @functools.wraps(f)
  @graph.update_context('scan')
  def scan_apply_wrapper(*args, **kwargs):
    # extract nodes
    (args, kwargs), input_graph_nodes = extract.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # extract carry arg
    carry_arg, args = _extract_carry_arg(args, carry_argnum)

    ctx = graph.current_update_context('scan')
    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *scan_states, carry_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # transpose axes arg
    flatdef, flat_scan, flat_carry = _transpose_and_split(
      (args, kwargs, scan_states),
      (in_axes, in_axes_kwargs, list(state_axes.values())),
    )

    # infer length
    lengths: set[int] = {
      x.shape[0]  # type: ignore
      for x, axis in zip(flat_scan, flatdef.flat_axes)
      if axis is not None
    }

    if len(lengths) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {lengths}'
      )
    elif len(lengths) == 0:
      if length is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      infered_length = length
    else:
      infered_length = lengths.pop()
      if length is not None and length != infered_length:
        raise ValueError(
          f'Specified length {length} is not the same as the inferred '
          f'length {infered_length}'
        )

    # split rng state
    split_rng_state, broadcast_rng_state = rng_state.split(split_rngs, ...)

    broadcasts = ScanBroadcasts(
      flatdef,
      flat_carry,
      graphdef,
      filters,
      f,
      # options
      carry_argnum,
      state_axes,
      split_rngs,
      transform_metadata,
      scan_output,
    )
    carry = (
      split_rng_state,
      broadcast_rng_state,
      carry_state,
      carry_arg,
      broadcasts,
    )
    scan = (flat_scan,)

    carry_out, scan_out = jax.lax.scan(
      scan_fn,
      carry,
      scan,
      length=infered_length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
    )
    (
      split_rng_state_out,
      broadcast_rng_state_out,
      carry_state_out,
      carry_arg_out,
      broadcasts,
    ) = carry_out
    graphdef_out, scan_args_out, scan_states_out = scan_out

    scan_args_out, scan_states_out = _transpose_tree(
      (scan_args_out, scan_states_out),
      (out_axes, list(state_axes.values())),
      move_front=False,
    )

    if carry_state_out:
      carry_state_out = State({0: carry_state_out._mapping})
    if split_rng_state_out:
      split_rng_state_out = State({0: split_rng_state_out._mapping})
    if broadcast_rng_state_out:
      broadcast_rng_state_out = State({0: broadcast_rng_state_out._mapping})

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *scan_states_out,
      carry_state_out,
      split_rng_state_out,
      broadcast_rng_state_out,
    )

    carry_arg_out, scan_args_out = extract.insert_graph_nodes(
      (carry_arg_out, scan_args_out), output_graph_nodes
    )

    rnglib.restore_rngs(input_rng_streams)

    if scan_output:
      scan_args_out = tp.cast(B, scan_args_out)
      return carry_arg_out, scan_args_out
    else:
      return carry_arg_out

  return scan_apply_wrapper  # type: ignore


class Scan(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
  ) -> tp.Callable[..., Scan[MA]]:
    def _create_scan(*args, **kwargs):
      return Scan(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        # base api
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        # extended api
        in_axes=in_axes,
        in_axes_kwargs=in_axes_kwargs,
        out_axes=out_axes,
        carry_argnum=carry_argnum,
        # nnx specific
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        scan_output=scan_output,
      )

    return _create_scan

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    # use Vmap to handle initialisation
    vmapped_module = Vmap.constructor(
      module_constructor,
      in_axes=in_axes,
      out_axes=None,
      axis_name=None,
      axis_size=length,
      spmd_axis_name=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      in_axes_kwargs=in_axes_kwargs,
      transform_metadata=transform_metadata,
    )(*module_init_args, **module_init_kwargs)
    self.scan_module = vmapped_module.vmap_module

    @functools.partial(
      scan,
      length=length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
      in_axes=in_axes,
      in_axes_kwargs=in_axes_kwargs,
      out_axes=out_axes,
      carry_argnum=carry_argnum,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
      scan_output=scan_output,
    )
    def scan_call(module, *args, _nnx_scan_accessor: DelayedAccessor, **kwargs):
      method = _nnx_scan_accessor(module)
      return method(*args, **kwargs)

    self.scan_call = scan_call

  @property
  def _submodule(self) -> M:
    return self.scan_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    return self.scan_call(
      self._submodule, *args, _nnx_scan_accessor=accessor, **kwargs
    )

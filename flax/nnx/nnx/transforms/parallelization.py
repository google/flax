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

import functools
import typing as tp

import jax
import jax.core
import jax.stages
from jax._src.tree_util import broadcast_prefix

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx.nnx import (
  filterlib,
  graph,
  rnglib,
  spmd,
)
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.proxy_caller import (
  DelayedAccessor,
)
from flax.nnx.nnx.state import State
from flax.nnx.nnx.transforms.transforms import LiftedModule
from flax.typing import Leaf

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

# -------------------------------
# vmap
# -------------------------------


def _get_axis_sizes(pytree, axes):
  axes = broadcast_prefix(axes, pytree, is_leaf=lambda x: x is None)
  leaves = jax.tree_util.tree_leaves(pytree)
  axis_sizes = {
    leaf.shape[axis] for axis, leaf in zip(axes, leaves) if axis is not None
  }
  return axis_sizes


def vmap_fn(
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  split_keys: State,
  split_counts: State,
  broadcast_keys: State,
  broadcast_counts: State,
  vectorized_states: list[State],
  broadcast_state: State,
  transform_metadata: tp.Mapping[str, tp.Any],
  state_axes: tp.Mapping[filterlib.Filter, int],
  f: tp.Callable[..., tp.Any],
  filters: tp.Tuple[filterlib.Filter, ...],
  split_rngs: filterlib.Filter,
):
  ctx = graph.current_update_context('vmap')
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states = [
      spmd.remove_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states, state_axes.values())
    ]

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef,
    *vectorized_states,
    broadcast_state,
    split_keys,
    split_counts,
    broadcast_keys,
    broadcast_counts,
  )

  (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = graph.extract_graph_nodes(out)

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *vectorized_states_out,
    broadcast_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  not_keys_out, split_keys_out, broadcast_keys_out = rng_state_out.split(
    rnglib.NotKey, split_rngs, ...
  )

  broadcast_state_out = State.merge(
    broadcast_state_out, broadcast_keys_out, not_keys_out
  )

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states_out = [
      spmd.add_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states_out, state_axes.values())
    ]

  return (
    graphdef_out,
    broadcast_state_out,
    vectorized_states_out,
    split_keys_out,
    out,
  )


def vmap(
  f: F,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  vectorized_states_axes = list(state_axes.values())

  vmapped_fn = jax.vmap(
    vmap_fn,
    in_axes=(
      in_axes,  # args_axes
      in_axes_kwargs,  # kwargs_axes
      None,  # graphdef_axes
      0,  # split_keys_axes
      None,  # split_counts_axes
      None,  # broadcast_keys_axes
      None,  # broadcast_counts_axes
      vectorized_states_axes,  # vectorized_states_axes
      None,  # broadcast_state_axes
      None,  # transform_metadata_axes
      None,  # states_axes_axes
      None,  # f_axes
      None,  # filters_axes
      None,  # split_rngs_axes
    ),
    out_axes=(
      None,  # graphdef_out_axes
      None,  # broadcast_state_axes
      vectorized_states_axes,
      0,  # keys_axes_out
      out_axes,  # out_axes
    ),
    axis_name=axis_name,
    axis_size=axis_size,
    spmd_axis_name=spmd_axis_name,
  )

  @functools.wraps(f)
  @graph.update_context('vmap')
  def vmap_wrapper(*args, **kwargs):
    ctx = graph.current_update_context('vmap')

    (args, kwargs), input_graph_nodes = graph.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # infer length
    axis_sizes: tp.Set[int] = set()
    axis_sizes.update(_get_axis_sizes(args, in_axes))
    axis_sizes.update(_get_axis_sizes(kwargs, in_axes_kwargs))
    for state, state_axis in zip(vectorized_states, state_axes.values()):
      axis_sizes.update(_get_axis_sizes(state, state_axis))

    if len(axis_sizes) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {axis_sizes}'
      )
    elif len(axis_sizes) == 0:
      if axis_size is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      _axis_size = axis_size
    else:
      _axis_size = axis_sizes.pop()
      if axis_size is not None and axis_size != _axis_size:
        raise ValueError(
          f'Specified axis_size {axis_size} is not the same as the'
          f' inferred length {_axis_size}'
        )

    split_keys, split_counts, broadcast_keys, broadcast_counts = rnglib.fork(
      rng_state,
      split_rngs,
      _axis_size,
    )

    (
      graphdef_out,
      broadcast_state,
      vectorized_states,
      split_keys_out,
      out,
    ) = vmapped_fn(
      args,
      kwargs,
      graphdef,
      split_keys,
      split_counts,
      broadcast_keys,
      broadcast_counts,
      vectorized_states,
      broadcast_state,
      transform_metadata,
      state_axes,
      f,
      filters,
      split_rngs,
    )

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *vectorized_states,
      broadcast_state,
      split_keys_out,
    )

    out = graph.insert_graph_nodes(out, output_graph_nodes)

    rnglib.restore_keys(input_rng_streams)

    return out

  return vmap_wrapper  # type: ignore


class Vmap(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  ) -> tp.Callable[..., Vmap[MA]]:
    def _create_vmap(*args, **kwargs):
      return Vmap(
        module_constructor=module_constructor,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=axis_size,
        axis_name=axis_name,
        spmd_axis_name=spmd_axis_name,
        # nnx specific
        in_axes_kwargs=in_axes_kwargs,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_vmap

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor

    @functools.partial(
      vmap,
      in_axes=None,
      out_axes=None,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      in_axes_kwargs=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def vmap_init(*args, **kwargs):
      return module_constructor(*args, **kwargs)

    self.vmap_module = vmap_init(*module_init_args, **module_init_kwargs)

    @functools.partial(
      vmap,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def vmap_call(module, *args, _nnx_vmap_accessor: DelayedAccessor, **kwargs):
      method = _nnx_vmap_accessor(module)
      return method(*args, **kwargs)

    self.vmap_call = vmap_call

  @property
  def _submodule(self) -> M:
    return self.vmap_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs):
    return self.vmap_call(
      self._submodule, *args, _nnx_vmap_accessor=accessor, **kwargs
    )


# -------------------------------
# pmap
# -------------------------------
@struct.dataclass
class PmapInputs:
  transform_metadata: tp.Mapping[str, tp.Any] = struct.field(pytree_node=False)
  state_axes: tp.Mapping[filterlib.Filter, int] = struct.field(
    pytree_node=False
  )
  f: tp.Callable[..., tp.Any] = struct.field(pytree_node=False)
  filters: tp.Tuple[filterlib.Filter, ...] = struct.field(pytree_node=False)
  split_rngs: filterlib.Filter = struct.field(pytree_node=False)


def pmap_fn(
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  split_keys: State,
  split_counts: State,
  broadcast_keys: State,
  broadcast_counts: State,
  vectorized_states: list[State],
  broadcast_state: State,
  pmap_inputs: PmapInputs,
):
  transform_metadata = pmap_inputs.transform_metadata
  state_axes = pmap_inputs.state_axes
  f = pmap_inputs.f
  filters = pmap_inputs.filters
  split_rngs = pmap_inputs.split_rngs
  ctx = graph.current_update_context('pmap')
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states = [
      spmd.remove_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states, state_axes.values())
    ]

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef,
    *vectorized_states,
    broadcast_state,
    split_keys,
    split_counts,
    broadcast_keys,
    broadcast_counts,
  )

  (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = graph.extract_graph_nodes(out)

  # split module state
  (
    graphdef_out,
    rng_state_out,
    *vectorized_states_out,
    broadcast_state_out,
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  not_keys_out, split_keys_out, broadcast_keys_out = rng_state_out.split(
    rnglib.NotKey, split_rngs, ...
  )

  broadcast_state_out = State.merge(
    broadcast_state_out, broadcast_keys_out, not_keys_out
  )

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states_out = [
      spmd.add_axis(state, index, transform_metadata)
      for state, index in zip(vectorized_states_out, state_axes.values())
    ]

  return (
    graphdef_out,
    broadcast_state_out,
    vectorized_states_out,
    split_keys_out,
    out,
  )


def pmap(
  f: F,
  axis_name: AxisName | None = None,
  *,
  in_axes: tp.Any = 0,
  out_axes: tp.Any = 0,
  static_broadcasted_argnums: int | tp.Iterable[int] = (),
  devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
  backend: str | None = None,
  axis_size: int | None = None,
  donate_argnums: int | tp.Iterable[int] = (),
  global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
  # nnx specific
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  if static_broadcasted_argnums:
    raise NotImplementedError(
      'static_broadcasted_argnums is not yet supported in nnx.pmap'
    )
  if donate_argnums != ():
    raise NotImplementedError('donate_argnums is not yet supported in nnx.pmap')

  if global_arg_shapes is not None:
    raise NotImplementedError(
      'global_arg_shapes is not yet supported in nnx.pmap'
    )

  vectorized_states_axes = list(state_axes.values())

  pmapped_fn = jax.pmap(
    pmap_fn,
    axis_name=axis_name,
    in_axes=(
      in_axes,  # args_axes
      in_axes_kwargs,  # kwargs_axes
      None,  # graphdef_axes
      0,  # split_keys_axes
      None,  # split_counts_axes
      None,  # broadcast_keys_axes
      None,  # broadcast_counts_axes
      vectorized_states_axes,  # vectorized_states_axes
      None,  # broadcast_state_axes
      None,  # pmap_inputs_axes
    ),  # type: ignore
    out_axes=(
      None,  # graphdef_out_axes
      None,  # broadcast_state_axes
      vectorized_states_axes,
      0,  # keys_axes_out
      out_axes,  # out_axes
    ),  # type: ignore
    devices=devices,
    backend=backend,
    axis_size=axis_size,
  )

  @functools.wraps(f)
  @graph.update_context('pmap')
  def pmap_wrapper(*args, **kwargs):
    ctx = graph.current_update_context('pmap')

    (args, kwargs), input_graph_nodes = graph.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # infer length
    axis_sizes: tp.Set[int] = set()
    axis_sizes.update(_get_axis_sizes(args, in_axes))
    axis_sizes.update(_get_axis_sizes(kwargs, in_axes_kwargs))
    for state, state_axis in zip(vectorized_states, state_axes.values()):
      axis_sizes.update(_get_axis_sizes(state, state_axis))

    if len(axis_sizes) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {axis_sizes}'
      )
    elif len(axis_sizes) == 0:
      if axis_size is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      _axis_size = axis_size
    else:
      _axis_size = axis_sizes.pop()
      if axis_size is not None and axis_size != _axis_size:
        raise ValueError(
          f'Specified axis_size {axis_size} is not the same as the'
          f' inferred length {_axis_size}'
        )

    split_keys, split_counts, broadcast_keys, broadcast_counts = rnglib.fork(
      rng_state,
      split_rngs,
      _axis_size,
    )

    (
      graphdef_out,
      broadcast_state,
      vectorized_states,
      split_keys_out,
      out,
    ) = pmapped_fn(
      args,
      kwargs,
      graphdef,
      split_keys,
      split_counts,
      broadcast_keys,
      broadcast_counts,
      vectorized_states,
      broadcast_state,
      PmapInputs(
        transform_metadata=transform_metadata,
        state_axes=state_axes,
        f=f,
        filters=filters,
        split_rngs=split_rngs,
      ),
    )

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *vectorized_states,
      broadcast_state,
      split_keys_out,
    )

    out = graph.insert_graph_nodes(out, output_graph_nodes)

    rnglib.restore_keys(input_rng_streams)

    return out

  return pmap_wrapper  # type: ignore


class Pmap(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  ) -> tp.Callable[..., Pmap[MA]]:
    def _create_pmap(*args, **kwargs):
      return Pmap(
        module_constructor=module_constructor,
        axis_name=axis_name,
        in_axes=in_axes,
        out_axes=out_axes,
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        axis_size=axis_size,
        # nnx specific
        in_axes_kwargs=in_axes_kwargs,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_pmap

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    axis_name: AxisName | None = None,
    in_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    static_broadcasted_argnums: int | tp.Iterable[int] = (),
    devices: tp.Sequence[jax.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor

    @functools.partial(
      pmap,
      axis_name=axis_name,
      in_axes=None,
      out_axes=None,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=(),
      global_arg_shapes=None,
      in_axes_kwargs=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def pmap_init(*args, **kwargs):
      return module_constructor(*args, **kwargs)

    self.pmap_module = pmap_init(*module_init_args, **module_init_kwargs)

    @functools.partial(
      pmap,
      axis_name=axis_name,
      in_axes=in_axes,
      out_axes=out_axes,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums,
      global_arg_shapes=global_arg_shapes,
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )
    def pmap_call(module, *args, _nnx_vmap_accessor: DelayedAccessor, **kwargs):
      method = _nnx_vmap_accessor(module)
      return method(*args, **kwargs)

    self.pmap_call = pmap_call

  @property
  def _submodule(self) -> M:
    return self.pmap_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs):
    return self.pmap_call(
      self._submodule, *args, _nnx_vmap_accessor=accessor, **kwargs
    )
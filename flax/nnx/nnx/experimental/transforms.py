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

from dataclasses import MISSING
import functools
import typing as tp

import numpy as np

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
from flax.nnx.nnx.transforms import LiftedModule
from flax.typing import Leaf
import jax
from jax._src.tree_util import broadcast_prefix
import jax.core
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


class Lift(tp.Mapping[A, B]):
  def __init__(
    self,
    mapping: tp.Mapping[A, B],
    /,
    *,
    rngs: filterlib.Filter | Missing = MISSING,
  ):
    self.mapping = dict(mapping)
    self.rngs = rngs

  def get_rngs(self, default: filterlib.Filter, /):
    return self.rngs if not isinstance(self.rngs, Missing) else default

  def __getitem__(self, key: A) -> B:
    return self.mapping[key]

  def __iter__(self):
    return iter(self.mapping)

  def __len__(self):
    return len(self.mapping)

  def __contains__(self, key):
    return key in self.mapping


# -------------------------------
# vmap
# -------------------------------


def vmap_fn(
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
  graphdef: GraphDef[list[tp.Any]],
  split_key_states: list[State],
  vectorized_states: dict[Index, list[State]],
  broadcast_states: list[State],
  transform_metadata: tp.Mapping[str, tp.Any],
  f: tp.Callable[..., tp.Any],
):
  ctx = graph.current_update_context('vmap')
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    vectorized_states = {
      index: [
        spmd.remove_axis(state, index, transform_metadata) for state in states
      ]
      for index, states in vectorized_states.items()
    }

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef,
    *(state for states in vectorized_states.values() for state in states),
    *broadcast_states,
    *split_key_states,
  )

  (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes, _ = graph.extract_graph_nodes(out)

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
  # state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  # split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  vectorized_states_axes: dict[Index, Index] = {}

  for index in jax.tree.leaves((in_axes, in_axes_kwargs)):
    if isinstance(index, int):
      vectorized_states_axes[index] = index
    elif isinstance(index, Lift):
      index = tp.cast(Lift[filterlib.Filter, Index], index)
      vectorized_states_axes.update({i: i for i in index.values()})

  vmapped_fn = jax.vmap(
    vmap_fn,
    in_axes=(
      in_axes,  # args_axes
      in_axes_kwargs,  # kwargs_axes
      None,  # graphdef_axes
      0,  # split_key_states_axes
      vectorized_states_axes,  # vectorized_states_axes
      None,  # broadcast_states_axes
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

    (args, kwargs), input_graph_nodes, node_prefixes = (
      graph.extract_graph_nodes(
        (args, kwargs), prefix=(in_axes, in_axes_kwargs)
      )
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # split module state
    # filters = (*state_axes.keys(), ...)
    graphdef, state = ctx.split(input_graph_nodes)  # type: ignore[misc]
    broadcast_states: list[State] = []
    vectorized_states: dict[int, list[State]] = {
      i: [] for i in vectorized_states_axes
    }
    split_key_states: list[State] = []
    _axis_size = axis_size
    axis_sizes: tp.Set[int] = set()

    if axis_size is not None:
      axis_sizes.add(axis_size)

    # extract states
    for i, state_i in state.items():
      assert isinstance(i, int)
      assert isinstance(state_i, State)
      state_i = State({i: state_i.raw_mapping})
      prefix = node_prefixes[i]

      if prefix is None:
        broadcast_states.append(state_i)
      elif isinstance(prefix, int):
        axis_sizes.update(
          x.shape[prefix] for x in jax.tree_util.tree_leaves(state_i)
        )
        if len(axis_sizes) > 1:
          raise ValueError(f'Inconsistent lengths for axis_size: {axis_sizes}')

        key_state, rest = state_i.split(rnglib.RngKey, ...)
        split_key_states.append(key_state)
        vectorized_states[prefix].append(rest)

      elif isinstance(prefix, Lift):
        prefix = tp.cast(Lift[filterlib.Filter, Index], prefix)
        filters = tuple(prefix.keys())
        indexes = tuple(prefix.values())
        split_rngs = prefix.get_rngs(...)

        split_keys_state, *vectorized_substates, broadcast_substate = (
          state_i.split(filterlib.All(rnglib.RngKey, split_rngs), *filters, ...)
        )
        split_key_states.append(split_keys_state)
        broadcast_states.append(broadcast_substate)

        for index, state_i in zip(indexes, vectorized_substates):
          axis_sizes.update(
            x.shape[index] for x in jax.tree_util.tree_leaves(state_i)
          )
          if len(axis_sizes) > 1:
            raise ValueError(
              f'Inconsistent lengths for axis_size: {axis_sizes}'
            )
          vectorized_states[index].append(state_i)

    prefixes = broadcast_prefix(
      (in_axes, in_axes_kwargs), (args, kwargs), is_leaf=lambda x: x is None
    )
    leaves, _ = jax.tree.flatten((args, kwargs))
    axis_sizes.update(
      x.shape[prefix]
      for x, prefix in zip(leaves, prefixes)
      if isinstance(x, (np.ndarray, jax.Array)) and isinstance(prefix, int)
    )

    # infer length
    if len(axis_sizes) > 1:
      raise ValueError(f'Inconsistent lengths for axis_size: {axis_sizes}')
    elif len(axis_sizes) == 0:
      raise ValueError(
        'Cannot infer axis_size from the inputs, please specify axis_size'
      )
    else:
      _axis_size = axis_sizes.pop()
      if axis_size is not None and axis_size != _axis_size:
        raise ValueError(
          f'Specified axis_size {axis_size} is not the same as the'
          f' inferred length {_axis_size}'
        )

    split_key_states = jax.tree.map(
      lambda x: jax.random.split(x, _axis_size), split_key_states
    )

    (
      graphdef_out,
      broadcast_states_out,
      vectorized_states_out,
      split_key_state_out,
      out,
    ) = vmapped_fn(
      args,
      kwargs,
      graphdef,
      split_key_states,
      vectorized_states,
      broadcast_states,
      transform_metadata,
      f,
    )

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *vectorized_states_out,
      *broadcast_states_out,
      *split_key_state_out,
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
  ) -> tp.Callable[..., 'Vmap[MA]']:
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

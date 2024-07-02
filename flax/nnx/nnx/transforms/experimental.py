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
from flax.nnx.nnx import (
  extract,
  filterlib,
  graph,
  spmd,
)
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.state import State
from flax.typing import Leaf
import jax
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

# -------------------------------
# vmap
# -------------------------------


class VmapArgState(extract.ExtractionIndex, extract.ExtractableStates):
  _graphdef: GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: State = struct.field(pytree_node=True)
  filter: filterlib.Predicate = struct.field(pytree_node=False)
  axis: Index | None = struct.field(pytree_node=False)

  @property
  def states(self) -> tp.Iterable[State]:
    yield self.state

  @property
  def graphdef(self) -> GraphDef[tp.Any]:
    return self._graphdef

  @property
  def arg_states(self) -> tp.Sequence[VmapArgState]:
    return (self,)


@struct.dataclass
class VmapStates(tp.Generic[A], extract.ExtractableStates):
  arg_states: tuple[A, ...] = struct.field(pytree_node=True)

  @property
  def index(self) -> int:
    first = self.arg_states[0]
    if not isinstance(first, VmapArgState):
      raise RuntimeError(
        f'Expected type VmapArgState, got {type(first)}, this is a bug.'
      )
    return first.index

  @property
  def graphdef(self) -> GraphDef[tp.Any]:
    first = self.arg_states[0]
    if not isinstance(first, VmapArgState):
      raise RuntimeError(
        f'Expected type VmapArgState, got {type(first)}, this is a bug.'
      )
    return first.graphdef

  @property
  def states(self) -> tp.Iterable[State]:
    for arg_state in self.arg_states:
      if not isinstance(arg_state, VmapArgState):
        raise RuntimeError(
          f'Expected type VmapArgState, got {type(arg_state)}, this is a bug.'
        )
      yield arg_state.state


class StateAxes:
  def __init__(
    self,
    filter_axes: tp.Mapping[filterlib.Filter, Index | None]
    | tp.Iterable[tuple[filterlib.Filter, Index | None]],
    /,
  ):
    iterable = (
      filter_axes.items()
      if isinstance(filter_axes, tp.Mapping)
      else filter_axes
    )
    self.filters: tuple[filterlib.Filter, ...] = tuple(
      filter for filter, _ in iterable
    )
    self.axes: tuple[Index | None, ...] = tuple(axis for _, axis in iterable)

  def __repr__(self):
    return f'StateAxes({dict(zip(self.filters, self.axes))})'

  def __eq__(self, other):
    return (
      isinstance(other, StateAxes)
      and self.filters == other.filters
      and self.axes == other.axes
    )

  def __hash__(self):
    return hash((self.filters, self.axes))


@dataclasses.dataclass(frozen=True)
class VmapInputs:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any


def _index_to_state(
  x: extract.Extractable,
  *,
  graphdef: GraphDef,
  states: State,
  axes: tuple[tp.Any, ...],
):
  node_axis = axes[x.index]
  node_state = states[x.index] if x.index in states else State({})
  assert isinstance(node_state, State)
  if isinstance(node_axis, StateAxes):
    substates = node_state.split(*node_axis.filters)
    return VmapStates(
      tuple(
        VmapArgState(x.index, graphdef, substate, filter, axis)
        for substate, filter, axis in zip(
          substates, node_axis.filters, node_axis.axes
        )
      ),
    )
  else:
    return VmapArgState(
      x.index, graphdef, node_state, filterlib.Everything(), node_axis
    )


def vmap_fn(
  args: tuple[tp.Any, ...],
  vmap_inputs: VmapInputs,
):
  f = vmap_inputs.f
  transform_metadata = vmap_inputs.transform_metadata
  extracted_states: tuple[VmapStates | VmapArgState, ...] = (
    extract.extract_indexes(args, types=(VmapStates, VmapArgState))
  )
  ctx = graph.current_update_context('vmap')

  # remove metadata axis name from Variable.sharding
  def remove_axis_fn(arg_state):
    if (
      isinstance(arg_state, VmapArgState)
      and arg_state.axis is not None
      and spmd.PARTITION_NAME in transform_metadata
    ):
      state = arg_state.state
      state = spmd.remove_axis(state, arg_state.axis, transform_metadata)
      return arg_state.replace(state=state)
    return arg_state

  extracted_states = jax.tree.map(remove_axis_fn, extracted_states)

  if extracted_states:
    graphdef, states = extract.merge_extractable_states(extracted_states)
    inputs_graph_nodes = ctx.merge(graphdef, states)
    args = extract.insert_graph_nodes(args, inputs_graph_nodes)
  else:
    inputs_graph_nodes = ()

  out = f(*args)

  (args_out, out), output_nodes, output_node_axis = extract.extract_graph_nodes(
    (args, out), prefix=(vmap_inputs.in_axes, vmap_inputs.out_axes)
  )
  extract.check_consistent_aliasing(output_nodes, output_node_axis)

  graphdef_out, states_out = ctx.split(output_nodes)

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in transform_metadata:
    for index in states_out:
      assert isinstance(index, int)
      if output_node_axis[index] is not None:
        states_out[index] = spmd.add_axis(
          states_out[index], output_node_axis[index], transform_metadata
        )

  replace_fn = functools.partial(
    _index_to_state,
    graphdef=graphdef_out,
    states=states_out,
    axes=output_node_axis,
  )
  out = extract.replace_indexes(out, replace_fn)
  args_out = extract.replace_indexes(args_out, replace_fn, clear=True)

  return args_out, out


@tp.overload
def vmap(
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]: ...


@tp.overload
def vmap(
  f: F,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F: ...


def vmap(
  f: F | Missing = MISSING,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  """Vectorizing map. Creates a function which maps ``f`` over argument axes.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over.

      If each positional argument to ``f`` is an array, then ``in_axes`` can
      be an integer, a None, or a tuple of integers and Nones with length equal
      to the number of positional arguments to ``f``. An integer or ``None``
      indicates which array axis to map over for all arguments (with ``None``
      indicating not to map any axis), and a tuple indicates which axis to map
      for each corresponding positional argument. Axis integers must be in the
      range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
      dimensions (axes) of the corresponding input array.

      If the positional arguments to ``f`` are container (pytree) types, ``in_axes``
      must be a sequence with length equal to the number of positional arguments to
      ``f``, and for each argument the corresponding element of ``in_axes`` can
      be a container with a matching pytree structure specifying the mapping of its
      container elements. In other words, ``in_axes`` must be a container tree prefix
      of the positional argument tuple passed to ``f``. See this link for more detail:
      https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

      Either ``axis_size`` must be provided explicitly, or at least one
      positional argument must have ``in_axes`` not None. The sizes of the
      mapped input axes for all mapped positional arguments must all be equal.

      Arguments passed as keywords are always mapped over their leading axis
      (i.e. axis index 0).

      See below for examples.

    out_axes: An integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof indicating where the mapped axis should appear
      in the output. All outputs with a mapped axis must have a non-None
      ``out_axes`` specification. Axis integers must be in the range ``[-ndim,
      ndim)`` for each output array, where ``ndim`` is the number of dimensions
      (axes) of the array returned by the :func:`vmap`-ed function, which is one
      more than the number of dimensions (axes) of the corresponding array
      returned by ``f``.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``f`` with arguments that correspond to
    those of ``f``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``f``, but
    with extra array axes at positions indicated by ``out_axes``.


  """
  if isinstance(f, Missing):
    return functools.partial(
      vmap,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      transform_metadata=transform_metadata,
    )

  jax_in_axes = jax.tree.map(
    lambda x: VmapStates(x.axes) if isinstance(x, StateAxes) else x,
    in_axes,
  )
  jax_out_axes = jax.tree.map(
    lambda x: VmapStates(x.axes) if isinstance(x, StateAxes) else x,
    out_axes,
  )

  vmapped_fn = jax.vmap(
    vmap_fn,
    in_axes=(
      jax_in_axes,  # args
      None,  # vmap_inputs
    ),
    out_axes=(
      jax_in_axes,  # args_out
      jax_out_axes,  # out_axes
    ),
    axis_name=axis_name,
    axis_size=axis_size,
    spmd_axis_name=spmd_axis_name,
  )

  @functools.wraps(f)
  @graph.update_context('vmap')
  def vmap_wrapper(*args):
    ctx = graph.current_update_context('vmap')

    args, input_graph_nodes, input_node_axis = extract.extract_graph_nodes(
      args, prefix=in_axes
    )
    extract.check_consistent_aliasing(input_graph_nodes, input_node_axis)
    graphdef, states = ctx.split(input_graph_nodes)
    args = extract.replace_indexes(
      args,
      functools.partial(
        _index_to_state, graphdef=graphdef, states=states, axes=input_node_axis
      ),
    )

    args_out, out = vmapped_fn(
      args,
      VmapInputs(
        f,
        transform_metadata,
        in_axes,
        out_axes,
      ),
    )

    extracted_states_out = extract.extract_indexes(
      (args_out, out), types=(VmapStates, VmapArgState)
    )
    if extracted_states_out:
      graphdef_out, states_out = extract.merge_extractable_states(
        extracted_states_out
      )
      output_nodes = ctx.merge(graphdef_out, states_out)
      out = extract.insert_graph_nodes(out, output_nodes)

    return out

  return vmap_wrapper  # type: ignore


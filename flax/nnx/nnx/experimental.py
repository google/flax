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


@dataclasses.dataclass(frozen=True)
class HashablePartial:
  f: tp.Callable[..., tp.Any]
  args: tuple[tp.Any, ...]
  kwargs: tp.Mapping[str, tp.Any]

  if tp.TYPE_CHECKING:
    create = functools.partial
  else:

    @classmethod
    def create(cls, f, *args, **kwargs):
      return cls(f, args, FrozenDict(kwargs))

  def __call__(self, *args, **kwargs):
    return self.f(*self.args, *args, **self.kwargs, **kwargs)


# -------------------------------
# vmap
# -------------------------------


class VmapArgState(extract.ExtractionIndex, extract.ExtractableStates):
  _graphdef: GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: State = struct.field(pytree_node=True)
  filter: filterlib.Filter = struct.field(pytree_node=False)
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
class VmapStates(extract.ExtractableStates):
  arg_states: tuple[VmapArgState, ...] = struct.field(pytree_node=True)

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


def vmap_fn(vmap_inputs: VmapInputs, *args: tuple[tp.Any, ...]):
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
  """Reference-aware version of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over (see `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
      In addition to integers and None, :class:`StateAxes`  can be used to control how
      graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis should appear
      in the output (see `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``f`` with arguments that correspond to
    those of ``f``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``f``, but
    with extra array axes at positions indicated by ``out_axes``.

  Example::

    >>> from flax import nnx
    >>> from jax import random, numpy as jnp
    ...
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((5, 2))
    ...
    >>> @nnx.experimental.vmap(in_axes=(None, 0), out_axes=0)
    ... def forward(model, x):
    ...   return model(x)
    ...
    >>> y = forward(model, x)
    >>> y.shape
    (5, 3)

  >>> class LinearEnsemble(nnx.Module):
  ...   def __init__(self, num, rngs):
  ...     self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))
  ...
  >>> model = LinearEnsemble(5, rngs=nnx.Rngs(0))
  >>> x = jnp.ones((2,))
  ...
  >>> @nnx.experimental.vmap(in_axes=(0, None), out_axes=0)
  ... def forward(model, x):
  ...   return jnp.dot(x, model.w.value)
  ...
  >>> y = forward(model, x)
  >>> y.shape
  (5, 3)

  To control control how graph node substates are vectorized, ``StateAxes``
  can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
  applied to each substate given a filter. The following example shows how to
  share the parameters between the ensemble members which keeping different
  batch statistics and dropout random state::

    >>> class Foo(nnx.Module):
    ...   def __init__(self):
    ...     self.a = nnx.Param(jnp.arange(4))
    ...     self.b = nnx.BatchStat(jnp.arange(4))
    ...
    >>> state_axes = nnx.StateAxes({nnx.Param: 0, nnx.BatchStat: None})
    >>> @nnx.experimental.vmap(in_axes=(state_axes,), out_axes=0)
    ... def mul(foo):
    ...   return foo.a * foo.b
    ...
    >>> foo = Foo()
    >>> y = mul(foo)
    >>> y
    Array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]], dtype=int32)
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
    lambda x: VmapStates(x.axes)  # type: ignore
    if isinstance(x, StateAxes)
    else x,
    in_axes,
  )
  jax_out_axes = jax.tree.map(
    lambda x: VmapStates(x.axes)  # type: ignore
    if isinstance(x, StateAxes)
    else x,
    out_axes,
  )

  vmapped_fn = jax.vmap(
    HashablePartial.create(
      vmap_fn,
      VmapInputs(
        f,
        transform_metadata,
        in_axes,
        out_axes,
      ),
    ),
    in_axes=jax_in_axes,
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

    args_out, out = vmapped_fn(*args)

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
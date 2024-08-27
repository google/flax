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
# pytype: skip-file

from collections import deque
import dataclasses
import functools
import typing as tp

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx.nnx import extract, filterlib, graph, spmd
from flax.nnx.nnx.module import Module
from flax.nnx.nnx.state import State
from flax.nnx.nnx.transforms.transforms import resolve_kwargs
from flax.typing import Leaf, MISSING, Missing, PytreeDeque
import jax
import jax.core
import jax.numpy as jnp
import jax.stages
import numpy as np

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


class Carry:
  pass


# -------------------------------
# vmap
# -------------------------------


class StateAxes:

  def __init__(
      self,
      filter_axes: (
          tp.Mapping[filterlib.Filter, Index | type[Carry] | None]
          | tp.Iterable[tuple[filterlib.Filter, Index | type[Carry] | None]]
      ),
      /,
  ):
    iterable = tuple(
        filter_axes.items()
        if isinstance(filter_axes, tp.Mapping)
        else filter_axes
    )
    self._filters = tuple(filter for filter, _ in iterable)
    self._axes = tuple(axis for _, axis in iterable)

  @property
  def filters(self) -> tuple[filterlib.Filter, ...]:
    return self._filters

  @property
  def axes(self) -> tuple[Index | type[Carry] | None, ...]:
    return self._axes

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


AxisFn = tp.Callable[
    [extract.GraphDefState, int, tp.Mapping], extract.GraphDefState
]


def _update_variable_sharding_metadata(
    tree, transform_metadata, axis_fn: AxisFn
):
  def _update_axes_fn(tree_node):
    if isinstance(tree_node, extract.TreeNode) and isinstance(
        tree_node.metatata, StateAxes
    ):
      graphdef_states_out: list[extract.GraphDefState] = []
      for graphdef_state, axis in zip(
          tree_node.graphdef_states, tree_node.metatata.axes
      ):
        assert isinstance(graphdef_state, extract.GraphDefState)
        if isinstance(axis, int):
          graphdef_state = axis_fn(graphdef_state, axis, transform_metadata)
        graphdef_states_out.append(graphdef_state)
      return tree_node.replace(graphdef_states=tuple(graphdef_states_out))
    return tree_node

  return jax.tree.map(
      _update_axes_fn, tree, is_leaf=lambda x: isinstance(x, extract.TreeNode)
  )


def _vmap_split_fn(ctx: graph.SplitContext, path, prefix, x):
  if isinstance(prefix, StateAxes):
    return extract.TreeNode.from_split(
        *ctx.split(x, *prefix.filters), metadata=prefix
    )
  return extract.TreeNode.from_split(*ctx.split(x))


@dataclasses.dataclass(eq=False)
class VmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )
    args = extract.from_tree(pure_args, ctxtag='vmap')

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
        (args_out, out),
        prefix=(self.in_axes, self.out_axes),
        split_fn=_vmap_split_fn,
        ctxtag='vmap',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_out = _update_variable_sharding_metadata(
          (pure_args_out, pure_out), self.transform_metadata, spmd.add_axis
      )
    return pure_args_out, pure_out


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
) -> tp.Callable[[F], F]:
  ...


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
) -> F:
  ...


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
      array axes to map over (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__). In
      addition to integers and None, :class:`StateAxes`  can be used to control
      how graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter
      <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis
      should appear in the output (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
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
    >>> @nnx.vmap(in_axes=(None, 0), out_axes=0)
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
  >>> @nnx.vmap(in_axes=(0, None), out_axes=0)
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
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=0)
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
    )  # type: ignore[return-value]

  jax_in_axes = jax.tree.map(
      lambda x: extract.TreeNode.from_prefixes(x.axes, metadata=x)
      if isinstance(x, StateAxes)
      else x,
      in_axes,
  )
  jax_out_axes = jax.tree.map(
      lambda x: extract.TreeNode.from_prefixes(x.axes, metadata=x)
      if isinstance(x, StateAxes)
      else x,
      out_axes,
  )
  vmapped_fn = jax.vmap(
      VmapFn(f, transform_metadata, in_axes, out_axes),
      in_axes=jax_in_axes,
      out_axes=(jax_in_axes, jax_out_axes),
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
  )

  @functools.wraps(f)
  @graph.update_context('vmap')
  def vmap_wrapper(*args, **kwargs):
    args = resolve_kwargs(f, args, kwargs)
    pure_args = extract.to_tree(
        args, prefix=in_axes, split_fn=_vmap_split_fn, ctxtag='vmap'
    )
    pure_args_out, pure_out = vmapped_fn(*pure_args)
    _args_out, out = extract.from_tree((pure_args_out, pure_out), ctxtag='vmap')
    return out

  return vmap_wrapper  # type: ignore


# -------------------------------
# pmap
# -------------------------------


@dataclasses.dataclass(eq=False)
class PmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )
    args = extract.from_tree(pure_args, ctxtag='pmap')

    out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    pure_args_out, pure_out = extract.to_tree(
        (args_out, out),
        prefix=(self.in_axes, self.out_axes),
        split_fn=_vmap_split_fn,
        ctxtag='pmap',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_out = _update_variable_sharding_metadata(
          (pure_args_out, pure_out), self.transform_metadata, spmd.add_axis
      )
    return pure_args_out, pure_out


@tp.overload
def pmap(
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
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]:
  ...


@tp.overload
def pmap(
    f: F,
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
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  ...


def pmap(
    f: F | Missing = MISSING,
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
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  """Reference-aware version of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__). In
      addition to integers and None, :class:`StateAxes`  can be used to control
      how graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter
      <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis
      should appear in the output (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
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
    >>> @nnx.vmap(in_axes=(None, 0), out_axes=0)
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
  >>> @nnx.vmap(in_axes=(0, None), out_axes=0)
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
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=0)
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
        transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  jax_in_axes = jax.tree.map(
      lambda x: extract.TreeNode.from_prefixes(x.axes, metadata=x)
      if isinstance(x, StateAxes)
      else x,
      in_axes,
  )
  jax_out_axes = jax.tree.map(
      lambda x: extract.TreeNode.from_prefixes(x.axes, metadata=x)
      if isinstance(x, StateAxes)
      else x,
      out_axes,
  )
  pmapped_fn = jax.pmap(
      PmapFn(f, transform_metadata, in_axes, out_axes),
      axis_name=axis_name,
      in_axes=jax_in_axes,
      out_axes=(jax_in_axes, jax_out_axes),
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums,
      global_arg_shapes=global_arg_shapes,
  )

  @functools.wraps(f)
  @graph.update_context('pmap')
  def vmap_wrapper(*args):
    pure_args = extract.to_tree(
        args, prefix=in_axes, split_fn=_vmap_split_fn, ctxtag='pmap'
    )
    pure_args_out, pure_out = pmapped_fn(*pure_args)
    _args_out, out = extract.from_tree((pure_args_out, pure_out), ctxtag='pmap')
    return out

  return vmap_wrapper  # type: ignore


# -------------------------------
# scan
# -------------------------------


class Broadcasted(struct.PyTreeNode):
  data: tp.Any


def _scan_split_in(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    broadcast_arrays: PytreeDeque[Broadcasted],
    /,
    ctx: graph.SplitContext,
    path,
    prefix,
    x,
):
  if graph.is_graph_node(x):
    vectorized_states: list[State] = []
    carry_states: list[State] = []
    broadcast_states: list[State] = []
    if isinstance(prefix, StateAxes):
      graphdef, *states = ctx.split(x, *prefix.filters)

      for state, axis in zip(states, prefix.axes):
        if axis is None:
          broadcast_states.append(state)
        elif isinstance(axis, int):
          state = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0), state)
          vectorized_states.append(state)
        else:  # axis is Carry
          carry_states.append(state)

      if not vectorized_states:
        vectorized_states.append(State({}))
      carry_deque.append(carry_states)
      broadcast_deque.append(broadcast_states)
      return extract.TreeNode.from_split(
          graphdef, *vectorized_states, metadata=prefix
      )
    elif isinstance(prefix, int):
      graphdef, state = ctx.split(x)
      state = jax.tree.map(lambda x: jnp.moveaxis(x, prefix, 0), state)
      vectorized_states.append(state)
    elif prefix is None:
      graphdef, state = ctx.split(x)
      broadcast_states.append(state)
      vectorized_states.append(State({}))
    elif prefix is Carry:
      graphdef, state = ctx.split(x)
      carry_states.append(state)
      vectorized_states.append(State({}))
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )

    if not vectorized_states:
      vectorized_states.append(State({}))
    carry_deque.append(carry_states)
    broadcast_deque.append(broadcast_states)
    return extract.TreeNode.from_split(
        graphdef, *vectorized_states, metadata=prefix
    )
  else:
    if isinstance(prefix, StateAxes):
      raise ValueError(
          'Cannot use StateAxes on non-graph nodes, '
          f'found {prefix} at path {jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      broadcast_arrays.append(Broadcasted(x))
      return Broadcasted(None)
    elif isinstance(prefix, int):
      if not isinstance(x, (jax.Array, np.ndarray)):
        raise ValueError(
            f'Expected an array, got {type(x).__name__} at path '
            f'{jax.tree_util.keystr(path)}'
        )
      return jnp.moveaxis(x, prefix, 0)
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )


def _scan_split_out(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    /,
    ctx: graph.SplitContext,
    path: extract.KeyPath,
    prefix,
    x,
):
  assert isinstance(path[0], jax.tree_util.SequenceKey)
  is_input_arg = path[0].idx == 0

  if graph.is_graph_node(x):
    vectorized_states: list[State] = []
    carry_states: list[State] = []
    broadcast_states: list[State] = []
    if isinstance(prefix, StateAxes):
      graphdef, *states = ctx.split(x, *prefix.filters)

      for state, filter, axis in zip(states, prefix.filters, prefix.axes):
        if axis is None:
          if is_input_arg:
            broadcast_states.append(state)
          elif state:
            raise ValueError(
                f'Cannot broadcast output state. Got filter {filter} and axis'
                f' None at path {jax.tree_util.keystr(path)}'
            )
        elif isinstance(axis, int):
          vectorized_states.append(state)
        else:  # axis is Carry
          if is_input_arg:
            carry_states.append(state)
          elif state:
            raise ValueError(
                f'Cannot carry output state. Got filter {filter} and axis'
                f' {axis} at path {jax.tree_util.keystr(path)}'
            )
      if not vectorized_states:
        vectorized_states.append(State({}))
      if is_input_arg:
        carry_deque.append(carry_states)
        broadcast_deque.append(broadcast_states)
      return extract.TreeNode.from_split(
          graphdef, *vectorized_states, metadata=prefix
      )
    elif isinstance(prefix, int):
      graphdef, state = ctx.split(x)
      vectorized_states.append(state)
    elif prefix is None:
      graphdef, state = ctx.split(x)
      if is_input_arg:
        broadcast_states.append(state)
        vectorized_states.append(State({}))
      elif state:
        raise ValueError(
            'Cannot broadcast output state. '
            f'Got out_axes=None at path {jax.tree_util.keystr(path)}'
        )
    elif prefix is Carry:
      graphdef, state = ctx.split(x)
      if is_input_arg:
        carry_states.append(state)
        vectorized_states.append(State({}))
      elif state:
        raise ValueError(
            'Cannot carry output state. '
            f'Got out_axes=carry at path {jax.tree_util.keystr(path)}'
        )
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )
    if not vectorized_states:
      vectorized_states.append(State({}))
    if is_input_arg:
      carry_deque.append(carry_states)
      broadcast_deque.append(broadcast_states)
    return extract.TreeNode.from_split(
        graphdef, *vectorized_states, metadata=prefix
    )
  else:
    if isinstance(prefix, StateAxes):
      raise ValueError(
          'Cannot use StateAxes on non-graph nodes, '
          f'found {prefix} at path {jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      if not is_input_arg:
        raise ValueError(
            'Cannot broadcast outputs. '
            f'Got out_axes=None at path {jax.tree_util.keystr(path)}'
        )
      return Broadcasted(None)
    elif isinstance(prefix, int):
      return x
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )


def _scan_merge_in(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    broadcast_arrays: PytreeDeque[Broadcasted],
    /,
    ctx: graph.MergeContext,
    path,
    prefix,
    x,
):
  if isinstance(x, extract.TreeNode):
    carry_states = carry_deque.popleft()
    broadcast_states = broadcast_deque.popleft()
    return ctx.merge(x.graphdef, *x.states, *carry_states, *broadcast_states)
  elif isinstance(x, Broadcasted):
    assert x.data is None
    return broadcast_arrays.popleft().data
  else:
    return x


def _scan_merge_out(
    carry_deque: PytreeDeque[list[State]],
    broadcast_deque: PytreeDeque[list[State]],
    /,
    ctx: graph.MergeContext,
    path,
    prefix,
    x,
):
  assert isinstance(path[0], jax.tree_util.SequenceKey)
  is_input_arg = path[0].idx == 0

  if isinstance(x, extract.TreeNode):
    states: list[State] = []
    if is_input_arg:
      carry_states = deque(carry_deque.popleft())
      broadcast_states = deque(broadcast_deque.popleft())
    else:
      carry_states = deque[State]()
      broadcast_states = deque[State]()
    if isinstance(prefix, StateAxes):
      vectorized_states = deque(x.states)
      assert len(prefix.axes) == len(vectorized_states) + len(
          carry_states
      ) + len(broadcast_states)
      for axis in prefix.axes:
        if isinstance(axis, int):
          state = vectorized_states.popleft()
          state = jax.tree.map(lambda x: jnp.moveaxis(x, 0, axis), state)
          states.append(state)
        elif axis is None:
          states.append(broadcast_states.popleft())
        else:  # axis is Carry
          states.append(carry_states.popleft())
      assert not vectorized_states and not carry_states and not broadcast_states
    elif isinstance(prefix, int):
      state = jax.tree.map(lambda x: jnp.moveaxis(x, 0, prefix), x.state)
      states.extend((state, *carry_states, *broadcast_states))
    elif prefix is None:
      assert is_input_arg
      states.extend(broadcast_states)
    elif prefix is Carry:
      assert is_input_arg
      states.extend(carry_states)
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )

    return ctx.merge(x.graphdef, *states)
  else:
    if isinstance(prefix, StateAxes):
      raise ValueError(
          'Cannot use StateAxes on non-graph nodes, '
          f'found {prefix} at path {jax.tree_util.keystr(path)}'
      )
    elif prefix is Carry:
      return x
    elif prefix is None:
      return x
    elif isinstance(prefix, int):
      if not isinstance(x, (jax.Array, np.ndarray)):
        raise ValueError(
            f'Expected an array, got {type(x).__name__} at path '
            f'{jax.tree_util.keystr(path)}'
        )
      return jnp.moveaxis(x, 0, prefix)
    else:
      raise ValueError(
          f'Invalid axes {prefix} at path {jax.tree_util.keystr(path)}'
      )


@dataclasses.dataclass(eq=False)
class ScanFn:
  f: tp.Callable[..., tp.Any]
  carry_argnum: int
  in_axes: tp.Any
  out_axes: tp.Any
  transform_metadata: tp.Mapping[str, tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(
      self,
      carry: tuple[
          tp.Any,  # carry_arg
          PytreeDeque[list[State]],  # carry_deque
          PytreeDeque[list[State]],  # broadcast_deque
          PytreeDeque[Broadcasted],  # broadcast_arrays
      ],
      pure_args: list[tp.Any],
  ):
    pure_carry_arg, carry_deque, broadcast_deque, broadcast_arrays = carry
    pure_args[self.carry_argnum] = pure_carry_arg
    broadcast_deque_out = PytreeDeque(broadcast_deque)
    broadcast_arrays_out = PytreeDeque(broadcast_arrays)

    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _update_variable_sharding_metadata(
          pure_args, self.transform_metadata, spmd.remove_axis
      )

    args = extract.from_tree(
        pure_args,
        prefix=self.in_axes,
        merge_fn=functools.partial(
            _scan_merge_in, carry_deque, broadcast_deque, broadcast_arrays
        ),
        is_leaf=lambda x: isinstance(x, (extract.TreeNode, Broadcasted)),
        map_non_graph_nodes=True,
        ctxtag='scan',
    )
    assert not carry_deque and not broadcast_deque and not broadcast_arrays

    carry_arg_out, scan_out = self.f(*args)

    args_out = extract.clear_non_graph_nodes(args)
    args_out[self.carry_argnum] = carry_arg_out

    carry_deque_out = PytreeDeque[list[State]]()
    _broadcast_deque_out_tmp = PytreeDeque[list[State]]()  # discarded
    pure_args_out, pure_scan_out = extract.to_tree(
        (args_out, scan_out),
        prefix=(self.in_axes, self.out_axes),
        split_fn=functools.partial(
            _scan_split_out, carry_deque_out, _broadcast_deque_out_tmp
        ),
        map_non_graph_nodes=True,
        ctxtag='scan',
    )
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args_out, pure_scan_out = _update_variable_sharding_metadata(
          (pure_args_out, pure_scan_out),
          self.transform_metadata,
          spmd.add_axis,
      )

    pure_carry_arg_out = pure_args_out[self.carry_argnum]
    pure_args_out[self.carry_argnum] = None

    # next we have to remove all the index_mappings from the NodeDefs
    # in the carry outputs because they are not present in the inputs
    carry_index_mappings = list[FrozenDict[int, int]]()

    def extract_index_mappings(x):
      if isinstance(x, extract.GraphDefState) and isinstance(
          x.graphdef, graph.NodeDef
      ):
        index_mapping = x.graphdef.index_mapping
        assert index_mapping is not None
        carry_index_mappings.append(index_mapping)
        x = x.replace(
            graphdef=dataclasses.replace(x.graphdef, index_mapping=None)
        )
      return x

    pure_carry_arg_out = jax.tree.map(
        extract_index_mappings,
        pure_carry_arg_out,
        is_leaf=lambda x: isinstance(x, extract.GraphDefState),
    )

    carry_arg_out = (
        pure_carry_arg_out,
        carry_deque_out,
        broadcast_deque_out,
        broadcast_arrays_out,
    )
    return carry_arg_out, (
        graph.Static(tuple(carry_index_mappings)),
        pure_args_out,
        pure_scan_out,
    )


@tp.overload
def scan(
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: tp.Sequence[tp.Any] = (Carry, 0),
    out_axes: tp.Any = 0,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> tp.Callable[[F], F]:
  ...


@tp.overload
def scan(
    f: F,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: tp.Sequence[tp.Any] = (Carry, 0),
    out_axes: tp.Any = 0,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  ...


def scan(
    f: F | Missing = MISSING,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: tp.Sequence[tp.Any] = (Carry, 0),
    out_axes: tp.Any = 0,
    # nnx specific
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
        scan,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        in_axes=in_axes,
        out_axes=out_axes,
        transform_metadata=transform_metadata,
    )  # type: ignore[return-value]

  carry_argnum: int = -1
  for key, x in jax.tree_util.tree_leaves_with_path(in_axes):
    if x is not Carry:
      continue
    assert isinstance(key[0], jax.tree_util.SequenceKey)
    i = key[0].idx
    if len(key) >= 2:
      raise ValueError(
          'Carry must be used direcly on an input, it cannot be nested. '
          f'Found {in_axes=}'
      )
    if carry_argnum >= 0:
      raise ValueError(f'Found multiple Carry axes in in_axes: {in_axes}')
    carry_argnum = i
  if carry_argnum < 0:
    raise ValueError(f'No Carry axis specified in in_axes: {in_axes}')

  in_axes = list(in_axes)

  scan_fn = ScanFn(
      f,
      carry_argnum,
      in_axes,
      out_axes,
      transform_metadata,
  )

  @functools.wraps(f)
  @graph.update_context('scan')
  def scan_wrapper(*args, **kwargs):
    args = list(resolve_kwargs(f, args, kwargs))
    carry_deque = PytreeDeque()
    broadcast_deque = PytreeDeque()
    broadcast_arrays = PytreeDeque()
    pure_args = extract.to_tree(
        args,
        prefix=in_axes,
        split_fn=functools.partial(
            _scan_split_in, carry_deque, broadcast_deque, broadcast_arrays
        ),
        map_non_graph_nodes=True,
        ctxtag='scan',
    )
    pure_carry_arg = pure_args[carry_argnum]
    pure_args[carry_argnum] = None

    carry = (pure_carry_arg, carry_deque, broadcast_deque, broadcast_arrays)

    carry_out, (static_carry_index_mappings, pure_args_out, pure_scan_out) = (
        jax.lax.scan(
            scan_fn,
            carry,
            pure_args,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
    )
    (
        pure_carry_arg_out,
        carry_deque_out,
        broadcast_deque_out,
        broadcast_arrays_out,
    ) = carry_out

    # next we have to insert all the index_mappings back into the NodeDefs
    # in the carry outputs
    carry_index_mappings = deque(static_carry_index_mappings.value)

    def insert_index_mappings(x):
      if isinstance(x, extract.GraphDefState) and isinstance(
          x.graphdef, graph.NodeDef
      ):
        index_mapping = carry_index_mappings.popleft()
        x = x.replace(
            graphdef=dataclasses.replace(
                x.graphdef, index_mapping=index_mapping
            )
        )
      return x

    pure_carry_arg_out = jax.tree.map(
        insert_index_mappings,
        pure_carry_arg_out,
        is_leaf=lambda x: isinstance(x, extract.GraphDefState),
    )

    pure_args_out[carry_argnum] = pure_carry_arg_out
    args_out, scan_out = extract.from_tree(
        (pure_args_out, pure_scan_out),
        prefix=(in_axes, out_axes),
        merge_fn=functools.partial(
            _scan_merge_out, carry_deque_out, broadcast_deque_out
        ),
        is_leaf=lambda x: isinstance(x, (extract.TreeNode, Broadcasted)),
        map_non_graph_nodes=True,
        ctxtag='scan',
    )
    carry_out = args_out[carry_argnum]

    return carry_out, scan_out

  return scan_wrapper  # type: ignore

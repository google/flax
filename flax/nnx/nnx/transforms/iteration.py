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
from flax.nnx.nnx.transforms.transforms import LiftedModule, resolve_kwargs
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
# vmap
# -------------------------------

class StateAxes:
  def __init__(
    self,
    filter_axes: tp.Mapping[filterlib.Filter, Index | None]
    | tp.Iterable[tuple[filterlib.Filter, Index | None]],
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
  def axes(self) -> tuple[Index | None, ...]:
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


def _vmap_update_variable_axes(tree, transform_metadata, axis_fn: AxisFn):
  def _update_axes_fn(tree_node):
    if isinstance(tree_node, extract.TreeNode) and isinstance(
      tree_node.metatata, StateAxes
    ):
      graphdef_states_out: list[extract.GraphDefState] = []
      for graphdef_state, axis in zip(
        tree_node.graphdef_states, tree_node.metatata.axes
      ):
        if axis is not None:
          graphdef_state = axis_fn(graphdef_state, axis, transform_metadata)
        graphdef_states_out.append(graphdef_state)
      return tree_node.replace(graphdef_states=tuple(graphdef_states_out))
    return tree_node

  return jax.tree_map(
    _update_axes_fn, tree, is_leaf=lambda x: isinstance(x, extract.TreeNode)
  )


def _vmap_split_fn(ctx: graph.SplitContext, path, prefix, x):
  if isinstance(prefix, StateAxes):
    return extract.TreeNode.from_split(
      *ctx.split(x, *prefix.filters), metadata=prefix
    )
  return extract.TreeNode.from_split(*ctx.split(x))


@dataclasses.dataclass(frozen=True)
class VmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _vmap_update_variable_axes(
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
      pure_args_out, pure_out = _vmap_update_variable_axes(
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
    )

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

@dataclasses.dataclass(frozen=True)
class PmapFn:
  f: tp.Callable[..., tp.Any]
  transform_metadata: tp.Mapping[str, tp.Any]
  in_axes: tp.Any
  out_axes: tp.Any

  def __call__(self, *pure_args: tuple[tp.Any, ...]):
    if spmd.PARTITION_NAME in self.transform_metadata:
      pure_args = _vmap_update_variable_axes(
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
      pure_args_out, pure_out = _vmap_update_variable_axes(
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
) -> tp.Callable[[F], F]: ...
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
) -> F: ...
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
    )

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
) -> tp.Callable[[F], F]: ...
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

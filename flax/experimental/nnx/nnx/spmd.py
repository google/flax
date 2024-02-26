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

import functools
import typing as tp

import jax
from jax.experimental import maps
from jax.sharding import Mesh, PartitionSpec

from flax.experimental.nnx.nnx import variables
from flax.experimental.nnx.nnx.pytreelib import TreeNode
from flax.experimental.nnx.nnx.state import State
from flax.typing import (
  Array,
  ArrayPytree,  # pylint: disable=invalid-name
  PartitionSpecPytree,  # pylint: disable=invalid-name
  Sharding,
)

A = tp.TypeVar('A')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
PARTITION_NAME = 'partition_name'


@tp.runtime_checkable
class HasSharding(tp.Protocol):
  sharding: tp.Optional[Sharding]


def add_axis(
  state: State, index: int, params: tp.Mapping[tp.Any, tp.Any]
) -> State:
  axis_name = _get_partition_name(params)

  def _add_axis(x: tp.Any):
    if isinstance(x, variables.Variable):
      if isinstance(x, HasSharding) and x.sharding is not None:
        sharding = list(x.sharding)
        while len(sharding) < index:
          sharding.append(None)
        sharding.insert(index, axis_name)
        x.sharding = tuple(sharding)

      x.add_axis(axis_name, index)
    return x

  return jax.tree_map(
    _add_axis, state, is_leaf=lambda x: isinstance(x, variables.Variable)
  )


def remove_axis(
  state: State, index: int, params: tp.Mapping[tp.Any, tp.Any]
) -> State:
  axis_name = _get_partition_name(params)

  def _remove_axis(x: tp.Any):
    if isinstance(x, variables.Variable):
      if isinstance(x, HasSharding) and x.sharding is not None:
        sharding = list(x.sharding)
        assert sharding.pop(index) == axis_name
        x.sharding = tuple(sharding)
      x.remove_axis(axis_name, index)
    return x

  return jax.tree_map(
    _remove_axis, state, is_leaf=lambda x: isinstance(x, variables.Variable)
  )


def _get_partition_name(params: tp.Mapping[tp.Any, tp.Any]) -> str:
  if PARTITION_NAME not in params:
    raise ValueError(
      'Trying to transform a Partitioned variable but "partition_name" '
      f'is not specified in scan_metadata: {params}'
    )
  return params[PARTITION_NAME]


def get_partition_spec(tree: A) -> A:
  """Extracts a PartitionSpec tree from a PyTree containing ``Variable`` values."""

  def _maybe_replicate(x):
    if hasattr(x, 'shape'):
      return PartitionSpec()
    else:
      return None

  def f(x):
    if isinstance(x, variables.Variable):
      if isinstance(x, HasSharding) and x.sharding:
        return x.replace(raw_value=PartitionSpec(*x.sharding))
      else:
        return x.replace(raw_value=_maybe_replicate(x.raw_value))

    return _maybe_replicate(x)

  return jax.tree_map(
    f,
    tree,
    is_leaf=lambda x: isinstance(x, variables.Variable)
    and not isinstance(x, TreeNode),
  )


def get_named_sharding(tree: A, mesh: jax.sharding.Mesh) -> A:
  spec = get_partition_spec(tree)
  sharding = jax.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
  return sharding


# Dynamic Axis Mapping Rngs
# ------------------------------------------------------------------------------


def _global_mesh_defined() -> bool:
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = maps.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def _with_sharding_constraint(
  x: Array,
  axis_resources: tp.Optional[jax.sharding.PartitionSpec],
  mesh: tp.Optional[jax.sharding.Mesh] = None,
):
  # if jax.devices()[0].platform == "cpu" or (
  if not _global_mesh_defined() and mesh is None:
    return x
  else:
    if mesh is not None and axis_resources is not None:
      sharding = jax.sharding.NamedSharding(mesh, axis_resources)
      return jax.lax.with_sharding_constraint(x, sharding)
    return jax.lax.with_sharding_constraint(x, axis_resources)


def _is_spec(x):
  return x is None or (
    isinstance(x, tuple) and all(isinstance(e, str) or e is None for e in x)
  )


def with_sharding_constraint(
  x: ArrayPytree,
  axis_resources: PartitionSpecPytree,
  mesh: tp.Optional[jax.sharding.Mesh] = None,
):
  # If no axis binding is set, this is a no-op.
  if axis_resources is None:
    return x
  # Translate logical names to mesh assignments.
  return jax.tree_util.tree_map(
    functools.partial(_with_sharding_constraint, mesh=mesh),
    x,
    axis_resources,
    is_leaf=_is_spec,
  )


# -------------------------------------
# Partitioning Axis Metadata
# -------------------------------------


@tp.runtime_checkable
class Partitioned(tp.Protocol):
  get_value_hooks: tp.Callable[[variables.Variable[tp.Any]], tp.Any]
  sharding: Sharding
  mesh: tp.Optional[Mesh]


def sharding_hook(
  node: variables.Variable[tp.Any],
  value: tp.Any,
  /,
) -> tp.Any:
  if _global_mesh_defined() or (
    isinstance(node, Partitioned) and node.mesh is not None
  ):
    spec = get_partition_spec(node).raw_value
    return with_sharding_constraint(value, spec, mesh=node.mesh)
  return value


def with_partitioning(
  initializer: F,
  sharding: Sharding,
  mesh: tp.Optional[jax.sharding.Mesh] = None,
  get_value_hooks: tp.Union[
    variables.GetValueHook[A], tp.Sequence[variables.GetValueHook[A]]
  ] = (),
  create_value_hooks: tp.Union[
    variables.CreateValueHook[A], tp.Sequence[variables.CreateValueHook[A]]
  ] = (),
  **metadata: tp.Any,
) -> F:
  if callable(get_value_hooks):
    get_value_hooks = (get_value_hooks, sharding_hook)
  else:
    get_value_hooks = (*get_value_hooks, sharding_hook)

  if callable(create_value_hooks):
    create_value_hooks = (create_value_hooks, sharding_hook)
  else:
    create_value_hooks = (*create_value_hooks, sharding_hook)

  return variables.with_metadata(
    initializer,
    get_value_hooks=get_value_hooks,
    create_value_hooks=create_value_hooks,
    sharding=sharding,
    mesh=mesh,
    **metadata,
  )

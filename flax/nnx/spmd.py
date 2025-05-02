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

import flax.core.spmd as core_spmd
from flax.nnx import variablelib
from flax.typing import (
  Array,
  ArrayPytree,  # pylint: disable=invalid-name
  PartitionSpecPytree,  # pylint: disable=invalid-name
  Sharding,
)
import jax
from jax.interpreters import pxla
from jax.sharding import PartitionSpec

A = tp.TypeVar('A')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
PARTITION_NAME = 'partition_name'

class HasSharding(tp.Protocol):
  sharding: tuple[str | None, ...] | None


def _has_sharding(x: tp.Any) -> tp.TypeGuard[HasSharding]:
  return hasattr(x, 'sharding') and x.sharding is not None

def add_axis(tree: A, index: int, transform_metadata: tp.Mapping) -> A:
  axis_name, other_meta = _get_partition_name_and_metadata(transform_metadata)

  def insert_field(fields, index, value):
    iterable = list(fields)
    while len(iterable) < index:
      iterable.append(None)
    iterable.insert(index, value)
    return tuple(iterable)

  def _add_axis(x: tp.Any):
    if isinstance(x, variablelib.Variable):
      # NOTE(cgarciae): change the default name set by with_partitioning to
      # something other than 'sharding' because this clashes with the new Array.sharding
      metadata = x.get_metadata()
      if (sharding := metadata.get('sharding')) is not None:
        metadata['sharding'] = insert_field(sharding, index, axis_name)

      for k, v in other_meta.items():
        if hasattr(x, k) and (t := getattr(x, k)) and isinstance(t, tuple):
          setattr(x, k, insert_field(t, index, v))

      x.add_axis(index, axis_name)
    return x

  return jax.tree.map(
    _add_axis, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def remove_axis(tree: A, index: int, transform_metadata: tp.Mapping[tp.Any, tp.Any]) -> A:
  axis_name, other_meta = _get_partition_name_and_metadata(transform_metadata)

  def remove_field(fields, index, value):
    iterable = list(fields)
    assert iterable.pop(index) == value
    return tuple(iterable)

  def _remove_axis(x: tp.Any):
    if isinstance(x, variablelib.Variable):
      metadata = x.get_metadata()
      if (sharding := metadata.get('sharding')) is not None:
        metadata['sharding'] = remove_field(sharding, index, axis_name)

      for k, v in other_meta.items():
        if hasattr(x, k) and (t := getattr(x, k)) and isinstance(t, tuple):
          setattr(x, k, remove_field(t, index, v))

      x.remove_axis(index, axis_name)
    return x

  return jax.tree.map(
    _remove_axis,
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable),
  )


def _get_partition_name_and_metadata(
  transform_metadata: tp.Mapping[tp.Any, tp.Any]
) -> tuple[str, tp.Mapping[tp.Any, tp.Any]]:
  if PARTITION_NAME not in transform_metadata:
    raise ValueError(
      'Trying to transform a Partitioned variable but "partition_name" '
      f'is not specified in transform_metadata: {transform_metadata}'
    )
  other_meta = dict(transform_metadata)  # shallow copy
  other_meta.pop(PARTITION_NAME)
  return transform_metadata[PARTITION_NAME], other_meta


def get_partition_spec(tree):
  """Extracts a PartitionSpec tree from a PyTree containing ``Variable`` values."""

  def _maybe_replicate(x):
    if hasattr(x, 'shape'):
      return PartitionSpec()
    else:
      return None

  def f(x):
    if isinstance(x, variablelib.Variable):
      metadata = x.get_metadata()
      if (sharding := metadata.get('sharding')) is not None:
        if core_spmd.get_logical_axis_rules() or 'sharding_rules' in metadata:
          context_rules = core_spmd.get_logical_axis_rules()
          local_rules = metadata.get('sharding_rules', ())
          rules = core_spmd.composite_rules(context_rules, local_rules)
          return PartitionSpec(
            *core_spmd.from_sharding_rules(x.sharding, rules)
          )
        return PartitionSpec(*sharding)
      else:
        return _maybe_replicate(x.value)

    return _maybe_replicate(x)

  return jax.tree.map(
    f, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def get_named_sharding(tree: A, mesh: jax.sharding.Mesh) -> A:
  spec = get_partition_spec(tree)
  sharding = jax.tree.map(
    lambda p: jax.sharding.NamedSharding(mesh, p), spec
  )
  return sharding


# Dynamic Axis Mapping Rngs
# ------------------------------------------------------------------------------


def _global_mesh_defined() -> bool:
  """Checks if global mesh resource environment is defined."""
  env = pxla.thread_resources.env
  return env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


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
  return (
    x is None
    or isinstance(x, variablelib.Variable)
    or (
      isinstance(x, tuple) and all(isinstance(e, str) or e is None for e in x)
    )
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
  return jax.tree.map(
    functools.partial(_with_sharding_constraint, mesh=mesh),
    x,
    axis_resources,
    is_leaf=_is_spec,
  )


def with_partitioning(
  initializer: F,
  sharding: Sharding,
  mesh: tp.Optional[jax.sharding.Mesh] = None,
  **metadata: tp.Any,
) -> F:
  return variablelib.with_metadata(
    initializer,
    sharding=sharding,
    mesh=mesh,
    **metadata,
  )

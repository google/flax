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

from functools import partial
import typing as tp

from flax.core import meta
import flax.core.spmd as core_spmd
from flax.nnx import graph
from flax.nnx import module
from flax.nnx import variablelib
from flax.typing import (
  Sharding,
)
import jax
from jax.sharding import PartitionSpec

A = tp.TypeVar('A')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
PARTITION_NAME = 'partition_name'


# Transform axis change helpers
# ------------------------------------------------------------------------------


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
      metadata = x.get_metadata()
      if 'sharding_names' in metadata and metadata['sharding_names']:
        sharding = metadata['sharding_names']
        x.sharding_names = insert_field(sharding, index, axis_name)

      for k, v in other_meta.items():
        if hasattr(x, k) and (t := getattr(x, k)) and isinstance(t, tuple):
          setattr(x, k, insert_field(t, index, v))

      assert isinstance(x, variablelib.Variable)
      x.add_axis(index, axis_name)
    return x

  return jax.tree.map(
    _add_axis, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def remove_axis(
  tree: A, index: int, transform_metadata: tp.Mapping[tp.Any, tp.Any]
) -> A:
  axis_name, other_meta = _get_partition_name_and_metadata(transform_metadata)

  def remove_field(fields, index, value):
    iterable = list(fields)
    assert iterable.pop(index) == value
    return tuple(iterable)

  def _remove_axis(x: tp.Any):
    if isinstance(x, variablelib.Variable):
      if hasattr(x, 'sharding_names') and x.sharding_names is not None:
        x.sharding_names = remove_field(x.sharding_names, index, axis_name)

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
  transform_metadata: tp.Mapping[tp.Any, tp.Any],
) -> tuple[str, tp.Mapping[tp.Any, tp.Any]]:
  if PARTITION_NAME not in transform_metadata:
    raise ValueError(
      'Trying to transform a Partitioned variable but "partition_name" '
      f'is not specified in transform_metadata: {transform_metadata}'
    )
  other_meta = dict(transform_metadata)  # shallow copy
  other_meta.pop(PARTITION_NAME)
  return transform_metadata[PARTITION_NAME], other_meta


# Annotation handling
# ------------------------------------------------------------------------------


def with_partitioning(
  initializer: F,
  sharding: Sharding,
  mesh: tp.Optional[jax.sharding.Mesh] = None,
  **metadata: tp.Any,
) -> F:
  """A wrapper over any initializer to add sharding annotation data to a `Variable`."""
  return variablelib.with_metadata(
    initializer,
    sharding_names=sharding,
    mesh=mesh,
    **metadata,
  )


def get_var_pspec(v: variablelib.Variable) -> PartitionSpec:
  """Given an `nnx.Variable`, return its `PartitionSpec`."""
  metadata = v.get_metadata()
  if 'sharding_names' in metadata and metadata['sharding_names']:
    sharding = metadata['sharding_names']
    if core_spmd.get_logical_axis_rules() or 'sharding_rules' in metadata:
      context_rules = core_spmd.get_logical_axis_rules()
      local_rules = metadata.get('sharding_rules', ())
      rules = core_spmd.composite_rules(context_rules, local_rules)
      return PartitionSpec(*core_spmd.from_sharding_rules(sharding, rules))
    return PartitionSpec(*sharding)
  elif hasattr(v, 'shape'):
      return PartitionSpec()
  return None


def get_partition_spec(tree: A) -> A:
  """Extracts a PartitionSpec tree from a PyTree containing ``Variable`` values."""

  def f(x):
    if isinstance(x, variablelib.Variable):
      return x.replace(get_var_pspec(x))
    elif hasattr(x, 'shape'):
        return PartitionSpec()
    return None

  return jax.tree.map(
    f, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def get_named_sharding(tree: A, mesh: jax.sharding.Mesh) -> A:
  spec = get_partition_spec(tree)
  sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
  return sharding


# Sharding constraints
# ------------------------------------------------------------------------------


def shard_variable(v: variablelib.Variable, mesh=None, require_mesh=False) -> variablelib.Variable:
  """Apply sharding constraint on a variable according to its annotations.

  Recommend to only use inside a `jax.jit` under a mesh."""
  if mesh is None and not meta.global_mesh_defined():
    if require_mesh:
      raise ValueError(
        'A mesh or mesh context is required if running `nnx.shard_model()` or'
        ' `nnx.sharded_init()` to initialize a model distributedly.')
    return v
  sharding = get_var_pspec(v)
  if mesh is not None:
    sharding = jax.sharding.NamedSharding(mesh, sharding)
  if sharding is not None:
    v.value = jax.lax.with_sharding_constraint(v.value, sharding)
  return v


def shard_model(model, mesh=None) -> module.Module:
  """Recommend to only use inside a `jax.jit`."""
  gdef, state = graph.split(model)
  state = jax.tree.map(partial(shard_variable, mesh=mesh, require_mesh=True), state,
                       is_leaf=lambda x: isinstance(x, variablelib.Variable))
  return graph.merge(gdef, state)


def sharded_init(init_fn, mesh=None) -> module.Module:
  @jax.jit
  def f():
    model = init_fn()
    return graph.split(shard_model(model, mesh))
  return graph.merge(*f())



# Other utilities
# ------------------------------------------------------------------------------


def get_abstract_state(init_fn, mesh):
  abs_state = jax.eval_shape(lambda: graph.state(init_fn()))
  abs_state = jax.tree.map(
    lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
    abs_state, get_named_sharding(abs_state, mesh)
  )
  return abs_state
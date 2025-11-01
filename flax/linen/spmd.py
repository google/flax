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

"""Utilities for working with jit and partitioned models.

This module introduces ``axis_rules``, ``logical_to_mesh_axes``,
``logical_to_mesh``, ``with_logical_constraint`` for appyling jit sharding
constraints in terms of "logical named axes" rather than jit's default mesh
axes.

Additionally the ``LogicallyPartitioned`` metadata wrapper is defined as
well as the initializer function wrapper ``with_logical_partitioning``for
introducing logical axis metadata into a model's variables.
"""

import dataclasses
import enum
import functools
from typing import Any
from collections.abc import Callable

import jax
from jax import lax

from flax import struct
from flax.core import meta
from flax.core.spmd import (
    _logical_to_mesh_axes,
    _unassigned_axis,
    get_logical_axis_rules,
    logical_to_mesh_axes,
)
from flax.typing import (
  Array,
  LogicalNames,
  LogicalRules,
  ArrayPytree,  # pylint: disable=invalid-name
  LogicalPartitionSpec,  # pylint: disable=unused-import
  LogicalPartitionSpecPytree,  # pylint: disable=invalid-name
  )





def logical_to_mesh(tree: Any, rules: LogicalRules | None = None) -> Any:
  """Applies logical_to_mesh_axes to pytrees of logical PartitionSpecs."""
  return jax.tree_util.tree_map(
      lambda x: logical_to_mesh_axes(x, rules),
      tree,
      is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec),
  )


def logical_to_mesh_sharding(
  tree: Any,
  mesh: jax.sharding.Mesh,
  rules: LogicalRules | None = None,
) -> Any:
  """Convert pytrees of logical PartitionSpecs to shardings."""
  return jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(mesh, x),
      logical_to_mesh(tree, rules),
      is_leaf=lambda x: isinstance(x, jax.sharding.PartitionSpec),
  )


class RulesFallback(enum.Enum):
  """How a sharding constraint should behave when no matching rule is found."""

  AXIS_IS_UNSHARDED = 'axis_is_unsharded'
  RAISE_ERROR = 'raise_error'
  NO_CONSTRAINT = 'no_constraint'


def _with_sharding_constraint(
  x: Array,
  axis_resources: jax.sharding.PartitionSpec | None,
  mesh: jax.sharding.Mesh | None = None,
):
  """Wrapper for lax.with_sharding_constraint, no-op on cpu or outside jit."""
  if not meta.global_mesh_defined() and mesh is None:
    return x
  else:
    if mesh is not None and axis_resources is not None:
      sharding = jax.sharding.NamedSharding(mesh, axis_resources)
      return lax.with_sharding_constraint(x, sharding)
    return lax.with_sharding_constraint(x, axis_resources)


def _with_sharding_constraint_one_fallback(
  axis_resources: LogicalPartitionSpec,
  x: Array,
  fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED,
  rules: LogicalRules | None = None,
  mesh: jax.sharding.Mesh | None = None,
):
  """Either imposes a sharding constraint or applies fallback."""
  mesh_axes = _logical_to_mesh_axes(axis_resources, rules)
  if mesh_axes is None:
    return _with_sharding_constraint(x, None, mesh=mesh)

  if fallback == RulesFallback.AXIS_IS_UNSHARDED:
    mesh_axes = [None if x is _unassigned_axis else x for x in mesh_axes]
  else:
    if any(x is _unassigned_axis for x in mesh_axes):
      if fallback == RulesFallback.RAISE_ERROR:
        raise ValueError(f'Axis names {axis_resources} did not match a rule')
      else:
        return x
  return _with_sharding_constraint(
    x, jax.sharding.PartitionSpec(*mesh_axes), mesh=mesh
  )


def _is_axis_spec(x):
  return (
      isinstance(x, str)
      or x is jax.sharding.PartitionSpec.UNCONSTRAINED
      or x is None
  )


def _is_logical_spec(x):
  return x is None or (
      isinstance(x, tuple) and all(_is_axis_spec(e) for e in x)
  )


def with_logical_constraint(
  x: ArrayPytree,
  logical_axis_resources: LogicalPartitionSpecPytree,
  rules: LogicalRules | None = None,
  mesh: jax.sharding.Mesh | None = None,
  fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED,
):
  """Version of jit's with_sharding_constraint that uses logical axis names."""
  # If no axis binding is set, this is a no-op.
  if rules is None:
    rules = get_logical_axis_rules()
  if not rules or logical_axis_resources is None:
    return x
  # Translate logical names to mesh assignments.
  return jax.tree_util.tree_map(
    functools.partial(
      _with_sharding_constraint_one_fallback,
      fallback=fallback,
      rules=rules,
      mesh=mesh,
    ),
    logical_axis_resources,
    x,
    is_leaf=_is_logical_spec,
  )


# Logical Partitioning Axis Metadata
# ------------------------------------------------------------------------------


class LogicallyPartitioned(meta.Partitioned):
  rules: LogicalRules | None = struct.field(default=None, pytree_node=False)

  def unbox(self, apply_constraint=True) -> Any:
    """Returns the wrapped value with the partitioning constraint applied."""
    if apply_constraint and (meta.global_mesh_defined() or self.mesh is not None):
      return with_logical_constraint(
        self.value,
        self.get_partition_spec(),
        rules=self.rules,
        mesh=self.mesh,
      )
    else:
      return self.value

  def to_nnx_metadata(self) -> dict[str, Any]:
    """Return a dict of metadata that can translate into an `nnx.Variable`."""
    metadata = vars(self)
    if 'names' in metadata:
      metadata['sharding_names'] = metadata.pop('names')
    if 'rules' in metadata:
      metadata['sharding_rules'] = metadata.pop('rules')
    return metadata

  @classmethod
  def from_nnx_metadata(cls, metadata: dict[str, Any]):
    """Given a dict of `nnx.Variable` format metadata, create a `nn.LogicallyPartitioned`."""
    metadata['names'] = metadata.pop('sharding_names')
    metadata['rules'] = metadata.pop('sharding_rules')
    fields = {x.name for x in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in metadata.items() if k in fields})


def with_logical_partitioning(
  fn: Callable[..., Any],
  names: LogicalNames,
  mesh: jax.sharding.Mesh | None = None,
  rules: LogicalRules | None = None,
) -> Callable[..., LogicallyPartitioned]:
  """Wraps a function's return value with LogicallyPartitioned.

  Example::

    >>> import flax.linen as nn
    >>> kernel_init = nn.with_logical_partitioning(
    ...     nn.initializers.lecun_normal(), (None, "data"))
    >>> partitioned_dense = nn.Dense(features=3, kernel_init=kernel_init)

  Args:
    fn: The function to be wrapped. Typically this is an initializer.
    names: The logical axis passed to ``LogicallyPartitioned``.
    mesh: The mesh to use for the partitioning. If None, the global mesh
      resource is used if available.
    rules: Optional logical to mesh rules use. If None, the global rules
      are used if available.
  Returns:
    A function wrapping ``fn`` that will return an instance of
    ``LogicallyPartitioned``.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return LogicallyPartitioned(
      fn(*args, **kwargs), names, mesh=mesh, rules=rules
    )  # pytype: disable=wrong-keyword-args

  return wrapper

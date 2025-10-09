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

import collections
import contextlib
import dataclasses
import threading
import typing as tp

import jax
from jax.sharding import PartitionSpec, NamedSharding
from flax.core import meta
from flax.typing import (
    LogicalRules,
)

def get_pspec(sharding_names, sharding_rules = None) -> PartitionSpec:
  """Given an `nnx.Variable`, return its `PartitionSpec`."""
  if get_logical_axis_rules() or sharding_rules:
    sharding_names = logical_to_mesh_axes(sharding_names, sharding_rules)
  return PartitionSpec(*sharding_names)


def shard_value(value, sharding_names, sharding_rules, mesh):
  if not sharding_names:
    return value
  if not mesh and not meta.global_mesh_defined():
    raise ValueError(
      'An auto mesh context or metadata is required if creating a variable'
      f' with annotation {sharding_names=}. '
      'If running this on CPU for debugging, make a'
      ' dummy mesh like `jax.make_mesh(((1, 1)), (<your axis names>))`. '
      'If running on explicit mode, remove `sharding_names=` annotation.')
  pspec = get_pspec(sharding_names, sharding_rules)
  if mesh is not None:
    jax.lax.with_sharding_constraint(value, NamedSharding(mesh, pspec))
  return jax.lax.with_sharding_constraint(value, pspec)




# Dynamic Axis Mapping Context
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class _AxisRules(threading.local):
  """Dynamic logical axis to mesh axis binding context."""

  rules: LogicalRules = ()


# Global axis binding context.
_axis_rules = _AxisRules()


def set_logical_axis_rules(rules: LogicalRules):
  """Sets the global logical axis to mesh axis binding."""
  _axis_rules.rules = rules


def get_logical_axis_rules() -> LogicalRules:
  """Returns the global logical axis to mesh axis binding."""
  return _axis_rules.rules


@contextlib.contextmanager
def logical_axis_rules(rules: LogicalRules):
  """Context manager for setting the logical to mesh axis bindings."""
  old_rules = _axis_rules.rules
  try:
    _axis_rules.rules = rules
    yield
  finally:
    _axis_rules.rules = old_rules


def composite_rules(rule1, rule2):
  if not rule1 and not rule2:
    return ()
  if rule1 and not rule2:
    return rule1
  if rule2 and not rule1:
    return rule2
  rules = {alias: value for alias, value in rule1}
  for alias, value in rule2:
    if alias in rules and rules[alias] != value:
      raise ValueError(
          f'Inconsistent logical axis annotations for {alias}: '
          f'{rules[alias]} vs {value}'
      )
    rules[alias] = value
  return tuple(rules.items())



class _UnassignedAxis:
  """Sentinel class for unassigned logical axis name."""

  def __repr__(self):
    return 'UnassignedAxis'

  def __bool__(self):
    return False


_unassigned_axis = _UnassignedAxis()


def _mesh_assignment_free(new_assignment, existing_assignments):
  """Determines if a given mesh axis has already been assigned."""
  new = set(jax.tree_util.tree_leaves(new_assignment))
  existing = set(jax.tree_util.tree_leaves(existing_assignments))
  if existing.intersection(new):
    return False
  return True


def _logical_to_mesh_axes(
    array_dim_names: tp.Sequence[str | None] | None,
    rules: LogicalRules | None = None,
) -> list[_UnassignedAxis | None | str | tuple[str, ...]] | None:
  """Same as logical_to_mesh_axes, but doesn't fill in _unassigned_axis."""
  if array_dim_names is None:
    return None
  if rules is None:
    rules = get_logical_axis_rules()
  axis_name_counts = collections.Counter(array_dim_names)
  # None and special values such as PartitionSpec.UNCONSTRAINED can appear more
  # then once.
  dups = tuple(
      k for k, v in axis_name_counts.items() if v > 1 and isinstance(k, str)
  )
  if dups:
    raise ValueError(
      f'Unsupported: Dimensions {dups} occur more than once in array names.'
    )
  if not isinstance(rules, (tuple, list)):
    raise ValueError('Unknown axis rule specification type.')
  # We assign mesh axes using a priority based ruleset over logical axis names.
  result: list[_UnassignedAxis | None | str | tuple[str, ...]]
  result = [
      (_unassigned_axis if isinstance(name, str) else name)
      for name in array_dim_names
  ]
  for rule_model_name, rule_mesh_names in rules:
    if rule_model_name in array_dim_names:
      pos = array_dim_names.index(rule_model_name)
      if (
        _mesh_assignment_free(rule_mesh_names, result)
        and result[pos] == _unassigned_axis
      ):
        result[pos] = rule_mesh_names
  return result


def logical_to_mesh_axes(
  array_dim_names: tp.Sequence[str | None] | None,
  rules: LogicalRules | None = None,
) -> jax.sharding.PartitionSpec | None:
  """Compute layout for an array.

  The rules are in order of precedence, and consist of pairs:
  ``(ArrayDimensionName, MeshDimensionName)``, meaning that the given array
  dimension (if present and unused) should be sharded across the given
  mesh dimension (if present and unused).

  A Layout of an Array is expressed as a tuple with one element for each
  dimension in the Array. The element is either None, or is the name of a
  mesh-dimension, meaning that this dimension of the array is sharded across
  this dimension of the mesh.

  For example, given an array with::

    array_dim_names = ('batch', 'length', 'heads', 'features')

  and the layout rules are::

    rules = (('batch', 'X'),
             ('features', 'X'),
             ('heads', 'Y'),
             ('batch', 'Z'))

  then this function will return::

    PartitionSpec('X', None, 'Y', None)

  Args:
    array_dim_names: Tuple of array dimension names or None.
    rules: Optional logical to mesh rules override.  Defaults to using the
      rules defined in the dynamic context set from the ``axis_rules`` function.

  Returns:
    PartitionSpec for the parameter.
  """
  result = _logical_to_mesh_axes(array_dim_names, rules)
  if result is None:
    return None
  # We default to None - ie unsharded along the dimension.
  result = [None if x is _unassigned_axis else x for x in result]
  return jax.sharding.PartitionSpec(*result)



# def from_sharding_rules(
#     sharding_names: Sharding, sharding_rules: LogicalRules
# ) -> Sharding:
#   rules = {alias: on_mesh for (alias, on_mesh) in sharding_rules}
#   return tuple(
#       rules[str(s)] if (s and str(s) in rules) else s for s in sharding_names
#   )

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

import contextlib
import dataclasses
import threading

import jax
from jax.sharding import PartitionSpec, NamedSharding
from flax.core import meta
from jax.experimental.layout import Format
from flax.typing import (
    LogicalRules,
    Sharding,
)

def get_pspec(sharding, sharding_rules = None) -> PartitionSpec:
  """Given an `nnx.Variable`, return its `PartitionSpec`."""
  return apply_rules(sharding, sharding_rules)   # type: ignore

def map_sharding(f, sharding):
  if isinstance(sharding, PartitionSpec) or isinstance(sharding, tuple):
    return PartitionSpec(*map(f, sharding))
  elif isinstance(sharding, NamedSharding):
    return NamedSharding(sharding.mesh, map_sharding(f, sharding.spec)) # type: ignore
  elif isinstance(sharding, Format):
    return Format(sharding.layout, map_sharding(f, sharding.sharding))  # type: ignore

def get_mesh(sharding):
  if isinstance(sharding, PartitionSpec) or isinstance(sharding, tuple):
    return None
  elif isinstance(sharding, NamedSharding):
    return sharding.mesh
  elif isinstance(sharding, Format):
    return get_mesh(sharding.sharding)

def apply_rules(sharding, sharding_rules):
  """Rename the axes of a sharding specification (which can include `PartitionSpec`, `NamedSharding` or `Format` objects)."""
  if get_logical_axis_rules() or sharding_rules:
    context_rules = get_logical_axis_rules()
    rules = {alias: on_mesh for (alias, on_mesh) in composite_rules(context_rules, sharding_rules)}
  else:
    rules = {}
  return map_sharding(lambda a: rules.get(a, a), sharding)

def _apply_sharding(value, sharding, mesh):
  if isinstance(sharding, Format):
    return jax.device_put(value, sharding)
  if mesh.are_all_axes_explicit:
    return jax.sharding.reshard(value, sharding)
  elif mesh.are_all_axes_auto:
    return jax.lax.with_sharding_constraint(value, sharding)
  else:
    raise ValueError(
        'Mesh must have all axes as Explicit or all axes as Auto. '
        f'Got mixed axis types: {mesh.axis_types}')

def shard_value(value, out_sharding, sharding_rules, mesh):
  if not out_sharding:
    return value

  if mesh is None:
    mesh = meta.get_global_mesh()

  out_sharding = apply_rules(out_sharding, sharding_rules)

  sharding_mesh = get_mesh(out_sharding)

  if sharding_mesh:
    if mesh:
      assert mesh == out_sharding.mesh
    mesh = sharding_mesh

  if mesh is None:
    raise ValueError(
      'An auto mesh context or metadata is required if creating a variable'
      f' with annotation {out_sharding=}. '
      'For more guidance, see https://flax.readthedocs.io/en/latest/flip/4844-var-eager-sharding.html.')

  if isinstance(out_sharding, PartitionSpec):
    out_sharding = NamedSharding(mesh, out_sharding)

  return _apply_sharding(value, out_sharding, mesh)




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


def from_sharding_rules(
    sharding: Sharding, sharding_rules: LogicalRules
) -> Sharding:
  rules = {alias: on_mesh for (alias, on_mesh) in sharding_rules}
  return tuple(
      rules[str(s)] if (s and str(s) in rules) else s for s in sharding
  )

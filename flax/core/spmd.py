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
from flax.typing import (
    LogicalRules,
    Sharding,
)

def get_pspec(sharding_names, sharding_rules = None) -> PartitionSpec:
  """Given an `nnx.Variable`, return its `PartitionSpec`."""
  if get_logical_axis_rules() or sharding_rules:
    context_rules = get_logical_axis_rules()
    rules = composite_rules(context_rules, sharding_rules)
    return PartitionSpec(*from_sharding_rules(sharding_names, rules))
  return PartitionSpec(*sharding_names)

def _apply_sharding(value, sharding):
  with jax.disable_jit(False):
    return jax.jit(lambda x: x, out_shardings=sharding)(value)

def shard_value(value, sharding_names, sharding_rules, mesh):
  if not sharding_names:
    return value
  if not mesh and not meta.global_mesh_defined():
    raise ValueError(
      'An auto mesh context or metadata is required if creating a variable'
      f' with annotation {sharding_names=}. '
      'For more guidance, see https://flax.readthedocs.io/en/latest/flip/4844-var-eager-sharding.html.')
  pspec = get_pspec(sharding_names, sharding_rules)
  if mesh is not None:
    return _apply_sharding(value, NamedSharding(mesh, pspec))
  return _apply_sharding(value, pspec)




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

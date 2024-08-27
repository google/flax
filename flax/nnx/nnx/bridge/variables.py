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

from typing import Any, TypeVar

import jax
from flax import struct
from flax.core import meta
from flax.nnx.nnx import variables as variableslib
import typing as tp


A = TypeVar('A')
B = TypeVar('B')


#######################################################
### Variable type <-> Linen collection name mapping ###
#######################################################
# Assumption: the mapping is 1-1 and unique.

VariableTypeCache: dict[str, tp.Type[variableslib.Variable[tp.Any]]] = {}


def variable_type(name: str) -> tp.Type[variableslib.Variable[tp.Any]]:
  """Given a Linen-style collection name, get or create its corresponding NNX Variable type."""
  if name not in VariableTypeCache:
    VariableTypeCache[name] = type(name, (variableslib.Variable,), {})
  return VariableTypeCache[name]


def variable_type_name(typ: tp.Type[variableslib.Variable[tp.Any]]) -> str:
  """Given an NNX Variable type, get or create its Linen-style collection name.

  Should output the exact inversed result of `variable_type()`."""
  for name, t in VariableTypeCache.items():
    if typ == t:
      return name
  name = typ.__name__
  if name in VariableTypeCache:
    raise ValueError(
      'Name {name} is already registered in the registry as {VariableTypeCache[name]}. '
      'It cannot be linked with this type {typ}.'
    )
  register_variable_name_type_pair(name, typ)
  return name


def register_variable_name_type_pair(name, typ):
  """Register a pair of variable type name (like Linen collections) and its NNX type."""
  VariableTypeCache[name] = typ


# add known variable type names
register_variable_name_type_pair('params', variableslib.Param)
register_variable_name_type_pair('batch_stats', variableslib.BatchStat)
register_variable_name_type_pair('cache', variableslib.Cache)
register_variable_name_type_pair('intermediates', variableslib.Intermediate)


def sort_variable_types(types: list[type]):
  def _variable_parents_count(t: type):
    return sum(1 for p in t.mro() if issubclass(p, variableslib.Variable))
  parent_count = {t: _variable_parents_count(t) for t in types}
  return sorted(types, key=lambda t: -parent_count[t])


#############################################
### NNX Variable <-> Linen metadata boxes ###
#############################################


class NNXMeta(struct.PyTreeNode, meta.AxisMetadata[A]):
  """Default Flax metadata class for `nnx.VariableState`.
  """

  var_type: type[variableslib.Variable[tp.Any]] = struct.field(pytree_node=False)
  value: Any = struct.field(pytree_node=True)
  metadata: dict[str, tp.Any] = struct.field(pytree_node=False)

  def unbox(self) -> A:
    return self.value

  def replace_boxed(self, val: B) -> 'NNXMeta[B]':
    return self.replace(value=val)  # type: ignore

  def add_axis(self, index: int, params: dict[Any, Any]) -> 'NNXMeta[A]':
    # TODO: implement this, supporting hooks
    return self

  def remove_axis(self, index: int, params: dict[Any, Any]) -> 'NNXMeta[A]':
    # TODO: implement this, supporting hooks
    return self


def to_linen_var(vs: variableslib.VariableState) -> meta.AxisMetadata:
  metadata = vs.get_metadata()
  if 'linen_meta_type' in metadata:
    if metadata['linen_meta_type'] is not meta.Partitioned:
      raise ValueError('Not supporting Linen metadata types other than nn.Partitioned')
    return meta.Partitioned(vs.value, names=metadata['sharding'], mesh=metadata['mesh'])
  return NNXMeta(vs.type, vs.value, vs.get_metadata())


def to_nnx_var(kp: tp.Sequence[Any], x: meta.AxisMetadata | Any) -> variableslib.Variable:
  """This function needs jax.tree_util.tree_map_with_path to map over. """
  if isinstance(x, NNXMeta):
    return x.var_type(x.value, **x.metadata)
  # Infer variable type from the linen collection name
  assert isinstance(kp[0], jax.tree_util.DictKey)
  vtype = variable_type(str(kp[0].key))
  if isinstance(x, meta.AxisMetadata):
    if isinstance(x, meta.Partitioned):
      return vtype(x.value, sharding=x.names, mesh=x.mesh, linen_meta_type=meta.Partitioned)
    raise ValueError('Not yet supporting metadata types other than nn.Partitioned and NNXMeta')
  return vtype(x)

def to_nnx_var_state(kp: tp.Sequence[Any], x: meta.AxisMetadata | Any) -> variableslib.VariableState:
  """This function needs jax.tree_util.tree_map_with_path to map over. """
  return to_nnx_var(kp, x).to_state()
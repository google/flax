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
import typing as tp

import jax
from flax import struct
from flax.core import meta
from flax.nnx import spmd
from flax.nnx import variablelib
from flax.typing import LogicalNames


A = TypeVar('A')
B = TypeVar('B')


def sort_variable_types(types: tp.Iterable[type]):
  def _variable_parents_count(t: type):
    return sum(1 for p in t.mro() if issubclass(p, variablelib.Variable))
  parent_count = {t: _variable_parents_count(t) for t in types}
  return sorted(types, key=lambda t: -parent_count[t])


#############################################
### NNX Variable <-> Linen metadata boxes ###
#############################################


class NNXMeta(struct.PyTreeNode, meta.AxisMetadata[A]):
  """Default Flax metadata class for `nnx.Variable`."""

  var_type: type[variablelib.Variable[tp.Any]] = struct.field(pytree_node=False)
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

  def get_partition_spec(self) -> jax.sharding.PartitionSpec:
    """Returns the ``Partitionspec`` for this partitioned value."""
    nnx_var = self.to_nnx_variable()
    spec = spmd.get_partition_spec(nnx_var).raw_value
    assert isinstance(spec, jax.sharding.PartitionSpec)
    return spec

  def to_nnx_variable(self) -> variablelib.Variable:
    return self.var_type(self.value, **self.metadata)


def with_partitioning(
    fn: tp.Callable[..., tp.Any],
    names: LogicalNames,
    mesh: jax.sharding.Mesh | None = None,
) -> tp.Callable[..., meta.Partitioned[tp.Any]]:
  """Same interface as Linen, but calls NNX `with_partitioning` within."""
  return spmd.with_partitioning(fn, names, mesh,
                                linen_meta_type=meta.Partitioned)
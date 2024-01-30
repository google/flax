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

from typing import (
  Any,
  Callable,
  Dict,
  Generic,
  Mapping,
  Optional,
  Sequence,
  Tuple,
  TypeVar,
  Union,
)

import jax
from flax.core import FrozenDict

import dataclasses


# General

Array = Union[jax.Array, Any]
PRNGKey = jax.Array
RNGSequences = Dict[str, PRNGKey]
Dtype = Union[jax.typing.DTypeLike, Any]
Shape = Sequence[int]

Path = str
PathParts = Tuple[str, ...]

Leaf = Any


# Linear

PrecisionLike = Union[
  None,
  str,
  jax.lax.Precision,
  Tuple[str, str],
  Tuple[jax.lax.Precision, jax.lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


# Initializers

Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]


# Collections

Collection = Mapping[str, Any]
MutableCollection = Dict[str, Any]


# Dicts

VariableDict = Mapping[str, Collection]
FrozenVariableDict = FrozenDict[str, Collection]
MutableVariableDict = Dict[str, MutableCollection]

PRNGFoldable = Union[int, str]


# Axes

T = TypeVar('T')

@dataclasses.dataclass(frozen=True)
class In(Generic[T]):
  """Specifies a variable collection should only be lifted as input."""

  axis: T

@dataclasses.dataclass(frozen=True)
class Out(Generic[T]):
  """Specifies a variable collection should only be lifted as output."""

  axis: T

Axis = Optional[int]
InOutAxis = Union[Axis, In[Axis], Out[Axis]]

ScanAxis = int
InOutScanAxis = Union[ScanAxis, In[ScanAxis], Out[ScanAxis]]

Axes = Union[int, Sequence[int]]


# SPMD

LogicalNames = Tuple[Union[str, None], ...]

LogicalRules = Sequence[Tuple[str, Union[str, Tuple[str], None]]]
ArrayPytree = Any  # pylint: disable=invalid-name
LogicalPartitionSpec = Any  # pylint: disable=invalid-name
LogicalPartitionSpecPytree = Any  # pylint: disable=invalid-name
PartitionSpecPytree = Any  # pylint: disable=invalid-name

Sharding = Tuple[Optional[str], ...]
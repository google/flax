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

import typing as tp

RefMap = tp.MutableMapping[tp.Any, int]
IndexMap = dict[int, tp.Any]

class NodeDef:
  type: type
  index: int | None
  outer_index: int | None
  num_attributes: int
  metadata: tp.Any

  def with_no_outer_index(self) -> NodeDef: ...
  def with_same_outer_index(self) -> NodeDef: ...
  def __eq__(self, other: tp.Any) -> bool: ...
  def __hash__(self) -> int: ...
  def __getstate__(
    self,
  ) -> tuple[tp.Any, tp.Any, tp.Any, tp.Any, tp.Any]: ...
  @staticmethod
  def __setstate__(
    nodedef: NodeDef, state: tuple[tp.Any, tp.Any, tp.Any, tp.Any, tp.Any]
  ) -> None: ...

class VariableDef:
  type: type
  index: int
  outer_index: int | None
  metadata: tp.Any

  def with_no_outer_index(self) -> VariableDef: ...
  def with_same_outer_index(self) -> VariableDef: ...
  def __eq__(self, other: tp.Any) -> bool: ...
  def __hash__(self) -> int: ...
  def __getstate__(
    self,
  ) -> tuple[tp.Any, int, tp.Any, tp.Any]: ...
  @staticmethod
  def __setstate__(
    variabledef: 'VariableDef', state: tuple[tp.Any, int, tp.Any, tp.Any]
  ) -> None: ...

class NodeRef:
  index: int

  def __eq__(self, other: tp.Any) -> bool: ...
  def __hash__(self) -> int: ...
  def __getstate__(self) -> tuple[int]: ...
  @staticmethod
  def __setstate__(noderef: NodeRef, state: tuple[int]) -> None: ...


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

import builtins
import dataclasses
from flax.typing import Path
import typing as tp

if tp.TYPE_CHECKING:
  ellipsis = builtins.ellipsis
else:
  ellipsis = tp.Any

Predicate = tp.Callable[[Path, tp.Any], bool]
FilterLiteral = tp.Union[type, str, Predicate, bool, ellipsis, None]
Filter = tp.Union[FilterLiteral, tuple[FilterLiteral, ...], list[FilterLiteral]]


def to_predicate(filter: Filter) -> Predicate:
  if isinstance(filter, str):
    return AtPath(filter)
  elif isinstance(filter, type):
    return OfType(filter)
  elif isinstance(filter, bool):
    return Everything() if filter else Nothing()
  elif filter is Ellipsis:
    return Everything()
  elif filter is None:
    return Nothing()
  elif callable(filter):
    return filter
  elif isinstance(filter, (list, tuple)):
    return Any(*filter)
  else:
    raise TypeError(f'Invalid collection filter: {filter:!r}. ')


@dataclasses.dataclass
class AtPath:
  path: str

  def __call__(self, path: Path, x: tp.Any):
    return self.path == path


@dataclasses.dataclass
class OfType:
  type: type

  def __call__(self, path: Path, x: tp.Any):
    return isinstance(x, self.type)


class Any:
  def __init__(self, *filters: Filter):
    self.predicates = tuple(
      to_predicate(collection_filter) for collection_filter in filters
    )

  def __call__(self, path: Path, x: tp.Any):
    return any(predicate(path, x) for predicate in self.predicates)


class All:
  def __init__(self, *filters: Filter):
    self.predicates = tuple(
      to_predicate(collection_filter) for collection_filter in filters
    )

  def __call__(self, path: Path, x: tp.Any):
    return all(predicate(path, x) for predicate in self.predicates)


class Not:
  def __init__(self, collection_filter: Filter):
    self.predicate = to_predicate(collection_filter)

  def __call__(self, path: Path, x: tp.Any):
    return not self.predicate(path, x)


class Everything:
  def __call__(self, path: Path, x: tp.Any):
    return True


class Nothing:
  def __call__(self, path: Path, x: tp.Any):
    return False

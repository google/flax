# Copyright 2023 The Flax Authors.
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

from __future__ import annotations

import dataclasses
import threading
import typing as tp
from contextlib import contextmanager
from types import MappingProxyType

A = tp.TypeVar('A')


@dataclasses.dataclass
class FlagsContext(threading.local):
  flags_stack: tp.List[MappingProxyType[str, tp.Any]] = dataclasses.field(
    default_factory=lambda: [MappingProxyType({})]
  )


FLAGS_CONTEXT = FlagsContext()


class Flags(tp.Mapping[str, tp.Any]):
  __slots__ = ()

  def __getitem__(self, name: str) -> tp.Any:
    current_flags = FLAGS_CONTEXT.flags_stack[-1]
    if name not in current_flags:
      raise ValueError(f'Unknown Flag: {name}')
    return current_flags[name]

  __getattr__ = __getitem__

  def __iter__(self) -> tp.Iterator[str]:
    return iter(FLAGS_CONTEXT.flags_stack[-1])

  def __len__(self) -> int:
    return len(FLAGS_CONTEXT.flags_stack[-1])

  def __contains__(self, name: tp.Any) -> bool:
    return name in FLAGS_CONTEXT.flags_stack[-1]

  @contextmanager
  def __call__(self, **kwargs: tp.Any):
    current_flags = FLAGS_CONTEXT.flags_stack[-1]
    FLAGS_CONTEXT.flags_stack.append(
      MappingProxyType(dict(current_flags, **kwargs))
    )
    try:
      yield
    finally:
      FLAGS_CONTEXT.flags_stack.pop()

  @tp.overload
  def get(self, name: str) -> tp.Any:
    ...

  @tp.overload
  def get(self, name: str, default: A) -> A:
    ...

  def get(self, name: str, default: A = None) -> A | None:
    return FLAGS_CONTEXT.flags_stack[-1].get(name, default)


flags = Flags()

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
import typing as tp

import jax
import jax.numpy as jnp

from flax.experimental.nnx.nnx import graph
from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx.variables import Variable
from flax.experimental.nnx.nnx import filterlib
from flax.experimental.nnx.nnx.filterlib import All
from flax.experimental.nnx.nnx.object import Object

Counts = list[int]
AxesValue = tp.Union[int, None]
SplitPattern = tp.Union[AxesValue, tuple[AxesValue, ...]]


class Missing:
  pass


MISSING = Missing()


class RngState(Variable[jax.Array]):
  pass


class RngCount(RngState):
  tag: str


class RngKey(RngState):
  tag: str

class RngKeyBackup(RngState):
  pass


NotKey = filterlib.All(RngState, filterlib.Not(RngKey))


@dataclasses.dataclass(repr=False)
class RngStream(Object):
  def __init__(
    self,
    tag: str,
    key: jax.Array,
    count: jax.Array,
  ):
    self.key = RngKey(key, tag=tag)
    self.count = RngCount(count, tag=tag)
    self.key_backups: list[RngKeyBackup] = []

  def __post_init__(self):
    if not isinstance(self.key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(self.key)}')

  def __call__(self) -> jax.Array:
    self.check_valid_context(
      'Cannot call RngStream from a different trace level'
    )
    key = jax.random.fold_in(self.key.value, self.count.value)
    self.count.value += 1
    return key


RngValue = tp.Union[int, jax.Array]
RngDict = tp.Union[
  tp.Mapping[str, int],
  tp.Mapping[str, jax.Array],
  tp.Mapping[str, RngValue],
]


class Rngs(Object, tp.Mapping[str, tp.Callable[[], jax.Array]]):
  def __init__(
    self,
    default: RngValue | RngDict | None = None,
    /,
    **rngs: RngValue,
  ):
    if default is not None:
      if isinstance(default, tp.Mapping):
        rngs = {**default, **rngs}
      else:
        rngs['default'] = default

    for name, value in rngs.items():
      stream = RngStream(
        tag=name,
        key=jax.random.key(value) if isinstance(value, int) else value,
        count=jnp.array(0, dtype=jnp.uint32),
      )
      setattr(self, name, stream)

  def _get_stream(self, name: str, error_type: type[Exception]) -> RngStream:
    rngs_vars = vars(self)
    if name not in rngs_vars:
      if 'default' not in rngs_vars:
        raise error_type(f"No RNG named {name!r} or 'default' found in Rngs.")
      stream = rngs_vars['default']
    else:
      stream = rngs_vars[name]

    return stream

  def __getitem__(self, name: str):
    return self._get_stream(name, KeyError)

  def __getattr__(self, name: str):
    return self._get_stream(name, AttributeError)

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    for name in vars(self):
      if name != '_object__state':
        yield name

  def __len__(self) -> int:
    return len(vars(self)) - 1

  def __contains__(self, name: tp.Any) -> bool:
    return name in vars(self)

class ForkStates(tp.NamedTuple):
  split_keys: State
  split_counts: State
  broadcast_keys: State
  broadcast_counts: State


def fork(
  state: State,
  split_filter: filterlib.Filter,
  split_pattern: SplitPattern,
) -> ForkStates:
  if split_pattern is None:
    raise RuntimeError('Split pattern cannot be None, this is a bug.')

  num_splits: int | tuple[int, ...]
  if isinstance(split_pattern, int):
    num_splits = split_pattern
  else:
    num_splits = tuple(x if x is not None else 1 for x in split_pattern)

  split_keys, split_counts, broadcast_keys, broadcast_counts = state.split(
    All(split_filter, RngKey),
    All(split_filter, RngCount),
    [RngKey, RngKeyBackup],  # Any
    RngCount,
  )

  def split_key(key: tp.Any) -> jax.Array:
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')

    return jax.random.split(key, num_splits)

  split_keys = jax.tree.map(split_key, split_keys)

  return ForkStates(split_keys, split_counts, broadcast_keys, broadcast_counts)


def backup_keys(node: tp.Any, /):
  streams: list[RngStream] = []
  for _, stream in graph.iter_nodes(node):
    if isinstance(stream, RngStream):
      stream.key_backups.append(RngKeyBackup(stream.key.value))
      streams.append(stream)
  return streams


def restore_keys(streams: list[RngStream], /):
  for stream in streams:
    if not stream.key_backups:
      raise RuntimeError('No key backups found.')
    backup = stream.key_backups.pop()
    stream.key.value = backup.value
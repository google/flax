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
import functools
import typing as tp

import jax

from flax.experimental.nnx.nnx import errors, filterlib, tracers

Counts = list[int]
AxesValue = tp.Union[int, None]
Pattern = tp.Union[AxesValue, tuple[AxesValue, ...]]


class Missing:
  pass


MISSING = Missing()


@dataclasses.dataclass
class RngStream:
  key: jax.Array  # dynamic
  count: int  # static

  def __post_init__(self):
    if not isinstance(self.key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(self.key)}')

  def make_rng(self) -> jax.Array:
    count = self.count
    self.count += 1
    return jax.random.fold_in(self.key, count)

  def fork(self, pattern: Pattern) -> jax.Array:
    if pattern is None:
      # broadcast key
      key = self.make_rng()
    else:
      if isinstance(pattern, int):
        num_splits = pattern
      else:
        num_splits = tuple(x if x is not None else 1 for x in pattern)
      key = jax.random.split(self.key, num_splits)
      self.count += 1
    return key


RngValue = tp.Union[int, jax.Array]
RngDict = tp.Union[
  tp.Mapping[str, int],
  tp.Mapping[str, jax.Array],
  tp.Mapping[str, RngValue],
]


class Rngs(tp.Mapping[str, tp.Callable[[], jax.Array]]):
  __slots__ = ('_trace_state', '_rngs', '_counts')

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

    self._rngs = {
      name: RngStream(
        key=jax.random.key(value) if isinstance(value, int) else value,
        count=0,
      )
      for name, value in rngs.items()
    }
    self._trace_state = tracers.TraceState()

  def _make_rng(self, name: str, error_type: Exception) -> jax.Array:
    if not self.is_valid():
      raise errors.TraceContextError(
        'Cannot use Rngs from a different trace level'
      )
    if name not in self._rngs:
      if 'default' not in self._rngs:
        raise error_type(f"No RNG named {name!r} or 'default' found in Rngs.")
      stream = self._rngs['default']
    else:
      stream = self._rngs[name]

    return stream.make_rng()

  def __getitem__(self, name: str) -> tp.Callable[[], jax.Array]:
    return lambda: self._make_rng(name, KeyError)

  def __getattr__(self, name: str) -> tp.Callable[[], jax.Array]:
    return lambda: self._make_rng(name, AttributeError)

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    return iter(self._rngs)

  def __len__(self) -> int:
    return len(self._rngs)

  def __contains__(self, name: tp.Any) -> bool:
    return name in self._rngs

  def replace(self, **kwargs: tp.Union[int, jax.Array, RngStream]) -> 'Rngs':
    rngs: dict[str, tp.Any] = self._rngs.copy()
    rngs.update(kwargs)
    return Rngs(**rngs)

  def is_valid(self) -> bool:
    return self._trace_state.is_valid()

  def fork(
    self,
    _default: Pattern | dict[filterlib.Filter, Pattern] | Missing = MISSING,
    /,
    **patterns: Pattern,
  ) -> ForkedKeys:
    if not self.is_valid():
      raise errors.TraceContextError(
        'Cannot use Rngs from a different trace level'
      )

    filter_patterns: list[tuple[filterlib.Filter, Pattern]]
    if isinstance(_default, dict):
      # merge default and patterns
      filter_patterns = [
        *_default.items(),
        *patterns.items(),
        (..., None),  # broadcast all remaining
      ]
    else:
      default = None if isinstance(_default, Missing) else _default
      filter_patterns = [
        *patterns.items(),
        (..., default),  # split all remaining with default
      ]

    predicate_pattern = [
      (filterlib.to_predicate(filter_), pattern)
      for filter_, pattern in filter_patterns
    ]

    splits: dict[str, jax.Array] = {}
    broadcasts: dict[str, jax.Array] = {}

    for name, stream in self._rngs.items():
      for predicate, pattern in predicate_pattern:
        if predicate(name, stream):
          fork = stream.fork(pattern)
          if pattern is None:
            broadcasts[name] = fork
          else:
            splits[name] = fork
          break
      else:
        raise RuntimeError(
          f'Strea {name!r} did not match any predicate, this is a bug.'
        )

    return ForkedKeys(broadcasts, splits)


class ForkedKeys(tp.Mapping[str, jax.Array]):
  def __init__(
    self,
    broadcast_rngs: dict[str, jax.Array],
    split_rngs: dict[str, jax.Array],
  ):
    self.broadcasts = broadcast_rngs
    self.splits = split_rngs

  def __getitem__(self, key: str) -> jax.Array:
    if key in self.broadcasts:
      return self.broadcasts[key]
    elif key in self.splits:
      return self.splits[key]
    else:
      raise KeyError(f'Key "{key}" not found in SplitRng.')

  def __iter__(self) -> tp.Iterator[str]:
    yield from self.broadcasts
    yield from self.splits

  def __len__(self) -> int:
    return len(self.broadcasts) + len(self.splits)


def _split_rng_flatten(rngs: ForkedKeys, *, with_keys: bool):
  broadcast_names = sorted(rngs.broadcasts.keys())
  split_names = sorted(rngs.splits.keys())

  items = [(name, rngs.broadcasts[name]) for name in broadcast_names]
  items += [(name, rngs.splits[name]) for name in split_names]

  if with_keys:
    nodes = tuple((jax.tree_util.DictKey(name), value) for name, value in items)
  else:
    nodes = tuple(value for _, value in items)

  metadata = (broadcast_names, split_names)

  return nodes, metadata


def _split_rng_unflatten(
  metadata: tuple[tuple[str, ...], tuple[str, ...]],
  nodes: tuple[jax.Array, ...],
):
  broadcast_names, split_names = metadata
  num_broadcasts = len(broadcast_names)
  rngs = ForkedKeys(
    dict(zip(broadcast_names, nodes[:num_broadcasts])),
    dict(zip(split_names, nodes[num_broadcasts:])),
  )
  return rngs


jax.tree_util.register_pytree_with_keys(
  ForkedKeys,
  functools.partial(_split_rng_flatten, with_keys=True),
  _split_rng_unflatten,
  flatten_func=functools.partial(_split_rng_flatten, with_keys=False),
)

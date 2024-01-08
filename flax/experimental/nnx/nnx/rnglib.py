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
import hashlib
import typing as tp

import jax
import numpy as np

from flax.experimental.nnx.nnx import errors, filterlib, tracers

Counts = list[int]
AxesValue = tp.Union[int, None]
Pattern = tp.Union[AxesValue, tuple[AxesValue, ...]]


class Missing:
  pass


MISSING = Missing()


def _stable_hash(data: tp.Sequence[tp.Hashable]) -> int:
  hash_str = ' '.join(str(x) for x in data)
  _hash = hashlib.blake2s(hash_str.encode())
  hash_bytes = _hash.digest()
  # uint32 is represented as 4 bytes in big endian
  return int.from_bytes(hash_bytes[:4], byteorder='big')


@dataclasses.dataclass
class RngStream:
  key: jax.Array  # dynamic
  counts: list[int]  # static

  def make_rng(self) -> jax.Array:
    fold_data = _stable_hash(self.counts)
    self.counts[-1] += 1
    return jax.random.fold_in(self.key, fold_data)

  def fork(self, pattern: Pattern) -> 'RngStream':
    if pattern is None:
      # broadcast key
      key = self.key
      count_path = [*self.counts, 0]
      self.counts[-1] += 1
    else:
      key = self.make_rng()
      # split key
      if isinstance(pattern, int):
        key = jax.random.split(key, pattern)
      else:
        num_splits = int(np.prod([x for x in pattern if x is not None]))
        axis_size = tuple(x if x is not None else 1 for x in pattern)
        # reshape key
        key = jax.random.split(key, num_splits).reshape(*axis_size)
      count_path = [0]
    return RngStream(key, count_path)

  def copy(self) -> 'RngStream':
    return RngStream(self.key, self.counts.copy())


jax.tree_util.register_pytree_node(
  RngStream,
  lambda rng: ((rng.key,), tuple(rng.counts)),
  lambda counts, nodes: RngStream(nodes[0], list(counts)),
)

RngValue = tp.Union[int, jax.Array, RngStream]
RngDict = tp.Union[
  dict[str, int],
  dict[str, jax.Array],
  dict[str, RngStream],
  dict[str, RngValue],
]


class Rngs(tp.Mapping[str, tp.Callable[[], jax.Array]]):
  __slots__ = ('_trace_state', '_rngs', '_counts')

  def __init__(
    self,
    default: RngValue | RngDict | None = None,
    **rngs: RngValue,
  ):
    if default is not None:
      if isinstance(default, dict):
        rngs = {**default, **rngs}
      else:
        rngs['default'] = default

    self._rngs = {
      name: (
        RngStream(jax.random.key(value), [0])
        if isinstance(value, int)
        else RngStream(value, [0])
        if isinstance(value, jax.Array)
        else value.copy()
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

  def copy(self) -> 'Rngs':
    return Rngs(**self._rngs)

  def replace(self, **kwargs: tp.Union[int, jax.Array, RngStream]) -> 'Rngs':
    rngs: dict[str, tp.Any] = self._rngs.copy()
    rngs.update(kwargs)
    return Rngs(**rngs)

  def is_valid(self) -> bool:
    return self._trace_state.is_valid()

  @tp.overload
  def fork(self) -> dict[str, RngStream]:
    ...

  @tp.overload
  def fork(self, __default: Pattern) -> dict[str, RngStream]:
    ...

  @tp.overload
  def fork(
    self,
    __default: Pattern | dict[filterlib.Filter, Pattern] | Missing = MISSING,
    **patterns: Pattern,
  ) -> tuple[dict[str, RngStream], dict[str, RngStream]]:
    ...

  def fork(
    self,
    _default: Pattern | dict[filterlib.Filter, Pattern] | Missing = MISSING,
    **patterns: Pattern,
  ) -> dict[str, RngStream] | tuple[dict[str, RngStream], dict[str, RngStream]]:
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

    splits: dict[str, RngStream] = {}
    broadcasts: dict[str, RngStream] = {}

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

    if isinstance(_default, dict) or patterns:
      return splits, broadcasts
    else:
      return {**splits, **broadcasts}

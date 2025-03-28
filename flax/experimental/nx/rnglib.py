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
from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp
from flax.experimental.nx.objectlib import Object
from flax.experimental.nx.variablelib import Variable


class RngState(Variable):
  tag: str


class RngCount(RngState): ...


class RngKey(RngState): ...


class RngStream(Object):
  __nodes__ = ('key', 'count')

  def __init__(
    self,
    tag: str,
    key: jax.Array,
    count: jax.Array,
  ):
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')

    self.key = RngKey(key, tag=tag)
    self.count = RngCount(count, tag=tag)

  def __call__(self) -> jax.Array:
    key = jax.random.fold_in(self.key.value, self.count.value)
    self.count.value += 1
    return key


class Rngs(Object):
  def __init__(
    self,
    default: int | jax.Array | None = None,
    /,
    **rngs: int | jax.Array,
  ):
    if default is not None:
      rngs['default'] = default

    self._streams: dict[str, RngStream] = {}
    for name, value in rngs.items():
      if isinstance(value, int):
        key = jax.random.key(value)
      elif isinstance(value, jax.Array):
        if value.dtype == jnp.uint32:
          key = jax.random.wrap_key_data(value)
        else:
          key = value
      else:
        raise ValueError(f'Invalid rng value: {value}')

      self._streams[name] = RngStream(
        tag=name,
        key=key,
        count=jnp.zeros(key.shape, dtype=jnp.uint32),
      )

  def __getitem__(self, name: str):
    if name not in self._streams:
      if 'default' not in self._streams:
        raise KeyError(f"No RNG named {name!r} or 'default' found in Rngs.")
      stream = self._streams['default']
    else:
      stream = self._streams[name]

    return stream

  def __getattr__(self, name: str):
    if name not in self._streams:
      if 'default' not in self._streams:
        raise AttributeError(
          f"No RNG named {name!r} or 'default' found in Rngs."
        )
      stream = self._streams['default']
    else:
      stream = self._streams[name]

    return stream

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    return iter(self._streams.keys())

  def __len__(self) -> int:
    return len(self._streams) - 1

  def __contains__(self, name: tp.Any) -> bool:
    return name in self._streams

  def items(self):
    return self._streams.items()

  # pickle support
  def __getstate__(self):
    return vars(self).copy()

  def __setstate__(self, state):
    vars(self).update(state)

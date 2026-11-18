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
from flax.core.frozen_dict import FrozenDict
from flax.experimental.nx import filterlib, statelib
from flax.experimental.nx.pytreelib import Pytree
from flax.experimental.nx.variablelib import Variable

A = tp.TypeVar('A')


class RngState(Variable):
  tag: str


class RngCount(RngState): ...


class RngKey(RngState): ...


class RngStream(Pytree):
  __nodes__ = ('key', 'count')

  def __init__(
    self,
    tag: str,
    key: jax.Array | int,
    mutable: bool = True,
  ):
    if isinstance(key, int):
      key = jax.random.key(key)
    elif isinstance(key, jax.Array):
      if key.dtype == jnp.uint32:
        key = jax.random.wrap_key_data(key)
    else:
      raise ValueError(f'Invalid key: {key}')

    assert isinstance(key, jax.Array)
    count = jnp.zeros(key.shape, dtype=jnp.uint32)

    self.tag = tag
    self.key = RngKey(key, tag=tag, mutable=mutable)
    self.count = RngCount(count, tag=tag, mutable=mutable)

  def __call__(self) -> jax.Array:
    key = jax.random.fold_in(self.key[...], self.count[...])
    self.count[...] += 1
    return key

  def fork(self, mutable: bool = False):
    key = self()
    stream = type(self)(
      tag=self.tag,
      key=key,
      mutable=mutable,
    )
    return stream


class Rngs(Pytree):
  __nodes__ = ('streams',)

  def __init__(
    self,
    default: int | jax.Array | None = None,
    /,
    mutable: bool = True,
    **rngs: int | jax.Array,
  ):
    if default is not None:
      rngs['default'] = default

    self.streams: tp.Mapping[str, RngStream] = FrozenDict(
      {
        name: RngStream(tag=name, key=key, mutable=mutable)
        for name, key in rngs.items()
      }
    )

  def fork(self, mutable: bool = False):
    keys = {name: stream() for name, stream in self.streams.items()}
    return type(self)(mutable=mutable, **keys)

  def __getitem__(self, name: str):
    if name not in self.streams:
      if 'default' not in self.streams:
        raise KeyError(f"No RNG named {name!r} or 'default' found in Rngs.")
      stream = self.streams['default']
    else:
      stream = self.streams[name]

    return stream

  def __getattr__(self, name: str):
    if name not in self.streams:
      if 'default' not in self.streams:
        raise AttributeError(
          f"No RNG named {name!r} or 'default' found in Rngs."
        )
      stream = self.streams['default']
    else:
      stream = self.streams[name]

    return stream

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    return iter(self.streams.keys())

  def __len__(self) -> int:
    return len(self.streams) - 1

  def __contains__(self, name: tp.Any) -> bool:
    return name in self.streams

  def items(self):
    return self.streams.items()

  # pickle support
  def __getstate__(self):
    return vars(self).copy()

  def __setstate__(self, state):
    vars(self).update(state)


def split_rngs(
  tree: A, num_splits: int, /, *, only: filterlib.Filter = ...
) -> A:
  do_split = filterlib.to_predicate(only)

  def split_fn(path, x):
    path = statelib._normalize_path(path)
    if isinstance(x, RngStream) and do_split(path, x):
      x.count[...] += 1
      keys = x.key.copy(jax.random.split(x.key[...], num_splits))
      counts = x.count.copy(jnp.zeros(keys[...].shape, dtype=jnp.uint32))
      return x.replace(key=keys, count=counts)
    return x

  tree = jax.tree.map_with_path(
    split_fn, tree, is_leaf=lambda x: isinstance(x, RngStream)
  )
  return tree

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

from flax.nnx.nnx import graph
from flax.nnx.nnx.state import State
from flax.nnx.nnx.variables import Variable
from flax.nnx.nnx import filterlib
from flax.nnx.nnx.filterlib import All
from flax.nnx.nnx.object import Object

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
  """NNX rng container class. To instantiate the ``Rngs``, pass
  in an integer, specifying the starting seed. ``Rngs`` can have
  different "streams", allowing the user to generate different
  rng keys. For example, to generate a key for the ``params``
  and ``dropout`` stream::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> rng1 = nnx.Rngs(0, params=1)
    >>> rng2 = nnx.Rngs(0)

    >>> assert rng1.params() != rng2.dropout()

  Because we passed in ``params=1``, the starting seed for
  ``params`` is ``1``, whereas the starting seed for ``dropout``
  defaults to the ``0`` we passed in, since we didn't specify
  a seed for ``dropout``. If we didn't specify a seed for ``params``,
  then both streams will default to using the ``0`` we passed in::

    >>> rng1 = nnx.Rngs(0)
    >>> rng2 = nnx.Rngs(0)

    >>> assert rng1.params() == rng2.dropout()

  The ``Rngs`` container class contains a separate counter for
  each stream. Every time the stream is called to generate a new rng
  key, the counter increments by ``1``. To generate a new rng key,
  we fold in the counter value for the current rng stream into its
  corresponding starting seed. If we try to generate an rng key for
  a stream we did not specify on instantiation, then the ``default``
  stream is used (i.e. the first positional argument passed to ``Rngs``
  during instantiation is the ``default`` starting seed)::

    >>> rng1 = nnx.Rngs(100, params=42)
    >>> # `params` stream starting seed is 42, counter is 0
    >>> assert rng1.params() == jax.random.fold_in(jax.random.key(42), 0)
    >>> # `dropout` stream starting seed is defaulted to 100, counter is 0
    >>> assert rng1.dropout() == jax.random.fold_in(jax.random.key(100), 0)
    >>> # empty stream starting seed is defaulted to 100, counter is 1
    >>> assert rng1() == jax.random.fold_in(jax.random.key(100), 1)
    >>> # `params` stream starting seed is 42, counter is 1
    >>> assert rng1.params() == jax.random.fold_in(jax.random.key(42), 1)

  Let's see an example of using ``Rngs`` in a :class:`Module` and
  verifying the output by manually threading the ``Rngs``::

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     # Linear uses the `params` stream twice for kernel and bias
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     # Dropout uses the `dropout` stream once
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.dropout(self.linear(x))

    >>> def assert_same(x, rng_seed, **rng_kwargs):
    ...   model = Model(rngs=nnx.Rngs(rng_seed, **rng_kwargs))
    ...   out = model(x)
    ...
    ...   # manual forward propagation
    ...   rngs = nnx.Rngs(rng_seed, **rng_kwargs)
    ...   kernel = nnx.initializers.lecun_normal()(rngs.params(), (2, 3))
    ...   assert (model.linear.kernel.value==kernel).all()
    ...   bias = nnx.initializers.zeros_init()(rngs.params(), (3,))
    ...   assert (model.linear.bias.value==bias).all()
    ...   mask = jax.random.bernoulli(rngs.dropout(), p=0.5, shape=(1, 3))
    ...   # dropout scales the output proportional to the dropout rate
    ...   manual_out = mask * (jnp.dot(x, kernel) + bias) / 0.5
    ...   assert (out == manual_out).all()

    >>> x = jnp.ones((1, 2))
    >>> assert_same(x, 0)
    >>> assert_same(x, 0, params=1)
    >>> assert_same(x, 0, params=1, dropout=2)
  """
  def __init__(
    self,
    default: RngValue | RngDict | None = None,
    /,
    **rngs: RngValue,
  ):
    """
    Args:
      default: the starting seed for the ``default`` stream. Any
        key generated from a stream that isn't specified in the
        ``**rngs`` key-word arguments will default to using this
        starting seed.
      **rngs: optional key-word arguments to specify starting
        seeds for different rng streams. The key-word is the
        stream name and its value is the corresponding starting
        seed for that stream.
    """
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
    RngKey,
    RngCount,
  )

  def split_key(key: tp.Any) -> jax.Array:
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')

    return jax.random.split(key, num_splits)

  split_keys = jax.tree.map(split_key, split_keys)

  return ForkStates(split_keys, split_counts, broadcast_keys, broadcast_counts)


def backup_keys(node: tp.Any, /):
  backups: list[tuple[RngStream, jax.Array]] = []
  for _, stream in graph.iter_graph(node):
    if isinstance(stream, RngStream):
      backups.append((stream, stream.key.value))
  return backups


def restore_keys(backups: list[tuple[RngStream, jax.Array]], /):
  for stream, key in backups:
    stream.key.value = key
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
import jax.numpy as jnp

from flax.experimental.nnx.nnx import graph_utils
from flax.experimental.nnx.nnx.variables import Variable
from flax.experimental.nnx.nnx import filterlib
from flax.experimental.nnx.nnx.graph_utils import GraphNode
from flax.typing import Dtype, Shape

A = tp.TypeVar('A')
Counts = list[int]
AxesValue = tp.Union[int, None]
Pattern = tp.Union[AxesValue, tuple[AxesValue, ...]]


class Missing:
  pass


MISSING = Missing()


class RngState(Variable[jax.Array]):
  pass


class RngCount(RngState):
  pass


class RngKey(RngState):
  tag: str


@dataclasses.dataclass
class RngStream(GraphNode):
  def __init__(
    self,
    tag: str,
    key: jax.Array,
    count: jax.Array,
  ):
    self.key = RngKey(key, tag=tag)
    self.count = RngCount(count)

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

  def fork(self, pattern: Pattern) -> jax.Array:
    if pattern is None:
      # broadcast key
      key = self()
    else:
      if isinstance(pattern, int):
        num_splits = pattern
      else:
        num_splits = tuple(x if x is not None else 1 for x in pattern)
      key = jax.random.split(self.key.value, num_splits)
      self.count.value += 1
    return key


RngValue = tp.Union[int, jax.Array]
RngDict = tp.Union[
  tp.Mapping[str, int],
  tp.Mapping[str, jax.Array],
  tp.Mapping[str, RngValue],
]


class Rngs(GraphNode, tp.Mapping[str, tp.Callable[[], jax.Array]]):
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

  def _get_stream(self, name: str, error_type: Exception) -> RngStream:
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
      if name != '_graph_node__state':
        yield name

  def __len__(self) -> int:
    return len(vars(self)) - 1

  def __contains__(self, name: tp.Any) -> bool:
    return name in vars(self)

  def replace(self, **kwargs: tp.Union[int, jax.Array, RngStream]) -> 'Rngs':
    rngs: dict[str, tp.Any] = vars(self).copy()
    del rngs['_graph_node__state']
    rngs.update(kwargs)
    return Rngs(**rngs)


  def fork(
    self,
    _default: Pattern | dict[filterlib.Filter, Pattern] | Missing = MISSING,
    /,
    **patterns: Pattern,
  ) -> ForkedKeys:
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

    for name, stream in self.items():
      for predicate, pattern in predicate_pattern:
        stream_path = (name,)
        # here we check if the stream's RngKey tag matches the predicate
        # the stream_path is no longer needed, but we keep it for consistency
        if predicate(stream_path, stream.key):
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

@tp.runtime_checkable
class _HasRngInit(tp.Protocol):
  def rng_init(self, rngs: Rngs):
    ...


@tp.overload
def init(node: A, rngs: Rngs, /) -> A:
  ...


@tp.overload
def init(
  node: A,
  default: RngValue | RngDict | None = None,
  /,
  **rngs: RngValue,
) -> A:
  ...


def init(node: A, *args, **kwargs) -> A:
  if len(args) > 0 and isinstance(args[0], Rngs):
    if len(args) > 1:
      raise ValueError(
        'Too many positional arguments, expected at most 1 Rngs.'
      )
    if len(kwargs) > 0:
      raise ValueError(
        'Cannot use keyword arguments with Rngs positional argument.'
      )
    rngs = args[0]
  else:
    rngs = Rngs(*args, **kwargs)
  for _, value in graph_utils._iter_node_or_variable(node, set(), ()):
    if isinstance(value, _HasRngInit):
      value.rng_init(rngs)
  return node


def empty(shape: Shape, dtype: Dtype = jax.numpy.float32, /) -> jax.Array:
  return jax.ShapeDtypeStruct(shape, dtype)  # type: ignore

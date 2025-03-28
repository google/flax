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

import dataclasses
import hashlib
import typing as tp

from flax.core import FrozenDict
from flax.experimental.nx.filterlib import Filter, to_predicate
from flax.experimental.nx.variablelib import Variable
from flax.typing import PathParts
import jax
from jax._src import core as jax_core

A = tp.TypeVar('A')
B = tp.TypeVar('B')


# ------------------------------------------------------------------------------
# state
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TreeDef(tp.Generic[A]):
  pytreedef: jax.tree_util.PyTreeDef
  path_index: FrozenDict[PathParts, int]


def _key_path_to_key(x: tp.Any) -> tp.Any:
  if isinstance(x, jax.tree_util.SequenceKey):
    return x.idx
  elif isinstance(x, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
    return x.key
  elif isinstance(x, jax.tree_util.GetAttrKey):
    return x.name
  else:
    return str(x)


def _normalize_path(path: tuple[tp.Any, ...]) -> PathParts:
  return tuple(_key_path_to_key(part) for part in path)


def init(tree: A, key: int | jax.Array) -> jax.Array:
  """Initializes a pytree of Variables."""

  if isinstance(key, int):
    key = jax.random.key(key)

  def _init_fn(path, x):
    if isinstance(x, Variable) and isinstance(x.value, jax.ShapeDtypeStruct):
      if not hasattr(x, 'initializer'):
        raise ValueError(f'Variable at {path} has no initializer.')

      path_str = '/'.join(str(_key_path_to_key(part)) for part in path)
      path_bytes = hashlib.blake2s(path_str.encode('utf-8')).digest()
      hash_int = int.from_bytes(path_bytes[:4], byteorder='big')
      path_key = jax.random.fold_in(key, hash_int)
      concrete_array = x.initializer(path_key, x.value.shape, x.value.dtype)
      return x.replace(value=concrete_array)
    return x

  tree = jax.tree.map_with_path(
      _init_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  return tree


def is_non_mutable_array(path, x):
  return not isinstance(x.value, jax_core.MutableArray)


def mutable(
    tree: A,
    do_mutate: tp.Callable[[PathParts, tp.Any], bool] = is_non_mutable_array,
) -> A:
  def _mutable_fn(path, x):
    path = _normalize_path(path)
    if isinstance(x, Variable) and do_mutate(path, x):
      if not isinstance(x.value, jax.Array | jax_core.MutableArray):
        raise ValueError(
            f'Variable at {path} is not a jax.Array or jax_core.MutableArray,'
            f' got {type(x.value)}.'
        )
      return x.replace(value=jax_core.mutable_array(x.value))
    return x

  tree = jax.tree.map_with_path(
      _mutable_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  return tree


def is_mutable_array(path, x):
  return True


def freeze(
    tree: A,
    do_freeze: tp.Callable[[PathParts, tp.Any], bool] = is_mutable_array,
) -> A:
  def _freeze_fn(path, x):
    path = _normalize_path(path)
    if isinstance(x, Variable) and do_freeze(path, x):
      if not isinstance(x.value, jax_core.MutableArray | jax.Array):
        raise ValueError(
            f'Variable at {path} is not a jax.Array or jax_core.MutableArray,'
            f' got {type(x.value)}.'
        )
      return x.replace(value=x.value)
    return x


@tp.overload
def split(tree: A) -> tuple[TreeDef[A], dict]:
  ...


@tp.overload
def split(tree: A, first: Filter, /) -> tuple[TreeDef[A], dict]:
  ...


@tp.overload
def split(
    tree: A, first: Filter, second: Filter, /
) -> tuple[TreeDef[A], dict, dict]:
  ...


@tp.overload
def split(
    tree: A, first: Filter, second: Filter, third: Filter, /
) -> tuple[TreeDef[A], tp.Any, dict, dict]:
  ...


def split(tree, *filters: Filter) -> tp.Any:
  """Splits a pytree of Variables into a tree of Variables and a tree of Variables."""

  flat_state, pytreedef = jax.tree.flatten_with_path(
      tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  flat_state = [(_normalize_path(path), x) for path, x in flat_state]
  treedef = TreeDef(
      pytreedef, FrozenDict({path: i for i, (path, _) in enumerate(flat_state)})
  )

  if filters:
    flat_states = _split_state(flat_state, filters)
  else:
    flat_states = (flat_state,)

  flat_states = tuple(
      _unflatten_sequence(flat_state) for flat_state in flat_states
  )
  return treedef, *flat_states  # type: ignore


def _split_state(
    flat_state: tp.Iterable[tuple[PathParts, tp.Any]],
    filters: tp.Sequence[Filter],
) -> tuple[list[tuple[PathParts, tp.Any]], ...]:
  predicates = tuple(map(to_predicate, filters))

  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  flat_states: tuple[list[tuple[PathParts, tp.Any]], ...] = tuple(
      [] for _ in range(len(predicates))
  )

  for path, value in flat_state:
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i].append((path, value))  # type: ignore[index] # mypy is wrong here?
        break
    else:
      raise ValueError(f'No predicate matched {path}, {value}')

  return flat_states


def _unflatten_sequence(xs: tp.Iterable[tuple[PathParts, tp.Any]]) -> dict:
  result = {}
  for path, value in xs:
    cursor = result
    for key in path[:-1]:
      if key not in cursor:
        cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return result  # type: ignore


def merge(
    treedef: TreeDef[A],
    *states: dict,
) -> A:
  """Merges a tree of Variables with a tree of Variables."""

  flat_state: list[tuple[PathParts, tp.Any]] = []
  for state in states:
    flat_state.extend(_flatten_to_sequence(state))

  def key_fn(path_value: tuple[PathParts, tp.Any]):
    path, _ = path_value
    return treedef.path_index[path]

  flat_state.sort(key=key_fn)

  return treedef.pytreedef.unflatten(value for _, value in flat_state)


def _flatten_to_sequence(xs: dict) -> list[tuple[PathParts, tp.Any]]:
  result = []

  def _flatten(xs: Any, prefix: tuple[Any, ...]):
    if not isinstance(xs, dict):
      result.append((prefix, xs))
    else:
      for key, value in xs.items():
        _flatten(value, (*prefix, key))

  _flatten(xs, ())  # type: ignore
  return result


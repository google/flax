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
import typing as tp

import jax
from jax._src import core as jax_core
from jax._src.state.types import AbstractRef
from jax._src.tree_util import _registry as JAX_PYTREE_REGISTRY

from flax.core import FrozenDict
from flax.experimental.nx.filterlib import Filter, to_predicate
from flax.experimental.nx.variablelib import Variable

PathParts = tuple[str, ...]
A = tp.TypeVar('A')
B = tp.TypeVar('B')
State = dict[str, tp.Any]

# ------------------------------------------------------------------------------
# state
# ------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class VariableInfo:
  type: type[Variable]
  metadata: tp.Mapping[str, tp.Hashable]


@dataclasses.dataclass(frozen=True, slots=True)
class LeafInfo:
  index: int
  variable_info: VariableInfo | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class TreeDef(tp.Generic[A]):
  pytreedef: jax.tree_util.PyTreeDef = dataclasses.field(
    metadata=dict(static=True)
  )
  path_index: tp.Mapping[PathParts, LeafInfo] = dataclasses.field(
    metadata=dict(static=True)
  )


def _key_path_to_str(x: tp.Any) -> tp.Any:
  if isinstance(x, str):
    return x
  elif isinstance(x, jax.tree_util.SequenceKey):
    return str(x.idx)
  elif isinstance(x, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
    return str(x.key)
  elif isinstance(x, jax.tree_util.GetAttrKey):
    return str(x.name)
  else:
    return str(x)


def _normalize_path(path: tuple[tp.Any, ...]) -> tuple[str, ...]:
  return tuple(_key_path_to_str(part) for part in path)


# def init(tree: A, key: int | jax.Array) -> jax.Array:
#   """Initializes a pytree of Variables."""

#   if isinstance(key, int):
#     key = jax.random.key(key)

#   def _init_fn(path, x):
#     if isinstance(x, Variable) and isinstance(x.value, jax.ShapeDtypeStruct):
#       if not hasattr(x, 'initializer'):
#         raise ValueError(f'Variable at {path} has no initializer.')

#       path_str = '/'.join(str(_key_path_to_key(part)) for part in path)
#       path_bytes = hashlib.blake2s(path_str.encode('utf-8')).digest()
#       hash_int = int.from_bytes(path_bytes[:4], byteorder='big')
#       path_key = jax.random.fold_in(key, hash_int)
#       concrete_array = x.initializer(path_key, x.value.shape, x.value.dtype)
#       return x.replace(value=concrete_array)
#     return x

#   tree = jax.tree.map_with_path(
#     _init_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
#   )
#   return tree


def is_mutable_array(x):
  return isinstance(jax.typeof(x), AbstractRef | jax_core.MutableArray)


def _is_immutable(path, x):
  return isinstance(x, Variable) and not x.mutable or isinstance(x, jax.Array)


def mutable(
  tree: A,
  /,
  do_mutate: tp.Callable[[PathParts, Variable], bool] = _is_immutable,
) -> A:
  def _mutable_fn(path, x):
    if do_mutate(path, x):
      if isinstance(x, Variable) and not x.mutable:
        return x.copy(mutable=True)
      elif isinstance(x, jax.Array) and not is_mutable_array(x):
        return jax_core.mutable_array(x)
    return x

  tree = jax.tree.map_with_path(
    _mutable_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  return tree


def _is_mutable(path, x):
  return isinstance(x, Variable) and x.mutable or is_mutable_array(x)


def freeze(
  tree: A,
  /,
  do_freeze: tp.Callable[[str, Variable], bool] = _is_mutable,
) -> A:
  def _freeze_fn(path, x):
    if do_freeze(path, x):
      if isinstance(x, Variable) and x.mutable:
        return x.copy(mutable=False)
      elif is_mutable_array(x):
        return x[...]
    return x

  tree = jax.tree.map_with_path(
    _freeze_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  return tree


@tp.overload
def split(tree: A) -> tuple[TreeDef[A], State]: ...


@tp.overload
def split(tree: A, first: Filter, /) -> tuple[TreeDef[A], State]: ...


@tp.overload
def split(
  tree: A, first: Filter, second: Filter, /
) -> tuple[TreeDef[A], State, State]: ...


@tp.overload
def split(
  tree: A, first: Filter, second: Filter, third: Filter, /, *filters: Filter
) -> tuple[TreeDef[A], State, State, State, tp.Unpack[tuple[State, ...]]]: ...


def split(
  tree: A, *filters: Filter, keep_variables: bool = False
) -> tuple[TreeDef[A], State, tp.Unpack[tuple[State, ...]]]:
  """Splits a pytree of Variables into a tree of Variables and a tree of Variables."""
  if not filters:
    filters = (...,)

  flat_state, pytreedef = jax.tree.flatten_with_path(
    tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  flat_state = [(_normalize_path(path), x) for path, x in flat_state]
  treedef = TreeDef(
    pytreedef,
    FrozenDict(
      {
        path: LeafInfo(
          i,
          VariableInfo(type(x), x.metadata)
          if isinstance(x, Variable)
          else None,
        )
        for i, (path, x) in enumerate(flat_state)
      },
    ),
  )

  if filters:
    flat_states = _split_state(flat_state, filters)
  else:
    flat_states = (flat_state,)

  def remove_variable(x):
    if isinstance(x, Variable) and not keep_variables:
      return x.raw_value
    return x

  flat_states = tuple(
    _unflatten_sequence((path, remove_variable(x)) for path, x in flat_state)
    for flat_state in flat_states
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
  /,
  *states: State,
) -> A:
  """Merges a tree of Variables with a tree of Variables."""

  flat_state: list[tuple[PathParts, tp.Any]] = []
  for state in states:
    flat_state.extend(_flatten_to_sequence(state))

  def key_fn(path_value: tuple[PathParts, tp.Any]):
    path, _ = path_value
    return treedef.path_index[path].index

  flat_state.sort(key=key_fn)

  def make_variable(path: PathParts, value, /):
    variable_info = treedef.path_index[path].variable_info
    if not isinstance(value, Variable) and variable_info is not None:
      value = variable_info.type.from_array_metadata(
        value, variable_info.metadata
      )
    return value

  flat_state = [
    (path, make_variable(path, value)) for path, value in flat_state
  ]
  tree = treedef.pytreedef.unflatten(value for _, value in flat_state)
  return tree


def _flatten_to_sequence(xs: dict) -> list[tuple[PathParts, tp.Any]]:
  result = []

  def _flatten(xs: tp.Any, prefix: tuple[tp.Any, ...]):
    if not isinstance(xs, dict):
      result.append((prefix, xs))
    else:
      for key, value in xs.items():
        _flatten(value, (*prefix, key))

  _flatten(xs, ())  # type: ignore
  return result


def update(
  tree: A, /, state: State, *states: State, reuse_mutable: bool = True
) -> A:
  treedef, current_state = split(tree)
  states = (state, *states)

  for state in states:
    _update_recursive(current_state, state, reuse_mutable)

  tree = merge(treedef, current_state)
  return tree


def _update_recursive(current: State, updates: State, reuse_mutable: bool):
  for key, new_value in updates.items():
    if key not in current:
      raise ValueError(f'Path {key} not found in tree.')
    current_value = current[key]
    if isinstance(new_value, dict):
      if not isinstance(current_value, dict):
        raise ValueError(f'Expected dict at {key}, got {type(current_value)}.')
      _update_recursive(current_value, new_value, reuse_mutable)
    elif isinstance(current_value, dict):
      raise ValueError(f'Expected non-dict at {key}, got dict.')
    elif isinstance(current_value, Variable) and (
      isinstance(new_value, jax.Array)
    ):
      if reuse_mutable and current_value.mutable:
        current_value[...] = new_value[...]
      else:
        current[key] = current_value.copy(new_value[...])
    elif is_mutable_array(current_value) and isinstance(new_value, jax.Array):
      if reuse_mutable:
        current_value[...] = new_value[...]
      else:
        current[key] = jax_core.mutable_array(new_value[...])
    else:
      current[key] = new_value


@tp.overload
def state(tree, /) -> State: ...


@tp.overload
def state(tree, first: Filter, /) -> State: ...


@tp.overload
def state(tree, first: Filter, second: Filter, /) -> tuple[State, State]: ...


@tp.overload
def state(
  tree, first: Filter, second: Filter, third: Filter, /, *filters: Filter
) -> tuple[State, State, State, tp.Unpack[tuple[State, ...]]]: ...


def state(tree, /, *filters: Filter) -> State | tuple[State, ...]:
  if not filters:
    filters = (...,)

  _, *states, _ = split(tree, *filters, ...)

  if len(states) == 1:
    return states[0]
  else:
    return tuple(states)


def _is_pytree_node(x: tp.Any) -> bool:
  if type(x) in JAX_PYTREE_REGISTRY:
    return True
  elif isinstance(x, tuple):
    return True
  else:
    return False


def recursive_map(tree, fn: tp.Callable[[PathParts, tp.Any], tp.Any]):
  tree = _recursive_map((), tree, fn)
  return tree


def _recursive_map(
  path: PathParts,
  node: tp.Any,
  fn: tp.Callable[[PathParts, tp.Any], tp.Any],
):
  if _is_pytree_node(node):
    keys_leaves, treedef = jax.tree.flatten_with_path(
      node, is_leaf=lambda x: x is not node
    )
    leaves = []
    for (key,), leaf in keys_leaves:
      key = _key_path_to_str(key)
      leaf = _recursive_map((*path, key), leaf, fn)
      leaves.append(leaf)

    new_node = treedef.unflatten(leaves)
  else:
    new_node = node

  new_node = fn(path, new_node)
  return new_node


def _all_values(path: PathParts, x):
  return True


def pure(
  tree: A, /, do_remove: tp.Callable[[PathParts, tp.Any], bool] = _all_values
) -> A:
  def _remove_fn(path, x):
    path = _normalize_path(path)
    if (isinstance(x, Variable) or is_mutable_array(x)) and do_remove(path, x):
      return x[...]
    return x

  tree = jax.tree.map_with_path(
    _remove_fn, tree, is_leaf=lambda x: isinstance(x, Variable)
  )
  return tree
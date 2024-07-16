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

import abc
import contextlib
import dataclasses
import threading
import typing as tp

import jax
from jax._src.tree_util import broadcast_prefix

from flax import struct
from flax.nnx.nnx.object import Object
from flax.nnx.nnx.state import State
from flax.typing import PathParts
from flax.nnx.nnx import graph


class Missing:
  pass


MISSING = Missing()
A = tp.TypeVar('A')
E = tp.TypeVar('E', bound='Extractable')
Index = int
KeyEntry = tp.TypeVar('KeyEntry', bound=tp.Hashable)
KeyPath = tuple[KeyEntry, ...]
Prefix = tp.Any
Leaf = tp.Any


class Extractable(abc.ABC):
  @property
  @abc.abstractmethod
  def index(self) -> Index: ...


class ExtractableStates(Extractable):
  @property
  @abc.abstractmethod
  def states(self) -> tp.Iterable[State]: ...

  @property
  @abc.abstractmethod
  def graphdef(self) -> graph.GraphDef[tp.Any]: ...


class ExtractionIndex(struct.PyTreeNode, Extractable):
  """Index of a graph node in a Pytree structure."""

  _index: Index = struct.field(pytree_node=False)

  @property
  def index(self) -> Index:
    return self._index


@tp.overload
def extract_graph_nodes(
  pytree: A,
  /,
  *,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> tuple[A, tuple[tp.Any, ...]]: ...


@tp.overload
def extract_graph_nodes(
  pytree: A,
  /,
  *,
  prefix: tp.Any,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> tuple[A, tuple[tp.Any, ...], tuple[tp.Any, ...]]: ...


def extract_graph_nodes(
  pytree: A,
  /,
  *,
  prefix: tp.Any = MISSING,
  validate_fn: tp.Callable[[KeyPath, Prefix, Leaf], None] | None = None,
) -> (
  tuple[A, tuple[tp.Any, ...]]
  | tuple[A, tuple[tp.Any, ...], tuple[tp.Any, ...]]
):
  """Extracts all graph nodes from a pytree."""
  nodes = graph.RefMap[tp.Any, Index]()
  node_prefixes = []
  leaves = []

  prefix_leaves = broadcast_prefix(
    prefix,
    pytree,
    is_leaf=lambda x: x is None,
  )
  key_leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)

  assert len(key_leaves) == len(prefix_leaves)

  for (keypath, leaf), prefix_leaf in zip(key_leaves, prefix_leaves):
    if validate_fn:
      validate_fn(keypath, prefix_leaf, leaf)
    if graph.is_graph_node(leaf):
      if leaf not in nodes:
        index = nodes[leaf] = len(nodes)
        node_prefixes.append(prefix_leaf)
      else:
        index = nodes[leaf]
        # check consistent aliasing
        if prefix_leaf != node_prefixes[index]:
          path_str = jax.tree_util.keystr(keypath)
          raise ValueError(
            f'Inconsistent aliasing detected. Node {type(leaf)} at path {path_str} '
            f'has different prefixes: {prefix_leaf} and {node_prefixes[index]}.'
          )
      leaves.append(ExtractionIndex(index))
    else:
      leaves.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves)

  if prefix is MISSING:
    return pytree_out, tuple(nodes)  # type: ignore[bad-return-type]
  else:
    return pytree_out, tuple(nodes), tuple(node_prefixes)  # type: ignore[bad-return-type]


def insert_graph_nodes(pytree: A, nodes: tuple[tp.Any, ...], /) -> A:
  """Inserts graph nodes into a pytree."""

  def _maybe_insert(x):
    if isinstance(x, Extractable):
      return nodes[x.index]
    return x

  return jax.tree_util.tree_map(
    _maybe_insert, pytree, is_leaf=lambda x: isinstance(x, Extractable)
  )


def extract_indexes(
  pytree,
  /,
  types: tuple[type[E], ...] | type[E] = Extractable,  # type: ignore[assignment]
) -> tuple[E, ...]:
  """Extracts all indexes from a pytree."""
  indexes: list[E] = []
  for x in jax.tree.leaves(
    pytree, is_leaf=lambda x: isinstance(x, Extractable)
  ):
    if isinstance(x, Extractable):
      if not isinstance(x, types):
        raise ValueError(f'Expected Extractable of type {types}, got {type(x)}')
      indexes.append(x)  # type: ignore[arg-type]
  return tuple(indexes)


def replace_indexes(
  pytree: A,
  replace_fn: tp.Callable[[Extractable], tp.Any],
  /,
  clear: bool = False,
) -> A:
  def _replace_map_fn(x):
    if isinstance(x, Extractable):
      return replace_fn(x)
    elif clear:
      return None
    return x

  return jax.tree_util.tree_map(
    _replace_map_fn, pytree, is_leaf=lambda x: isinstance(x, Extractable)
  )


def merge_extractable_states(
  extractable_states: tp.Sequence[ExtractableStates], /
):
  if len(extractable_states) == 0:
    raise ValueError('Expected at least one ExtractableStates object')

  graphdef = extractable_states[0].graphdef
  flat_state: list[tuple[PathParts, tp.Any]] = []

  for extractable_state in extractable_states:
    flat_state.extend(
      ((extractable_state.index, *path), value)
      for state in extractable_state.states
      for path, value in state.flat_state().items()
    )

  state = State.from_flat_path(flat_state)
  return graphdef, state


def check_consistent_aliasing(
  nodes: tuple[tp.Any, ...], prefixes: tuple[tp.Any, ...]
):
  node_prefixes = graph.RefMap[tp.Any, list[tuple[PathParts, tp.Any]]]()

  # collect all paths and prefixes for each node
  for node, prefix in zip(nodes, prefixes):
    for path, value in graph.iter_graph(node):
      if graph.is_graph_node(value):
        if isinstance(value, Object):
          value.check_valid_context(
            f'Trying to extract graph node from different trace level, got {value!r}'
          )
        if value in node_prefixes:
          paths_prefixes = node_prefixes[value]
          paths_prefixes.append((path, prefix))
        else:
          node_prefixes[value] = [(path, prefix)]

  # check for inconsistent aliasing
  node_msgs = []
  for node, paths_prefixes in node_prefixes.items():
    unique_prefixes = {prefix for _, prefix in paths_prefixes}
    if len(unique_prefixes) > 1:
      path_prefix_repr = '\n'.join(
        f'  {"/".join(map(str,path)) if path else "<root>"}: {prefix}'
        for path, prefix in paths_prefixes
      )
      nodes_msg = f'Node: {type(node)}\n{path_prefix_repr}'
      node_msgs.append(nodes_msg)

  if node_msgs:
    raise ValueError(
      'Inconsistent aliasing detected. The following nodes have different prefixes:\n'
      + '\n'.join(node_msgs)
    )

# -----------------------------
# broadcast
# -----------------------------


@dataclasses.dataclass
class BroadcastContext(threading.local):
  broadcast_state_stacks: dict[str, list[tp.Any]] = dataclasses.field(
    default_factory=dict
  )


BROADCAST_CONTEXT = BroadcastContext()


@contextlib.contextmanager
def broadcast_state(tag: str, state: tp.Any):
  if tag in BROADCAST_CONTEXT.broadcast_state_stacks:
    stack = BROADCAST_CONTEXT.broadcast_state_stacks[tag]
  else:
    stack = BROADCAST_CONTEXT.broadcast_state_stacks[tag] = []
  stack.append(state)
  try:
    yield
  finally:
    stack.pop()
    if not stack:
      del BROADCAST_CONTEXT.broadcast_state_stacks[tag]


def get_broadcast_state(tag: str) -> tp.Any:
  if tag not in BROADCAST_CONTEXT.broadcast_state_stacks:
    raise ValueError(f'No broadcast state found for {tag!r}')

  stack = BROADCAST_CONTEXT.broadcast_state_stacks[tag]

  if not stack:
    raise RuntimeError(
      f'Empty broadcast state stack for {tag!r}, this is a bug'
    )

  return stack[-1]

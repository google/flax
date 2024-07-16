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

import contextlib
import dataclasses
import threading
import typing as tp

import jax
# from jax._src.tree_util import broadcast_prefix

from flax import struct
from flax.nnx.nnx.object import Object
from flax.typing import MISSING, PathParts
from flax.nnx.nnx import graph


A = tp.TypeVar('A')
Index = int
KeyEntry = tp.TypeVar('KeyEntry', bound=tp.Hashable)
KeyPath = tuple[KeyEntry, ...]
Prefix = tp.Any
Leaf = tp.Any


class ExtractionIndex(struct.PyTreeNode):
  """Index of a graph node in a Pytree structure."""

  index: Index = struct.field(pytree_node=False)


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
    prefix_is_leaf=lambda x: x is None,
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
    if isinstance(x, ExtractionIndex):
      return nodes[x.index]
    return x

  return jax.tree.map(
    _maybe_insert, pytree, is_leaf=lambda x: isinstance(x, ExtractionIndex)
  )


def check_consistent_aliasing(
  node: tuple[tp.Any, ...],
  prefix: tuple[tp.Any, ...],
  /,
  *,
  node_prefixes: graph.RefMap[tp.Any, list[tuple[PathParts, tp.Any]]]
  | None = None,
):
  if node_prefixes is None:
    node_prefixes = graph.RefMap()

  # collect all paths and prefixes for each node
  for path, value in graph.iter_graph(node):
    if graph.is_graph_node(value) or isinstance(value, graph.Variable):
      if isinstance(value, Object):
        value.check_valid_context(
          lambda: f'Trying to extract graph node from different trace level, got {value!r}'
        )
      if isinstance(value, graph.Variable):
        if not value._trace_state.is_valid():
          raise ValueError(
            f'Cannot extract graph node from different trace level, got {value!r}'
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

# -----------------------------
# to_tree/from_tree
# -----------------------------

def broadcast_prefix(
  prefix_tree: tp.Any,
  full_tree: tp.Any,
  prefix_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
  tree_is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> list[tp.Any]:
  # If prefix_tree is not a tree prefix of full_tree, this code can raise a
  # ValueError; use prefix_errors to find disagreements and raise more precise
  # error messages.
  result = []
  num_leaves = lambda t: jax.tree_util.tree_structure(
    t, is_leaf=tree_is_leaf
  ).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=prefix_is_leaf)
  return result


class GraphDefState(struct.PyTreeNode):
  graphdef: graph.GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: graph.GraphState = struct.field(pytree_node=True)

class StateOnly(struct.PyTreeNode):
  state: graph.GraphState = struct.field(pytree_node=True)

  @property
  def graphdef(self) -> graph.GraphDef[tp.Any]:
    raise ValueError('No graphdef available in StateOnly')


@dataclasses.dataclass(frozen=True)
class StateSequence(tp.Sequence[graph.GraphState]):
  graphdef_states: tuple[GraphDefState | StateOnly, ...]

  @tp.overload
  def __getitem__(self, index: int) -> graph.GraphState: ...
  @tp.overload
  def __getitem__(self, index: slice) -> 'StateSequence': ...
  def __getitem__(self, index):
    if isinstance(index, slice):
      return StateSequence(self.graphdef_states[index])
    elif isinstance(index, int):
      return self.graphdef_states[index].state
    else:
      raise TypeError(f'Invalid index type: {type(index)}')

  def __len__(self):
    return len(self.graphdef_states)

  def __iter__(self):
    return (s.state for s in self.graphdef_states)


class TreeNode(struct.PyTreeNode):
  metatata: tp.Any = struct.field(pytree_node=False)
  graphdef_states: tuple[GraphDefState | StateOnly, ...] = struct.field(
    pytree_node=True
  )

  @property
  def graphdef(self) -> graph.GraphDef[tp.Any]:
    return self.graphdef_states[0].graphdef

  @property
  def state(self) -> graph.GraphState:
    if len(self.graphdef_states) != 1:
      raise ValueError(
        f'Expected exactly one GraphDefState, got {len(self.graphdef_states)}'
      )
    return self.graphdef_states[0].state

  @property
  def states(self) -> tp.Sequence[graph.GraphState]:
    return StateSequence(self.graphdef_states)

  @classmethod
  def from_split(
    cls,
    graphdef: graph.GraphDef[tp.Any],
    state: graph.GraphState,
    /,
    *states: graph.GraphState,
    metadata: tp.Any = None,
  ):
    states = (state, *states)
    return cls(
      metadata, tuple(GraphDefState(graphdef, state) for state in states)
    )

  @classmethod
  def from_states(cls, state: graph.GraphState, *states: graph.GraphState):
    states = (state, *states)
    return cls(None, tuple(StateOnly(state) for state in states))

  @classmethod
  def from_prefixes(
    cls,
    prefixes: tp.Iterable[tp.Any],
    /,
    *,
    metadata: tp.Any = None,
  ):
    return cls(metadata, tuple(prefixes))


def default_split_fn(
  ctx: graph.SplitContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  return TreeNode.from_split(*ctx.split(leaf))


def to_tree(
  tree,
  /,
  *,
  prefix: tp.Any = MISSING,
  split_fn: tp.Callable[
    [graph.SplitContext, KeyPath, Prefix, Leaf], tp.Any
  ] = default_split_fn,
  map_non_graph_nodes: bool = False,
  ctxtag: str | None = None,
) -> tp.Any:
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None,
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(tree)

  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []
  node_prefixes = graph.RefMap[tp.Any, list[tuple[PathParts, tp.Any]]]()

  with graph.split_context(ctxtag) as split_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if graph.is_graph_node(leaf):
        check_consistent_aliasing(
          leaf, leaf_prefix, node_prefixes=node_prefixes
        )
        tree_node = split_fn(split_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(tree_node)
      else:
        if map_non_graph_nodes:
          leaf = split_fn(split_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves_out)
  return pytree_out


def merge_tree_node(
  ctx: graph.MergeContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  if not isinstance(leaf, TreeNode):
    raise ValueError(f'Expected TreeNode, got {type(leaf)} at path {path}')
  return ctx.merge(leaf.graphdef, *leaf.states)


def is_tree_node(x):
  return isinstance(x, TreeNode)


def from_tree(
  tree: tp.Any,
  /,
  *,
  prefix: tp.Any = MISSING,
  merge_fn: tp.Callable[
    [graph.MergeContext, KeyPath, Prefix, Leaf], tp.Any
  ] = merge_tree_node,
  is_node_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  is_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  map_non_graph_nodes: bool = False,
  ctxtag: str | None = None,
) -> tp.Any:
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None or is_leaf(x),
    tree_is_leaf=is_leaf,
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(
    tree, is_leaf=is_leaf
  )
  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []

  with graph.merge_context(ctxtag) as merge_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if is_node_leaf(leaf):
        leaf_out = merge_fn(merge_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(leaf_out)
      else:
        if map_non_graph_nodes:
          leaf = merge_fn(merge_ctx, keypath, leaf_prefix, leaf)
        leaves_out.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves_out)
  return pytree_out

def clear_non_graph_nodes(tree):
  return jax.tree.map(lambda x: x if graph.is_graph_node(x) else None, tree)
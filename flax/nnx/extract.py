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
import typing as tp

import jax

from flax import struct
from flax.nnx.object import Object
from flax.typing import Missing, PathParts
from flax.nnx import graph, variablelib


A = tp.TypeVar('A')
Index = int
KeyEntry = tp.TypeVar('KeyEntry', bound=tp.Hashable)
KeyPath = tuple[KeyEntry, ...]
Prefix = tp.Any
Leaf = tp.Any


class PrefixMapping(abc.ABC):
  @abc.abstractmethod
  def map_prefix(
    self,
    path: variablelib.PathParts,
    variable: variablelib.Variable,
    /,
  ) -> tp.Any: ...

def check_consistent_aliasing(
  node: tp.Any,
  prefix: tp.Any,
  /,
  *,
  node_prefixes: dict[tp.Any, list[tuple[PathParts, tp.Any]]] | None = None,
):
  if node_prefixes is None:
    node_prefixes = {}

  # collect all paths and prefixes for each node
  for path, value in graph.iter_graph(node):
    if graph.is_graph_node(value) or isinstance(value, graph.Variable):
      if isinstance(value, Object):
        value._check_valid_context(
          lambda: f'Trying to extract graph node from different trace level, got {value!r}'
        )
      if isinstance(value, graph.Variable):
        if not value._trace_state.is_valid():
          raise ValueError(
            f'Cannot extract graph node from different trace level, got {value!r}'
          )
        if isinstance(prefix, PrefixMapping):
          variable_prefix = prefix.map_prefix(path, value)
        else:
          variable_prefix = prefix

        if value in node_prefixes:
          paths_prefixes = node_prefixes[value]
          paths_prefixes.append((path, variable_prefix))
        else:
          node_prefixes[value] = [(path, variable_prefix)]

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
  jax.tree.map(
    add_leaves,
    prefix_tree,
    full_tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graph.is_graph_node(x)
    or (prefix_is_leaf is not None and prefix_is_leaf(x)),
  )
  return result


class GraphDefState(struct.PyTreeNode):
  graphdef: graph.GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: graph.GraphState = struct.field(pytree_node=True)

S = tp.TypeVar(
  'S', bound=graph.GraphState | graph.GraphFlatState | list[tp.Any]
)

class NodeStates(struct.PyTreeNode):
  _graphdef: graph.GraphDef[tp.Any] | None
  states: tuple[tp.Any, ...]
  metadata: tp.Any = struct.field(pytree_node=False)

  @property
  def graphdef(self) -> graph.GraphDef[tp.Any]:
    if self._graphdef is None:
      raise ValueError('No graphdef available')
    return self._graphdef

  @property
  def state(self) -> tp.Any:
    if len(self.states) != 1:
      raise ValueError(
        f'Expected exactly one GraphDefState, got {len(self.states)}'
      )
    return self.states[0]

  @classmethod
  def from_split(
    cls,
    graphdef: graph.GraphDef[tp.Any],
    state: tp.Any,
    /,
    *states: tp.Any,
    metadata: tp.Any = None,
  ):
    return cls(_graphdef=graphdef, states=(state, *states), metadata=metadata)

  @classmethod
  def from_states(
    cls,
    state: tp.Any,
    *states: tp.Any,
  ):
    return cls(_graphdef=None, states=(state, *states), metadata=None)

  @classmethod
  def from_prefixes(
    cls,
    prefixes: tp.Iterable[tp.Any],
    /,
    *,
    metadata: tp.Any = None,
  ):
    return cls(_graphdef=None, states=tuple(prefixes), metadata=metadata)


def default_split_fn(
  ctx: graph.SplitContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  return NodeStates.from_split(*ctx.split(leaf))


def to_tree(
  tree,
  /,
  *,
  prefix: tp.Any = Missing,
  split_fn: tp.Callable[
    [graph.SplitContext, KeyPath, Prefix, Leaf], tp.Any
  ] = default_split_fn,
  map_non_graph_nodes: bool = False,
  ctxtag: tp.Hashable | None = None,
  check_aliasing: bool = True,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graph.split_context(ctxtag) as split_ctx:
      return jax.tree.map(
        lambda x: split_fn(split_ctx, (), prefix, x)
        if map_non_graph_nodes
        or graph.is_graph_node(x)
        or isinstance(x, variablelib.Variable)
        else x,
        tree,
        is_leaf=lambda x: isinstance(x, variablelib.Variable)
        or graph.is_graph_node(x),
      )
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None
    or isinstance(x, variablelib.Variable)
    or graph.is_graph_node(x),
    tree_is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graph.is_graph_node(x),
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graph.is_graph_node(x),
  )

  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []
  node_prefixes: dict[tp.Any, list[tuple[PathParts, tp.Any]]] = {}

  with graph.split_context(ctxtag) as split_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if graph.is_graph_node(leaf) or isinstance(leaf, variablelib.Variable):
        if check_aliasing:
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
  if not isinstance(leaf, NodeStates):
    raise ValueError(f'Expected TreeNode, got {type(leaf)} at path {path}')
  return ctx.merge(leaf.graphdef, *leaf.states)


def is_tree_node(x):
  return isinstance(x, NodeStates)


def from_tree(
  tree: tp.Any,
  /,
  *,
  prefix: tp.Any = Missing,
  merge_fn: tp.Callable[
    [graph.MergeContext, KeyPath, Prefix, Leaf], tp.Any
  ] = merge_tree_node,
  is_node_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  is_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  map_non_graph_nodes: bool = False,
  is_inner: bool | None = None,
  ctxtag: tp.Hashable | None = None,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graph.merge_context(ctxtag, is_inner) as merge_ctx:

      def maybe_split(x):
        if (
          map_non_graph_nodes
          or is_node_leaf(x)
          or isinstance(x, variablelib.Variable)
        ):
          return merge_fn(merge_ctx, (), prefix, x)
        return x

      return jax.tree.map(maybe_split, tree, is_leaf=is_leaf)
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

  with graph.merge_context(ctxtag, is_inner) as merge_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if (
        map_non_graph_nodes
        or is_node_leaf(leaf)
        or isinstance(leaf, variablelib.Variable)
      ):
        leaf = merge_fn(merge_ctx, keypath, leaf_prefix, leaf)
      leaves_out.append(leaf)

  pytree_out = jax.tree.unflatten(treedef, leaves_out)
  return pytree_out

def clear_non_graph_nodes(tree):
  return jax.tree.map(
    lambda x: x
    if graph.is_graph_node(x) or isinstance(x, variablelib.Variable)
    else None,
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graph.is_graph_node(x),
  )
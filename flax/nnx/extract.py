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
import functools
import typing as tp
from collections import namedtuple

import jax

from flax import struct
from flax import typing
from flax.nnx.pytreelib import Pytree
from flax.typing import Missing, PathParts
from flax.nnx import graphlib, variablelib


A = tp.TypeVar('A')
Index = int
KeyPath = tuple[tp.Hashable, ...]
Prefix = tp.Any
Leaf = tp.Any


class PrefixMapping(abc.ABC):
  @abc.abstractmethod
  def map_prefix(
    self,
    path: typing.PathParts,
    variable: variablelib.Variable,
    /,
  ) -> tp.Any: ...

def check_consistent_aliasing(
  node: tp.Any,
  prefix: tp.Any,
  /,
  *,
  node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]] | None = None,
):
  """Check for consistent aliasing of nodes when extracting graph."""
  if node_prefixes is None:
    node_prefixes = {}

  # Store variable references for error messages
  node_id_to_variable: dict[int, tp.Any] = {}

  # collect all paths and prefixes for each node
  for path, value in graphlib.iter_graph(node, graph=True):
    if graphlib.is_graph_node(value) or isinstance(value, graphlib.Variable):
      if isinstance(value, Pytree):
        value._check_valid_context(
          lambda: f'Trying to extract graph node from different trace level, got {value!r}'
        )
      if isinstance(value, graphlib.Variable):
        if not value._can_update:
          raise ValueError(
            f'Cannot extract graph node from different trace level, got {value!r}'
          )
        if isinstance(prefix, PrefixMapping):
          variable_prefix = prefix.map_prefix(path, value)
        else:
          variable_prefix = prefix

        value_id = id(value)
        node_id_to_variable[value_id] = value
        if value_id in node_prefixes:
          paths_prefixes = node_prefixes[value_id]
          paths_prefixes.append((path, variable_prefix))
        else:
          node_prefixes[value_id] = [(path, variable_prefix)]

  # check for inconsistent aliasing
  node_msgs = []
  for node_id, paths_prefixes in node_prefixes.items():
    unique_prefixes = {prefix for _, prefix in paths_prefixes}
    if len(unique_prefixes) > 1:
      path_prefix_repr = '\n'.join(
        f'  {"/".join(map(str,path)) if path else "<root>"}: {prefix}'
        for path, prefix in paths_prefixes
      )
      # Get the variable type name if available
      if node_id in node_id_to_variable:
        variable = node_id_to_variable[node_id]
        node_type_name = type(variable).__name__
      else:
        node_type_name = f'Node ID: {node_id}'

      nodes_msg = f'Node: {node_type_name}\n{path_prefix_repr}'
      node_msgs.append(nodes_msg)

  if node_msgs:
    raise ValueError(
      'Inconsistent aliasing detected. The following nodes have different prefixes:\n'
      + '\n'.join(node_msgs)
    )


def check_consistent_aliasing2(
  node: tp.Any,
  prefix: tp.Any,
  /,
  *,
  base_path: tuple[tp.Any, ...] = (),
  node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]] | None = None,
):
  if node_prefixes is None:
    node_prefixes = {}

  node_id_to_variable: dict[int, tp.Any] = {}

  for path, value in graphlib.iter_graph(node, graph=True):
    path = base_path + path
    if graphlib.is_graph_node(value) or isinstance(value, graphlib.Variable):
      value_id = id(value)
      node_id_to_variable[value_id] = value
      if value_id in node_prefixes:
        node_prefixes[value_id].append((path, prefix))
      else:
        node_prefixes[value_id] = [(path, prefix)]

  node_msgs = []
  for node_id, paths_prefixes in node_prefixes.items():
    unique_prefixes = {p for _, p in paths_prefixes}
    if len(unique_prefixes) > 1:
      path_prefix_repr = '\n'.join(
        f'  {"/".join(map(str,path)) if path else "<root>"}: {p}'
        for path, p in paths_prefixes
      )
      if node_id in node_id_to_variable:
        variable = node_id_to_variable[node_id]
        node_type_name = type(variable).__name__
      else:
        node_type_name = f'Node ID: {node_id}'

      node_msgs.append(f'Node: {node_type_name}\n{path_prefix_repr}')

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
    or graphlib.is_graph_node(x)
    or (prefix_is_leaf is not None and prefix_is_leaf(x)),
  )
  return result

def broadcast_prefix2(
  prefix_tree: tp.Any,
  full_tree: tp.Any,
  is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tuple[list[KeyPath], list[tp.Any]]:
  paths: list[KeyPath] = []
  leaves: list[tp.Any] = []
  num_leaves = lambda t: jax.tree_util.tree_structure(t).num_leaves
  def add_leaves(path, x, subtree):
    n = num_leaves(subtree)
    paths.extend([path] * n)
    leaves.extend([x] * n)
  jax.tree.map_with_path(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
  return paths, leaves

def broadcast_prefix_map(
  f: tp.Callable[..., tp.Any],
  prefix_tree: tp.Any,
  full_tree: tp.Any,
  *rest: tp.Any,
  is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tp.Any:
  paths, prefix_leaves = broadcast_prefix2(prefix_tree, full_tree, is_leaf=is_leaf)
  leaves, treedef = jax.tree_util.tree_flatten(full_tree, is_leaf=is_leaf)
  full_prefix_tree = treedef.unflatten(prefix_leaves)
  return jax.tree.map_with_path(f, full_prefix_tree, full_tree, *rest, is_leaf=is_leaf)


class GraphDefState(struct.PyTreeNode):
  graphdef: graphlib.GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: graphlib.GraphState = struct.field(pytree_node=True)

S = tp.TypeVar(
  'S', bound=graphlib.GraphState | graphlib.GraphFlatState | list[tp.Any]
)

class NodeStates(struct.PyTreeNode):
  _graphdef: graphlib.GraphDef[tp.Any] | None
  states: tuple[tp.Any, ...]
  metadata: tp.Any = struct.field(pytree_node=False)

  @property
  def graphdef(self) -> graphlib.GraphDef[tp.Any]:
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
    graphdef: graphlib.GraphDef[tp.Any] | None,
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
  ctx: graphlib.SplitContext, path: KeyPath, prefix: Prefix, leaf: Leaf
) -> tp.Any:
  return NodeStates.from_split(*ctx.split(leaf))


def to_tree(
  tree,
  /,
  *,
  prefix: tp.Any = Missing,
  split_fn: tp.Callable[
    [graphlib.SplitContext, KeyPath, Prefix, Leaf], tp.Any
  ] = default_split_fn,
  map_non_graph_nodes: bool = False,
  ctxtag: tp.Hashable | None = None,
  check_aliasing: bool = True,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graphlib.split_context(ctxtag) as split_ctx:
      return jax.tree.map(
        lambda x: split_fn(split_ctx, (), prefix, x)
        if map_non_graph_nodes
        or graphlib.is_graph_node(x)
        or isinstance(x, variablelib.Variable)
        else x,
        tree,
        is_leaf=lambda x: isinstance(x, variablelib.Variable)
        or graphlib.is_graph_node(x),
      )
  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None
    or isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x),
    tree_is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x),
  )
  leaf_keys, treedef = jax.tree_util.tree_flatten_with_path(
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x),
  )

  assert len(leaf_keys) == len(leaf_prefixes)
  leaves_out = []
  node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]] = {}

  with graphlib.split_context(ctxtag) as split_ctx:
    for (keypath, leaf), leaf_prefix in zip(leaf_keys, leaf_prefixes):
      if graphlib.is_graph_node(leaf) or isinstance(leaf, variablelib.Variable):
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


def to_tree2(
  tree,
  /,
  *,
  prefix: tp.Any = Missing,
  check_aliasing: bool = True,
) -> tp.Any:
  """to_tree2 has two main tasks:

  1. Convert all graph nodes to NodeStates (a tree representation).
  2. Check all Variables are aliased consistently given the prefix tree,
    e.g. vmap's in/out_axes arguments.

  Each NodeState contains the `GraphDef` and State for each object, these
  are generated using `graphlib.flatten`. `extract.broadcast_prefix` is used
  to calculate the prefix for each node, `check_consistent_aliasing2` traverses
  the nodes subgraph and checks for Variable aliasing.
  """
  ref_index: graphlib.RefMap = graphlib.RefMap()

  def _to_node_states(leaf):
    if not (graphlib.is_graph_node(leaf) or isinstance(leaf, variablelib.Variable)):
      return leaf
    graphdef, flat_state = graphlib.flatten(
      leaf, ref_index=ref_index, graph=True
    )
    (state,) = graphlib._to_nested_state(graphdef, (flat_state,))
    return NodeStates.from_split(graphdef, state)

  is_leaf = lambda x: (
    isinstance(x, variablelib.Variable) or graphlib.is_graph_node(x)
  )

  if prefix is Missing or prefix is None:
    return jax.tree.map(_to_node_states, tree, is_leaf=is_leaf)

  leaf_prefixes = broadcast_prefix(
    prefix,
    tree,
    prefix_is_leaf=lambda x: x is None or is_leaf(x),
    tree_is_leaf=is_leaf,
  )
  leaf_paths, treedef = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)

  assert len(leaf_paths) == len(leaf_prefixes)
  leaves_out = []
  node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]] = {}

  for (keypath, leaf), leaf_prefix in zip(leaf_paths, leaf_prefixes):
    if is_leaf(leaf):
      if check_aliasing:
        base_path = graphlib.jax_to_nnx_path(keypath)
        check_consistent_aliasing2(
          leaf, leaf_prefix, base_path=base_path, node_prefixes=node_prefixes
        )
      leaves_out.append(_to_node_states(leaf))
    else:
      leaves_out.append(leaf)

  return jax.tree.unflatten(treedef, leaves_out)


def from_tree2(tree: tp.Any, /) -> tp.Any:
  index_ref = graphlib.IndexMap()

  def _from_node_states(x):
    if not isinstance(x, NodeStates):
      return x
    state = graphlib._merge_to_flat_state(x.states)
    return graphlib.unflatten(
      x.graphdef, state, index_ref=index_ref,
    )

  return jax.tree.map(
    _from_node_states,
    tree,
    is_leaf=lambda x: (
      isinstance(x, NodeStates)
      or graphlib.is_graph_node(x)
      or isinstance(x, variablelib.Variable)
    ),
  )


def merge_tree_node(
  ctx: graphlib.MergeContext, path: KeyPath, prefix: Prefix, leaf: Leaf
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
    [graphlib.MergeContext, KeyPath, Prefix, Leaf], tp.Any
  ] = merge_tree_node,
  is_node_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  is_leaf: tp.Callable[[Leaf], bool] = is_tree_node,
  map_non_graph_nodes: bool = False,
  is_inner: bool | None = None,
  ctxtag: tp.Hashable | None = None,
) -> tp.Any:
  if prefix is Missing or prefix is None:
    # fast path, no need for prefix broadcasting or consistent aliasing checks
    with graphlib.merge_context(ctxtag, is_inner) as merge_ctx:

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

  with graphlib.merge_context(ctxtag, is_inner) as merge_ctx:
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
    if graphlib.is_graph_node(x) or isinstance(x, variablelib.Variable)
    else None,
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x),
  )

class Mask(tp.NamedTuple):
  pass

def mask_at(t: tuple, index: int | None) -> tuple:
  if index is None:
    return t
  return tuple(
    Mask() if i == index else x
    for i, x in enumerate(t)
  )

def replace_at(t: tuple, index: int, value: tp.Any) -> tuple:
  return tuple(
    value if i == index else x
    for i, x in enumerate(t)
  )

def updates_and_snapshot(args: A) -> tuple[A, A]:
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  leaves, treedef = jax.tree.flatten(args, is_leaf=is_leaf)
  updates_leaves: list[variablelib.Variable | Mask] = []
  snapshot_leaves: list[variablelib.Variable | Mask] = []
  for leaf in leaves:
    if isinstance(leaf, variablelib.Variable):
      updates_leaves.append(leaf)
      # don't snapshot hijax or ref Variables as their updates are automatically
      # masked out in mask_variable_updates. However, the leaf is kept in the
      # updates to check for aliasing. This avoids a copy operation which has
      # significance for ref Variables.
      if leaf.hijax or leaf.ref:
        snapshot_leaves.append(Mask())
      else:
        snapshot_leaves.append(leaf.copy())
    else:
      updates_leaves.append(Mask())
      snapshot_leaves.append(Mask())
  updates = jax.tree.unflatten(treedef, updates_leaves)
  snapshot = jax.tree.unflatten(treedef, snapshot_leaves)
  return updates, snapshot


def check_no_aliases(fn_name: str, /, **kwargs):
  Attrs = namedtuple('Attrs', kwargs.keys())  # type: ignore[misc]
  container = Attrs(**kwargs)
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  seen: dict[int, jax.tree_util.KeyPath] = {}
  for path, leaf in jax.tree.leaves_with_path(
    container, is_leaf=is_leaf
  ):
    if not isinstance(leaf, variablelib.Variable):
      continue
    var_id = id(leaf)
    if var_id in seen:
      path_str = jax.tree_util.keystr(path)
      seen_path_str = jax.tree_util.keystr(seen[var_id])
      raise ValueError(
        f'Duplicate {leaf}\nfound at paths:\n\n'
        f'  - {seen_path_str}\n'
        f'  - {path_str}\n\n'
        f'nnx.{fn_name} with graph_updates=False does not support '
        'returning input Variables as outputs. '
        f'Consider the following options:\n\n'
        f'1. Remove the duplicate Variables.\n'
        f'2. Create new Variables via nnx.clone() and use those instead.\n'
        f'3. Enable graph mode and graph updates by passing graph=True and '
        f'graph_updates=True to {fn_name}\n\n'
        f'  nnx.{fn_name}(..., graph=True, graph_updates=True)\n\n'
        f'4. Use nnx.compat.{fn_name} (sets graph and graph_updates to True '
        f'automatically)\n\n'
        f'  nnx.compat.{fn_name}(...)'
      )
    seen[var_id] = path


def check_prefix(prefix: tp.Any, prefix_name: str, fn_name: str):
  def _check(path, leaf):
    if graphlib.is_graph_node(leaf) or isinstance(leaf, variablelib.Variable):
      raise ValueError(
        f'Found graph node or Variable of type {type(leaf).__name__} '
        f'at path {jax.tree_util.keystr(path)} in `{prefix_name}` '
        f'for nnx.{fn_name}. Graph nodes and Variables are not allowed '
        f'as prefixes when graph=True and graph_updates=False'
      )
  jax.tree.map_with_path(
    _check, prefix,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x),
  )


def variable_changed(post: variablelib.Variable, pre: variablelib.Variable) -> bool:
  post_leaves, post_td = jax.tree.flatten(post)
  pre_leaves, pre_td = jax.tree.flatten(pre)
  return post_td != pre_td or any(  # type: ignore[operator]
    a is not b for a, b in zip(post_leaves, pre_leaves)
  )

KeepFn = tp.Callable[
    [PathParts, tp.Any, variablelib.Variable, variablelib.Variable], bool
]

def mask_variable_updates(
    current_tree: A,
    snapshot_tree: A,
    *,
    prefix: tp.Any = Missing,
    keep_fn: KeepFn | None = None,
) -> A:
  if keep_fn is None:
    keep_fn = lambda _, _pfx, cur, snap: variable_changed(cur, snap)

  def _mask_updates(path, prefix_leaf, current, snapshot):
    if current is None:
      # None leaves should remain None, they only appear here because
      # is_leaf catches None values for the prefix
      return None
    if isinstance(current, variablelib.Variable):
      if current.hijax or current.ref:
        return Mask()
      if keep_fn(path, prefix_leaf, current, snapshot):
        return current
    return Mask()
  is_leaf = lambda x: isinstance(x, variablelib.Variable) or x is None
  if prefix is Missing:
    return jax.tree.map_with_path(
        lambda path, cur, snap: _mask_updates(path, None, cur, snap),
        current_tree, snapshot_tree, is_leaf=is_leaf,
    )
  return broadcast_prefix_map(
      _mask_updates, prefix, current_tree, snapshot_tree, is_leaf=is_leaf
  )


def apply_variable_updates(args_tree: A, updates_tree: A):
  is_leaf = lambda x: isinstance(x, variablelib.Variable) or isinstance(x, Mask)
  args_leaves = jax.tree.leaves(args_tree, is_leaf=is_leaf)
  _, treedef = jax.tree.flatten(args_tree, is_leaf=is_leaf)
  updates_leaves = treedef.flatten_up_to(updates_tree)
  for variable, update in zip(args_leaves, updates_leaves, strict=True):
    if isinstance(update, variablelib.Variable):
      assert isinstance(variable, variablelib.Variable)
      variable.update_from_state(update)


def treemap_copy_args(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    args, kwargs = jax.tree.map(lambda x: x, (args, kwargs))
    return f(*args, **kwargs)
  return wrapper


def check_same_variables(inputs, outputs, transform_name: str = ''):
  def _check(in_leaf, out_leaf):
    if isinstance(in_leaf, variablelib.Variable) and in_leaf is not out_leaf:
      raise ValueError(
        f'{transform_name} Variable identity must be preserved '
        'across iterations.'
      )
  is_leaf = lambda x: isinstance(x, (Mask, variablelib.Variable))
  jax.tree.map(
    _check, inputs, outputs,
    is_leaf=is_leaf,
  )


def update_carry_variables(init_val, val_out):
  def _update(in_leaf, out_leaf):
    if isinstance(in_leaf, variablelib.Variable):
      in_leaf.update_from_state(out_leaf)
      return in_leaf
    return out_leaf

  return jax.tree.map(
    _update, init_val, val_out,
    is_leaf=lambda x: isinstance(x, variablelib.Variable),
  )

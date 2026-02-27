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

import jax

from flax import struct
from flax import typing
from flax.nnx.pytreelib import Pytree
from flax.typing import Missing, PathParts
from flax.nnx import graphlib, variablelib


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


def updates_and_snapshot(args: A) -> tuple[A, A]:
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  leaves, treedef = jax.tree.flatten(args, is_leaf=is_leaf)
  updates_leaves: list[variablelib.Variable | None] = []
  snapshot_leaves: list[variablelib.Variable | None] = []
  for leaf in leaves:
    if isinstance(leaf, variablelib.Variable):
      updates_leaves.append(leaf)
      snapshot_leaves.append(leaf.copy())
    else:
      updates_leaves.append(None)
      snapshot_leaves.append(None)
  updates = jax.tree.unflatten(treedef, updates_leaves)
  snapshot = jax.tree.unflatten(treedef, snapshot_leaves)
  return updates, snapshot


class _InputsAndOutputs(tp.NamedTuple):
  args: tuple
  kwargs: dict | None
  output: tp.Any


@tp.overload
def check_no_aliases(args: tuple[tp.Any, ...], output: tp.Any) -> None:
  ...


@tp.overload
def check_no_aliases(
    args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], output: tp.Any
) -> None:
  ...


def check_no_aliases(*positional_args):
  if len(positional_args) == 2:
    args, output = positional_args  # pytype: disable=bad-unpacking
    kwargs = None
  elif len(positional_args) == 3:
    args, kwargs, output = positional_args  # pytype: disable=bad-unpacking
  else:
    raise TypeError(
      f'check_no_aliases expects 2 or 3 arguments, got {len(positional_args)}'
    )

  container = _InputsAndOutputs(args=args, kwargs=kwargs, output=output)
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
        f'Variable at {path_str} is the same instance as at '
        f'{seen_path_str}. tree-mode transforms do not support '
        f'returning input Variables as outputs.'
      )
    seen[var_id] = path


def _variable_changed(post: variablelib.Variable, pre: variablelib.Variable) -> bool:
  post_leaves, post_td = jax.tree.flatten(post)
  pre_leaves, pre_td = jax.tree.flatten(pre)
  return post_td != pre_td or any(  # type: ignore[operator]
    a is not b for a, b in zip(post_leaves, pre_leaves)
  )

MaskFn = tp.Callable[
    [PathParts, variablelib.Variable, variablelib.Variable], bool
]

def mask_variable_updates(
    current_tree: A,
    snapshot_tree: A,
    *,
    keep_fn: MaskFn | None = None,
) -> A:
  if keep_fn is None:
    keep_fn = lambda _, cur, snap: False

  def _mask_updates(jax_path, current, snapshot):
    path = graphlib.jax_to_nnx_path(jax_path)
    if isinstance(current, variablelib.Variable) and (
        keep_fn(path, current, snapshot) or _variable_changed(current, snapshot)
    ):
      return current
    return None
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  return jax.tree.map_with_path(
      _mask_updates, current_tree, snapshot_tree, is_leaf=is_leaf
  )


def apply_variable_updates(args_tree: A, updates_tree: A) -> None:
  is_leaf = lambda x: isinstance(x, variablelib.Variable) or x is None
  args_leaves, treedef = jax.tree.flatten_with_path(args_tree, is_leaf=is_leaf)
  updates_leaves = treedef.flatten_up_to(updates_tree)
  seen: dict[int, jax.tree_util.KeyPath] = {}
  for (path, variable), update in zip(args_leaves, updates_leaves):
    if not isinstance(variable, variablelib.Variable):
      continue
    var_id = id(variable)
    if var_id in seen:
      path_str = jax.tree_util.keystr(path)
      seen_path_str = jax.tree_util.keystr(seen[var_id])
      raise ValueError(
        f'Variable at {path_str} was already seen at {seen_path_str}. '
        'tree-mode jit does not support shared Variable references.'
      )
    seen[var_id] = path
    if update is not None:
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
  is_leaf = lambda x: x is None or isinstance(x, variablelib.Variable)
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
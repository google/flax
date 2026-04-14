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
from collections import namedtuple
import dataclasses
import functools
import typing as tp

from flax import struct
from flax import typing
from flax.nnx import filterlib, graphlib, variablelib
from flax.nnx.pytreelib import Pytree
from flax.typing import Missing, PathParts
import jax


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
    node_prefixes: dict[int, list[tuple[PathParts, tp.Any]]],
):
  node_id_to_variable: dict[int, tp.Any] = {}

  for local_path, value in graphlib.iter_graph(node, graph=True):
    path = base_path + local_path
    if isinstance(value, variablelib.Variable):
      value_id = id(value)
      node_id_to_variable[value_id] = value
      # If prefix is a TreeState (e.g. from nnx.prefix(graph=True)),
      # extract the actual prefix value for this Variable using local_path.
      if isinstance(prefix, TreeState):
        prefix_fn = prefix.prefix_fn.value
        if not callable(prefix_fn):
          raise ValueError(
              'When passing a TreeState object as a prefix (e.g. for'
              ' `in_axes`), it must have been produced by `nnx.prefix()` or'
              ' contain a callable in `TreeState.metadata` with signature'
              ' `(path: tuple[Any, ...], value: Variable) -> Any`. Got'
              f' metadata of type {type(prefix_fn).__name__}.'
          )
        leaf_prefix = prefix_fn(local_path, value)
      else:
        leaf_prefix = prefix
      if value_id in node_prefixes:
        node_prefixes[value_id].append((path, leaf_prefix))
      else:
        node_prefixes[value_id] = [(path, leaf_prefix)]

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
  prefix_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tuple[list[KeyPath], list[tp.Any]]:
  _prefix_leaf: tp.Callable[[tp.Any], bool] | None
  if prefix_leaf is not None and is_leaf is not None:
    _prefix_leaf = lambda x: prefix_leaf(x) or is_leaf(x)
  elif prefix_leaf is not None:
    _prefix_leaf = prefix_leaf
  else:
    _prefix_leaf = is_leaf

  paths: list[KeyPath] = []
  leaves: list[tp.Any] = []
  num_leaves = lambda t: jax.tree.structure(t, is_leaf=is_leaf).num_leaves
  def add_leaves(path, x, subtree):
    n = num_leaves(subtree)
    paths.extend([path] * n)
    leaves.extend([x] * n)
  jax.tree.map_with_path(add_leaves, prefix_tree, full_tree, is_leaf=_prefix_leaf)
  return paths, leaves

def broadcast_prefix_map(
  f: tp.Callable[..., tp.Any],
  prefix_tree: tp.Any,
  full_tree: tp.Any,
  *rest: tp.Any,
  is_leaf: tp.Callable[[tp.Any], bool] | None = None,
  prefix_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tp.Any:
  _, prefix_leaves = broadcast_prefix2(prefix_tree, full_tree, is_leaf=is_leaf, prefix_leaf=prefix_leaf)
  full_leaves_with_path, treedef = jax.tree.flatten_with_path(full_tree, is_leaf=is_leaf)
  rest_flat = [treedef.flatten_up_to(r) for r in rest]
  out_leaves = []
  for (path, full_leaf), p_leaf, *r_leaves in zip(full_leaves_with_path, prefix_leaves, *rest_flat):
    out_leaf = f(path, p_leaf, full_leaf, *r_leaves)
    out_leaves.append(out_leaf)
  return jax.tree.unflatten(treedef, out_leaves)


class GraphDefState(struct.PyTreeNode):
  graphdef: graphlib.GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: graphlib.State = struct.field(pytree_node=True)

S = tp.TypeVar(
  'S', bound=graphlib.State | graphlib.GraphFlatState | list[tp.Any]
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


@dataclasses.dataclass(frozen=True, slots=True)
class Opaque(tp.Generic[A]):
  value: A

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return 0


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['__state__'],
    meta_fields=['graphdef', 'prefix_fn'],
)
@dataclasses.dataclass(frozen=True, slots=True)
class TreeState:
  graphdef: graphlib.GraphDef[tp.Any] | None
  __state__: tp.Any
  prefix_fn: Opaque[tp.Callable[[PathParts, tp.Any], tp.Any] | None] = Opaque(
      None
  )

  @property
  def state(self) -> tp.Any:
    return self.__state__

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def to_tree2(
    tree,
    /,
    *,
    prefix: tp.Any = Missing,
    check_aliasing: bool = True,
    prefix_fn: tp.Callable[[PathParts, tp.Any], tp.Any] | None = None,
) -> tp.Any:
  """to_tree2 has two main tasks:

  1. Convert all graph nodes to TreeState (a tree representation).
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
    return TreeState(graphdef, state, prefix_fn=Opaque(prefix_fn))

  is_leaf = lambda x: (
    isinstance(x, variablelib.Variable) or graphlib.is_graph_node(x)
  )

  if prefix is Missing or prefix is None:
    return jax.tree.map(_to_node_states, tree, is_leaf=is_leaf)

  leaf_prefixes = broadcast_prefix(
      prefix,
      tree,
      prefix_is_leaf=lambda x: x is None
      or isinstance(x, TreeState)
      or is_leaf(x),
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
    if not isinstance(x, TreeState):
      return x
    state = graphlib._merge_to_flat_state((x.state,))
    return graphlib.unflatten(
      x.graphdef, state, index_ref=index_ref,
    )

  return jax.tree.map(
      _from_node_states,
      tree,
      is_leaf=lambda x: (
          isinstance(x, TreeState)
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


def slice_at(t: tuple, index: int | None) -> tuple[tp.Any, tuple]:
  if index is None:
    return None, t
  return t[index], t[:index] + t[index + 1 :]


def insert_at(t: tuple, index: int | None, value: tp.Any) -> tuple:
  if index is None:
    return t
  xs = list(t)
  xs.insert(index, value)
  return tuple(xs)


def find(t: tuple, value: tp.Any) -> int | None:
  return next((i for i, x in enumerate(t) if x == value), None)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class ExtractIndex:
  index: int


def extract(
  f: tp.Callable[[jax.tree_util.KeyPath, tp.Any, tp.Any], bool],
  prefix: tp.Any,
  tree: tp.Any,
  *,
  is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tuple[tp.Any, list[tp.Any]]:
  extracted: list[tp.Any] = []
  def _leaf_fn(path: jax.tree_util.KeyPath, prefix_leaf: tp.Any, leaf: tp.Any):
    if f(path, prefix_leaf, leaf):
      idx = len(extracted)
      extracted.append(leaf)
      return ExtractIndex(idx)
    return leaf

  full_prefix = jax.tree.broadcast(prefix, tree, is_leaf=is_leaf)
  new_tree = jax.tree.map_with_path(_leaf_fn, full_prefix, tree, is_leaf=is_leaf)
  return new_tree, extracted


def insert(
    tree: tp.Any,
    extracted: list[tp.Any],
    is_leaf: tp.Callable[[tp.Any], bool] | None = None,
) -> tp.Any:
  if is_leaf is None:
    _is_leaf = lambda x: isinstance(x, ExtractIndex)
  else:
    _is_leaf = lambda x: isinstance(x, ExtractIndex) or is_leaf(x)

  def _leaf_fn(leaf: tp.Any):
    if isinstance(leaf, ExtractIndex):
      return extracted[leaf.index]
    return leaf

  return jax.tree.map(_leaf_fn, tree, is_leaf=_is_leaf)


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


def check_no_aliases(
    fn_name: str, /, *, check_can_update: tp.Iterable[str] = (), **kwargs
):
  Attrs = namedtuple('Attrs', kwargs.keys())  # type: ignore[misc]
  container = Attrs(**kwargs)
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  seen: dict[int, jax.tree_util.KeyPath] = {}
  for path, leaf in jax.tree.leaves_with_path(container, is_leaf=is_leaf):
    if not isinstance(leaf, variablelib.Variable):
      continue

    assert isinstance(path[0], jax.tree_util.GetAttrKey)
    kwarg_name = path[0].name

    if kwarg_name in check_can_update:
      if not leaf._can_update:
        path_str = jax.tree_util.keystr(path)
        raise ValueError(
            f'Cannot return captured Variable of type {type(leaf).__name__} '
            f'from nnx.{fn_name}.\n'
            f'Found at path: {path_str}'
        )

    var_id = id(leaf)
    if var_id in seen:
      path_str = jax.tree_util.keystr(path)
      seen_path_str = jax.tree_util.keystr(seen[var_id])
      raise ValueError(
        f'Duplicate {leaf}\nfound at paths:\n\n'
        f'  - {seen_path_str}\n'
        f'  - {path_str}\n\n'
        f'nnx.{fn_name} with graph_updates=False does not support '
        'Variable aliasing (duplicate inputs, duplicate outputs, or '
        'input Variables returned as outputs). '
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


def check_prefix(
  prefix: tp.Any,
  prefix_name: str,
  fn_name: str,
  graph: bool,
  graph_updates: bool,
):
  def _check(path, leaf):
    if isinstance(leaf, variablelib.Variable):
      raise ValueError(
        f'Found Variable of type {type(leaf).__name__} '
        f'at path {jax.tree_util.keystr(path)} in `{prefix_name}` '
        f'for nnx.{fn_name}. Variables prefixes are not supported.'
        f'Pass a prefix for the entire Variable instead of passing a '
        f'Variable with a prefix for its value.'
      )
    if isinstance(leaf, PrefixMapping) and not (graph and graph_updates):
      raise ValueError(
        f'`{prefix_name}` cannot contain `{type(leaf).__name__}` objects '
        f'when `graph=False` or `graph_updates=False`. '
        f'Consider the following options:\n\n'
        f'1. Remove `{type(leaf).__name__}` objects from `{prefix_name}`.\n'
        f'2. Enable graph mode and graph updates by passing graph=True and '
        f'graph_updates=True to {fn_name} e.g.\n\n'
        f'  nnx.{fn_name}(..., graph=True, graph_updates=True)\n\n'
        f'3. Use nnx.compat.{fn_name} instead e.g.\n\n'
        f'  nnx.compat.{fn_name}(...)'
      )
    if graphlib.is_graph_node(leaf) and graph:
      raise ValueError(
        f'Found graph node of type {type(leaf).__name__} '
        f'at path {jax.tree_util.keystr(path)} in `{prefix_name}` '
        f'for nnx.{fn_name}. Graph nodes are not allowed as prefixes when '
        f'graph=True.'
        f'Consider the following options:\n\n'
        f'1. Remove graph nodes from `{prefix_name}`.\n'
        f'2. Enable tree mode by passing graph=False to {fn_name} e.g.\n\n'
        f'  nnx.{fn_name}(..., graph=False)\n\n'
        f'3. If you using nnx.prefix to create the prefix, pass graph=True:\n\n'
        f'  prefix = nnx.prefix(..., graph=True)'
      )
    if isinstance(leaf, TreeState) and (not graph or graph_updates):
      msg = (
        f'Found `TreeState` object at path {jax.tree_util.keystr(path)} in '
        f'`{prefix_name}` for nnx.{fn_name}. `TreeState` objects are only '
        f'allowed as prefixes when `graph=True` and `graph_updates=False`.'
        f'Consider the following options:\n\n'
        f'1. Enable graph mode and graph updates by passing graph=True and '
        f'graph_updates=True to {fn_name} e.g.\n\n'
        f'  nnx.{fn_name}(..., graph=True, graph_updates=True)\n\n'
        f'2. Use nnx.compat.{fn_name} instead e.g.\n\n'
        f'  nnx.compat.{fn_name}(...)'
      )
      if graph_updates:
        msg += (
          f'\n\n3. If you using nnx.prefix to create the prefix, pass graph=False:\n\n'
          f'  prefix = nnx.prefix(..., graph=False)'
        )
      raise ValueError(msg)

  jax.tree.map_with_path(
    _check,
    prefix,
    is_leaf=lambda x: isinstance(x, variablelib.Variable)
    or graphlib.is_graph_node(x)
    or isinstance(x, PrefixMapping)
    or isinstance(x, TreeState),
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
    if isinstance(current, variablelib.Variable):
      if current.hijax or current.ref:
        return Mask()
      if keep_fn(path, prefix_leaf, current, snapshot):
        return current
    return Mask()
  prefix_leaf = lambda x: x is None
  is_leaf = lambda x: isinstance(x, variablelib.Variable)
  if prefix is Missing:
    return jax.tree.map_with_path(
        lambda path, cur, snap: _mask_updates(path, None, cur, snap),
        current_tree, snapshot_tree, is_leaf=is_leaf
    )
  return broadcast_prefix_map(
      _mask_updates, prefix, current_tree, snapshot_tree, is_leaf=is_leaf,
      prefix_leaf=prefix_leaf,
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


def prefix(
    node,
    filter_map: tp.Mapping[filterlib.Filter, tp.Any] | tp.Callable[..., tp.Any],
    /,
    *,
    graph: bool | None = None,
):
  """Replaces leaves in a graph node with prefix values.

  ``prefix`` replaces each leaf in ``node`` with a prefix value computed by
  ``filter_map(path, leaf)``. In graph mode (``graph=True``), the node is
  first converted to a tree and the prefix is applied to
  the resulting structure so it can be used directly as axes arguments for
  transforms like ``nnx.vmap``.

  Example usage::

    from flax import nnx
    import jax.numpy as jnp

    d = {'a': nnx.Param(jnp.array(2)), 'b': nnx.BatchStat(jnp.arange(5))}
    prefix = nnx.prefix(d, lambda path, x: 0 if 'b' in path else None)

    @nnx.vmap(in_axes=(prefix,))
    def f(d):
      return d['a'] * d['b']

    f(d)  # Array([0, 2, 4, 6, 8])

  ``filter_map`` can also be a mapping from :class:`Filter` to prefix values.
  Filters are checked in order and the first match determines the prefix::

    d = {'a': nnx.Param(jnp.array(2)), 'b': nnx.BatchStat(jnp.arange(5))}
    prefix = nnx.prefix(d, {nnx.Param: None, nnx.BatchStat: 0})

  Calculating prefixes for graph mode transforms is a bit more involved as
  the graph nodes are first converted to a trees in an order-dependent manner.
  This means prefixes should be calculated jointly between all graph nodes in
  the transform in the same order they appear in the arguments. For example::

    import jax
    import jax.numpy as jnp

    @nnx.vmap
    def create_model(rngs):
      return nnx.Linear(2, 3, rngs=rngs)

    model = create_model(nnx.Rngs(0).split(4))
    px1, px2 = nnx.prefix((model, model), {nnx.Param: 0}, graph=True)

    @nnx.vmap(in_axes=(px1, px2, None), graph=True)
    def forward(m1, m2, x):
      assert m1 is m2
      return m1(x) + m2(x)

    y = forward(model, model, jnp.ones(2))
    assert y.shape == (4, 3)

  The prefixes might be invalid if all graph node involved in the transform
  aren't passed to `nnx.prefix`.

  Args:
    node: A graph node object.
    filter_map: A callable ``(path, leaf) -> prefix`` that computes the prefix
      for each leaf, or a mapping from :class:`Filter` to prefix values (filters
      are checked in order; the first match determines the prefix).
    graph: If ``True``, uses graph-mode which supports the full NNX feature set
      including shared references. If ``False``, uses tree-mode which treats
      Modules as regular JAX pytrees, avoiding the overhead of the graph
      protocol.

  Returns:
    A new tree with prefix values replacing the leaves.
  """
  if graph is None:
    graph = graphlib.set_graph_mode.current_value()

  if isinstance(filter_map, tp.Mapping):
    predicates = tuple(
        (filterlib.to_predicate(f), value) for f, value in filter_map.items()
    )
    filters = list(filter_map.keys())

    def prefix_fn(path, leaf):
      for predicate, _prefix in predicates:
        if predicate(path, leaf):
          return _prefix
      raise ValueError(
          f'No filter matched leaf at path {path!r} with value {leaf!r}. '
          f'Filters: {filters}'
      )

  else:
    prefix_fn = filter_map

  is_leaf = lambda x: isinstance(x, variablelib.Variable)

  if graph:
    node = to_tree2(node, prefix_fn=prefix_fn)

  def _apply_prefix(jax_path, leaf):
    path = graphlib.jax_to_nnx_path(jax_path)
    if graph:
      # remove __state__ resulting from TreeState from path
      # to match the path you get on graph=False
      path = tuple(k for k in path if k != '__state__')
    return prefix_fn(path, leaf)

  return jax.tree.map_with_path(_apply_prefix, node, is_leaf=is_leaf)

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
import enum
import functools
import threading
import typing as tp
from copy import deepcopy

import jax
import numpy as np
import typing_extensions as tpe

from flax.nnx.nnx import filterlib, reprlib
from flax.nnx.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.nnx.state import FlatState, State
from flax.nnx.nnx.variables import Variable, VariableState
from flax.typing import Key, PathParts

A = tp.TypeVar('A')
B = tp.TypeVar('B')
C = tp.TypeVar('C')
F = tp.TypeVar('F', bound=tp.Callable)

HA = tp.TypeVar('HA', bound=tp.Hashable)
HB = tp.TypeVar('HB', bound=tp.Hashable)

Index = int
Names = tp.Sequence[int]
Node = tp.TypeVar('Node')
Leaf = tp.TypeVar('Leaf')
AuxData = tp.TypeVar('AuxData')

StateLeaf = tp.Union[VariableState[tp.Any], np.ndarray, jax.Array]
GraphState = State[Key, StateLeaf]
GraphFlatState = FlatState[StateLeaf]


def is_state_leaf(x: tp.Any) -> tpe.TypeGuard[StateLeaf]:
  return isinstance(x, (VariableState, np.ndarray, jax.Array))


@dataclasses.dataclass
class GraphContext(threading.local):
  update_context_stacks: dict[str, list[UpdateContext]] = dataclasses.field(
    default_factory=dict
  )


GRAPH_CONTEXT = GraphContext()


def is_node_leaf(x: tp.Any) -> tpe.TypeGuard[StateLeaf]:
  return isinstance(x, (Variable, np.ndarray, jax.Array))


class _HashById(tp.Hashable, tp.Generic[A]):
  """A wrapper around a value that uses its id for hashing and equality.
  This is used by RefMap to explicitly use object id as the hash for the keys.
  """

  __slots__ = ('_value',)

  def __init__(self, value: A):
    self._value = value

  @property
  def value(self) -> A:
    return self._value

  def __hash__(self) -> int:
    return id(self._value)

  def __eq__(self, other: tp.Any) -> bool:
    return isinstance(other, _HashById) and self._value is other._value


class RefMap(tp.MutableMapping[A, B], reprlib.MappingReprMixin[A, B]):
  """A mapping that uses object id as the hash for the keys."""

  def __init__(
    self, mapping: tp.Mapping[A, B] | tp.Iterable[tuple[A, B]] = (), /
  ):
    self._mapping: dict[_HashById[A], B] = {}
    self.update(mapping)

  def __getitem__(self, key: A) -> B:
    return self._mapping[_HashById(key)]

  def __contains__(self, key: object) -> bool:
    return _HashById(key) in self._mapping

  def __setitem__(self, key: A, value: B):
    self._mapping[_HashById(key)] = value

  def __delitem__(self, key: A):
    del self._mapping[_HashById(key)]

  def __iter__(self) -> tp.Iterator[A]:
    return (x.value for x in self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __str__(self) -> str:
    return repr(self)


@dataclasses.dataclass(frozen=True)
class NodeImplBase(tp.Generic[Node, Leaf, AuxData]):
  type: type
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]]

  def node_dict(self, node: Node) -> dict[Key, Leaf]:
    nodes, _ = self.flatten(node)
    return dict(nodes)


@dataclasses.dataclass(frozen=True)
class GraphNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  set_key: tp.Callable[[Node, Key, Leaf], None]
  pop_key: tp.Callable[[Node, Key], Leaf]
  create_empty: tp.Callable[[AuxData], Node]
  clear: tp.Callable[[Node], None]

  def init(self, node: Node, items: tuple[tuple[Key, Leaf], ...]):
    for key, value in items:
      self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PytreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  unflatten: tp.Callable[[tuple[tuple[Key, Leaf], ...], AuxData], Node]


NodeImpl = tp.Union[
  GraphNodeImpl[Node, Leaf, AuxData], PytreeNodeImpl[Node, Leaf, AuxData]
]


_node_impl_for_type: dict[type, NodeImpl[tp.Any, tp.Any, tp.Any]] = {}


def register_graph_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
  set_key: tp.Callable[[Node, Key, Leaf], None],
  pop_key: tp.Callable[[Node, Key], Leaf],
  create_empty: tp.Callable[[AuxData], Node],
  clear: tp.Callable[[Node], None],
):
  _node_impl_for_type[type] = GraphNodeImpl(
    type=type,
    flatten=flatten,
    set_key=set_key,
    pop_key=pop_key,
    create_empty=create_empty,
    clear=clear,
  )


def is_node(x: tp.Any) -> bool:
  if type(x) in _node_impl_for_type:
    return True
  return is_pytree_node(x)


def is_graph_node(x: tp.Any) -> bool:
  return type(x) in _node_impl_for_type


def is_node_type(x: type[tp.Any]) -> bool:
  return x in _node_impl_for_type or x is PytreeType


def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any]:
  if isinstance(x, Variable):
    raise ValueError(f'Variable is not a node: {x}')

  node_type = type(x)

  if node_type not in _node_impl_for_type:
    if is_pytree_node(x):
      return PYTREE_NODE_IMPL
    else:
      raise ValueError(f'Unknown node type: {x}')

  return _node_impl_for_type[node_type]


def get_node_impl_for_type(x: type[Node]) -> NodeImpl[Node, tp.Any, tp.Any]:
  if x is PytreeType:
    return PYTREE_NODE_IMPL
  return _node_impl_for_type[x]


class _HashableMapping(tp.Mapping[HA, HB], tp.Hashable):
  def __init__(self, mapping: tp.Mapping[HA, HB] | tp.Iterable[tuple[HA, HB]]):
    self._mapping = dict(mapping)

  def __contains__(self, key: object) -> bool:
    return key in self._mapping

  def __getitem__(self, key: HA) -> HB:
    return self._mapping[key]

  def __iter__(self) -> tp.Iterator[HA]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __hash__(self) -> int:
    return hash(tuple(sorted(self._mapping.items())))

  def __eq__(self, other: tp.Any) -> bool:
    return (
      isinstance(other, _HashableMapping) and self._mapping == other._mapping
    )

  def __repr__(self) -> str:
    return repr(self._mapping)


@dataclasses.dataclass(frozen=True, repr=False)
class NodeDef(tp.Generic[Node], reprlib.Representable):
  type: tp.Type[Node]
  index: int
  attributes: tuple[Key, ...]
  subgraphs: _HashableMapping[Key, tp.Union[NodeDef[tp.Any], Index]]
  static_fields: _HashableMapping[Key, tp.Any]
  leaves: _HashableMapping[Key, Index | None]
  metadata: tp.Any

  @classmethod
  def create(
    cls,
    type: tp.Type[Node],
    index: int,
    attributes: tuple[Key, ...],
    subgraphs: tp.Iterable[tuple[Key, tp.Union[NodeDef[tp.Any], Index]]],
    static_fields: tp.Iterable[tuple[Key, tp.Any]],
    leaves: tp.Iterable[tuple[Key, Index | None]],
    metadata: tp.Any,
  ):
    return cls(
      type=type,
      index=index,
      attributes=attributes,
      subgraphs=_HashableMapping(subgraphs),
      static_fields=_HashableMapping(static_fields),
      leaves=_HashableMapping(leaves),
      metadata=metadata,
    )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('attributes', self.attributes)
    yield reprlib.Attr('subgraphs', reprlib.PrettyMapping(self.subgraphs))
    yield reprlib.Attr(
      'static_fields', reprlib.PrettyMapping(self.static_fields)
    )
    yield reprlib.Attr('leaves', reprlib.PrettyMapping(self.leaves))
    yield reprlib.Attr('metadata', self.metadata)

  def __penzai_repr__(self, path, subtree_renderer):
    from penzai.treescope import repr_lib as pz_repr_lib  # type: ignore[import-not-found,import-untyped]
    return pz_repr_lib.render_object_constructor(
        object_type=type(self),
        attributes={
            'type': self.type,
            'index': self.index,
            'attributes': self.attributes,
            'subgraphs': dict(self.subgraphs),
            'static_fields': dict(self.static_fields),
            'leaves': dict(self.leaves),
            'metadata': self.metadata,
        },
        path=path,
        subtree_renderer=subtree_renderer,
    )


@dataclasses.dataclass(frozen=True, repr=False)
class GraphDef(tp.Generic[Node], reprlib.Representable):
  """A dataclass that denotes the tree structure of a
  :class:`Module`. A ``GraphDef`` can be generated by either
  calling :func:`split` or :func:`graphdef` on the :class:`Module`."""

  nodedef: NodeDef[Node]
  index_mapping: dict[Index, Index] | None

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('nodedef', self.nodedef)
    yield reprlib.Attr('index_mapping', self.index_mapping)

  def __penzai_repr__(self, path, subtree_renderer):
    from penzai.treescope import repr_lib as pz_repr_lib  # type: ignore[import-not-found,import-untyped]
    return pz_repr_lib.render_object_constructor(
        object_type=type(self),
        attributes={
            'nodedef': self.nodedef,
            'index_mapping': self.index_mapping,
        },
        path=path,
        subtree_renderer=subtree_renderer,
    )

  def __deepcopy__(self, memo=None):
    nodedef = deepcopy(self.nodedef, memo)
    index_mapping = deepcopy(self.index_mapping, memo)
    return GraphDef(nodedef, index_mapping)

  def __hash__(self):
    return hash(self.nodedef)

  def __eq__(self, other):
    return isinstance(other, GraphDef) and self.nodedef == other.nodedef

  def apply(
    self, state: GraphState, *states: GraphState
  ) -> ApplyCaller[tuple[GraphDef[Node], GraphState]]:
    accessor = DelayedAccessor()

    def _apply(
      accessor: DelayedAccessor, *args, **kwargs
    ) -> tuple[tp.Any, tuple[GraphDef[Node], GraphState]]:
      module = merge(self, state, *states)
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, flatten(module)[:2]

    return CallableProxy(_apply, accessor)  # type: ignore

def _graphdef_flatten(graphdef: GraphDef[Node]):
  # refmap is opaque, we don't propagate it
  static = (graphdef.nodedef, graphdef.index_mapping)
  return (), static


def _graphdef_unflatten(
  static: tuple[NodeDef[Node], dict[Index, Index] | None], _nodes: tuple[()]
):
  nodedef, index_mapping = static
  return GraphDef(nodedef, index_mapping)


jax.tree_util.register_pytree_node(
  GraphDef,
  _graphdef_flatten,
  _graphdef_unflatten,
)


def flatten(
  x: Node,
  /,
  *,
  idxmap: dict[Index, tp.Any] | None = None,
) -> tuple[GraphDef[Node], GraphState, RefMap[tp.Any, Index]]:
  refmap = RefMap[tp.Any, Index]()
  flat_state: dict[PathParts, StateLeaf] = {}
  nodedef = _graph_flatten((), refmap, flat_state, x)
  assert not isinstance(nodedef, int)
  if idxmap is not None:
    index_to_index = compose_mapping(idxmap, refmap)
  else:
    index_to_index = None
  graphdef = GraphDef(nodedef, index_to_index)
  return graphdef, GraphState.from_flat_path(flat_state), refmap


def _graph_flatten(
  path: PathParts,
  refmap: RefMap[tp.Any, Index],
  flat_state: dict[PathParts, StateLeaf],
  node: Node,
) -> NodeDef[Node] | int:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if node in refmap:
    return refmap[node]

  node_impl = get_node_impl(node)

  # only cache graph nodes
  if isinstance(node_impl, GraphNodeImpl):
    index = len(refmap)
    refmap[node] = index
  else:
    index = -1

  subgraphs: list[tuple[Key, tp.Union[NodeDef[Node], Index]]] = []
  static_fields: list[tuple[Key, tp.Any]] = []
  leaves: list[tuple[Key, Index | None]] = []

  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if is_node(value):
      nodedef = _graph_flatten((*path, key), refmap, flat_state, value)
      subgraphs.append((key, nodedef))
    elif isinstance(value, Variable):
      if value in refmap:
        leaves.append((key, refmap[value]))
      else:
        flat_state[(*path, key)] = value.to_state()
        variable_index = refmap[value] = len(refmap)
        leaves.append((key, variable_index))
    elif is_state_leaf(value):
      flat_state[(*path, key)] = value
      leaves.append((key, None))
    else:
      static_fields.append((key, value))

  nodedef = NodeDef.create(
    type=node_impl.type,
    index=index,
    attributes=tuple(key for key, _ in values),
    subgraphs=subgraphs,
    static_fields=static_fields,
    leaves=leaves,
    metadata=metadata,
  )
  return nodedef


def unflatten(
  graphdef: GraphDef[Node],
  state: GraphState,
  /,
  *,
  idxmap: dict[Index, tp.Any] | None = None,
) -> tuple[Node, dict[Index, tp.Any]]:
  """Unflattens a graphdef into a node with the given state.

  Args:
    graphdef: A NodeDef instance.
    state: A State instance.
    ref_cache: A mapping from indexes to existing nodes that can be reused.
      When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the graphdef.
  """
  index_to_ref: dict[Index, tp.Any] = {}
  node = _graph_unflatten(
    graphdef.nodedef, state.raw_mapping, index_to_ref, idxmap
  )
  return node, index_to_ref


def _graph_unflatten(
  nodedef: tp.Union[NodeDef[Node], int],
  state: tp.Mapping[Key, StateLeaf | tp.Mapping[Key, tp.Any]],
  index_to_ref: dict[Index, tp.Any],
  idxmap: dict[Index, tp.Any] | None,
) -> Node:
  """Recursive helper for graph_unflatten.

  Args:
    nodedef: A NodeDef instance or an index to a node in the cache.
    state: A mapping from attribute names to variables or subgraphs.
    index_to_ref: A mapping from indexes to nodes that have been traversed.
      If a node is already in the cache, it won't be traversed again.
    ref_cache: A mapping from indexes to existing nodes that can be reused.
      When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the nodedef.
  """
  if isinstance(nodedef, int):
    return index_to_ref[nodedef]

  if not is_node_type(nodedef.type):
    raise RuntimeError(f'Unsupported type: {nodedef.type}, this is a bug.')

  if nodedef.index in index_to_ref:
    raise RuntimeError(f'NodeDef index {nodedef.index} already used.')

  node_impl = get_node_impl_for_type(nodedef.type)

  def _get_children():
    children: dict[Key, StateLeaf | Node] = {}

    # NOTE: we could allw adding new StateLeafs here
    if unkown_keys := set(state) - set(nodedef.attributes):
      raise ValueError(f'Unknown keys: {unkown_keys}')

    # for every key in attributes there are 6 possible cases:
    #  - (2) the key can either be present in the state or not
    #  - (3) the key can be a subgraph, a leaf, or a static attribute
    for key in nodedef.attributes:
      if key not in state:
        # TODO(cgarcia): maybe we shouldn't support unflattening with missing keys?
        # if key is not present create an empty types
        if key in nodedef.static_fields:
          children[key] = nodedef.static_fields[key]
        elif key in nodedef.subgraphs:
          # if the key is a subgraph we create an empty node
          subgraphdef = nodedef.subgraphs[key]
          if isinstance(subgraphdef, int):
            # subgraph exists, take it from the cache
            children[key] = index_to_ref[subgraphdef]
          else:
            # create a node from an empty state, reasoning:
            # * its a node with no state
            # * its a node with state but only through references of already
            #   created nodes
            substate = {}
            children[key] = _graph_unflatten(
              subgraphdef, substate, index_to_ref, idxmap
            )
        elif key in nodedef.leaves:
          leaf_index = nodedef.leaves[key]
          if leaf_index is not None and leaf_index in index_to_ref:
            # variable exists, take it from the cache
            children[key] = index_to_ref[leaf_index]
          else:
            # key for a variable is missing, raise an error
            raise ValueError(
              f'Expected key {key!r} in state while building node of type '
              f'{nodedef.type.__name__}.'
            )
        else:
          raise RuntimeError(f'Unknown static field: {key!r}')
      else:
        value = state[key]
        if key in nodedef.static_fields:
          raise ValueError(
            f'Got state for static field {key!r}, this is not supported.'
          )
        if key in nodedef.subgraphs:
          if is_state_leaf(value):
            raise ValueError(
              f'Expected value of type {nodedef.subgraphs[key]} for '
              f'{key!r}, but got {value!r}'
            )
          assert isinstance(value, dict)
          subgraphdef = nodedef.subgraphs[key]

          if isinstance(subgraphdef, int):
            children[key] = index_to_ref[subgraphdef]
          else:
            children[key] = _graph_unflatten(
              subgraphdef, value, index_to_ref, idxmap
            )

        elif key in nodedef.leaves:
          if not is_state_leaf(value):
            raise ValueError(f'Expected a leaf for {key!r}, but got {value!r}')

          leaf_index = nodedef.leaves[key]

          if leaf_index is None:
            # if the leaf is None, it means that the value was originally
            # a non-VariableState leaf, however we allow providing a
            # VariableState presumbly created by modifying the State
            if isinstance(value, VariableState):
              value = value.to_variable()
            children[key] = value
          elif leaf_index in index_to_ref:
            # add an existing variable
            children[key] = index_to_ref[leaf_index]
          else:
            # its a unseen variable, create a new one
            if not isinstance(value, VariableState):
              raise ValueError(
                f'Expected a Variable type for {key!r}, but got {type(value)}.'
              )
            # when idxmap is present, check if the Varable exists there
            # and update existing variables if it does
            if idxmap is not None and leaf_index in idxmap:
              variable = idxmap[leaf_index]
              if not isinstance(variable, Variable):
                raise ValueError(
                  f'Expected a Variable type for {key!r}, but got {type(variable)}.'
                )
              variable.copy_from_state(value)
            else:  # if it doesn't, create a new variable
              assert isinstance(value, VariableState)
              variable = value.to_variable()
            children[key] = variable
            index_to_ref[leaf_index] = variable
        else:
          raise RuntimeError(f'Unknown key: {key!r}, this is a bug.')

    return children

  if isinstance(node_impl, GraphNodeImpl):
    # we create an empty node first and add it to the index
    # this avoids infinite recursion when there is a reference cycle
    if idxmap is not None and nodedef.index in idxmap:
      node = idxmap[nodedef.index]
      if type(node) != nodedef.type:
        raise ValueError(
          f'Expected a node of type {nodedef.type} for index '
          f'{nodedef.index}, but got a node of type {type(node)}.'
        )
      node_impl.clear(node)
    else:
      node = node_impl.create_empty(nodedef.metadata)
    index_to_ref[nodedef.index] = node
    children = _get_children()
    node_impl.init(node, tuple(children.items()))
  else:
    # if the node type does not support the creation of an empty object it means
    # that it cannot reference itself, so we can create its children first
    children = _get_children()
    node = node_impl.unflatten(tuple(children.items()), nodedef.metadata)

  return node


def graph_pop(
  node: tp.Any,
  filters: tuple[filterlib.Filter, ...],
) -> tuple[GraphState, ...]:
  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[GraphFlatState, ...] = tuple({} for _ in predicates)
  _graph_pop(node, id_to_index, path_parts, flat_states, predicates)
  return tuple(
    GraphState.from_flat_path(flat_state) for flat_state in flat_states
  )


def _graph_pop(
  node: tp.Any,
  id_to_index: dict[int, Index],
  path_parts: PathParts,
  flat_states: tuple[GraphFlatState, ...],
  predicates: tuple[filterlib.Predicate, ...],
) -> None:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if id(node) in id_to_index:
    return

  id_to_index[id(node)] = len(id_to_index)
  node_impl = get_node_impl(node)
  node_dict = node_impl.node_dict(node)

  for name, value in node_dict.items():
    if is_node(value):
      _graph_pop(
        node=value,
        id_to_index=id_to_index,
        path_parts=(*path_parts, name),
        flat_states=flat_states,
        predicates=predicates,
      )
      continue
    elif not is_node_leaf(value):
      continue
    elif id(value) in id_to_index:
      continue

    node_path = (*path_parts, name)
    node_impl = get_node_impl(node)
    for state, predicate in zip(flat_states, predicates):
      if predicate(node_path, value):
        if isinstance(node_impl, PytreeNodeImpl):
          raise ValueError(
            f'Cannot pop key {name!r} from node of type {type(node).__name__}'
          )
        id_to_index[id(value)] = len(id_to_index)
        node_impl.pop_key(node, name)
        if isinstance(value, Variable):
          value = value.to_state()
        state[node_path] = value  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # NOTE: should we raise an error here?
      pass


def _graph_update_dynamic(node: tp.Any, state: tp.Mapping[Key, tp.Any]):
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}')

  node_impl = get_node_impl(node)
  node_dict = node_impl.node_dict(node)
  for key, value in state.items():
    # case 1: new state is being added
    if key not in node_dict:
      if isinstance(node_impl, PytreeNodeImpl):
        raise ValueError(
          f'Cannot set key {key!r} on immutable node of '
          f'type {type(node).__name__}'
        )
      if isinstance(value, Variable):
        value = value.copy()
      node_impl.set_key(node, key, value)
      continue

    # check values are of the same type
    current_value = node_dict[key]

    # case 2: subgraph is being updated
    if is_node(current_value):
      if is_state_leaf(value):
        raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
      _graph_update_dynamic(current_value, value)
    elif isinstance(value, VariableState):
      # case 3: state leaf is being updated
      if not isinstance(current_value, Variable):
        raise ValueError(
          f'Trying to update a non-Variable attribute {key!r} with a Variable: '
          f'{value!r}'
        )
      current_value.copy_from_state(value)
    elif is_state_leaf(value):
      # case 4: state field is being updated
      if isinstance(node_impl, PytreeNodeImpl):
        raise ValueError(
          f'Cannot set key {key!r} on immutable node of '
          f'type {type(node).__name__}'
        )
      node_impl.set_key(node, key, value)
    else:
      raise ValueError(
        f'Unsupported update type: {type(value)} for key {key!r}'
      )


class _StaticModuleStatus(enum.Enum):
  NEW = enum.auto()
  UPDATED = enum.auto()


# TODO(cgarciae): remove once transform init are reimplemented
def update_from(node: Node, updates: Node) -> None:
  graph_update_static(node, updates)
  _, state = split(updates)
  update(node, state)


# TODO(cgarciae): remove once transform init are reimplemented
def graph_update_static(node: Node, updates: Node) -> None:
  cache: dict[int, _StaticModuleStatus] = {}
  _graph_update_static(node, updates, cache, _StaticModuleStatus.UPDATED, ())


def _graph_update_static(
  node: Node,
  updates: Node,
  cache: dict[int, _StaticModuleStatus],
  status: _StaticModuleStatus,
  path: PathParts,
) -> None:
  if type(node) != type(updates):
    raise ValueError(
      f'Trying to update a node with a different type: '
      f'expected {type(node).__name__!r}, '
      f'but got {type(updates).__name__!r}'
    )
  if not is_node(node):
    raise ValueError(f'Unsupported node type: {type(node)}')

  if id(updates) in cache:
    if cache[id(updates)] != status:
      str_path = '/'.join(str(p) for p in path)
      if status is _StaticModuleStatus.NEW:
        raise ValueError(
          f'Trying to add a new node at path {str_path!r} but a'
          ' node with the same reference has been updated'
        )
      else:
        raise ValueError(
          f'Trying to update a node at path {str_path!r} but a new'
          ' node with the same reference has been added'
        )
    return

  cache[id(updates)] = status

  node_impl = get_node_impl(node)
  node_dict = node_impl.node_dict(node)
  updates_dict = node_impl.node_dict(updates)
  for name, value_updates in updates_dict.items():
    # case 1: trying to update a Variable, skip
    if is_state_leaf(value_updates):
      continue
    elif is_node(value_updates):
      # case 2: updating an existing subgraph
      if name in node_dict:
        _graph_update_static(
          node_dict[name],
          value_updates,
          cache,
          _StaticModuleStatus.UPDATED,
          (*path, name),
        )
      else:
        # case 3: adding a new subgraph
        if isinstance(node_impl, PytreeNodeImpl):
          raise ValueError(
            f'Cannot set key {name!r} on immutable node of '
            f'type {type(node).__name__}'
          )

        # check if the subgraph is already in the cache
        if id(value_updates) in cache:
          # if its in the cache, check its status is not NEW
          if cache[id(value_updates)] is not _StaticModuleStatus.NEW:
            raise ValueError(
              f'Trying to add a new node at path {name!r} but a '
              'node with the same reference has been updated'
            )
        else:
          cache[id(value_updates)] = _StaticModuleStatus.NEW

        node_impl.set_key(node, name, value_updates)
    else:  # static field
      if isinstance(node_impl, PytreeNodeImpl):
        if name in node_dict and node_dict[name] == value_updates:
          # if the value is the same, skip
          continue
        # if trying
        raise ValueError(
          f'Cannot update key {name!r} on immutable node of '
          f'type {type(node).__name__}. Current value is {node_dict[name]!r}, '
          f'new value is {value_updates!r}.'
        )

      node_impl.set_key(node, name, value_updates)


# --------------------------------------------------------
# UpdateContext
# --------------------------------------------------------


# --------------------------------------------------------
# UpdateContext
# --------------------------------------------------------


@dataclasses.dataclass
class UpdateContext:
  """A context manager for handling complex state updates."""

  tag: str
  refmap: RefMap[tp.Any, Index] | None
  idxmap: dict[Index, tp.Any] | None

  # define hash and eq to make this an opaque object
  def __hash__(self):
    return 0

  def __eq__(self, other):
    return isinstance(other, UpdateContext)

  @tp.overload
  def split(self, graph_node: A, /) -> tuple[GraphDef[A], GraphState]: ...

  @tp.overload
  def split(
    self, graph_node: A, first: filterlib.Filter, /
  ) -> tuple[GraphDef[A], GraphState]: ...

  @tp.overload
  def split(
    self,
    graph_node: A,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[GraphDef[A], GraphState, tpe.Unpack[tuple[GraphState, ...]]]: ...

  def split(
    self, node: A, *filters: filterlib.Filter
  ) -> tuple[GraphDef[A], GraphState, tpe.Unpack[tuple[GraphState, ...]]]:
    """Split a graph node into a :class:`GraphDef` and one or more :class:`State`s. State is
    a ``Mapping`` from strings or integers to ``Variables``, Arrays or nested States. GraphDef
    contains all the static information needed to reconstruct a ``Module`` graph, it is analogous
    to JAX’s ``PyTreeDef``. :func:`split` is used in conjunction with :func:`merge` to switch
    seamlessly between stateful and stateless representations of the graph.

    Example usage::

      >>> from flax.experimental import nnx
      >>> import jax, jax.numpy as jnp
      ...
      >>> class Foo(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...
      >>> node = Foo(nnx.Rngs(0))
      >>> graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
      ...
      >>> jax.tree.map(jnp.shape, params)
      State({
        'batch_norm': {
          'bias': VariableState(
            type=Param,
            value=(2,)
          ),
          'scale': VariableState(
            type=Param,
            value=(2,)
          )
        },
        'linear': {
          'bias': VariableState(
            type=Param,
            value=(3,)
          ),
          'kernel': VariableState(
            type=Param,
            value=(2, 3)
          )
        }
      })
      >>> jax.tree.map(jnp.shape, batch_stats)
      State({
        'batch_norm': {
          'mean': VariableState(
            type=BatchStat,
            value=(2,)
          ),
          'var': VariableState(
            type=BatchStat,
            value=(2,)
          )
        }
      })

    Arguments:
      node: graph node to split.
      *filters: some optional filters to group the state into mutually exclusive substates.
    Returns:
      :class:`GraphDef` and one or more :class:`State`'s equal to the number of filters passed. If no
      filters are passed, a single :class:`State` is returned.
    """
    graphdef, state, refmap = flatten(node, idxmap=self.idxmap)

    states: GraphState | tuple[GraphState, ...]
    if len(filters) == 0:
      states = (state,)
    elif len(filters) == 1:
      states = (state.split(filters[0]),)
    else:
      states = state.split(filters[0], filters[1], *filters[2:])

    if self.refmap is None:
      self.refmap = refmap

    if graphdef.index_mapping is not None:
      # clear idxmap to remove any references to tracers
      self.idxmap = None

    return graphdef, states[0], *states[1:]

  def merge(
    self,
    graphdef: GraphDef[A],
    state: GraphState,
    *states: GraphState,
  ) -> A:
    """merge"""
    if self.refmap is None:
      raise ValueError('Cannot update a graphdef without refmap.')

    if states:
      state = GraphState.merge(state, *states)

    if graphdef.index_mapping is None:
      node, self.idxmap = unflatten(graphdef, state)
    else:
      index_to_ref = compose_mapping_reversed(
        self.refmap, graphdef.index_mapping
      )
      node, _idxmap = unflatten(graphdef, state, idxmap=index_to_ref)
      # clear references
      self.refmap = None
      self.idxmap = None

    return node


jax.tree_util.register_static(UpdateContext)


@dataclasses.dataclass
class UpdateContextManager:
  tag: str

  def __enter__(self):
    ctx = UpdateContext(self.tag, None, None)
    if self.tag not in GRAPH_CONTEXT.update_context_stacks:
      GRAPH_CONTEXT.update_context_stacks[self.tag] = [ctx]
    else:
      GRAPH_CONTEXT.update_context_stacks[self.tag].append(ctx)
    return ctx

  def __exit__(self, *args):
    if self.tag not in GRAPH_CONTEXT.update_context_stacks:
      raise RuntimeError(
          f'No update context found for tag {self.tag!r}, this is a bug.'
      )
    stack = GRAPH_CONTEXT.update_context_stacks[self.tag]

    ctx = stack.pop()
    # clear references
    ctx.refmap = None
    ctx.idxmap = None

    if not stack:
      del GRAPH_CONTEXT.update_context_stacks[self.tag]

  def __call__(self, f: F) -> F:
    @functools.wraps(f)
    def update_context_manager_wrapper(*args, **kwargs):
      with self:
        return f(*args, **kwargs)

    return update_context_manager_wrapper  # type: ignore


def update_context(tag: str):
  """Creates an :class:`UpdateContext` context manager which can be used to handle
  more complex state updates beyond what ``nnx.update`` can handle, including
  updates to static properties and graph structure.

  UpdateContext exposes a ``split`` and ``merge`` API with the same
  signature as ``nnx.split`` / ``nnx.merge`` but performs some bookkeeping
  to have the necessary information in order to perfectly update the input
  objects based on the changes made inside the transform. The UpdateContext
  must call split and merge a total of 4 times, the first
  and last calls happen outside the transform and the second and third calls
  happen inside the transform as shown in the diagram below::


                          idxmap
    (2) merge ─────────────────────────────► split (3)
          ▲                                    │
          │               inside               │
          │. . . . . . . . . . . . . . . . . . │ index_mapping
          │               outside              │
          │                                    ▼
    (1) split──────────────────────────────► merge (4)
                          refmap


  The first call to split ``(1)`` creates a ``refmap`` which keeps track of the
  outer references, and the first call to merge ``(2)`` creates an ``idxmap`` which
  keeps track of the inner references. The second call to split ``(3)`` combines
  the refmap and idxmap to produce the ``index_mapping`` which indicates
  how the outer references map to the inner references. Finally, the last call to
  merge ``(4)`` uses the index_mapping and the refmap to reconstruct the
  output of the transform while reusing/updating the inner references. To avoid
  memory leaks, the idxmap is cleared after ``(3)`` and the refmap is
  cleared after ``(4)``, and both are cleared after the context manager exits.

  Here is a simple example showing the use of ``update_context``::

    >>> from flax import nnx
    ...
    >>> m1 = nnx.Dict({})
    >>> with nnx.update_context('example') as ctx:
    ...   graphdef, state = ctx.split(m1)
    ...   @jax.jit
    ...   def f(graphdef, state):
    ...     m2 = ctx.merge(graphdef, state)
    ...     m2.a = 1
    ...     m2.ref = m2  # create a reference cycle
    ...     return ctx.split(m2)
    ...   graphdef_out, state_out = f(graphdef, state)
    ...   m3 = ctx.merge(graphdef_out, state_out)
    ...
    >>> assert m1 is m3
    >>> assert m1.a == 1
    >>> assert m1.ref is m1

  Note that ``update_context`` takes in a ``tag`` argument which is used
  primarily as a safety mechanism reduce the risk of accidentally using the
  wrong UpdateContext when using :func:`current_update_context` to access the
  current active context. current_update_context can be used as a way of
  accessing the current active context without having to pass it as a capture::

    >>> from flax import nnx
    ...
    >>> m1 = nnx.Dict({})
    >>> @jax.jit
    ... def f(graphdef, state):
    ...   ctx = nnx.current_update_context('example')
    ...   m2 = ctx.merge(graphdef, state)
    ...   m2.a = 1     # insert static attribute
    ...   m2.ref = m2  # create a reference cycle
    ...   return ctx.split(m2)
    ...
    >>> @nnx.update_context('example')
    ... def g(m1):
    ...   ctx = nnx.current_update_context('example')
    ...   graphdef, state = ctx.split(m1)
    ...   graphdef_out, state_out = f(graphdef, state)
    ...   return ctx.merge(graphdef_out, state_out)
    ...
    >>> m3 = g(m1)
    >>> assert m1 is m3
    >>> assert m1.a == 1
    >>> assert m1.ref is m1

  As shown in the code above, ``update_context`` can also be used as a
  decorator that creates/activates an UpdateContext context for the
  duration of the function. The context can be accessed using
  :func:`current_update_context`.

  Args:
    tag: A string tag to identify the context.
  """
  return UpdateContextManager(tag)


def current_update_context(tag: str) -> UpdateContext:
  """Returns the current active :class:`UpdateContext` for the given tag."""
  if tag not in GRAPH_CONTEXT.update_context_stacks:
    raise ValueError(f'No update context found for tag {tag!r}.')
  return GRAPH_CONTEXT.update_context_stacks[tag][-1]


# --------------------------------------------------------
# Functional API
# --------------------------------------------------------


@tp.overload
def split(graph_node: A, /) -> tuple[GraphDef[A], GraphState]: ...


@tp.overload
def split(
  graph_node: A,
  first: filterlib.Filter,
  /,
) -> tuple[GraphDef[A], GraphState]: ...


@tp.overload
def split(
  graph_node: A,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[GraphDef[A], GraphState, tpe.Unpack[tuple[GraphState, ...]]]: ...


def split(
  node: A, *filters: filterlib.Filter
) -> tuple[GraphDef[A], GraphState, tpe.Unpack[tuple[GraphState, ...]]]:
  """Split a graph node into a :class:`GraphDef` and one or more :class:`State`s. State is
  a ``Mapping`` from strings or integers to ``Variables``, Arrays or nested States. GraphDef
  contains all the static information needed to reconstruct a ``Module`` graph, it is analogous
  to JAX’s ``PyTreeDef``. :func:`split` is used in conjunction with :func:`merge` to switch
  seamlessly between stateful and stateless representations of the graph.

  Example usage::

    >>> from flax.experimental import nnx
    >>> import jax, jax.numpy as jnp
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...
    >>> node = Foo(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
    ...
    >>> jax.tree.map(jnp.shape, params)
    State({
      'batch_norm': {
        'bias': VariableState(
          type=Param,
          value=(2,)
        ),
        'scale': VariableState(
          type=Param,
          value=(2,)
        )
      },
      'linear': {
        'bias': VariableState(
          type=Param,
          value=(3,)
        ),
        'kernel': VariableState(
          type=Param,
          value=(2, 3)
        )
      }
    })
    >>> jax.tree.map(jnp.shape, batch_stats)
    State({
      'batch_norm': {
        'mean': VariableState(
          type=BatchStat,
          value=(2,)
        ),
        'var': VariableState(
          type=BatchStat,
          value=(2,)
        )
      }
    })

  :func:`split` and :func:`merge` are primarily used to interact directly with JAX
  transformations, see
  `Functional API <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Arguments:
    node: graph node to split.
    *filters: some optional filters to group the state into mutually exclusive substates.
  Returns:
    ``GraphDef`` and one or more ``States`` equal to the number of filters passed. If no
    filters are passed, a single ``State`` is returned.
  """
  graphdef, state, _ = flatten(node)

  states: GraphState | tuple[GraphState, ...]
  if len(filters) == 0:
    states = (state,)
  elif len(filters) == 1:
    states = (state.split(filters[0]),)
  else:
    states = state.split(filters[0], filters[1], *filters[2:])

  return graphdef, states[0], *states[1:]


def merge(
  graphdef: GraphDef[A],
  state: GraphState,
  /,
  *states: GraphState,
) -> A:
  """The inverse of :func:`split`.

  ``merge`` takes a :class:`GraphDef` and one or more :class:`State`'s and creates
  a new node with the same structure as the original node.

  Example usage::

    >>> from flax.experimental import nnx
    >>> import jax, jax.numpy as jnp
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...
    >>> node = Foo(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(node, nnx.Param, nnx.BatchStat)
    ...
    >>> new_node = nnx.merge(graphdef, params, batch_stats)
    >>> assert isinstance(new_node, Foo)
    >>> assert isinstance(new_node.batch_norm, nnx.BatchNorm)
    >>> assert isinstance(new_node.linear, nnx.Linear)

  :func:`split` and :func:`merge` are primarily used to interact directly with JAX
  transformations, see
  `Functional API <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Args:
    graphdef: A :class:`GraphDef` object.
    state: A :class:`State` object.
    *states: Additional :class:`State` objects.
  Returns:
    The merged :class:`Module`.
  """
  if states:
    state = GraphState.merge(state, *states)

  node, _ = unflatten(graphdef, state)
  return node


def update(node, state: State, /, *states: State) -> None:
  """Update the given graph node with a new :class:`State` in-place.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 3))
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    >>> def loss_fn(model, x, y):
    ...   return jnp.mean((y - model(x))**2)
    >>> prev_loss = loss_fn(model, x, y)

    >>> grads = nnx.grad(loss_fn)(model, x, y)
    >>> new_state = jax.tree.map(lambda p, g: p - 0.1*g, nnx.state(model), grads)
    >>> nnx.update(model, new_state)
    >>> assert loss_fn(model, x, y) < prev_loss

  Args:
    node: A graph node to update.
    state: A :class:`State` object.
    *states: Additional :class:`State` objects.
  """
  if states:
    state = GraphState.merge(state, *states)

  _graph_update_dynamic(node, state.raw_mapping)


@tp.overload
def state(node, /) -> GraphState: ...


@tp.overload
def state(node, first: filterlib.Filter, /) -> GraphState: ...


@tp.overload
def state(
  node,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[GraphState, ...]: ...


def state(
  node,
  *filters: filterlib.Filter,
) -> tp.Union[GraphState, tuple[GraphState, ...]]:
  """Similar to :func:`split` but only returns the :class:`State`'s indicated by the filters.

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batch_norm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> # get the learnable parameters from the batch norm and linear layer
    >>> params = nnx.state(model, nnx.Param)
    >>> # get the batch statistics from the batch norm layer
    >>> batch_stats = nnx.state(model, nnx.BatchStat)
    >>> # get them separately
    >>> params, batch_stats = nnx.state(model, nnx.Param, nnx.BatchStat)
    >>> # get them together
    >>> state = nnx.state(model)

  Args:
    node: A graph node object.
    *filters: One or more :class:`Variable` objects to filter by.
  Returns:
    One or more :class:`State` mappings.
  """
  state = flatten(node)[1]

  states: GraphState | tuple[GraphState, ...]
  if len(filters) == 0:
    states = state
  elif len(filters) == 1:
    states = state.filter(filters[0])
  else:
    states = state.filter(filters[0], filters[1], *filters[2:])

  return states


def graphdef(node: tp.Any, /) -> GraphDef[tp.Any]:
  """Get the :class:`GraphDef` of the given graph node.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, _ = nnx.split(model)
    >>> assert graphdef == nnx.graphdef(model)

  Args:
    node: A graph node object.
  Returns:
    The :class:`GraphDef` of the :class:`Module` object.
  """
  graphdef, _, _ = flatten(node)
  return graphdef


@tp.overload
def pop(
  node,
  filter: filterlib.Filter,
  /,
) -> GraphState: ...


@tp.overload
def pop(
  node,
  filter: filterlib.Filter,
  filter2: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[GraphState, ...]: ...


def pop(
  node, *filters: filterlib.Filter
) -> tp.Union[GraphState, tuple[GraphState, ...]]:
  """Pop one or more :class:`Variable` types from the graph node.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     self.sow(nnx.Intermediate, 'i', x)
    ...     x = self.linear2(x)
    ...     return x

    >>> x = jnp.ones((1, 2))
    >>> model = Model(rngs=nnx.Rngs(0))
    >>> assert not hasattr(model, 'i')
    >>> y = model(x)
    >>> assert hasattr(model, 'i')

    >>> intermediates = nnx.pop(model, nnx.Intermediate)
    >>> assert intermediates['i'].value[0].shape == (1, 3)
    >>> assert not hasattr(model, 'i')

  Args:
    node: A graph node object.
    *filters: One or more :class:`Variable` objects to filter by.
  Returns:
    The popped :class:`State` containing the :class:`Variable`
    objects that were filtered for.
  """
  if len(filters) == 0:
    raise ValueError('Expected at least one filter')

  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[GraphFlatState, ...] = tuple({} for _ in predicates)
  _graph_pop(
    node=node,
    id_to_index=id_to_index,
    path_parts=path_parts,
    flat_states=flat_states,
    predicates=predicates,
  )
  states = tuple(
    GraphState.from_flat_path(flat_state) for flat_state in flat_states
  )

  if len(states) == 1:
    return states[0]
  else:
    return states


def clone(node: Node) -> Node:
  """Create a deep copy of the given graph node.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> cloned_model = nnx.clone(model)
    >>> model.bias.value += 1
    >>> assert (model.bias.value != cloned_model.bias.value).all()

  Args:
    node: A graph node object.
  Returns:
    A deep copy of the :class:`Module` object.
  """
  graphdef, state = split(node)
  return merge(graphdef, state)


def iter_graph(node: tp.Any, /) -> tp.Iterator[tuple[PathParts, tp.Any]]:
  """Iterates over all nested nodes and leaves of the given graph node, including the current node.

  ``iter_graph`` creates a generator that yields path and value pairs, where
  the path is a tuple of strings or integers representing the path to the value from the
  root. Repeated nodes are visited only once. Leaves include static values.

  Example::
    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> class Linear(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.din, self.dout = din, dout
    ...     self.w = nnx.Param(jax.random.uniform(rngs.next(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...
    >>> module = Linear(3, 4, rngs=nnx.Rngs(0))
    >>> graph = [module, module]
    ...
    >>> for path, value in nnx.iter_graph(graph):
    ...   print(path, type(value).__name__)
    ...
    (0, 'b') Param
    (0, 'din') int
    (0, 'dout') int
    (0, 'w') Param
    (0,) Linear
    () list
  """
  visited: set[int] = set()
  path_parts: PathParts = ()
  yield from _iter_graph(node, visited, path_parts)


def _iter_graph(
  node: tp.Any, visited: set[int], path_parts: PathParts
) -> tp.Iterator[tuple[PathParts, tp.Any]]:
  if is_node(node):
    if id(node) in visited:
      return
    visited.add(id(node))
    node_dict = get_node_impl(node).node_dict(node)
    for key, value in node_dict.items():
      yield from _iter_graph(value, visited, (*path_parts, key))

  yield path_parts, node


def compose_mapping(
  map_ab: tp.Mapping[A, B], map_bc: tp.Mapping[B, C], /
) -> dict[A, C]:
  return {a: map_bc[b] for a, b in map_ab.items() if b in map_bc}


def compose_mapping_reversed(
  map_ab: tp.Mapping[A, B], map_bc: tp.Mapping[B, C], /
) -> dict[C, A]:
  return {map_bc[b]: a for a, b in map_ab.items() if b in map_bc}


@dataclasses.dataclass(frozen=True)
class Static(tp.Generic[A]):
  """An empty pytree node that treats its inner value as static.
  ``value`` must define ``__eq__`` and ``__hash__``.
  """

  value: A


jax.tree_util.register_static(Static)

# ---------------------------------------------------------
# Pytree
# ---------------------------------------------------------
class PytreeType: ...


def is_pytree_node(x: tp.Any) -> bool:
  return not jax.tree_util.all_leaves([x])


def _key_path_to_key(key: tp.Any) -> Key:
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(
    key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
  ):
    if not isinstance(key.key, Key):
      raise ValueError(
        f'Invalid key: {key.key}. May be due to its type not being hashable or comparable.'
      )
    return key.key
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return key.name
  else:
    return str(key)


def _flatten_pytree(pytree: tp.Any):
  leaves, treedef = jax.tree_util.tree_flatten_with_path(
    pytree, is_leaf=lambda x: x is not pytree
  )
  nodes = tuple((_key_path_to_key(path[0]), value) for path, value in leaves)

  return nodes, treedef


def _unflatten_pytree(
  nodes: tuple[tuple[Key, tp.Any], ...], treedef: jax.tree_util.PyTreeDef
):
  pytree = treedef.unflatten(value for _, value in nodes)
  return pytree


PYTREE_NODE_IMPL = PytreeNodeImpl(
  type=PytreeType,
  flatten=_flatten_pytree,
  unflatten=_unflatten_pytree,
)

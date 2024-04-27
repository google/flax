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
import threading
import typing as tp
from abc import ABCMeta
from copy import deepcopy


import jax
import numpy as np
import typing_extensions as tpe

from flax.experimental.nnx.nnx import (
  errors,
  filterlib,
  reprlib,
  tracers,
)
from flax.experimental.nnx.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.experimental.nnx.nnx.state import (
  FlatState,
  State,
  StateLeaf,
  is_state_leaf,
)
from flax.experimental.nnx.nnx.variables import Variable, VariableState
from flax.typing import PathParts, Key

A = tp.TypeVar('A')
B = tp.TypeVar('B')
C = tp.TypeVar('C')
G = tp.TypeVar('G', bound='GraphNode')
HA = tp.TypeVar('HA', bound=tp.Hashable)
HB = tp.TypeVar('HB', bound=tp.Hashable)

Index = int
Names = tp.Sequence[int]
Node = tp.TypeVar('Node')
Leaf = tp.TypeVar('Leaf')
AuxData = tp.TypeVar('AuxData')

Updates = tp.Union[
  A,
  'GraphDef[A]',
  tuple['GraphDef[A]', State],
  tuple['GraphDef[A]', tuple[State, ...]],
  State,
  tuple[State, ...],
]

NodeLeaf = tp.Union[Variable[tp.Any], np.ndarray, jax.Array]


def is_node_leaf(x: tp.Any) -> tpe.TypeGuard[NodeLeaf]:
  return isinstance(x, (Variable, np.ndarray, jax.Array))


@dataclasses.dataclass
class GraphUtilsContext(threading.local):
  node_types: dict[
    type, 'NodeImpl[tp.Any, tp.Any, tp.Any]'
  ] = dataclasses.field(default_factory=dict)
  seen_modules_repr: set[int] | None = None


CONTEXT = GraphUtilsContext()


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
  clear: tp.Callable[[Node, AuxData], None]

  def init(self, node: Node, items: tuple[tuple[Key, Leaf], ...]):
    for key, value in items:
      self.set_key(node, key, value)


@dataclasses.dataclass(frozen=True)
class PytreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  unflatten: tp.Callable[[tuple[tuple[Key, Leaf], ...], AuxData], Node]


NodeImpl = tp.Union[
  GraphNodeImpl[Node, Leaf, AuxData], PytreeNodeImpl[Node, Leaf, AuxData]
]


def register_graph_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
  set_key: tp.Callable[[Node, Key, Leaf], None],
  pop_key: tp.Callable[[Node, Key], Leaf],
  create_empty: tp.Callable[[AuxData], Node],
  clear: tp.Callable[[Node, AuxData], None],
):
  CONTEXT.node_types[type] = GraphNodeImpl(
    type=type,
    flatten=flatten,
    set_key=set_key,
    pop_key=pop_key,
    create_empty=create_empty,
    clear=clear,
  )


def is_node(x: tp.Any) -> bool:
  if type(x) in CONTEXT.node_types:
    return True
  return is_pytree_node(x)


def is_graph_node(x: tp.Any) -> bool:
  return type(x) in CONTEXT.node_types


def is_node_type(x: type[tp.Any]) -> bool:
  return x in CONTEXT.node_types or x is PytreeType


def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any]:
  if isinstance(x, Variable):
    raise ValueError(f'Variable is not a node: {x}')

  node_type = type(x)

  if node_type not in CONTEXT.node_types:
    if is_pytree_node(x):
      return PYTREE_NODE_IMPL
    else:
      raise ValueError(f'Unknown node type: {x}')

  return CONTEXT.node_types[node_type]


def get_node_impl_for_type(x: type[Node]) -> NodeImpl[Node, tp.Any, tp.Any]:
  if x is PytreeType:
    return PYTREE_NODE_IMPL
  return CONTEXT.node_types[x]


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
  subgraphs: _HashableMapping[Key, tp.Union['NodeDef[tp.Any]', Index]]
  static_fields: _HashableMapping[Key, tp.Any]
  variables: _HashableMapping[Key, Index]
  metadata: tp.Any

  @classmethod
  def create(
    cls,
    type: tp.Type[Node],
    index: int,
    attributes: tuple[Key, ...],
    subgraphs: tp.Iterable[tuple[Key, tp.Union['GraphDef[tp.Any]', Index]]],
    static_fields: tp.Iterable[tuple[Key, tp.Any]],
    variables: tp.Iterable[tuple[Key, Index]],
    metadata: tp.Any,
  ):
    return cls(
      type=type,
      index=index,
      attributes=attributes,
      subgraphs=_HashableMapping(subgraphs),
      static_fields=_HashableMapping(static_fields),
      variables=_HashableMapping(variables),
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
    yield reprlib.Attr('variables', reprlib.PrettyMapping(self.variables))
    yield reprlib.Attr('metadata', self.metadata)


@dataclasses.dataclass(frozen=True, repr=False)
class GraphDef(tp.Generic[Node], reprlib.Representable):
  nodedef: NodeDef[Node]
  index_mapping: dict[Index, Index] | None

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('nodedef', self.nodedef)
    yield reprlib.Attr('index_mapping', self.index_mapping)

  def __deepcopy__(self, memo=None):
    nodedef = deepcopy(self.nodedef, memo)
    index_mapping = deepcopy(self.index_mapping, memo)
    return GraphDef(nodedef, index_mapping)

  def __hash__(self):
    return hash(self.nodedef)

  def __eq__(self, other):
    return isinstance(other, GraphDef) and self.nodedef == other.nodedef

  def apply(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple['GraphDef[Node]', State]]:
    accessor = DelayedAccessor()

    def _apply(
      accessor: DelayedAccessor, *args, **kwargs
    ) -> tuple[tp.Any, tuple[GraphDef[Node], State]]:
      module = merge(self, state, *states)
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, flatten(module)[:2]

    return CallableProxy(_apply, accessor)  # type: ignore

  def make_empty(self) -> Node:
    return merge(self, State({}))


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
) -> tuple[GraphDef[Node], State, RefMap[tp.Any, Index]]:
  refmap = RefMap[tp.Any, Index]()
  flat_state: dict[PathParts, StateLeaf] = {}
  nodedef = _graph_flatten((), refmap, flat_state, x)
  assert not isinstance(nodedef, int)
  if idxmap is not None:
    index_to_index = compose_mapping(idxmap, refmap)
  else:
    index_to_index = None
  graphdef = GraphDef(nodedef, index_to_index)
  return graphdef, State.from_flat_path(flat_state), refmap


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

  subgraphs: list[tuple[Key, tp.Union[NodeDef[Node], int]]] = []
  static_fields: list[tuple[Key, tp.Any]] = []
  variables: list[tuple[Key, int]] = []

  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if is_node(value):
      nodedef = _graph_flatten((*path, key), refmap, flat_state, value)
      subgraphs.append((key, nodedef))
    elif isinstance(value, Variable):
      if value in refmap:
        variables.append((key, refmap[value]))
      else:
        flat_state[(*path, key)] = value.to_state()
        variable_index = refmap[value] = len(refmap)
        variables.append((key, variable_index))
    elif is_state_leaf(value):
      flat_state[(*path, key)] = value
    else:
      static_fields.append((key, value))

  nodedef = NodeDef.create(
    type=node_impl.type,
    index=index,
    attributes=tuple(key for key, _ in values),
    subgraphs=subgraphs,
    static_fields=static_fields,
    variables=variables,
    metadata=metadata,
  )
  return nodedef


def unflatten(
  graphdef: GraphDef[Node],
  state: State,
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
  state: dict[Key, StateLeaf | dict[Key, tp.Any]],
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
    children: dict[Key, NodeLeaf | Node] = {}

    if unkown_keys := set(state) - set(nodedef.attributes):
      raise ValueError(f'Unknown keys: {unkown_keys}')

    for key in nodedef.attributes:
      if key in nodedef.static_fields:
        children[key] = nodedef.static_fields[key]
      elif key not in state:
        # TODO(cgarcia): maybe we shouldn't support unflattening with missing keys?
        # if key is not present create an empty types
        if key in nodedef.subgraphs:
          # if the key is a subgraph we create an empty node
          subgraphdef = nodedef.subgraphs[key]
          if isinstance(subgraphdef, int):
            # subgraph exists, take it from the cache
            children[key] = index_to_ref[subgraphdef]
          else:
            # create an empty node
            substate = {}
            children[key] = _graph_unflatten(
              subgraphdef, substate, index_to_ref, idxmap
            )
        elif key in nodedef.variables:
          variable_index = nodedef.variables[key]
          if variable_index in index_to_ref:
            # variable exists, take it from the cache
            children[key] = index_to_ref[variable_index]
          else:
            # key for a variable is missing, raise an error
            raise ValueError(
              f'Expected key for Variable but was not found in state: {key!r}'
            )
        else:
          raise RuntimeError(f'Unknown static field: {key!r}')
      else:
        value = state[key]
        if key in nodedef.subgraphs:
          if is_state_leaf(value):
            raise ValueError(
              f'Expected value of type {nodedef.subgraphs[key]} for '
              f'{key!r}, but got {value!r}'
            )
          assert isinstance(value, dict)
          subgraphdef = nodedef.subgraphs[key]

          if isinstance(subgraphdef, int):
            node = index_to_ref[subgraphdef]
          else:
            node = children[key] = _graph_unflatten(
              subgraphdef, value, index_to_ref, idxmap
            )

        elif key in nodedef.variables:
          variable_index = nodedef.variables[key]
          if variable_index in index_to_ref:
            children[key] = index_to_ref[variable_index]
          else:
            if not isinstance(value, VariableState):
              raise ValueError(
                f'Expected a Variable type for {key!r}, but got {type(value)}.'
              )
            if idxmap is not None and variable_index in idxmap:
              variable = idxmap[variable_index]
              if not isinstance(variable, Variable):
                raise ValueError(
                  f'Expected a Variable type for {key!r}, but got {type(variable)}.'
                )
              variable.copy_from_state(value)
            else:
              assert isinstance(value, VariableState)
              variable = value.to_variable()
            children[key] = variable
            index_to_ref[variable_index] = variable
        elif is_state_leaf(value):
          if isinstance(value, VariableState):
            value = value.to_variable()
          children[key] = value
        else:
          raise RuntimeError

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
      node_impl.clear(node, nodedef.metadata)
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
) -> tuple[State, ...]:
  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[FlatState, ...] = tuple({} for _ in predicates)
  _graph_pop(node, id_to_index, path_parts, flat_states, predicates)
  return tuple(State.from_flat_path(flat_state) for flat_state in flat_states)


def _graph_pop(
  node: tp.Any,
  id_to_index: dict[int, Index],
  path_parts: PathParts,
  flat_states: tuple[FlatState, ...],
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
        state[node_path] = value
        break
    else:
      # NOTE: should we raise an error here?
      pass


def _graph_update_dynamic(node: tp.Any, state: dict[Key, tp.Any]):
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
      str_path = '/'.join(path)
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


@dataclasses.dataclass
class UpdateContext:
  refmap: RefMap[tp.Any, Index] | None = None
  idxmap: dict[Index, tp.Any] | None = None

  # define context manager to clean up refmap and idxmap
  # on exit
  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    self.refmap = None
    self.idxmap = None

  # define hash and eq to make this an opaque object
  def __hash__(self):
    return 0

  def __eq__(self, other):
    return isinstance(other, UpdateContext)

  @tp.overload
  def split(self, graph_node: A, /) -> tuple[GraphDef[A], State]:
    ...

  @tp.overload
  def split(
    self, graph_node: A, first: filterlib.Filter, /
  ) -> tuple[GraphDef[A], State]:
    ...

  @tp.overload
  def split(
    self,
    graph_node: A,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[GraphDef[A], State, tpe.Unpack[tuple[State, ...]]]:
    ...

  def split(
    self, node: A, *filters: filterlib.Filter
  ) -> tuple[GraphDef[A], State, tpe.Unpack[tuple[State, ...]]]:
    if self.refmap is not None and self.idxmap is None:
      raise ValueError(
        "'merge' was not called in-between the first and second call to 'split'"
      )
    graphdef, state, refmap = flatten(node, idxmap=self.idxmap)

    if len(filters) == 0:
      states = (state,)
    elif len(filters) == 1:
      states = (state.split(filters[0]),)
    else:
      states = state.split(filters[0], filters[1], *filters[2:])

    if self.refmap is None:
      self.refmap = refmap

    return graphdef, states[0], *states[1:]

  def merge(
    self,
    graphdef: GraphDef[A],
    state: State,
    *states: State,
  ) -> A:
    # TODO: add docstring of example usage
    if states:
      state = State.merge(state, *states)

    node, self.idxmap = unflatten(graphdef, state)
    return node

  def update(
    self,
    new_graphdef: GraphDef[A],
    state: State,
    /,
    *states: State,
  ):
    if self.refmap is None:
      raise ValueError('Cannot update a graphdef without refmap.')
    if new_graphdef.index_mapping is None:
      raise ValueError('Cannot update a graphdef without index_mapping.')

    if states:
      state = State.merge(state, *states)

    index_to_ref = compose_mapping_reversed(
      self.refmap, new_graphdef.index_mapping
    )
    return unflatten(new_graphdef, state, idxmap=index_to_ref)[0]


jax.tree_util.register_static(UpdateContext)


@tp.overload
def split(graph_node: A, /) -> tuple[GraphDef[A], State]:
  ...


@tp.overload
def split(
  graph_node: A,
  first: filterlib.Filter,
  /,
) -> tuple[GraphDef[A], State]:
  ...


@tp.overload
def split(
  graph_node: A,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[GraphDef[A], State, tpe.Unpack[tuple[State, ...]]]:
  ...


def split(
  node: A, *filters: filterlib.Filter
) -> tuple[GraphDef[A], State, tpe.Unpack[tuple[State, ...]]]:
  graphdef, state, _ = flatten(node)

  if len(filters) == 0:
    states = (state,)
  elif len(filters) == 1:
    states = (state.split(filters[0]),)
  else:
    states = state.split(filters[0], filters[1], *filters[2:])

  return graphdef, states[0], *states[1:]


def merge(
  graphdef: GraphDef[A],
  state: State,
  /,
  *states: State,
) -> A:
  if states:
    state = State.merge(state, *states)

  node, _ = unflatten(graphdef, state)
  return node


def update(node, state: State, /, *states: State) -> None:
  if states:
    state = State.merge(state, *states)

  _graph_update_dynamic(node, state.raw_mapping)


@tp.overload
def state(node, /) -> State:
  ...


@tp.overload
def state(node, first: filterlib.Filter, /) -> State:
  ...


@tp.overload
def state(
  node,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]:
  ...


def state(
  node,
  *filters: filterlib.Filter,
) -> tp.Union[State, tuple[State, ...]]:
  state = flatten(node)[1]

  if len(filters) == 0:
    states = state
  elif len(filters) == 1:
    states = state.filter(filters[0])
  else:
    states = state.filter(filters[0], filters[1], *filters[1:])

  return states


def graphdef(node: tp.Any, /) -> GraphDef[tp.Any]:
  graphdef, _, _ = flatten(node)
  return graphdef


@tp.overload
def pop(
  node,
  filter: filterlib.Filter,
  /,
) -> State:
  ...


@tp.overload
def pop(
  node,
  filter: filterlib.Filter,
  filter2: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]:
  ...


def pop(node, *filters: filterlib.Filter) -> tp.Union[State, tuple[State, ...]]:
  if len(filters) == 0:
    raise ValueError('Expected at least one filter')

  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[FlatState, ...] = tuple({} for _ in predicates)
  _graph_pop(
    node=node,
    id_to_index=id_to_index,
    path_parts=path_parts,
    flat_states=flat_states,
    predicates=predicates,
  )
  states = tuple(State.from_flat_path(flat_state) for flat_state in flat_states)

  if len(states) == 1:
    return states[0]
  else:
    return states


def clone(node: Node) -> Node:
  graphdef, state = split(node)
  return merge(graphdef, state)


def iter_nodes(node: tp.Any, /) -> tp.Iterator[tuple[PathParts, tp.Any]]:
  visited: set[int] = set()
  path_parts: PathParts = ()
  yield from _iter_nodes(node, visited, path_parts)


def _iter_nodes(
  node: tp.Any, visited: set[int], path_parts: PathParts
) -> tp.Iterator[tuple[PathParts, tp.Any]]:
  if not is_node(node):
    return
  if id(node) in visited:
    return
  visited.add(id(node))
  yield path_parts, node
  node_impl = get_node_impl(node)
  node_dict = node_impl.node_dict(node)
  for key, value in node_dict.items():
    yield from _iter_nodes(value, visited, (*path_parts, key))


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
# insert/extract_graph_nodes API
# ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GraphNodeIndex:
  """Index of a graph node in a Pytree structure."""

  index: Index


jax.tree_util.register_static(GraphNodeIndex)


def extract_graph_nodes(pytree: A, /) -> tuple[A, tuple[tp.Any, ...]]:
  """Extracts all graph nodes from a pytree."""
  nodes = RefMap[tp.Any, Index]()

  def _maybe_extract(x):
    if is_graph_node(x):
      if x not in nodes:
        index = nodes[x] = len(nodes)
      else:
        index = nodes[x]
      return GraphNodeIndex(index)
    return x

  return jax.tree_util.tree_map(_maybe_extract, pytree), tuple(nodes)


def insert_graph_nodes(pytree: A, nodes: tuple[tp.Any, ...], /) -> A:
  """Inserts graph nodes into a pytree."""

  def _maybe_insert(x):
    if isinstance(x, GraphNodeIndex):
      return nodes[x.index]
    return x

  return jax.tree_util.tree_map(
    _maybe_insert, pytree, is_leaf=lambda x: isinstance(x, GraphNodeIndex)
  )


# ---------------------------------------------------------
# GraphNode
# ---------------------------------------------------------


class ModuleState(reprlib.Representable):
  __slots__ = ('_trace_state', '_initializing')

  def __init__(self, initializing: bool = False):
    self._trace_state = tracers.TraceState()
    self._initializing = initializing

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  @property
  def initializing(self) -> bool:
    return self._initializing

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('trace_state', self._trace_state)


class GraphNodeMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _graph_node_meta_call(cls, *args, **kwargs)


def _graph_node_meta_call(cls: tp.Type[G], *args, **kwargs) -> G:
  node = cls.__new__(cls, *args, **kwargs)
  vars(node)['_graph_node__state'] = ModuleState()
  node.__init__(*args, **kwargs)

  return node

@dataclasses.dataclass(frozen=True, repr=False)
class Array:
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  def __repr__(self):
    return f'Array(shape={self.shape}, dtype={self.dtype.name})'


class GraphNode(reprlib.Representable, metaclass=GraphNodeMeta):
  if tp.TYPE_CHECKING:
    _graph_node__state: ModuleState

  def __init_subclass__(cls) -> None:
    super().__init_subclass__()

    register_graph_node_type(
      type=cls,
      flatten=cls._graph_node_flatten,
      set_key=cls._graph_node_set_key,
      pop_key=cls._graph_node_pop_key,
      create_empty=cls._graph_node_create_empty,
      clear=cls._graph_node_clear,
    )

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    self.check_valid_context(
      f"Cannot mutate '{type(self).__name__}' from different trace level"
    )
    object.__setattr__(self, name, value)

  def check_valid_context(self, error_msg: str) -> None:
    if not self._graph_node__state.trace_state.is_valid():
      raise errors.TraceContextError(error_msg)

  def __deepcopy__(self: G, memo=None) -> G:
    graphdef, state = split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return merge(graphdef, state)

  def __nnx_repr__(self):
    if CONTEXT.seen_modules_repr is None:
      CONTEXT.seen_modules_repr = set()
      clear_seen = True
    else:
      clear_seen = False

    if id(self) in CONTEXT.seen_modules_repr:
      yield reprlib.Object(type=type(self), empty_repr='...')
      return

    yield reprlib.Object(type=type(self))
    CONTEXT.seen_modules_repr.add(id(self))

    try:
      for name, value in vars(self).items():
        if name.startswith('_'):
          continue

        def to_shape_dtype(value):
          if isinstance(value, Variable):
            return value.replace(
              raw_value=jax.tree.map(to_shape_dtype, value.raw_value)
            )
          elif isinstance(value, (np.ndarray, jax.Array)):
            return Array(value.shape, value.dtype)
          return value

        value = jax.tree.map(to_shape_dtype, value)
        yield reprlib.Attr(name, repr(value))
    finally:
      if clear_seen:
        CONTEXT.seen_modules_repr = None

  # Graph Definition
  def _graph_node_flatten(self):
    nodes = sorted(
      (key, value)
      for key, value in vars(self).items()
      if key != '_graph_node__state'
    )
    return nodes, type(self)

  def _graph_node_set_key(self, key: Key, value: tp.Any):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    elif (
      hasattr(self, key)
      and isinstance(variable := getattr(self, key), Variable)
      and isinstance(value, VariableState)
    ):
      variable.copy_from_state(value)
    else:
      setattr(self, key, value)

  def _graph_node_pop_key(self, key: Key):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    return vars(self).pop(key)

  @staticmethod
  def _graph_node_create_empty(node_type: tp.Type[G]) -> G:
    node = object.__new__(node_type)
    vars(node).update(_graph_node__state=ModuleState())
    return node

  def _graph_node_clear(self, cls: tp.Type[G]):
    module_state = self._graph_node__state
    module_vars = vars(self)
    module_vars.clear()
    module_vars['_graph_node__state'] = module_state


# ---------------------------------------------------------
# Pytree
# ---------------------------------------------------------
class PytreeType:
  ...


def is_pytree_node(x: tp.Any) -> bool:
  return not jax.tree_util.all_leaves([x])


def _key_path_to_key(key: tp.Any) -> Key:
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(
    key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
  ):
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

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

import contextlib
import dataclasses
import functools
import threading
import typing as tp

import jax
import numpy as np
import typing_extensions as tpe

from flax.nnx import filterlib, reprlib
from flax.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.statelib import State
from flax.nnx import variablelib
from flax.nnx.variablelib import Variable, VariableState
from flax.typing import Key, PathParts, is_key_like

A = tp.TypeVar('A')
B = tp.TypeVar('B')
C = tp.TypeVar('C')
F = tp.TypeVar('F', bound=tp.Callable)

HA = tp.TypeVar('HA', bound=tp.Hashable)
HB = tp.TypeVar('HB', bound=tp.Hashable)
KeyT = tp.TypeVar('KeyT', bound=Key)

Index = int
Names = tp.Sequence[int]
Node = tp.TypeVar('Node')
Leaf = tp.TypeVar('Leaf')
AuxData = tp.TypeVar('AuxData')

StateLeaf = VariableState[tp.Any]
NodeLeaf = Variable[tp.Any]
GraphState = State[Key, StateLeaf]


def is_state_leaf(x: tp.Any) -> tpe.TypeGuard[StateLeaf]:
  return isinstance(x, VariableState)


def is_node_leaf(x: tp.Any) -> tpe.TypeGuard[NodeLeaf]:
  return isinstance(x, Variable)


class RefMap(tp.MutableMapping[A, B], reprlib.MappingReprMixin[A, B]):
  """A mapping that uses object id as the hash for the keys."""

  def __init__(
    self, mapping: tp.Mapping[A, B] | tp.Iterable[tuple[A, B]] = (), /
  ):
    self._mapping: dict[int, tuple[A, B]] = {}
    self.update(mapping)

  def __getitem__(self, key: A) -> B:
    return self._mapping[id(key)][1]

  def __contains__(self, key: object) -> bool:
    return id(key) in self._mapping

  def __setitem__(self, key: A, value: B):
    self._mapping[id(key)] = (key, value)

  def __delitem__(self, key: A):
    del self._mapping[id(key)]

  def __iter__(self) -> tp.Iterator[A]:
    return (key for key, _ in self._mapping.values())

  def __len__(self) -> int:
    return len(self._mapping)

  def __str__(self) -> str:
    return repr(self)


@dataclasses.dataclass(frozen=True, slots=True)
class NodeImplBase(tp.Generic[Node, Leaf, AuxData]):
  type: type[Node]
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]]

  def node_dict(self, node: Node) -> dict[Key, Leaf]:
    nodes, _ = self.flatten(node)
    return dict(nodes)


@dataclasses.dataclass(frozen=True, slots=True)
class GraphNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  set_key: tp.Callable[[Node, Key, Leaf], None]
  pop_key: tp.Callable[[Node, Key], Leaf]
  create_empty: tp.Callable[[AuxData], Node]
  clear: tp.Callable[[Node], None]
  init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None]


@dataclasses.dataclass(frozen=True, slots=True)
class PytreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  unflatten: tp.Callable[[tp.Sequence[tuple[Key, Leaf]], AuxData], Node]


NodeImpl = tp.Union[
  GraphNodeImpl[Node, Leaf, AuxData], PytreeNodeImpl[Node, Leaf, AuxData]
]


GRAPH_REGISTRY: dict[type, NodeImpl[tp.Any, tp.Any, tp.Any]] = {}
PYTREE_REGISTRY: dict[type, PytreeNodeImpl[tp.Any, tp.Any, tp.Any]] = {}


def register_graph_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
  set_key: tp.Callable[[Node, Key, Leaf], None],
  pop_key: tp.Callable[[Node, Key], Leaf],
  create_empty: tp.Callable[[AuxData], Node],
  clear: tp.Callable[[Node], None],
  init: tp.Callable[[Node, tp.Iterable[tuple[Key, Leaf]]], None],
):
  if type in GRAPH_REGISTRY:
    raise ValueError(f'Node type {type} is already registered.')

  GRAPH_REGISTRY[type] = GraphNodeImpl(
    type=type,
    flatten=flatten,
    set_key=set_key,
    pop_key=pop_key,
    create_empty=create_empty,
    clear=clear,
    init=init,
  )

def register_pytree_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]],
  unflatten: tp.Callable[[tp.Sequence[tuple[Key, Leaf]], AuxData], Node],
):
  if type in PYTREE_REGISTRY:
    raise ValueError(f'Node type {type} is already registered.')

  PYTREE_REGISTRY[type] = PytreeNodeImpl(
    type=type, flatten=flatten, unflatten=unflatten
  )

def is_node(x: tp.Any) -> bool:
  if type(x) in GRAPH_REGISTRY:
    return True
  return is_pytree_node(x)


def is_graph_node(x: tp.Any) -> bool:
  return type(x) in GRAPH_REGISTRY


def is_node_type(x: type[tp.Any]) -> bool:
  return x in GRAPH_REGISTRY or x in PYTREE_REGISTRY or x is GenericPytree


def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any]:
  if isinstance(x, Variable):
    raise ValueError(f'Variable is not a node: {x}')

  node_type = type(x)

  if node_type in GRAPH_REGISTRY:
    return GRAPH_REGISTRY[node_type]
  elif node_type in PYTREE_REGISTRY:
    return PYTREE_REGISTRY[node_type]
  elif is_pytree_node(x):
    return PYTREE_NODE_IMPL  # type: ignore
  else:
    raise ValueError(f'Unknown node type: {x}')


def get_node_impl_for_type(x: type[Node]) -> NodeImpl[Node, tp.Any, tp.Any]:
  if x is GenericPytree:
    return PYTREE_NODE_IMPL  # type: ignore
  elif x in PYTREE_REGISTRY:
    return PYTREE_REGISTRY[x]
  else:
    return GRAPH_REGISTRY[x]


class HashableMapping(tp.Mapping[HA, HB], tp.Hashable):
  def __init__(self, mapping: tp.Mapping[HA, HB], copy: bool = True):
    self._mapping = dict(mapping) if copy else mapping

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
      isinstance(other, HashableMapping) and self._mapping == other._mapping
    )

  def __repr__(self) -> str:
    return repr(self._mapping)


class GraphDef(tp.Generic[Node]):
  """A class that represents all the static, stateless, and Pythonic parts of a Flax
  :class:`Module`. A ``GraphDef`` can be generated by either calling :func:`split` or
  :func:`graphdef` on the :class:`Module`."""

  type: type[Node]
  index: int


@dataclasses.dataclass(frozen=True, repr=False)
class NodeRef(GraphDef[Node], reprlib.Representable):
  type: type[Node]
  index: int

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'type': self.type, 'index': self.index},
      path=path,
      subtree_renderer=subtree_renderer,
    )


jax.tree_util.register_static(NodeRef)

@dataclasses.dataclass(frozen=True, repr=False)
class VariableDef(reprlib.Representable):
  type: type[Variable]
  index: int
  metadata: HashableMapping[str, tp.Any]

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('metadata', reprlib.PrettyMapping(self.metadata))

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]

    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'type': self.type,
        'index': self.index,
        'metadata': self.metadata,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )


jax.tree_util.register_static(VariableDef)


@dataclasses.dataclass(frozen=True, slots=True)
class SubGraphAttribute:
  key: Key
  value: NodeDef[tp.Any] | NodeRef[tp.Any]


@dataclasses.dataclass(frozen=True, slots=True)
class StaticAttribute:
  key: Key
  value: tp.Any


@dataclasses.dataclass(frozen=True, slots=True)
class LeafAttribute:
  key: Key
  value: VariableDef | NodeRef[tp.Any]


@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class NodeDef(GraphDef[Node], reprlib.Representable):
  """A dataclass that denotes the tree structure of a
  :class:`Module`. A ``GraphDef`` can be generated by either
  calling :func:`split` or :func:`graphdef` on the :class:`Module`."""

  type: tp.Type[Node]
  index: int
  attributes: tuple[SubGraphAttribute | StaticAttribute | LeafAttribute, ...]
  metadata: tp.Any
  index_mapping: HashableMapping[Index, Index] | None

  @classmethod
  def create(
    cls,
    type: tp.Type[Node],
    index: int,
    attributes: tuple[SubGraphAttribute | StaticAttribute | LeafAttribute, ...],
    metadata: tp.Any,
    index_mapping: tp.Mapping[Index, Index] | None,
  ):
    return cls(
      type=type,
      index=index,
      attributes=attributes,
      metadata=metadata,
      index_mapping=HashableMapping(index_mapping)
      if index_mapping is not None
      else None,
    )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('attributes', reprlib.PrettySequence(self.attributes))
    yield reprlib.Attr('metadata', self.metadata)
    yield reprlib.Attr(
      'index_mapping',
      reprlib.PrettyMapping(self.index_mapping)
      if self.index_mapping is not None
      else None,
    )

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'type': self.type,
        'index': self.index,
        'attributes': self.attributes,
        'metadata': self.metadata,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )

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
      return out, flatten(module)

    return CallableProxy(_apply, accessor)  # type: ignore


jax.tree_util.register_static(NodeDef)

PureState = tuple[GraphDef[A], GraphState]


def flatten(
  node: Node, /, ref_index: RefMap[tp.Any, Index] | None = None
) -> tuple[GraphDef[Node], GraphState]:
  """Flattens a graph node into a (graphdef, state) pair.

  Args:
    x: A graph node.
    ref_index: A mapping from nodes to indexes, defaults to None. If not provided, a new
      empty dictionary is created. This argument can be used to flatten a sequence of graph
      nodes that share references.
  """
  if ref_index is None:
    ref_index = RefMap()
  flat_state: list[tuple[PathParts, StateLeaf]] = []
  graphdef = _graph_flatten((), ref_index, flat_state, node)
  return graphdef, GraphState.from_flat_path(flat_state)


def _graph_flatten(
  path: PathParts,
  ref_index: RefMap[tp.Any, Index],
  flat_state: list[tuple[PathParts, StateLeaf]],
  node: Node,
) -> NodeDef[Node] | NodeRef:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if node in ref_index:
    return NodeRef(type(node), ref_index[node])

  node_impl = get_node_impl(node)

  # only cache graph nodes
  if isinstance(node_impl, GraphNodeImpl):
    index = len(ref_index)
    ref_index[node] = index
  else:
    index = -1

  attributes: list[SubGraphAttribute | StaticAttribute | LeafAttribute] = []

  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if is_node(value):
      nodedef = _graph_flatten((*path, key), ref_index, flat_state, value)
      # subgraphs.append((key, nodedef))
      attributes.append(SubGraphAttribute(key, nodedef))
    elif isinstance(value, Variable):
      if value in ref_index:
        attributes.append(
          LeafAttribute(key, NodeRef(type(value), ref_index[value]))
        )
      else:
        flat_state.append(((*path, key), value.to_state()))
        variable_index = ref_index[value] = len(ref_index)
        variabledef = VariableDef(
          type(value), variable_index, HashableMapping(value._var_metadata)
        )
        attributes.append(LeafAttribute(key, variabledef))
    else:
      if isinstance(value, (jax.Array, np.ndarray)):
        path_str = '/'.join(map(str, (*path, key)))
        raise ValueError(
            f'Arrays leaves are not supported, at {path_str!r}: {value}'
        )
      # static_fields.append((key, value))
      attributes.append(StaticAttribute(key, value))

  nodedef = NodeDef.create(
    type=node_impl.type,
    index=index,
    attributes=tuple(attributes),
    metadata=metadata,
    index_mapping=None,
  )
  return nodedef


def unflatten(
  graphdef: GraphDef[Node],
  state: tp.Mapping[KeyT, StateLeaf | tp.Mapping[Key, tp.Any]],
  /,
  *,
  index_ref: dict[Index, tp.Any] | None = None,
  index_ref_cache: dict[Index, tp.Any] | None = None,
) -> Node:
  """Unflattens a graphdef into a node with the given state.

  Args:
    graphdef: A GraphDef instance.
    state: A State instance.
    index_ref: A mapping from indexes to nodes references found during the graph
      traversal, defaults to None. If not provided, a new empty dictionary is
      created. This argument can be used to unflatten a sequence of (graphdef, state)
      pairs that share the same index space.
    index_ref_cache: A mapping from indexes to existing nodes that can be reused.
      When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the graphdef.
  """
  if isinstance(state, State):
    state = state.raw_mapping  # type: ignore
  if index_ref is None:
    index_ref = {}
  assert isinstance(graphdef, (NodeDef, NodeRef))
  node = _graph_unflatten(graphdef, state, index_ref, index_ref_cache)
  return node

def _graph_unflatten(
  nodedef: NodeDef[Node] | NodeRef[Node],
  state: tp.Mapping[KeyT, StateLeaf | tp.Mapping[Key, tp.Any]],
  index_ref: dict[Index, tp.Any],
  index_ref_cache: dict[Index, tp.Any] | None,
) -> Node:
  """Recursive helper for graph_unflatten.

  Args:
    nodedef: A GraphDef instance or an index to a node in the cache.
    state: A mapping from attribute names to variables or subgraphs.
    index_to_ref: A mapping from indexes to nodes that have been traversed.
      If a node is already in the cache, it won't be traversed again.
    index_ref_cache: A mapping from indexes to existing nodes that can be reused.
      When an reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the nodedef.
  """
  if isinstance(nodedef, NodeRef):
    return index_ref[nodedef.index]

  if not is_node_type(nodedef.type):
    raise RuntimeError(f'Unsupported type: {nodedef.type}, this is a bug.')

  if nodedef.index in index_ref:
    raise RuntimeError(f'GraphDef index {nodedef.index} already used.')

  node_impl = get_node_impl_for_type(nodedef.type)

  def _get_children():
    children: list[tuple[Key, NodeLeaf | Node]] = []
    state_keys: set = set(state.keys())

    # for every key in attributes there are 6 possible cases:
    #  - (2) the key can either be present in the state or not
    #  - (3) the key can be a subgraph, a leaf, or a static attribute
    for attribute in nodedef.attributes:
      key = attribute.key
      if key not in state:
        # if key is not present create an empty types
        if type(attribute) is StaticAttribute:
          children.append((key, attribute.value))
        elif type(attribute) is SubGraphAttribute:
          # if the key is a subgraph we create an empty node
          subgraphdef = attribute.value
          assert not isinstance(subgraphdef, VariableDef)
          if isinstance(subgraphdef, NodeRef):
            # subgraph exists, take it from the cache
            children.append((key, index_ref[subgraphdef.index]))
          else:
            # create a node from an empty state, reasoning:
            # * its a node with no state
            # * its a node with state but only through references of already
            #   created nodes
            substate = {}
            subnode = _graph_unflatten(
              subgraphdef, substate, index_ref, index_ref_cache
            )
            children.append((key, subnode))
        elif type(attribute) is LeafAttribute:
          variabledef = attribute.value
          if variabledef.index in index_ref:
            # variable exists, take it from the cache
            children.append((key, index_ref[variabledef.index]))
          else:
            # key for a variable is missing, raise an error
            raise ValueError(
              f'Expected key {key!r} in state while building node of type '
              f'{nodedef.type.__name__}.'
            )
        else:
          raise RuntimeError(f'Unknown static field: {key!r}')
      else:
        state_keys.remove(key)
        value = state[key]
        # if key in nodedef.static_fields:
        if type(attribute) is StaticAttribute:
          raise ValueError(
            f'Got state for static field {key!r}, this is not supported.'
          )
        elif type(attribute) is SubGraphAttribute:
          if is_state_leaf(value):
            raise ValueError(
              f'Expected value of type {attribute.value} for '
              f'{key!r}, but got {value!r}'
            )
          assert isinstance(value, dict)
          subgraphdef = attribute.value

          if isinstance(subgraphdef, NodeRef):
            children.append((key, index_ref[subgraphdef.index]))
          else:
            subnode = _graph_unflatten(
              subgraphdef, value, index_ref, index_ref_cache
            )
            children.append((key, subnode))

        elif type(attribute) is LeafAttribute:
          variabledef = attribute.value

          if variabledef.index in index_ref:
            # add an existing variable
            assert isinstance(variabledef, NodeRef)
            children.append((key, index_ref[variabledef.index]))
          else:
            # its a unseen variable, create a new one
            assert isinstance(variabledef, VariableDef)
            # when idxmap is present, check if the Varable exists there
            # and update existing variables if it does
            if (
              index_ref_cache is not None
              and variabledef.index in index_ref_cache
            ):
              # if variable exists, update it
              variable = index_ref_cache[variabledef.index]
              if not isinstance(variable, Variable):
                raise ValueError(
                  f'Expected a Variable type for {key!r}, but got {type(variable)}.'
                )
              if isinstance(value, VariableState):
                variable.update_from_state(value)
              else:
                variable.raw_value = value
            else:  # if it doesn't, create a new variable
              if isinstance(value, VariableState):
                variable = value.to_variable()
              else:
                variable = variabledef.type.from_metadata(
                  value, variabledef.metadata
                )
            children.append((key, variable))
            index_ref[variabledef.index] = variable
        else:
          raise RuntimeError(f'Unknown key: {key!r}, this is a bug.')

    # NOTE: we could allw adding new StateLeafs here
    if state_keys:
      raise ValueError(f'Unknown keys: {state_keys}')

    return children

  if isinstance(node_impl, GraphNodeImpl):
    # we create an empty node first and add it to the index
    # this avoids infinite recursion when there is a reference cycle
    if index_ref_cache is not None and nodedef.index in index_ref_cache:
      node = index_ref_cache[nodedef.index]
      if type(node) != nodedef.type:
        raise ValueError(
          f'Expected a node of type {nodedef.type} for index '
          f'{nodedef.index}, but got a node of type {type(node)}.'
        )
      node_impl.clear(node)
    else:
      node = node_impl.create_empty(nodedef.metadata)
    index_ref[nodedef.index] = node
    node_impl.init(node, _get_children())
  else:
    # if the node type does not support the creation of an empty object it means
    # that it cannot reference itself, so we can create its children first
    node = node_impl.unflatten(_get_children(), nodedef.metadata)

  return node


def graph_pop(
  node: tp.Any,
  filters: tuple[filterlib.Filter, ...],
) -> tuple[GraphState, ...]:
  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[dict[PathParts, StateLeaf], ...] = tuple(
    {} for _ in predicates
  )
  _graph_pop(node, id_to_index, path_parts, flat_states, predicates)
  return tuple(
    GraphState.from_flat_path(flat_state) for flat_state in flat_states
  )


def _graph_pop(
  node: tp.Any,
  id_to_index: dict[int, Index],
  path_parts: PathParts,
  flat_states: tuple[dict[PathParts, StateLeaf], ...],
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


def _graph_update_dynamic(node: tp.Any, state: tp.Mapping[KeyT, tp.Any]):
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
    else:
      # case 3: state leaf is being updated
      if not isinstance(current_value, Variable):
        raise ValueError(
          f'Trying to update a non-Variable attribute {key!r} with a Variable: '
          f'{value!r}'
        )
      if isinstance(value, VariableState):
        # updated from VariableState
        current_value.update_from_state(value)
      else:
        # updated from raw value
        current_value.raw_value = value

# --------------------------------------------------------
# UpdateContext
# --------------------------------------------------------

@dataclasses.dataclass
class GraphContext(threading.local):
  update_context_stacks: dict[str, list[UpdateContext]] = dataclasses.field(
    default_factory=dict
  )
  ref_index_stack: list[SplitContext] = dataclasses.field(default_factory=list)
  index_ref_stack: list[MergeContext] = dataclasses.field(default_factory=list)


GRAPH_CONTEXT = GraphContext()


@dataclasses.dataclass
class SplitContext:
  ctxtag: str | None
  ref_index: RefMap[tp.Any, Index]

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
  ) -> tuple[GraphDef[A], tpe.Unpack[tuple[GraphState, ...]]]:
    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    graphdef, state = flatten(node, self.ref_index)
    states = _split_state(state, filters)
    if ctx is not None:
      if ctx.index_ref is not None and isinstance(graphdef, NodeDef):
        index_to_index = compose_mapping(ctx.index_ref, self.ref_index)
        graphdef = dataclasses.replace(
          graphdef, index_mapping=HashableMapping(index_to_index, copy=False)
        )

    return graphdef, *states


@contextlib.contextmanager
def split_context(ctxtag: str | None = None):
  index_ref: RefMap[tp.Any, Index] = RefMap()
  flatten_ctx = SplitContext(ctxtag, index_ref)
  GRAPH_CONTEXT.ref_index_stack.append(flatten_ctx)

  try:
    yield flatten_ctx
  finally:
    GRAPH_CONTEXT.ref_index_stack.pop()
    if ctxtag is not None:
      ctx = current_update_context(ctxtag)
      ctx.flatten_end(index_ref)
    del flatten_ctx.ref_index
    del flatten_ctx.ctxtag


@dataclasses.dataclass
class MergeContext:
  ctxtag: str | None
  index_ref: dict[Index, tp.Any]

  def merge(
    self, graphdef: GraphDef[A], state: GraphState, /, *states: GraphState
  ) -> A:
    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    if (
      ctx is not None
      and isinstance(graphdef, NodeDef)
      and graphdef.index_mapping is not None
    ):
      # outer merge (4), create index_ref_cache
      assert ctx.ref_index is not None
      index_ref_cache = compose_mapping_reversed(
        ctx.ref_index, graphdef.index_mapping
      )
    else:
      # inner merge (2)
      index_ref_cache = None

    state = State.merge(state, *states)
    node = unflatten(
      graphdef,
      state,
      index_ref=self.index_ref,
      index_ref_cache=index_ref_cache,
    )
    return node


@contextlib.contextmanager
def merge_context(ctxtag: str | None = None):
  index_ref: dict[Index, tp.Any] = {}

  unflatten_ctx = MergeContext(ctxtag, index_ref)
  GRAPH_CONTEXT.index_ref_stack.append(unflatten_ctx)

  try:
    yield unflatten_ctx
  finally:
    GRAPH_CONTEXT.index_ref_stack.pop()
    if ctxtag is not None:
      ctx = current_update_context(ctxtag)
      ctx.unflatten_end(index_ref)
    del unflatten_ctx.index_ref
    del unflatten_ctx.ctxtag


@dataclasses.dataclass
class UpdateContext:
  """A context manager for handling complex state updates."""

  tag: str
  ref_index: RefMap[tp.Any, Index] | None
  index_ref: dict[Index, tp.Any] | None

  # define hash and eq to make this an opaque object
  def __hash__(self):
    return 0

  def __eq__(self, other):
    return isinstance(other, UpdateContext)

  def flatten_end(self, ref_index: RefMap[tp.Any, Index]):
    if self.ref_index is None:
      # outer split (1), store the references
      self.ref_index = ref_index
    else:
      # inner split (3), clear index_ref
      self.index_ref = None

  def unflatten_end(self, index_ref: dict[Index, tp.Any]):
    self.index_ref = index_ref

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

      >>> from flax import nnx
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
    ref_index: RefMap[tp.Any, Index] = RefMap()
    graphdef, state = flatten(node, ref_index)
    states = _split_state(state, filters)

    if self.index_ref is not None and isinstance(graphdef, NodeDef):
      index_to_index = compose_mapping(self.index_ref, ref_index)
      graphdef = dataclasses.replace(
        graphdef, index_mapping=HashableMapping(index_to_index, copy=False)
      )

    self.flatten_end(ref_index)

    return graphdef, *states

  def merge(
    self,
    graphdef: GraphDef[A],
    state: GraphState,
    *states: GraphState,
  ) -> A:
    """merge"""
    if not isinstance(graphdef, NodeDef):
      raise ValueError(
        f'Expected a NodeDef instance, but got {type(graphdef)}.'
      )
    if self.ref_index is None:
      raise ValueError('Cannot merge without ref_index.')

    if graphdef.index_mapping is not None:
      # outer merge (4), create index_ref_cache
      assert self.ref_index is not None
      index_ref_cache = compose_mapping_reversed(
        self.ref_index, graphdef.index_mapping
      )
    else:
      # inner merge (2)
      index_ref_cache = None

    state = State.merge(state, *states)
    index_ref: dict[Index, tp.Any] = {}
    node = unflatten(
      graphdef, state, index_ref=index_ref, index_ref_cache=index_ref_cache
    )

    self.unflatten_end(index_ref)

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
    del ctx.ref_index
    del ctx.index_ref

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

def _split_state(
  state: GraphState,
  filters: tuple[filterlib.Filter, ...],
) -> tuple[GraphState, tpe.Unpack[tuple[GraphState, ...]]]:
  if not filters:
    return (state,)
  states = state.split(*filters)
  if isinstance(states, State):
    return (states,)
  assert len(states) > 0
  return states  # type: ignore[return-value]


@tp.overload
def split(graph_node: A, /) -> tuple[GraphDef[A], GraphState]: ...
@tp.overload
def split(
  graph_node: A, first: filterlib.Filter, /
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

    >>> from flax import nnx
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
  graphdef, state = flatten(node)
  states = _split_state(state, filters)
  return graphdef, *states

def merge(
  graphdef: GraphDef[A],
  state: tp.Mapping[KeyT, tp.Any],
  /,
  *states: tp.Mapping[KeyT, tp.Any],
) -> A:
  """The inverse of :func:`split`.

  ``merge`` takes a :class:`GraphDef` and one or more :class:`State`'s and creates
  a new node with the same structure as the original node.

  Example usage::

    >>> from flax import nnx
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
  state = State.merge(state, *states)
  node = unflatten(graphdef, state)
  return node


def update(
  node, state: tp.Mapping[KeyT, tp.Any], /, *states: tp.Mapping[KeyT, tp.Any]
) -> None:
  """Update the given graph node with a new state(s) in-place.

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
    state = State.merge(state, *states)
  if isinstance(state, State):
    state = state.raw_mapping
  _graph_update_dynamic(node, state)

def _variables_generator(node) -> tp.Iterable[tuple[PathParts, Variable]]:
  for path, value in iter_graph(node):
    if isinstance(value, Variable):
      yield path, value


@tp.overload
def variables(node, /) -> State[Key, Variable]: ...
@tp.overload
def variables(node, first: filterlib.Filter, /) -> State[Key, Variable]: ...
@tp.overload
def variables(
  node,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State[Key, Variable], ...]: ...
def variables(
  node,
  *filters: filterlib.Filter,
) -> tp.Union[State[Key, Variable], tuple[State[Key, Variable], ...]]:
  """Similar to :func:`state` but returns the current :class:`Variable` objects instead
  of new :class:`VariableState` instances.

  Example::

    >>> from flax import nnx
    ...
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> params = nnx.variables(model, nnx.Param)
    ...
    >>> assert params['kernel'] is model.kernel
    >>> assert params['bias'] is model.bias

  Args:
    node: A graph node object.
    *filters: One or more :class:`Variable` objects to filter by.
  Returns:
    One or more :class:`State` mappings containing the :class:`Variable` objects.
  """
  num_filters = len(filters)
  if num_filters == 0:
    filters = (..., ...)
  else:
    filters = (*filters, ...)

  variables_iterable = _variables_generator(node)
  flat_states = variablelib.split_flat_state(
    variables_iterable, (*filters, ...)
  )
  states = tuple(State.from_flat_path(flat_state) for flat_state in flat_states)
  if num_filters < 2:
    return states[0]
  return states

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
  _, state = flatten(node)

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
  graphdef, _ = flatten(node)
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
  flat_states: tuple[dict[PathParts, StateLeaf], ...] = tuple(
    {} for _ in predicates
  )
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


def call(
  graphdef_state: tuple[GraphDef[A], GraphState], /
) -> ApplyCaller[tuple[GraphDef[A], GraphState]]:
  """Calls a method underlying graph node defined by a (GraphDef, State) pair.

  ``call`` takes a ``(GraphDef, State)`` pair and creates a proxy object that can be
  used to call methods on the underlying graph node. When a method is called, the
  output is returned along with a new (GraphDef, State) pair that represents the
  updated state of the graph node. ``call`` is equivalent to :func:`merge` > ``method``
  > :func:`split`` but is more convenient to use in pure JAX functions.

  Example::

    >>> from flax import nnx
    >>> import jax
    >>> import jax.numpy as jnp
    ...
    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> linear = StatefulLinear(3, 2, nnx.Rngs(0))
    >>> linear_state = nnx.split(linear)
    ...
    >>> @jax.jit
    ... def forward(x, linear_state):
    ...   y, linear_state = nnx.call(linear_state)(x)
    ...   return y, linear_state
    ...
    >>> x = jnp.ones((1, 3))
    >>> y, linear_state = forward(x, linear_state)
    >>> y, linear_state = forward(x, linear_state)
    ...
    >>> linear = nnx.merge(*linear_state)
    >>> linear.count.value
    Array(2, dtype=uint32)

  The proxy object returned by ``call`` supports indexing and attribute access
  to access nested methods. In the example below, the ``increment`` method indexing
  is used to call the ``increment`` method of the ``StatefulLinear`` module
  at the ``b`` key of a ``nodes`` dictionary.

    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> rngs = nnx.Rngs(0)
    >>> nodes = dict(
    ...   a=StatefulLinear(3, 2, rngs),
    ...   b=StatefulLinear(2, 1, rngs),
    ... )
    ...
    >>> node_state = nnx.split(nodes)
    >>> # use attribute access
    >>> _, node_state = nnx.call(node_state)['b'].increment()
    ...
    >>> nodes = nnx.merge(*node_state)
    >>> nodes['a'].count.value
    Array(0, dtype=uint32)
    >>> nodes['b'].count.value
    Array(1, dtype=uint32)
  """

  def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
    node = merge(*graphdef_state)
    method = accessor(node)
    out = method(*args, **kwargs)
    return out, split(node)

  return CallableProxy(pure_caller)  # type: ignore


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
class GenericPytree: ...


def is_pytree_node(x: tp.Any) -> bool:
  t = type(x)
  if t in PYTREE_REGISTRY:
    return True
  elif t in GRAPH_REGISTRY:
    return False
  # known non-pytree types
  elif isinstance(x, Variable):
    return False
  # known pytree types
  elif type(x) is VariableState or type(x) is State:
    return True
  else:
    return not jax.tree_util.all_leaves((x,))


def _key_path_to_key(key: tp.Any) -> Key:
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(
    key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
  ):
    if not is_key_like(key.key):
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
  type=GenericPytree,
  flatten=_flatten_pytree,
  unflatten=_unflatten_pytree,  # type: ignore
)

# common pytrees
# list
register_pytree_node_type(
  list,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: [value for _, value in nodes],  # type: ignore
)
# tuple
register_pytree_node_type(
  tuple,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: tuple(value for _, value in nodes),  # type: ignore
)
# dict
register_pytree_node_type(
  dict,
  flatten=lambda x: (sorted(x.items()), None),
  unflatten=lambda nodes, _: {key: value for key, value in nodes},  # type: ignore
)
# None
register_pytree_node_type(
  type(None),
  flatten=lambda x: ([], None),
  unflatten=lambda _, __: None,  # type: ignore
)
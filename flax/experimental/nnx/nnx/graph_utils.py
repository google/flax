# Copyright 2023 The Flax Authors.
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
import typing as tp

import jax

from flax.experimental.nnx.nnx import filterlib, reprlib
from flax.experimental.nnx.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx.variables import EMPTY, Empty, Variable

HA = tp.TypeVar('HA', bound=tp.Hashable)
HB = tp.TypeVar('HB', bound=tp.Hashable)

Index = int
Names = tp.Sequence[int]
PathParts = tuple[str, ...]
Path = str
Node = tp.TypeVar('Node')
Leaf = tp.TypeVar('Leaf')
AuxData = tp.TypeVar('AuxData')

NODE_TYPES: dict[type, 'NodeImpl[tp.Any, tp.Any, tp.Any]'] = {}


@dataclasses.dataclass(frozen=True)
class NodeImpl(tp.Generic[Node, Leaf, AuxData]):
  type: type
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[str, Leaf]], AuxData]]
  get_key: tp.Callable[[Node, str], Leaf]
  set_key: tp.Callable[[Node, str, Leaf], Node]
  has_key: tp.Callable[[Node, str], bool]
  all_keys: tp.Callable[[Node], tuple[str, ...]]
  unflatten: tp.Callable[[tuple[tuple[str, Leaf], ...], AuxData], Node] | None
  create_empty: tp.Callable[[AuxData], Node] | None
  init: tp.Callable[[Node, tuple[tuple[str, Leaf], ...]], None] | None

  def items(self, node: Node) -> tp.Iterator[tuple[str, Leaf]]:
    for key in self.all_keys(node):
      yield key, self.get_key(node, key)


@tp.overload
def register_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[str, Leaf]], AuxData]],
  get_key: tp.Callable[[Node, str], Leaf],
  set_key: tp.Callable[[Node, str, Leaf], Node],
  has_key: tp.Callable[[Node, str], bool],
  all_keys: tp.Callable[[Node], tuple[str, ...]],
  *,
  unflatten: tp.Callable[[tuple[tuple[str, Leaf], ...], AuxData], Node],
):
  ...


@tp.overload
def register_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[str, Leaf]], AuxData]],
  get_key: tp.Callable[[Node, str], Leaf],
  set_key: tp.Callable[[Node, str, Leaf], Node],
  has_key: tp.Callable[[Node, str], bool],
  all_keys: tp.Callable[[Node], tuple[str, ...]],
  *,
  create_empty: tp.Callable[[AuxData], Node],
  init: tp.Callable[[Node, tuple[tuple[str, Leaf], ...]], None],
):
  ...


def register_node_type(
  type: type,
  flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[str, Leaf]], AuxData]],
  get_key: tp.Callable[[Node, str], Leaf],
  set_key: tp.Callable[[Node, str, Leaf], Node],
  has_key: tp.Callable[[Node, str], bool],
  all_keys: tp.Callable[[Node], tuple[str, ...]],
  *,
  unflatten: tp.Callable[[tuple[tuple[str, Leaf], ...], AuxData], Node]
  | None = None,
  create_empty: tp.Callable[[AuxData], Node] | None = None,
  init: tp.Callable[[Node, tuple[tuple[str, Leaf], ...]], None] | None = None,
):
  if type in NODE_TYPES:
    raise ValueError(f"Node type '{type}' already registered.")
  NODE_TYPES[type] = NodeImpl(
    type,
    flatten,
    get_key,
    set_key,
    has_key,
    all_keys,
    unflatten,
    create_empty,
    init,
  )


def is_node(x: tp.Any) -> bool:
  return type(x) in NODE_TYPES


def is_node_type(x: type[tp.Any]) -> bool:
  return x in NODE_TYPES


@tp.overload
def get_node_impl(x: type[Node]) -> NodeImpl[Node, tp.Any, tp.Any]:
  ...


@tp.overload
def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any]:
  ...


def get_node_impl(x: type[Node] | Node) -> NodeImpl[Node, tp.Any, tp.Any]:
  if not isinstance(x, type):
    x = type(x)
  if not is_node_type(x):
    raise ValueError(f'Unknown node type: {x}')
  return NODE_TYPES[x]


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
    return hash(tuple(self._mapping.items()))

  def __eq__(self, other: tp.Any) -> bool:
    if not isinstance(other, tp.Mapping):
      return False
    return self._mapping == other

  def __repr__(self) -> str:
    return repr(self._mapping)


@dataclasses.dataclass(repr=False)
class _MappingRepr(reprlib.Representable):
  mapping: tp.Mapping[str, tp.Any]

  def __nnx_repr__(self):
    yield reprlib.Object(type='', value_sep=': ', start='{', end='}')

    for key, value in self.mapping.items():
      yield reprlib.Attr(repr(key), value)


class VariableDef(reprlib.Representable):
  __slots__ = (
    '_type',
    '_index',
    '_metadata',
  )

  @classmethod
  def from_variable(cls, variable: Variable[tp.Any], index: int) -> VariableDef:
    metadata = vars(variable).copy()
    del metadata['value']
    return cls(type(variable), index, metadata)

  def to_variable(self, value: Node) -> Variable[Node]:
    variables = object.__new__(self._type)
    variables.value = value
    vars(variables).update(self._metadata)
    return variables

  def __init__(
    self,
    type: tp.Type[Variable[tp.Any]],
    index: int,
    metadata: dict[str, tp.Any],
  ):
    self._type = type
    self._index = index
    self._metadata = metadata

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self._type.__name__)
    yield reprlib.Attr('index', self._index)
    yield reprlib.Attr('metadata', _MappingRepr(self._metadata))

  @property
  def type(self):
    return self._type

  @property
  def index(self):
    return self._index

  @property
  def metadata(self):
    return self._metadata

  def __hash__(self):
    return hash((self._type, self._index, tuple(self._metadata.items())))

  def __eq__(self, other):
    if not isinstance(other, VariableDef):
      return False
    return (
      self._type == other._type
      and self._index == other._index
      and self._metadata == other._metadata
    )


class GraphDef(tp.Generic[Node], reprlib.Representable):
  __slots__ = (
    '_type',
    '_index',
    '_attributes',
    '_subgraphs',
    '_static_fields',
    '_variables',
    '_metadata',
  )

  def __init__(
    self,
    type: tp.Type[Node],
    index: int,
    attributes: tuple[str, ...],
    subgraphs: tp.Iterable[tuple[str, tp.Union['GraphDef[tp.Any]', int]]],
    static_fields: tp.Iterable[tuple[str, tp.Any]],
    variables: tp.Iterable[tuple[str, VariableDef | int]],
    metadata: tp.Any,
  ):
    self._type = type
    self._index = index
    self._attributes = attributes
    self._subgraphs = _HashableMapping(subgraphs)
    self._static_fields = _HashableMapping(static_fields)
    self._variables = _HashableMapping(variables)
    self._metadata = metadata

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self._type.__name__)
    yield reprlib.Attr('index', self._index)
    yield reprlib.Attr('attributes', self._attributes)
    yield reprlib.Attr('subgraphs', _MappingRepr(self._subgraphs))
    yield reprlib.Attr('static_fields', _MappingRepr(self._static_fields))
    yield reprlib.Attr('variables', _MappingRepr(self._variables))
    yield reprlib.Attr('metadata', self._metadata)

  def __hash__(self) -> int:
    return hash((self._type, self._subgraphs))

  def __eq__(self, other: tp.Any) -> bool:
    if not isinstance(other, GraphDef):
      return False
    return self._type == other._type and self._subgraphs == other._subgraphs

  @property
  def type(self) -> tp.Type[Node]:
    return self._type

  @property
  def index(self) -> int:
    return self._index

  @property
  def attributes(self) -> tuple[str, ...]:
    return self._attributes

  @property
  def subgraphs(self):
    return self._subgraphs

  @property
  def static_fields(self):
    return self._static_fields

  @property
  def variables(self):
    return self._variables

  @property
  def metadata(self) -> tp.Any:
    return self._metadata

  def merge(self, state: State, *states: State) -> Node:
    if states:
      state = State.merge(state, *states)
    return graph_unflatten(self, state)

  def apply(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple[State, 'GraphDef[Node]']]:
    accessor = DelayedAccessor()

    def _apply(
      accessor: DelayedAccessor, *args, **kwargs
    ) -> tuple[tp.Any, tuple[State, GraphDef[Node]]]:
      module = self.merge(state, *states)
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, graph_flatten(module)

    return CallableProxy(_apply, accessor)  # type: ignore

  def make_empty(self) -> Node:
    return self.merge(State({}))


def _gradphdef_flatten(graphdef: GraphDef[tp.Any]):
  return (), (
    graphdef._type,
    graphdef._index,
    graphdef._attributes,
    graphdef._subgraphs,
    graphdef._static_fields,
    graphdef._variables,
    graphdef._metadata,
  )


def _graphdef_unflatten(
  metadata: tuple[
    tp.Type[Node],
    int,
    tuple[str, ...],
    tuple[tuple[str, GraphDef[Node] | int], ...],
    tuple[tuple[str, tp.Any], ...],
    tuple[tuple[str, Variable[Empty] | int], ...],
    tp.Any,
  ],
  _,
) -> GraphDef[Node]:
  return GraphDef(*metadata)


jax.tree_util.register_pytree_node(
  GraphDef, _gradphdef_flatten, _graphdef_unflatten
)


def graph_flatten(x: Node) -> tuple[State, GraphDef[Node]]:
  id_to_index: dict[int, Index] = {}
  flat_state: dict[Path, Variable[tp.Any]] = {}
  dagdef = _graph_flatten((), id_to_index, flat_state, x)
  assert not isinstance(dagdef, int)
  return State.from_flat_path(flat_state), dagdef


def _graph_flatten(
  path: PathParts,
  id_to_index: dict[int, Index],
  flat_state: dict[Path, Variable[tp.Any]],
  node: Node,
) -> GraphDef[Node] | int:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if (node_id := id(node)) in id_to_index:
    return id_to_index[node_id]

  index = len(id_to_index)
  id_to_index[id(node)] = index

  subgraphs: list[tuple[str, tp.Union[GraphDef[Node], int]]] = []
  static_fields: list[tuple[str, tp.Any]] = []
  variables: list[tuple[str, VariableDef | int]] = []

  node_impl = get_node_impl(node)
  values, metadata = node_impl.flatten(node)
  for key, value in values:
    if not isinstance(key, str):
      raise TypeError(
        f'Node (of type {type(node).__name__}) has a key of non-string '
        f'type {type(key).__name__}.'
      )
    if is_node(value):
      graphdef = _graph_flatten((*path, key), id_to_index, flat_state, value)
      subgraphs.append((key, graphdef))
    elif isinstance(value, Variable):
      str_path = '/'.join((*path, key))
      variable_id = id(value)
      if variable_id in id_to_index:
        variables.append((key, id_to_index[variable_id]))
      else:
        flat_state[str_path] = value.copy()
        variable_index = id_to_index[variable_id] = len(id_to_index)
        variables.append(
          (key, VariableDef.from_variable(value, variable_index))
        )
    else:
      static_fields.append((key, value))

  graphdef = GraphDef(
    type=type(node),
    index=index,
    attributes=tuple(key for key, _ in values),
    subgraphs=subgraphs,
    static_fields=static_fields,
    variables=variables,
    metadata=metadata,
  )
  return graphdef


def graph_unflatten(graphdef: GraphDef[Node], state: State) -> Node:
  index_to_node: dict[Index, tp.Any] = {}
  return _graph_unflatten(graphdef, state.raw_mapping, index_to_node)


def _graph_unflatten(
  graphdef: tp.Union[GraphDef[Node], int],
  state: dict[str, Variable[tp.Any] | dict[str, tp.Any]],
  index_to_node: dict[Index, tp.Any],
) -> Node:
  if isinstance(graphdef, int):
    return index_to_node[graphdef]

  if not is_node_type(graphdef.type):
    raise RuntimeError(f'Unsupported type: {graphdef.type}, this is a bug.')

  if graphdef.index in index_to_node:
    raise RuntimeError(f'GraphDef index {graphdef.index} already used.')

  state = state.copy()
  node_impl = get_node_impl(graphdef.type)

  def _get_children():
    new_state: dict[str, tp.Any] = {}

    for key in graphdef.attributes:
      if key in graphdef.static_fields:
        new_state[key] = graphdef.static_fields[key]
      elif key not in state:
        # if key is not present create an empty types
        if key in graphdef.subgraphs:
          # if the key is a subgraph we create an empty node
          subgraphdef = graphdef.subgraphs[key]
          if isinstance(subgraphdef, int):
            # subgraph exists, take it from the cache
            new_state[key] = index_to_node[subgraphdef]
          else:
            # create an empty node and add it to the cache
            substate = {}
            node = new_state[key] = _graph_unflatten(
              subgraphdef, substate, index_to_node
            )
            index_to_node[subgraphdef.index] = node
        elif key in graphdef.variables:
          variable_def = graphdef.variables[key]
          if isinstance(variable_def, int):
            # variable exists, take it from the cache
            new_state[key] = index_to_node[variable_def]
          else:
            # create an empty variable and add it to the cache
            node = new_state[key] = variable_def.to_variable(EMPTY)
            index_to_node[variable_def.index] = node
        else:
          raise RuntimeError(f'Unknown static field: {key!r}')
      else:
        value = state[key]
        if key in graphdef.subgraphs:
          if isinstance(value, Variable):
            raise ValueError(
              f'Expected a subgraph for {key!r}, but got a Variable.'
            )
          subgraphdef = graphdef.subgraphs[key]

          if isinstance(subgraphdef, int):
            node = index_to_node[subgraphdef]
          else:
            node = new_state[key] = _graph_unflatten(
              subgraphdef, value, index_to_node
            )
            index_to_node[subgraphdef.index] = node

        elif key in graphdef.variables:
          variable_def = graphdef.variables[key]
          if isinstance(variable_def, int):
            new_state[key] = index_to_node[variable_def]
          else:
            if type(value) != variable_def.type:
              raise ValueError(
                f'Expected a Variable of type {variable_def.type} '
                f'for {key!r}, but got a Variable of type {type(value)}.'
              )
            assert isinstance(value, Variable)
            value = value.copy()
            new_state[key] = value
            index_to_node[variable_def.index] = value

    for new_key in set(state) - set(graphdef.attributes):
      new_state[new_key] = state[new_key]

    return new_state

  if node_impl.create_empty:
    assert node_impl.init is not None
    # we create an empty node first and add it to the index
    # this avoids infinite recursion when there is a reference cycle
    node = node_impl.create_empty(graphdef.metadata)
    index_to_node[graphdef.index] = node
    children = _get_children()
    node_impl.init(node, tuple(children.items()))
  else:
    # if the node type does not support the creation of an empty object it means
    # that it cannot reference itself, so we can create its children first
    assert node_impl.unflatten is not None
    children = _get_children()
    node = node_impl.unflatten(tuple(children.items()), graphdef.metadata)
    index_to_node[graphdef.index] = node

  return node


def graph_pop(
  node: tp.Any,
  filters: tuple[filterlib.Filter, ...],
) -> tuple[State, ...]:
  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  states = tuple({} for _ in predicates)
  _graph_pop(node, id_to_index, path_parts, states, predicates)
  return tuple(State(x) for x in states)


def _graph_pop(
  node: tp.Any,
  id_to_index: dict[int, Index],
  path_parts: PathParts,
  states: tuple[dict[Path, tp.Any], ...],
  predicates: tuple[filterlib.Predicate, ...],
) -> None:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if id(node) in id_to_index:
    return

  id_to_index[id(node)] = len(id_to_index)
  node_impl = get_node_impl(node)

  for name in node_impl.all_keys(node):
    value = node_impl.get_key(node, name)
    if is_node(value):
      _graph_pop(value, id_to_index, (*path_parts, name), states, predicates)
      continue
    elif not isinstance(value, Variable):
      continue
    elif value.is_empty:
      continue
    elif id(value) in id_to_index:
      continue

    path = '/'.join((*path_parts, name))
    node_impl = get_node_impl(node)
    for state, predicate in zip(states, predicates):
      if predicate(path, value):
        state[path] = value.copy()
        id_to_index[id(value)] = len(id_to_index)
        # empty Variable attributes
        node_impl.set_key(node, name, value.as_empty())
        break
    else:
      # NOTE: should we raise an error here?
      pass


def graph_update_dynamic(
  node: tp.Any,
  updates: State | tp.Sequence[State],
) -> None:
  if not is_node(node):
    raise ValueError(f'Unsupported type: {type(node)}')

  if isinstance(updates, State):
    new_states = (updates,)
  else:
    new_states = updates

  for state in new_states:
    _graph_update_dynamic(node, state.raw_mapping)


def _graph_update_dynamic(
  node: tp.Any, state: dict[str, Variable[tp.Any] | dict[str, tp.Any]]
):
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}')

  node_impl = get_node_impl(node)
  for key, value in state.items():
    if is_node(value):
      if isinstance(value, Variable):
        raise ValueError(
          f'Expected a subgraph for {key!r}, but got a Variable.'
        )
      _graph_update_dynamic(node_impl.get_key(node, key), value)
    else:
      # we skip if trying to replace a non-empty Variable with an empty one
      if (
        isinstance(value, Variable)
        and value.is_empty
        and isinstance(current_value := node_impl.get_key(node, key), Variable)
        and not current_value.is_empty
      ):
        continue

      node_impl.set_key(node, key, value)


class _StaticModuleStatus(enum.Enum):
  NEW = enum.auto()
  UPDATED = enum.auto()


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
  for name, value_updates in node_impl.items(updates):
    if isinstance(value_updates, Variable):
      continue
    elif is_node(value_updates):
      if node_impl.has_key(node, name):
        _graph_update_static(
          node_impl.get_key(node, name),
          value_updates,
          cache,
          _StaticModuleStatus.UPDATED,
          (*path, name),
        )
      else:
        if id(value_updates) in cache:
          if cache[id(value_updates)] is not _StaticModuleStatus.NEW:
            raise ValueError(
              f'Trying to add a new node at path {name!r} but a '
              'node with the same reference has been updated'
            )
        else:
          cache[id(value_updates)] = _StaticModuleStatus.NEW

        node_impl.set_key(node, name, value_updates)
    else:  # static field
      node_impl.set_key(node, name, value_updates)


def clone(node: Node) -> Node:
  state, static = graph_flatten(node)
  return static.merge(state)


def iter_nodes(node: tp.Any) -> tp.Iterator[tuple[Path, tp.Any]]:
  visited: set[int] = set()
  path_parts: PathParts = ()
  yield from _iter_nodes(node, visited, path_parts)


def _iter_nodes(
  node: tp.Any, visited: set[int], path_parts: PathParts
) -> tp.Iterator[tuple[Path, tp.Any]]:
  if not is_node(node):
    return
  if id(node) in visited:
    return
  visited.add(id(node))
  path = '/'.join(path_parts)
  yield path, node
  node_impl = get_node_impl(node)
  for key, value in node_impl.items(node):
    yield from _iter_nodes(value, visited, (*path_parts, key))


# -----------------------------
# register node types
# -----------------------------
# dict
def _flatten_dict(
  node: dict[str, tp.Any],
) -> tuple[tuple[tuple[str, tp.Any], ...], None]:
  return tuple(node.items()), None


def _get_key_dict(node: dict[str, tp.Any], key: str) -> tp.Any:
  return node[key]


def _set_key_dict(
  node: dict[str, tp.Any], key: str, value: tp.Any
) -> dict[str, tp.Any]:
  node[key] = value
  return node


def _has_key_dict(node: dict[str, tp.Any], key: str) -> bool:
  return key in node


def _all_keys_dict(node: dict[str, tp.Any]) -> tuple[str, ...]:
  return tuple(node.keys())


def _create_empty_dict(metadata: None) -> dict[str, tp.Any]:
  return {}


def _init_dict(node: dict[str, tp.Any], items: tuple[tuple[str, tp.Any], ...]):
  node.update(items)


register_node_type(
  dict,
  _flatten_dict,
  _get_key_dict,
  _set_key_dict,
  _has_key_dict,
  _all_keys_dict,
  create_empty=_create_empty_dict,
  init=_init_dict,
)


# list
def _flatten_list(
  node: list[tp.Any],
) -> tuple[tuple[tuple[str, tp.Any], ...], int]:
  return tuple((str(i), value) for i, value in enumerate(node)), len(node)


def _get_key_list(node: list[tp.Any], key: str) -> tp.Any:
  return node[int(key)]


def _set_key_list(node: list[tp.Any], key: str, value: tp.Any) -> list[tp.Any]:
  int_key = int(key)
  if int_key >= len(node):
    node.extend([EMPTY] * (int_key - len(node) + 1))
  node[int_key] = value
  return node


def _has_key_list(node: list[tp.Any], key: str) -> bool:
  return int(key) < len(node)


def _all_keys_list(node: list[tp.Any]) -> tuple[str, ...]:
  return tuple(str(i) for i in range(len(node)))


def _create_empty_list(length: int) -> list[tp.Any]:
  return [EMPTY] * length


def _init_list(node: list[tp.Any], items: tuple[tuple[str, tp.Any], ...]):
  for key, value in items:
    _set_key_list(node, key, value)


register_node_type(
  list,
  _flatten_list,
  _get_key_list,
  _set_key_list,
  _has_key_list,
  _all_keys_list,
  create_empty=_create_empty_list,
  init=_init_list,
)


# tuple
def _flatten_tuple(
  node: tuple[tp.Any, ...],
) -> tuple[tuple[tuple[str, tp.Any], ...], int]:
  return tuple((str(i), value) for i, value in enumerate(node)), len(node)


def _unflatten_tuple(
  items: tuple[tuple[str, tp.Any], ...], length: int
) -> tuple[tp.Any, ...]:
  node = [EMPTY] * length
  for key, value in items:
    node[int(key)] = value
  return tuple(node)


def _get_key_tuple(node: tuple[tp.Any, ...], key: str) -> tp.Any:
  return node[int(key)]


def _set_key_tuple(node: tuple[tp.Any, ...], key: str, value: tp.Any):
  current_value = _get_key_tuple(node, key)
  if current_value is value or current_value == value:
    return
  else:
    raise ValueError("'tuple' object is immutable, does not support assignment")


def _has_key_tuple(node: tuple[tp.Any, ...], key: str) -> bool:
  return int(key) < len(node)


def _all_keys_tuple(node: tuple[tp.Any, ...]) -> tuple[str, ...]:
  return tuple(str(i) for i in range(len(node)))


register_node_type(
  tuple,
  _flatten_tuple,
  _get_key_tuple,
  _set_key_tuple,
  _has_key_tuple,
  _all_keys_tuple,
  unflatten=_unflatten_tuple,
)

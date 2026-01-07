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

import jax.core

from flax import config
from flax.nnx import filterlib, reprlib, traversals, variablelib
from flax.nnx import statelib
from flax.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.statelib import FlatState, State, map_state
from flax.nnx.variablelib import Variable, is_array_ref, V
from flax.typing import HashableMapping, Key, PathParts, is_key_like
import jax
import numpy as np
import treescope  # type: ignore[import-not-found,import-untyped]
import typing_extensions as tpe

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


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class NoUpdate: ...


NO_UPDATE = NoUpdate()


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Repeated: ...


REPEATED = Repeated()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class ArrayRefOutput(reprlib.Representable):
  value: jax.Array

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('value', self.value)

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'value': self.value,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )


LeafType = tp.Union[
  Variable,
  jax.Array,
  np.ndarray,
  variablelib.Ref,
  ArrayRefOutput,
  NoUpdate,
]

GraphState = tp.Annotated[State[Key, LeafType], 'GraphState']
GraphState.__doc__ = """Alias for the state of a graph node."""

GraphFlatState = tp.Annotated[FlatState[LeafType], 'GraphFlatState']
GraphFlatState.__doc__ = """Alias for the flat state of a graph node."""


def is_node_leaf(x: tp.Any) -> tpe.TypeGuard[LeafType]:
  return isinstance(x, LeafType) or variablelib.is_array_ref(x)  # type: ignore[misc, arg-type]


class IndexMap(dict[Index, tp.Any]):
  @staticmethod
  def from_refmap(refmap: RefMap) -> IndexMap:
    return IndexMap((index, value) for value, index in refmap.items())


if config.flax_use_flaxlib:
  import flaxlib  # type: ignore[import]

  globals()['IndexMap'] = flaxlib.IndexMap


# RefMap = dict
class RefMap(tp.MutableMapping[tp.Any, int], reprlib.MappingReprMixin):
  """A mapping that hashes keys by their identity."""

  def __init__(
    self,
    mapping: tp.Mapping[tp.Any, int]
    | tp.Iterable[tuple[tp.Any, int]]
    | None = None,
    /,
  ):
    self._mapping: dict[int, tuple[tp.Any, int]] = dict()
    if mapping is not None:
      self.update(mapping)

  @staticmethod
  def from_indexmap(indexmap: IndexMap) -> RefMap:
    refmap = RefMap()
    refmap.update((value, index) for index, value in indexmap.items())
    return refmap

  def get(self, key: tp.Any, default: int | None = None) -> int | None:  # type: ignore[override]
    return self._mapping.get(id(key), (None, default))[1]

  def __getitem__(self, key: tp.Any) -> int:
    return self._mapping[id(key)][1]

  def __setitem__(self, key: tp.Any, value: int):
    self._mapping[id(key)] = (key, value)

  def __delitem__(self, key: tp.Any):
    del self._mapping[id(key)]

  def __len__(self) -> int:
    return len(self._mapping)

  def __contains__(self, key: tp.Any) -> bool:
    return id(key) in self._mapping

  def __iter__(self) -> tp.Iterator[tp.Any]:
    for key, _ in self._mapping.values():
      yield key

  def items(self) -> tp.ItemsView[tp.Any, int]:
    return self._mapping.values()  # type: ignore


# save python version
PythonRefMap = RefMap

if config.flax_use_flaxlib:
  import flaxlib  # type: ignore[import]

  globals()['RefMap'] = flaxlib.RefMap


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
  set_key: tp.Callable[[Node, Key, Leaf], None] | None
  pop_key: tp.Callable[[Node, Key], Leaf] | None


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
  *,
  set_key: tp.Callable[[Node, Key, Leaf], None] | None = None,
  pop_key: tp.Callable[[Node, Key], Leaf] | None = None,
):
  if type in PYTREE_REGISTRY:
    raise ValueError(f'Node type {type} is already registered.')

  PYTREE_REGISTRY[type] = PytreeNodeImpl(
    type=type,
    flatten=flatten,
    unflatten=unflatten,
    set_key=set_key,
    pop_key=pop_key,
  )


def is_node(x: tp.Any) -> bool:
  if isinstance(x, Variable):
    return False
  if type(x) in GRAPH_REGISTRY:
    return True
  return is_pytree_node(x)


def is_graph_node(x: tp.Any) -> bool:
  return (
    type(x) in GRAPH_REGISTRY
    or variablelib.is_array_ref(x)
    or isinstance(x, Variable)
  )


def is_node_type(x: type[tp.Any]) -> bool:
  return x in GRAPH_REGISTRY or x in PYTREE_REGISTRY or x is GenericPytree


def get_node_impl(x: Node) -> NodeImpl[Node, tp.Any, tp.Any] | None:
  if isinstance(x, Variable):
    return None

  node_type = type(x)

  if node_type in GRAPH_REGISTRY:
    return GRAPH_REGISTRY[node_type]
  elif node_type in PYTREE_REGISTRY:
    return PYTREE_REGISTRY[node_type]
  elif node_type in JAX_PYTREE_REGISTRY or issubclass(node_type, tuple):
    return PYTREE_NODE_IMPL  # type: ignore
  else:
    return None


def get_node_impl_for_type(
  x: type[Node],
) -> NodeImpl[Node, tp.Any, tp.Any] | None:
  if x is GenericPytree:
    return PYTREE_NODE_IMPL  # type: ignore
  elif x in PYTREE_REGISTRY:
    return PYTREE_REGISTRY[x]
  elif x in GRAPH_REGISTRY:
    return GRAPH_REGISTRY[x]
  else:
    return None



@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, repr=False)
class NodeRef(tp.Generic[Node], reprlib.Representable):
  index: int

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('index', self.index)

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'index': self.index},
      path=path,
      subtree_renderer=subtree_renderer,
    )


if config.flax_use_flaxlib:
  import flaxlib  # type: ignore[import]

  jax.tree_util.register_static(flaxlib.NodeRef)
  globals()['NodeRef'] = flaxlib.NodeRef


@dataclasses.dataclass(frozen=True, repr=False)
class VariableDef(reprlib.Representable, tp.Generic[Node]):
  type: type[Node]
  index: int
  outer_index: int | None
  metadata: HashableMapping[str, tp.Any]
  array_refdef: ArrayRefDef | NodeRef | None

  def with_no_outer_index(self) -> VariableDef:
    return VariableDef(
      type=self.type,
      index=self.index,
      outer_index=None,
      metadata=self.metadata,
      array_refdef=self.array_refdef.with_no_outer_index()
      if isinstance(self.array_refdef, ArrayRefDef)
      else self.array_refdef,
    )

  def with_same_outer_index(self) -> VariableDef:
    return VariableDef(
      type=self.type,
      index=self.index,
      outer_index=self.index,
      metadata=self.metadata,
      array_refdef=self.array_refdef.with_same_outer_index()
      if isinstance(self.array_refdef, ArrayRefDef)
      else self.array_refdef,
    )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('outer_index', self.outer_index)
    yield reprlib.Attr('metadata', reprlib.PrettyMapping(self.metadata))

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'type': self.type,
        'index': self.index,
        'outer_index': self.outer_index,
        'metadata': self.metadata,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )


if config.flax_use_flaxlib:
  import flaxlib  # type: ignore[import]

  jax.tree_util.register_static(flaxlib.VariableDef)
  globals()['VariableDef'] = flaxlib.VariableDef


@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRefDef(reprlib.Representable):
  index: int
  outer_index: int | None

  def with_no_outer_index(self):
    return ArrayRefDef(
      index=self.index,
      outer_index=None,
    )

  def with_same_outer_index(self):
    return ArrayRefDef(
      index=self.index,
      outer_index=self.index,
    )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('outer_index', self.outer_index)

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'index': self.index,
        'outer_index': self.outer_index,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class NodeDef(tp.Generic[Node], reprlib.Representable):
  """Definition for a single node within a :class:`GraphDef`.

  Contains metadata about a graph node including its type, index for
  reference tracking, number of attributes, and any additional metadata.
  Multiple ``NodeDef`` objects are combined in a :class:`GraphDef` to
  represent the complete graph structure.
  """

  type: tp.Type[Node]
  index: int | None
  outer_index: int | None
  num_attributes: int
  metadata: tp.Any

  def with_no_outer_index(self) -> NodeDef[Node]:
    return NodeDef(
      type=self.type,
      index=self.index,
      outer_index=None,
      num_attributes=self.num_attributes,
      metadata=self.metadata,
    )

  def with_same_outer_index(self) -> NodeDef[Node]:
    return NodeDef(
      type=self.type,
      index=self.index,
      outer_index=self.index,
      num_attributes=self.num_attributes,
      metadata=self.metadata,
    )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self.type.__name__)
    yield reprlib.Attr('index', self.index)
    yield reprlib.Attr('outer_index', self.outer_index)
    yield reprlib.Attr('num_attributes', self.num_attributes)
    yield reprlib.Attr('metadata', self.metadata)

  def __treescope_repr__(self, path, subtree_renderer):
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={
        'type': self.type,
        'index': self.index,
        'outer_index': self.outer_index,
        'num_attributes': self.num_attributes,
        'metadata': self.metadata,
      },
      path=path,
      subtree_renderer=subtree_renderer,
    )


if config.flax_use_flaxlib:
  import flaxlib  # type: ignore[import]

  jax.tree_util.register_static(flaxlib.NodeDef)
  globals()['NodeDef'] = flaxlib.NodeDef

NodeDefType = tp.Union[
  NodeDef[Node],
  NodeRef[Node],
  VariableDef[Node],
  ArrayRefDef,
]


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayAttr:
  pass


ARRAY_ATTR = ArrayAttr()


@dataclasses.dataclass(frozen=True, slots=True)
class MutableArrayAttr:
  pass


MUTABLE_ARRAY_ATTR = MutableArrayAttr()


@dataclasses.dataclass(frozen=True, slots=True)
class NodeAttr:
  pass


NODE_ATTR = NodeAttr()

AttrType = tp.Union[
  NodeAttr,
  ArrayAttr,
  MutableArrayAttr,
  'Static[tp.Any]',
]


# GraphDef = tp.Union[NodeDef[Node], NodeRef[Node], VariableDef[Node]]
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class GraphDef(tp.Generic[Node]):
  """Static structure definition of a graph node.

  Contains the structural information (node types, attributes, references)
  needed to reconstruct a graph node when combined with a :class:`GraphState`.

  This is a hashable object, treated as immutable, that can be used as a JAX static value.

  It is analogous to JAX’s ``PyTreeDef``.
  """

  nodes: list[NodeDefType[tp.Any]]
  attributes: list[tuple[Key, AttrType]]
  num_leaves: int

  def __hash__(self) -> int:
    return hash((tuple(self.nodes), tuple(self.attributes)))

  def with_no_outer_index(self) -> GraphDef[Node]:
    return GraphDef(
      nodes=[
        node.with_no_outer_index() if not isinstance(node, NodeRef) else node
        for node in self.nodes
      ],
      attributes=self.attributes,
      num_leaves=self.num_leaves,
    )

  def with_same_outer_index(self) -> GraphDef[Node]:
    return GraphDef(
      nodes=[
        node.with_same_outer_index() if not isinstance(node, NodeRef) else node
        for node in self.nodes
      ],
      attributes=self.attributes,
      num_leaves=self.num_leaves,
    )

  # TODO(cgarciae): remove this method
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
      graphdef, flat_state = flatten(module)
      state_ = statelib.from_flat_state(flat_state)
      return out, (graphdef, state_)

    return CallableProxy(_apply, accessor)  # type: ignore


PureState = tp.Annotated[tuple[GraphDef[Node], GraphState], 'PureState']
PureState.__doc__ = """Alias for a pure (functional) representation of a graph node."""


@tp.overload
def flatten(  # type: ignore[invalid-annotation]
  node: Node,
  /,
  *,
  ref_index: RefMap | None = None,
  ref_outer_index: RefMap | None = None,
) -> tuple[GraphDef[Node], FlatState[tp.Any]]: ...
@tp.overload
def flatten(  # type: ignore[invalid-annotation]
  node: Node,
  /,
  *,
  with_paths: tp.Literal[True],
  ref_index: RefMap | None = None,
  ref_outer_index: RefMap | None = None,
) -> tuple[
  GraphDef[Node],
  FlatState[tp.Any],
]: ...
@tp.overload
def flatten(  # type: ignore[invalid-annotation]
  node: Node,
  /,
  *,
  with_paths: tp.Literal[False],
  ref_index: RefMap | None = None,
  ref_outer_index: RefMap | None = None,
) -> tuple[
  GraphDef[Node],
  list[tp.Any],
]: ...
@tp.overload
def flatten(  # type: ignore[invalid-annotation]
  node: Node,
  /,
  *,
  with_paths: bool,
  ref_index: RefMap | None = None,
  ref_outer_index: RefMap | None = None,
) -> tuple[
  GraphDef[Node],
  FlatState[tp.Any] | list[tp.Any],
]: ...
def flatten(  # type: ignore[invalid-annotation]
  node: Node,
  /,
  *,
  with_paths: bool = True,
  ref_index: RefMap | None = None,
  ref_outer_index: RefMap | None = None,
) -> tuple[
  GraphDef[Node],
  FlatState[tp.Any] | list[tp.Any],
]:
  """Flatten a graph node into a :class:`GraphDef` and :class:`FlatState` tuple.

  Lower-level version of :func:`split` that returns a flat state representation
  instead of a nested one. Useful for advanced use cases requiring direct access
  to the flat structure.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, flat_state = nnx.flatten(model)
    >>> flat_state.paths
    (('bias',), ('kernel',))

  Args:
    node: A graph node.
    with_paths: Boolean which controls whether to return paths with the state.
      Defaults to ``True``.
    ref_index: A mapping from nodes to indices for flattening multiple graph
      nodes that share references. Defaults to ``None``.
    ref_outer_index: A mapping from nodes to outer indices. Defaults to ``None``.

  Returns:
    A ``GraphDef`` and either a ``FlatState`` or list of values (depending on
    ``with_paths``).
  """

  if ref_index is None:
    ref_index = RefMap()

  leaves: list[tp.Any] = []
  path: list[Key] | None = [] if with_paths else None
  paths: list[PathParts] | None = [] if with_paths else None
  nodes: list[NodeDefType[tp.Any]] = []
  attributes: list[tuple[Key, AttrType]] = []
  node_impl = get_node_impl(node)
  _graph_flatten(
    node,
    node_impl,
    path,
    ref_index,
    ref_outer_index,
    nodes,
    attributes,
    leaves,
    paths,
  )
  graphdef: GraphDef = GraphDef(
    nodes=nodes, attributes=attributes, num_leaves=len(leaves)
  )

  if paths is not None:
    return graphdef, FlatState.from_sorted_keys_values(tuple(paths), leaves)  # type: ignore[return-value]
  else:
    return graphdef, leaves


def _graph_flatten(
  node: Node,
  node_impl: NodeImpl[Node, Leaf, AuxData] | None,
  path: list[Key] | None,
  ref_index: RefMap,
  ref_outer_index: RefMap | None,
  nodes: list[NodeDefType[tp.Any]],
  attributes: list[tuple[Key, AttrType]],
  leaves: list[tp.Any],
  paths: list[PathParts] | None,
) -> None:
  is_pytree_node_ = type(node_impl) is PytreeNodeImpl

  index: int | None
  if not is_pytree_node_ and node in ref_index:
    nodes.append(NodeRef(index := ref_index[node]))
    return

  is_graph_node_ = type(node_impl) is GraphNodeImpl
  is_variable = isinstance(node, Variable)
  is_array_ref = variablelib.is_array_ref(node)

  # only cache graph nodes, we don't add array refs here
  # as they are added in the make_mutable_arraydef function
  if is_graph_node_ or is_variable:
    index = len(ref_index)
    ref_index[node] = index
  else:
    index = None

  def make_mutable_arraydef(value: variablelib.Ref):
    if value in ref_index:
      index = ref_index[value]
      return NodeRef(index), REPEATED
    else:
      index = len(ref_index)
      ref_index[value] = index
    output_value: NoUpdate | ArrayRefOutput | variablelib.Ref
    if ref_outer_index is not None:
      if value in ref_outer_index:
        outer_index = ref_outer_index[value]
        output_value = NO_UPDATE
        array_refdef = ArrayRefDef(index=index, outer_index=outer_index)
      else:
        output_value = ArrayRefOutput(value[...])
        array_refdef = ArrayRefDef(index=index, outer_index=None)
    else:
      output_value = value
      array_refdef = ArrayRefDef(index=index, outer_index=None)
    return array_refdef, output_value

  if is_variable:
    assert isinstance(node, Variable)
    assert index is not None
    prev_inner_value = node.get_raw_value()
    if variablelib.is_array_ref(prev_inner_value):
      array_refdef, inner_value = make_mutable_arraydef(prev_inner_value)
    else:
      array_refdef = None
      inner_value = prev_inner_value
    if path is None:
      leaf = inner_value
    else:
      leaf = node  # type: ignore[assignment]
      if inner_value is not prev_inner_value:
        leaf.set_raw_value(inner_value)

    variabledef = VariableDef(
      type=node.var_type,  # type: ignore
      index=index,
      outer_index=ref_outer_index.get(node, None) if ref_outer_index else None,
      metadata=HashableMapping(node.get_metadata()),
      array_refdef=array_refdef,
    )
    if type(inner_value) is not Repeated:
      assert not isinstance(leaf, Repeated)
      leaves.append(leaf)
      if path is not None:
        assert paths is not None
        paths.append(tuple(path))
    nodes.append(variabledef)
    return
  elif is_array_ref:
    array_refdef, leaf = make_mutable_arraydef(node)  # type: ignore[arg-type]
    if not isinstance(leaf, Repeated):
      leaves.append(leaf)
      if path is not None:
        assert paths is not None
        paths.append(tuple(path))
    nodes.append(array_refdef)
    return
  elif not is_pytree_node_ and not is_graph_node_:
    # unknown leaf
    leaves.append(node)
    if path is not None:
      assert paths is not None
      paths.append(tuple(path))
    return

  if node_impl is None:
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  values, metadata = node_impl.flatten(node)
  num_attributes = len(values)
  nodedef = NodeDef(
    node_impl.type,
    index,
    ref_outer_index[node]
    if is_graph_node_ and ref_outer_index and node in ref_outer_index
    else None,
    num_attributes,
    metadata,
  )
  nodes.append(nodedef)

  for key, value in values:
    value_node_impl = get_node_impl(value)
    if path is not None:
      path.append(key)
    if value_node_impl is not None or isinstance(value, Variable):
      attributes.append((key, NODE_ATTR))
      _graph_flatten(
        value,
        value_node_impl,
        path,
        ref_index,
        ref_outer_index,
        nodes,
        attributes,
        leaves,
        paths,
      )
    elif variablelib.is_array_ref(value):
      attributes.append((key, MUTABLE_ARRAY_ATTR))
      array_refdef, leaf = make_mutable_arraydef(value)
      if not isinstance(leaf, Repeated):
        leaves.append(leaf)
        if paths is not None:
          paths.append(tuple(path))  # type: ignore
      nodes.append(array_refdef)
    elif isinstance(value, (jax.Array, np.ndarray)):
      attributes.append((key, ARRAY_ATTR))
      if paths is not None:
        paths.append(tuple(path))  # type: ignore
      leaves.append(value)
    else:
      attributes.append((key, Static(value)))

    if path is not None:
      path.pop()

  return


def _get_sorted_leaves(
  xs: tp.Mapping[tp.Any, tp.Any],
) -> list[tp.Any]:
  if not isinstance(xs, tp.Mapping):  # type: ignore
    raise TypeError(f'expected Mapping; got {type(xs).__qualname__}')
  leaves: list[tp.Any] = []

  def _flatten(xs):
    if not isinstance(xs, tp.Mapping):
      leaves.append(xs)
    else:
      for _, value in sorted(xs.items()):
        _flatten(value)

  _flatten(xs)
  return leaves


def unflatten(  # type: ignore[invalid-annotation]
  graphdef: GraphDef[Node],
  state: State[Key, tp.Any] | FlatState[tp.Any] | list[tp.Any],
  /,
  *,
  index_ref: IndexMap | None = None,
  outer_index_outer_ref: IndexMap | None = None,
  copy_variables: bool = False,
) -> Node:
  """Unflatten a :class:`GraphDef` and state into a graph node.

  The inverse of :func:`flatten`. Lower-level version of :func:`merge` that
  accepts flat state representations.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, flat_state = nnx.flatten(model)
    >>> reconstructed = nnx.unflatten(graphdef, flat_state)
    >>> assert isinstance(reconstructed, nnx.Linear)

  Args:
    graphdef: A ``GraphDef`` object.
    state: A ``State``, ``dict``, ``FlatState``, or ``list`` of leaf values.
    index_ref: A mapping from indices to node references for unflattening
      multiple (graphdef, state) pairs that share the same index space.
      Defaults to ``None``.
    outer_index_outer_ref: A mapping from outer indices to outer node references.
      Defaults to ``None``.
    copy_variables: Boolean which controls whether to create new copies of
      Variables. Defaults to ``False``.

  Returns:
    A graph node reconstructed from the ``GraphDef`` and state.
  """

  if isinstance(state, (State, dict)):
    leaves = _get_sorted_leaves(state)
  elif isinstance(state, FlatState):
    leaves = state.leaves
  elif isinstance(state, list):  # type: ignore
    leaves = state
  else:
    raise ValueError(f'Unsupported state type: {type(state)}')
  if index_ref is None:
    index_ref = IndexMap()

  if len(leaves) != graphdef.num_leaves:
    raise ValueError(
      f'Incorrect number of leaves, expected {graphdef.num_leaves} leaves, but got {len(leaves)}.'
    )

  if len(graphdef.nodes) == 0:
    # unknown leaf
    return leaves[0]
  elif isinstance(nodedef := graphdef.nodes[0], NodeRef):
    node = index_ref[nodedef.index]
  else:
    node_iter = iter(graphdef.nodes)
    attribute_iter = iter(graphdef.attributes)
    leaves_iter = iter(leaves)
    nodedef = next(node_iter)
    assert not isinstance(nodedef, NodeRef)
    if isinstance(nodedef, ArrayRefDef):
      node_impl = None
    else:
      node_impl = get_node_impl_for_type(nodedef.type)
    node = _graph_unflatten(
      nodedef,
      node_impl,
      node_iter,
      attribute_iter,
      leaves_iter,
      index_ref,
      outer_index_outer_ref,
      copy_variables,
    )

    try:
      next(leaves_iter)
    except StopIteration:
      pass
    else:
      raise ValueError('Incorrect number of leaves in state.')

  return node


def _graph_unflatten(
  nodedef: NodeDefType[Node],
  node_impl: NodeImpl[Node, Leaf, AuxData] | None,
  node_iter: tp.Iterator[NodeDefType[Node]],
  attribute_iter: tp.Iterator[tuple[Key, AttrType]],
  leaves_iter: tp.Iterator[tp.Any],
  index_ref: IndexMap,
  outer_index_outer_ref: IndexMap | None,
  copy_variables: bool,
) -> Node:
  """Recursive helper for graph_unflatten.

  Args:
    nodedef: A NodeDef instance or an index to a node in the cache.
    node_impl: The node implementation for the node type.
    node_iter: Iterator over remaining node definitions.
    attribute_iter: Iterator over remaining attribute definitions.
    leaves_iter: Iterator over leaf values.
    index_ref: A mapping from indexes to nodes that have been traversed.
      If a node is already in the cache, it won't be traversed again.
    outer_index_outer_ref: A mapping from indexes to existing nodes that can be reused.
      When a reference is reused, ``GraphNodeImpl.clear`` is called to leave the
      object in an empty state and then filled by the unflatten process, as a result
      existing graph nodes are mutated to have the new content/topology
      specified by the nodedef.
    copy_variables: Whether to create copies of Variable objects.
  """

  def get_mutable_array(array_refdef: ArrayRefDef, leaf):
    assert type(array_refdef) is ArrayRefDef
    if (
      outer_index_outer_ref is not None
      and array_refdef.outer_index is not None
      and array_refdef.outer_index in outer_index_outer_ref
    ):
      # if array ref exists, update it
      array_ref = outer_index_outer_ref[array_refdef.outer_index]
      if not variablelib.is_array_ref(array_ref):
        raise RuntimeError(f'Expected an ArrayRef type but got {array_ref}.')
      if type(leaf) is not NoUpdate:
        raise RuntimeError(f'Expected a no update for ArrayRef but got {leaf}.')
    elif type(leaf) in (NoUpdate, Repeated):
      raise ValueError(
        f"Expected an ArrayRefOutput type but got '{leaf}.'"
      )
    elif type(leaf) is ArrayRefOutput:
      array_ref = jax.new_ref(leaf.value)
    elif variablelib.is_array_ref(leaf):
      array_ref = leaf
    else:
      # here we allow merging frozen arrays and will not create a new array ref
      array_ref = leaf

    index_ref[array_refdef.index] = array_ref
    return array_ref

  if type(nodedef) is NodeRef:
    return index_ref[nodedef.index]

  if type(nodedef) is VariableDef:
    variabledef = tp.cast(VariableDef[Variable], nodedef)
    # it's an unseen variable, create a new one

    if variabledef.array_refdef is not None:
      if type(variabledef.array_refdef) is NodeRef:
        value = index_ref[variabledef.array_refdef.index]
      else:
        value = next(leaves_iter)
        assert type(variabledef.array_refdef) is ArrayRefDef
        if isinstance(value, Variable):
          copy_ref = not isinstance(
            value.get_raw_value(), (NoUpdate, Repeated, ArrayRefOutput)
          )
          value = value.copy(_copy_ref=copy_ref) if copy_variables else value
          inner_value = value.get_raw_value()
          array_ref = get_mutable_array(variabledef.array_refdef, inner_value)
          if array_ref is not inner_value:
            value.set_raw_value(array_ref)
        else:
          # if value is an array or array ref, we need call get_mutable_array
          # to register it in the index_ref
          value = get_mutable_array(variabledef.array_refdef, value)
    else:
      value = next(leaves_iter)
      if isinstance(value, Variable) and copy_variables:
        copy_ref = not isinstance(
          value.get_raw_value(), (NoUpdate, Repeated, ArrayRefOutput)
        )
        value = value.copy(_copy_ref=copy_ref)

    # when idxmap is present, check if the Variable exists there
    # and update existing variables if it does
    if (
      outer_index_outer_ref is not None
      and variabledef.outer_index is not None
      and variabledef.outer_index in outer_index_outer_ref
    ):
      # if variable exists, update it
      variable = outer_index_outer_ref[variabledef.outer_index]
      if not isinstance(variable, Variable):
        raise ValueError(f'Expected a Variable type but got {type(variable)}.')
      elif isinstance(value, Variable):
        variable.update_from_state(value)
      else:
        variable.set_raw_value(value)
    else:  # variabledef.index not in index_ref_cache
      # variable reference does not exist outside, create a new one
      if isinstance(value, Variable):
        variable = value
      else:
        variable = variabledef.type.from_metadata(
          value, dict(variabledef.metadata)
        )
    index_ref[variabledef.index] = variable
    return variable  # type: ignore[return-value]

  if type(nodedef) is ArrayRefDef:
    leaf = next(leaves_iter)
    array_ref = get_mutable_array(nodedef, leaf)
    return array_ref  # type: ignore[return-value]

  assert type(nodedef) is NodeDef
  if node_impl is None:
    raise RuntimeError(f'Unsupported type: {nodedef.type}, this is a bug.')
  if nodedef.index is not None and nodedef.index in index_ref:
    raise RuntimeError(f'GraphDef index {nodedef.index} already used.')

  def _get_children() -> list[tuple[Key, tp.Any]]:
    children: list[tuple[Key, LeafType | Node]] = []  # type: ignore[invalid-annotation]

    assert type(nodedef) is NodeDef
    for _ in range(nodedef.num_attributes):
      key, value = next(attribute_iter)
      if type(value) is Static:
        children.append((key, value.value))  # type: ignore[attribute-error]
      elif type(value) is MutableArrayAttr:
        array_refdef = next(node_iter)
        assert (
          type(array_refdef) is ArrayRefDef or type(array_refdef) is NodeRef
        )
        if type(array_refdef) is NodeRef:
          array_ref = index_ref[array_refdef.index]
        else:
          assert type(array_refdef) is ArrayRefDef
          leaf = next(leaves_iter)
          array_ref = get_mutable_array(array_refdef, leaf)
        children.append((key, array_ref))
      elif type(value) is ArrayAttr:
        array = next(leaves_iter)
        children.append((key, array))
      elif type(value) is NodeRef:
        children.append((key, index_ref[value.index]))  # type: ignore[attribute-error]
      elif type(value) is NodeAttr:
        # if the key is a subgraph we create an empty node
        subgraphdef = next(node_iter)
        if type(subgraphdef) is NodeDef:
          value_node_impl = get_node_impl_for_type(subgraphdef.type)  # type: ignore[attribute-error]
        else:
          value_node_impl = None
        subnode = _graph_unflatten(
          subgraphdef,
          value_node_impl,
          node_iter,
          attribute_iter,
          leaves_iter,
          index_ref,
          outer_index_outer_ref,
          copy_variables,
        )
        children.append((key, subnode))
      else:
        raise RuntimeError(f'Unknown static field: {key!r}')

    return children

  if isinstance(node_impl, GraphNodeImpl):
    # we create an empty node first and add it to the index
    # this avoids infinite recursion when there is a reference cycle
    assert type(nodedef) is NodeDef
    if (
      outer_index_outer_ref is not None
      and nodedef.outer_index is not None
      and nodedef.outer_index in outer_index_outer_ref
    ):
      node = outer_index_outer_ref[nodedef.outer_index]
      if type(node) != nodedef.type:
        raise ValueError(
          f'Expected a node of type {nodedef.type} for index '
          f'{nodedef.index}, but got a node of type {type(node)}.'
        )
      node_impl.clear(node)
    else:
      node = node_impl.create_empty(nodedef.metadata)
    assert nodedef.index is not None
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
  flat_states: tuple[dict[PathParts, LeafType], ...] = tuple(
    {} for _ in predicates
  )
  _graph_pop(node, id_to_index, path_parts, flat_states, predicates)
  return tuple(
    statelib.from_flat_state(flat_state) for flat_state in flat_states
  )


def _graph_pop(
  node: tp.Any,
  id_to_index: dict[int, Index],
  path_parts: PathParts,
  flat_states: tuple[dict[PathParts, LeafType], ...],
  predicates: tuple[filterlib.Predicate, ...],
) -> None:
  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')

  if id(node) in id_to_index:
    return

  id_to_index[id(node)] = len(id_to_index)
  node_impl = get_node_impl(node)
  if node_impl is None:
    raise TypeError(f'Unknown node type: {type(node)}')
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
    if node_impl is None:
      raise TypeError(f'Unknown node type: {type(node)}')

    for state, predicate in zip(flat_states, predicates):
      if predicate(node_path, value):
        if node_impl.pop_key is None:
          raise ValueError(
            f'Cannot pop key {name!r} from node of type {type(node).__name__}'
          )
        id_to_index[id(value)] = len(id_to_index)
        node_impl.pop_key(node, name)
        if isinstance(value, Variable):
          value = value
        state[node_path] = value  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # NOTE: should we raise an error here?
      pass


def _graph_update_dynamic(node: tp.Any, state: tp.Mapping[KeyT, tp.Any]):
  def _update_variable(node: Variable, value):
    if isinstance(value, Variable):
      # updated from Variable
      node.update_from_state(value)
    else:
      # updated from raw value
      if isinstance(value, State) and not value:
        # NOTE: this is a special case when trying to update a Variable from state
        # created when flattening into a NodeRef, which creates an empty State. This
        # can happen when using standalone Variables with `grad`
        pass
      else:
        if is_array_ref(node.get_raw_value()) and (
          isinstance(value, jax.Array) or is_array_ref(value)
        ):
          node[...] = value[...]
        else:
          node.set_raw_value(value, _unsafe_bypass_check=True)

  if isinstance(node, Variable):
    _update_variable(node, state)
    return

  if not is_node(node):
    raise RuntimeError(f'Unsupported type: {type(node)}')

  node_impl = get_node_impl(node)
  if node_impl is None:
    raise TypeError(f'Unknown node type: {type(node)}')
  node_dict = node_impl.node_dict(node)
  for key, value in state.items():
    # case 1: new state is being added
    if key not in node_dict:
      if node_impl.set_key is None:
        raise ValueError(
          f'Cannot set key {key!r} on immutable node of '
          f'type {type(node).__name__}'
        )
      if isinstance(value, Variable):
        copy_ref = not isinstance(
          value.get_raw_value(), (NoUpdate, Repeated, ArrayRefOutput)
        )
        value = value.copy(_copy_ref=copy_ref)
      node_impl.set_key(node, key, value)
      continue

    current_value = node_dict[key]

    # case 2: subgraph is being updated
    if is_array_ref(current_value):
      current_value[...] = value
    elif is_node(current_value):
      if is_node_leaf(value):
        raise ValueError(f'Expected a subgraph for {key!r}, but got: {value!r}')
      _graph_update_dynamic(current_value, value)
    elif isinstance(current_value, Variable):
      _update_variable(current_value, value)
    elif node_impl.set_key is not None:
      node_impl.set_key(node, key, value)
    else:
      raise ValueError(
        f'Cannot set key {key!r} on immutable node of '
        f'type {type(node).__name__}'
      )


# --------------------------------------------------------
# UpdateContext
# --------------------------------------------------------


class StaticCache(tp.NamedTuple):
  graphdef: GraphDef[tp.Any]
  final_graphdef: GraphDef[tp.Any]
  paths: tuple[PathParts, ...]
  variables: list[Variable[tp.Any]]
  new_ref_index: RefMap
  new_index_ref: IndexMap

  @staticmethod
  def create(
    graphdef: GraphDef[tp.Any],
    paths: tuple[PathParts, ...],
    variables: list[Variable[tp.Any]],
    new_ref_index: RefMap,
  ):
    new_index_ref = IndexMap.from_refmap(new_ref_index)
    final_graphdef: GraphDef[tp.Any]
    final_graphdef = graphdef.with_same_outer_index()
    return StaticCache(
      graphdef=graphdef,
      final_graphdef=final_graphdef,
      paths=paths,
      variables=variables,
      new_ref_index=new_ref_index,
      new_index_ref=new_index_ref,
    )


@dataclasses.dataclass
class GraphContext(threading.local):
  update_context_stacks: dict[tp.Hashable, list[UpdateContext]] = (
    dataclasses.field(default_factory=dict)
  )
  ref_index_stack: list[SplitContext] = dataclasses.field(default_factory=list)
  index_ref_stack: list[MergeContext] = dataclasses.field(default_factory=list)
  tmp_static_cache: tp.MutableMapping[tp.Any, StaticCache] | None = None
  caching: bool = False


GRAPH_CONTEXT = GraphContext()


@contextlib.contextmanager
def static_cache(static_cache: tp.MutableMapping[tp.Any, StaticCache]):
  if GRAPH_CONTEXT.caching:
    yield
    return

  GRAPH_CONTEXT.tmp_static_cache = static_cache

  try:
    yield
  finally:
    if GRAPH_CONTEXT.tmp_static_cache is not None:
      raise ValueError(
        'GRAPH_CONTEXT.tmp_static_cache should be None, no context consumed it.'
      )


def _cached_partial(f: tp.Callable[..., tp.Any], *cached_args):
  """Create a partial function that caches graph node traversals for performance.

  Reduces Python overhead by caching the traversal of NNX graph nodes. Useful
  for functions called repeatedly with the same graph nodes (e.g., a train step
  with fixed model and optimizer).

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>> import optax

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    >>> @nnx.jit
    ... def train_step(model, optimizer, x, y):
    ...   def loss_fn(model):
    ...     return jnp.mean((model(x) - y) ** 2)
    ...   loss, grads = nnx.value_and_grad(loss_fn)(model)
    ...   optimizer.update(model, grads)
    ...   return loss

    >>> cached_step = nnx.cached_partial(train_step, model, optimizer)
    >>> loss = cached_step(jnp.ones((10, 2)), jnp.ones((10, 3)))

  Note:
    Cached graph nodes are cloned but share the same ``Variable`` objects, ensuring
    state propagates correctly. The graph structure must remain unchanged after
    each call (temporary mutations like ``sow`` are allowed if cleaned up via
    ``pop``).

  Args:
    f: The function to create a partial from.
    *cached_args: Arguments containing graph nodes to cache.

  Returns:
    A partial function expecting the remaining arguments.
  """

  cache: tp.MutableMapping[tp.Any, StaticCache] = PythonRefMap()  # type: ignore
  original_ref_index: RefMap = RefMap()
  index_ref: IndexMap = IndexMap()
  cached_ref_index: RefMap = RefMap()

  def create_static_cache(x):
    # TODO(cgarciae): support Array attribute updates for graph nodes
    if is_graph_node(x) or isinstance(x, Variable):
      graphdef, flat_state = flatten(
        x, with_paths=True, ref_index=original_ref_index
      )
      paths = flat_state.paths
      variables = flat_state.leaves
      # clone but keep the same variable references
      node_cache = unflatten(
        graphdef, flat_state, index_ref=index_ref, copy_variables=False
      )
      start_index = len(cached_ref_index)
      flatten(
        node_cache,
        ref_index=cached_ref_index,
        with_paths=False,
      )
      cached_new_ref_index = RefMap(
        (key, value)
        for key, value in cached_ref_index.items()
        if value >= start_index
      )
      cache[node_cache] = StaticCache.create(
        graphdef, paths, variables, cached_new_ref_index
      )
      return node_cache
    return x

  cached_args = jax.tree.map(
    create_static_cache,
    cached_args,
    is_leaf=lambda x: is_graph_node(x) or isinstance(x, Variable),
  )

  @functools.wraps(f)
  def cache_args_wrapper(*args, **kwargs):
    with static_cache(cache):
      return f(*cached_args, *args, **kwargs)

  return cache_args_wrapper


if tp.TYPE_CHECKING:
  cached_partial = functools.partial
else:
  cached_partial = _cached_partial


@dataclasses.dataclass
class SplitContext:
  """Context for splitting graph nodes within an :func:`update_context`.

  Provides ``split`` and ``flatten`` methods that track reference mappings
  for proper state synchronization across JAX transformation boundaries.

  Use via :func:`split_context` context manager.
  """

  ctxtag: tp.Hashable | None
  ref_index: RefMap
  is_inner: bool | None

  @tp.overload
  def split(self, graph_node: A, /) -> tuple[GraphDef[A], GraphState]: ...  # type: ignore[invalid-annotation]

  @tp.overload
  def split(  # type: ignore[invalid-annotation]
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
  ) -> tuple[GraphDef[A], GraphState, tpe.Unpack[tuple[GraphState, ...]]]: ...  # type: ignore[not-supported-yet]

  def split(
    self, node: A, *filters: filterlib.Filter
  ) -> tuple[GraphDef[A], tpe.Unpack[tuple[GraphState, ...]]]:  # type: ignore[not-supported-yet]
    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    inner_ref_outer_index = (
      ctx.inner_ref_outer_index if ctx and ctx.inner_ref_outer_index else None
    )
    graphdef, flat_state = flatten(
      node, ref_index=self.ref_index, ref_outer_index=inner_ref_outer_index
    )
    flat_states = _split_state(flat_state, filters)
    states = _to_nested_state(graphdef, flat_states)

    return graphdef, *states

  @tp.overload
  def flatten(  # type: ignore[invalid-annotation]
    self,
    graph_node: A,
    /,
    *,
    with_paths: tp.Literal[False],
  ) -> tuple[GraphDef[A], list[tp.Any]]: ...

  @tp.overload
  def flatten(  # type: ignore[invalid-annotation]
    self,
    graph_node: A,
    /,
  ) -> tuple[GraphDef[A], FlatState[tp.Any]]: ...

  @tp.overload
  def flatten(  # type: ignore[invalid-annotation]
    self,
    graph_node: A,
    first: filterlib.Filter,
    /,
  ) -> tuple[GraphDef[A], FlatState[tp.Any]]: ...

  @tp.overload
  def flatten(  # type: ignore[invalid-annotation]
    self,
    graph_node: A,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[
    GraphDef[A],
    FlatState[tp.Any],
    tpe.Unpack[tuple[FlatState[tp.Any], ...]],
  ]: ...

  def flatten(  # type: ignore[invalid-annotation]
    self,
    node: A,
    *filters: filterlib.Filter,
    with_paths: bool = True,
  ) -> tuple[
    GraphDef[A],
    FlatState[tp.Any] | list[tp.Any],
    tpe.Unpack[tuple[FlatState[tp.Any], ...]],
  ]:
    if not with_paths and filters:
      raise ValueError('Cannot use filters with with_paths=False')

    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    static_cache = (
      ctx.static_cache if ctx is not None and self.is_inner is False else None
    )
    ref_outer_index = (
      ctx.inner_ref_outer_index if ctx and ctx.inner_ref_outer_index else None
    )
    flat_state: FlatState[tp.Any] | list[tp.Any]
    leaves: list[tp.Any]
    if node in self.ref_index:
      # node is already in the ref_index, call flatten which will return a NodeRef
      graphdef, flat_state = flatten(
        node,
        ref_index=self.ref_index,
        ref_outer_index=ref_outer_index,
        with_paths=with_paths,
      )
      if with_paths:
        assert isinstance(flat_state, FlatState)
        paths = flat_state.paths
        leaves = flat_state.leaves
      else:
        assert isinstance(flat_state, list)
        paths = None
        leaves = flat_state
    elif static_cache is not None and node in static_cache:
      node_static_cache = static_cache[node]
      graphdef = node_static_cache.graphdef
      # add the new references to the ref_index
      self.ref_index.update(node_static_cache.new_ref_index)

      if with_paths:
        paths = node_static_cache.paths
        leaves = node_static_cache.variables
      else:
        paths = None
        leaves = [
          variable.get_raw_value() for variable in node_static_cache.variables
        ]
    else:
      graphdef, flat_state = flatten(
        node,
        ref_index=self.ref_index,
        ref_outer_index=ref_outer_index,
        with_paths=with_paths,
      )
      if with_paths:
        assert isinstance(flat_state, FlatState)
        paths = flat_state.paths
        leaves = flat_state.leaves
      else:
        assert isinstance(flat_state, list)
        paths = None
        leaves = flat_state

    if with_paths:
      assert paths is not None
      flat_state = FlatState.from_sorted_keys_values(paths, leaves)
      flat_states = _split_state(flat_state, filters)
      return graphdef, *flat_states  # type: ignore[bad-return-type]
    else:
      return graphdef, leaves


@contextlib.contextmanager
def split_context(ctxtag: tp.Hashable | None = None):
  """Create a context for splitting graph nodes with reference tracking.

  Used within :func:`update_context` to split nodes while maintaining
  reference mappings between outer and inner graph states.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> with nnx.split_context() as ctx:
    ...   graphdef, state = ctx.split(model)

  Args:
    ctxtag: Optional tag to link with an :func:`update_context`. If provided, there must be an active ``update_context`` with the same tag.
      Defaults to ``None`` .

  Yields:
    A ``SplitContext`` object with ``split`` and ``flatten`` methods.
  """

  ctx = current_update_context(ctxtag) if ctxtag is not None else None
  is_inner = ctx.outer_ref_outer_index is not None if ctx is not None else None
  GRAPH_CONTEXT.ref_index_stack.append(SplitContext(ctxtag, RefMap(), is_inner))

  try:
    yield GRAPH_CONTEXT.ref_index_stack[-1]
  finally:
    flatten_ctx = GRAPH_CONTEXT.ref_index_stack.pop()
    if ctxtag is not None:
      ctx = current_update_context(ctxtag)
      ctx.flatten_end(flatten_ctx.ref_index)
    del flatten_ctx.ref_index
    del flatten_ctx.ctxtag


@dataclasses.dataclass
class MergeContext:
  """Context for merging graph nodes within an :func:`update_context`.

  Provides ``merge`` and ``unflatten`` methods that track reference mappings
  for proper state synchronization across JAX transformation boundaries.

  Use via :func:`merge_context` context manager.
  """

  ctxtag: tp.Hashable | None
  index_ref: IndexMap
  is_inner: bool | None

  def merge(  # type: ignore[invalid-annotation]
    self,
    graphdef: GraphDef[A],
    state: GraphState,
    /,
    *states: GraphState,
  ) -> A:
    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    outer_index_outer_ref = (
      ctx.outer_index_outer_ref if ctx and ctx.outer_index_outer_ref else None
    )

    _state = _merge_to_flat_state((state, *states))
    node = unflatten(
      graphdef,
      _state,
      index_ref=self.index_ref,
      outer_index_outer_ref=outer_index_outer_ref,
      copy_variables=True,
    )
    return node

  def unflatten(  # type: ignore[invalid-annotation]
    self,
    graphdef: GraphDef[A],
    flat_state: GraphFlatState | list[tp.Any],
    /,
    *flat_states: GraphFlatState,
  ) -> A:
    ctx = (
      current_update_context(self.ctxtag) if self.ctxtag is not None else None
    )
    static_cache = (
      ctx.static_cache if ctx is not None and self.is_inner is False else None
    )
    state: FlatState[tp.Any] | list[tp.Any]
    if type(flat_state) is list:
      if flat_states:
        raise ValueError(
          'Cannot use multiple flat_states when flat_state is a list, '
          f'got flat_state: {flat_state!r}, flat_states: {flat_states!r}'
        )
      state = flat_state
    else:
      state = FlatState.merge(flat_state, *flat_states)

    if type(graphdef.nodes[0]) is NodeRef:
      node = unflatten(
        graphdef,
        state,
        index_ref=self.index_ref,
      )

    elif static_cache is not None:
      assert isinstance(graphdef.nodes[0], NodeDef)
      assert ctx is not None
      if (outer_index := graphdef.nodes[0].outer_index) is not None:
        outer_index_outer_ref = ctx.outer_index_outer_ref
        assert outer_index_outer_ref is not None
        node = outer_index_outer_ref[outer_index]

        if node in static_cache:
          static_cache_node = static_cache[node]
          if static_cache_node.final_graphdef != graphdef:
            raise ValueError(
              'The graph structure of a node added to cached_partial was mutated inside the transformation, '
              f'this is not allowed.\nNode: {node}\nOutput graphdef: {graphdef}\nExpected graphdef: {static_cache_node.final_graphdef}'
            )
          if type(state) is list:
            leaves = state
          elif type(state) is FlatState:
            leaves = state.leaves
          else:
            raise ValueError(f'Unsupported state type: {type(state)}')

          if len(leaves) != len(static_cache_node.variables):
            raise ValueError(
              f'Incorrect number of leaves: expected {len(static_cache_node.variables)} '
              f'leaves in the state, got {len(leaves)}'
            )
          for variable, leaf in zip(static_cache_node.variables, leaves):
            if isinstance(leaf, Variable):
              variable.update_from_state(leaf)
            else:
              variable.set_raw_value(leaf)
          self.index_ref.update(static_cache_node.new_index_ref)
        else:
          # uncached node, create it
          node = unflatten(
            graphdef,
            state,
            index_ref=self.index_ref,
            outer_index_outer_ref=outer_index_outer_ref,
          )
      else:  # graphdef.outer_index is None
        # it's a new node, create it
        node = unflatten(
          graphdef,
          state,
          index_ref=self.index_ref,
        )
    else:
      outer_index_outer_ref = (
        ctx.outer_index_outer_ref if ctx and ctx.outer_index_outer_ref else None
      )
      node = unflatten(
        graphdef,
        state,
        index_ref=self.index_ref,
        outer_index_outer_ref=outer_index_outer_ref,
      )
    return node


@tp.overload
@contextlib.contextmanager
def merge_context() -> tp.Generator[MergeContext, None, None]: ...  # type: ignore[bad-return-type]
@tp.overload
@contextlib.contextmanager
def merge_context(
  ctxtag: tp.Hashable | None, inner: bool | None
) -> tp.Generator[MergeContext, None, None]: ...  # type: ignore[bad-return-type]
@contextlib.contextmanager
def merge_context(ctxtag: tp.Hashable | None = None, inner: bool | None = None):
  """Create a context for merging graph nodes with reference tracking.

  Used within :func:`update_context` to merge nodes while maintaining
  reference mappings between outer and inner graph states.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, state = nnx.split(model)
    >>> with nnx.merge_context() as ctx:
    ...   new_model = ctx.merge(graphdef, state)

  Args:
    ctxtag: Optional tag to link with an :func:`update_context`. Defaults to ``None``.
    inner: Whether this is an inner merge (inside the transform). Required if
      ``ctxtag`` is provided. Defaults to ``None``.

  Yields:
    A ``MergeContext`` object with ``merge`` and ``unflatten`` methods.
  """

  GRAPH_CONTEXT.index_ref_stack.append(MergeContext(ctxtag, IndexMap(), inner))

  try:
    yield GRAPH_CONTEXT.index_ref_stack[-1]
  finally:
    unflatten_ctx = GRAPH_CONTEXT.index_ref_stack.pop()
    index_ref = unflatten_ctx.index_ref
    if ctxtag is not None:
      if inner is None:
        raise ValueError('inner must be specified when using ctxtag')
      ctx = current_update_context(ctxtag)
      ctx.unflatten_end(index_ref, inner)
    del unflatten_ctx.index_ref
    del unflatten_ctx.ctxtag


@jax.tree_util.register_static
@dataclasses.dataclass
class UpdateContext:
  """Context for tracking references across JAX transformation boundaries.

  Used internally by :func:`update_context` to maintain mappings between
  outer and inner graph references during split/merge cycles. This enables
  proper state synchronization when graph structure changes inside transforms.

  See :func:`update_context` for usage details.
  """

  tag: tp.Hashable
  outer_ref_outer_index: RefMap | None
  outer_index_inner_ref: IndexMap | None
  # reverse caches
  outer_index_outer_ref: IndexMap | None
  inner_ref_outer_index: RefMap | None
  static_cache: tp.MutableMapping[tp.Any, StaticCache] | None

  # define hash and eq to make this an opaque object
  def __hash__(self):
    return 0

  def __eq__(self, other):
    return isinstance(other, UpdateContext)

  def flatten_end(self, ref_index: RefMap):
    if self.outer_ref_outer_index is None:
      # outer split (1), store the references
      self.outer_ref_outer_index = ref_index
      self.outer_index_outer_ref = IndexMap.from_refmap(
        self.outer_ref_outer_index
      )
    else:
      # inner split (3), clear index_ref
      self.outer_index_inner_ref = None
      self.inner_ref_outer_index = None

  def unflatten_end(self, index_ref: IndexMap, inner_merge: bool):
    if inner_merge:
      # inner merge (2)
      self.outer_index_inner_ref = index_ref
      self.inner_ref_outer_index = RefMap.from_indexmap(index_ref)


@dataclasses.dataclass
class UpdateContextManager:
  tag: tp.Hashable

  def __enter__(self):
    if GRAPH_CONTEXT.tmp_static_cache is not None:
      # take current static cache
      static_cache = GRAPH_CONTEXT.tmp_static_cache
      GRAPH_CONTEXT.tmp_static_cache = None
    else:
      static_cache = None
    ctx = UpdateContext(
      tag=self.tag,
      outer_ref_outer_index=None,
      outer_index_inner_ref=None,
      outer_index_outer_ref=None,
      inner_ref_outer_index=None,
      static_cache=static_cache,
    )
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
    del ctx.outer_ref_outer_index
    del ctx.outer_index_inner_ref
    del ctx.outer_index_outer_ref
    del ctx.inner_ref_outer_index

    if not stack:
      del GRAPH_CONTEXT.update_context_stacks[self.tag]

  def __call__(self, f: F) -> F:
    @functools.wraps(f)
    def update_context_manager_wrapper(*args, **kwargs):
      with self:
        return f(*args, **kwargs)

    return update_context_manager_wrapper  # type: ignore


def update_context(tag: tp.Hashable):
  """Create a context for complex state updates across JAX transformation boundaries.

  Enables updates to static properties and graph structure that :func:`update`
  cannot handle. Used with :func:`split_context` and :func:`merge_context` to
  track reference mappings between outer and inner graph states.

  Can be used as a context manager or decorator::

                          idxmap
    (2) merge ─────────────────────────────► split (3)
          ▲                                    │
          │               inside               │
          │. . . . . . . . . . . . . . . . . . │ index_mapping
          │               outside              │
          │                                    ▼
    (1) split──────────────────────────────► merge (4)
                          refmap

  Example usage::

    >>> from flax import nnx
    >>> import jax

    >>> class Foo(nnx.Module): pass

    >>> m1 = Foo()
    >>> with nnx.update_context('example'):
    ...   with nnx.split_context('example') as ctx:
    ...     graphdef, state = ctx.split(m1)
    ...   @jax.jit
    ...   def f(graphdef, state):
    ...     with nnx.merge_context('example', inner=True) as ctx:
    ...       m2 = ctx.merge(graphdef, state)
    ...     m2.a = 1
    ...     with nnx.split_context('example') as ctx:
    ...       return ctx.split(m2)
    ...   graphdef_out, state_out = f(graphdef, state)
    ...   with nnx.merge_context('example', inner=False) as ctx:
    ...     m3 = ctx.merge(graphdef_out, state_out)

    >>> assert m1 is m3
    >>> assert m1.a == 1

  Note:
    Structural changes must be static with respect to JIT inputs.
    See `JIT Compilation <https://docs.jax.dev/en/latest/jit-compilation.html>`__
    for more information.

  Args:
    tag: A hashable identifier for this context (used to match split/merge calls).

  Returns:
    An ``UpdateContext`` object which can be used as a context manager (and decorator) for tracking graph state updates.
  """

  return UpdateContextManager(tag=tag)


def current_update_context(tag: tp.Hashable) -> UpdateContext:
  """Get the currently active :class:`UpdateContext` object for a given tag.

  Args:
    tag: The tag identifying the context (must match an active :func:`update_context`).

  Returns:
    The active ``UpdateContext`` object for the given tag.

  Raises:
    ValueError: If no context with the given tag is active.
  """
  if tag not in GRAPH_CONTEXT.update_context_stacks:
    raise ValueError(f'No update context found for tag {tag!r}.')
  return GRAPH_CONTEXT.update_context_stacks[tag][-1]


# --------------------------------------------------------
# Functional API
# --------------------------------------------------------


def _split_state(
  state: FlatState[tp.Any],
  filters: tuple[filterlib.Filter, ...],
) -> tuple[FlatState[tp.Any], tpe.Unpack[tuple[FlatState[tp.Any], ...]]]:
  if not filters:
    return (state,)  # type: ignore[bad-return-type]
  states = state.split(*filters)
  if not isinstance(states, tuple):
    return (states,)  # type: ignore[bad-return-type]
  assert len(states) > 0
  return states  # type: ignore[return-value]


@tp.overload
def split(  # type: ignore[invalid-annotation]
  graph_node: A, /
) -> tuple[GraphDef[A], GraphState]: ...
@tp.overload
def split(  # type: ignore[invalid-annotation]
  graph_node: A, first: filterlib.Filter, /
) -> tuple[GraphDef[A], GraphState]: ...
@tp.overload
def split(  # type: ignore[invalid-annotation]
  graph_node: A,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[
  GraphDef[A],
  GraphState,
  tpe.Unpack[tuple[GraphState, ...]],
]: ...
def split(  # type: ignore[invalid-annotation]
  node: A, *filters: filterlib.Filter
) -> tuple[
  GraphDef[A],
  GraphState,
  tpe.Unpack[tuple[GraphState, ...]],
]:
  """Split a graph node into a :class:`GraphDef` and one or more :class:`GraphState` objects.

  Used with :func:`merge` to convert between stateful and stateless representations,
  enabling compatibility with JAX transformations.

  For non-graph leaves, the ``state`` returned is the leaf value.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)

    >>> model = Model(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)

    >>> jax.tree.map(jnp.shape, params)
    State({
      'batch_norm': {
        'bias': Param(
          value=(2,)
        ),
        'scale': Param(
          value=(2,)
        )
      },
      'linear': {
        'bias': Param(
          value=(3,)
        ),
        'kernel': Param(
          value=(2, 3)
        )
      }
    })

    >>> jax.tree.map(jnp.shape, batch_stats)
    State({
      'batch_norm': {
        'mean': BatchStat(
          value=(2,)
        ),
        'var': BatchStat(
          value=(2,)
        )
      }
    })

  See the `Functional API guide <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Args:
    node: A graph node.
    *filters: Optional filters to partition the state into mutually exclusive substates.

  Returns:
    A ``GraphDef`` and one or more ``GraphState`` objects. The number of ``GraphState``
    objects equals the number of filters passed; if none are passed, a single
    ``GraphState`` is returned.
  """

  graphdef, flat_state = flatten(node)
  flat_states = _split_state(flat_state, filters)
  states = _to_nested_state(graphdef, flat_states)
  return graphdef, *states  # type: ignore[return-value]


def _to_nested_state(
  graphdef: GraphDef[A], flat_states: tp.Iterable[tp.Any]
) -> tuple[tp.Any, ...]:
  if not graphdef.nodes or type(graphdef.nodes[0]) in (
    VariableDef,
    ArrayRefDef,
  ):
    states = tuple(
      flat_state[0][1] if flat_state else State({})
      for flat_state in flat_states
    )
  else:
    states = tuple(
      statelib.from_flat_state(flat_state) for flat_state in flat_states
    )
  return states


def _merge_to_flat_state(states: tp.Iterable[tp.Any]):
  flat_state: list[tuple[PathParts, tp.Any]] = []

  for state in states:
    if isinstance(state, dict | State):
      flat_state.extend(traversals.flatten_to_sequence(state))
    elif isinstance(state, FlatState):
      flat_state.extend(state)
    else:
      flat_state.append(((), state))

  flat_state.sort()
  return [value for _, value in flat_state]


def merge(  # type: ignore[invalid-annotation]
  graphdef: GraphDef[A],
  state: tp.Any,
  /,
  *states: tp.Any,
  copy: bool = False,
) -> A:
  """Merge a :class:`GraphDef` and one or more :class:`GraphState` objects into a graph node.

  The inverse of :func:`split`. Used to reconstruct a graph node from its
  stateless representation after JAX transformations.

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)

    >>> model = Model(nnx.Rngs(0))
    >>> graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
    >>> new_model = nnx.merge(graphdef, params, batch_stats)

    >>> assert isinstance(new_model, Model)
    >>> assert isinstance(new_model.batch_norm, nnx.BatchNorm)
    >>> assert isinstance(new_model.linear, nnx.Linear)

  See the `Functional API guide <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`__
  for more information.

  Args:
    graphdef: A ``GraphDef`` object.
    state: A ``State``, ``dict``, ``FlatState``, or ``list`` of leaf values.
    *states: Additional state objects (if state was split with filters).
    copy: Boolean which controls whether to create new copies of the Variables.
      Defaults to ``False``.

  Returns:
    A graph node reconstructed from the ``GraphDef`` and state.
  """

  if isinstance(state, list):
    if len(states) != 0:
      raise ValueError(f'Only one state can be passed as a list.')
    _state = state
  else:
    _state = _merge_to_flat_state((state, *states))
  node = unflatten(graphdef, _state, copy_variables=copy)
  return node


def update(node, state: tp.Any, /, *states: tp.Any) -> None:
  """Update a graph node's state in-place.

  Applies new values from one or more :class:`GraphState` objects to the graph node,
  modifying it in-place rather than creating a new node.

  Example usage::

    >>> from flax import nnx
    >>> import jax

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> state = nnx.state(model)
    >>> new_state = jax.tree.map(lambda x: x + 1, state)
    >>> nnx.update(model, new_state)

  Args:
    node: A graph node.
    state: A ``GraphState`` object.
    *states: Additional ``GraphState`` objects (if state was split with filters).
  """

  if states:
    if isinstance(node, Variable):
      non_empty_states = [
        _state
        for _state in (state, *states)
        if not isinstance(_state, tp.Mapping) or _state
      ]
      if len(non_empty_states) != 1:
        all_states = (state, *states)
        raise ValueError(
          f'Expected exactly one non-empty state, got: {all_states!r}'
        )
      state = non_empty_states[0]
    else:
      state = statelib.merge_state(state, *states)
  _graph_update_dynamic(node, state)


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
  """Extract the :class:`GraphState` from a graph node without the :class:`GraphDef`.

  Supports filters to partition the state by variable type.

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batch_norm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)

    >>> model = Model(nnx.Rngs(0))

    >>> # Extract state together
    >>> _, state = nnx.split(model)
    >>> assert state == nnx.state(model)

    >>> # or separately
    >>> _, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)
    >>> assert params == nnx.state(model, nnx.Param)
    >>> assert batch_stats == nnx.state(model, nnx.BatchStat)

  Note:
    ``nnx.variables`` is an alias for this function.

  Args:
    node: A graph node.
    *filters: Optional filters to partition the state into mutually exclusive substates.

  Returns:
    A single ``GraphState`` if no filters are passed, otherwise a tuple of
    ``GraphState`` objects (one per filter).
  """

  _, flat_state = flatten(node)
  state = flat_state.to_nested_state()

  states: GraphState | tuple[GraphState, ...]
  if len(filters) == 0:
    states = state  # type: ignore[assignment]
  elif len(filters) == 1:
    states = statelib.filter_state(state, filters[0])
  else:
    states = statelib.filter_state(state, filters[0], filters[1], *filters[2:])

  return states


variables = state


def graphdef(node: tp.Any, /) -> GraphDef[tp.Any]:
  """Extract the :class:`GraphDef` from a graph node without the :class:`GraphState`.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, _ = nnx.split(model)
    >>> assert graphdef == nnx.graphdef(model)

  Args:
    node: A graph node.

  Returns:
    The ``GraphDef`` of the graph node.
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
  """Remove and return variables from a graph node by filter.

  Removes matching variables from the graph node in-place and returns them
  as a :class:`GraphState`. Commonly used to extract intermediate values collected
  via ``sow``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear(x)
    ...     self.sow(nnx.Intermediate, 'i', x)
    ...     return x

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> y = model(jnp.ones((1, 2)))
    >>> intermediates = nnx.pop(model, nnx.Intermediate)
    >>> assert not hasattr(model, 'i')

  Args:
    node: A graph node.
    *filters: One or more filters specifying which variable types to pop.

  Returns:
    A ``GraphState`` containing the popped variables (or tuple of ``GraphState``
    objects if multiple filters).
  """

  if len(filters) == 0:
    raise ValueError('Expected at least one filter')

  id_to_index: dict[int, Index] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  flat_states: tuple[dict[PathParts, LeafType], ...] = tuple(
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
    statelib.from_flat_state(flat_state) for flat_state in flat_states
  )

  if len(states) == 1:
    return states[0]
  else:
    return states


def clone(node: Node, variables: bool = True) -> Node:
  """Create a deep copy of a graph node.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> cloned_model = nnx.clone(model)

    >>> model.bias[...] += 1
    >>> assert (model.bias[...] != cloned_model.bias[...]).all()

  Args:
    node: A graph node.
    variables: Boolean which controls whether to create copies of Variables.
      If ``False``, Variables are shared between original and clone.
      Defaults to ``True``.

  Returns:
    A deep copy of the graph node.
  """

  graphdef, state = split(node)
  return merge(graphdef, state, copy=variables)


def vars_as(
  node: A,
  /,
  *,
  hijax: bool | None = None,
  ref: bool | None = None,
  mutable: bool | None = None,
  only: filterlib.Filter = ...,
  allow_duplicates: bool = False,
) -> A:
  """Convert :class:`Variable` objects in a graph node to use different settings.

  Args:
    node: A graph node.
    hijax: Boolean which controls whether variables use Hijax.
      Defaults to ``None`` (unchanged).
    ref: Boolean which controls whether variables use reference semantics.
      Defaults to ``None`` (unchanged).
    mutable: Boolean which controls whether variables are mutable.
      Defaults to ``None`` (unchanged).
    only: Filter to specify which variables to convert. Defaults to ``...``
      (all variables).
    allow_duplicates: Boolean which controls whether duplicate variables are
      allowed. Defaults to ``False``.

  Returns:
    The graph node with matching ``Variable`` objects converted to use specific settings.
  """

  new_attrs: dict[str, bool] = {}
  if hijax is not None:
    new_attrs['hijax'] = hijax
  if ref is not None:
    new_attrs['ref'] = ref
  if mutable is not None:
    new_attrs['mutable'] = mutable

  def _different_vars(path, x):
    return isinstance(x, Variable) and any(
      getattr(x, attr) != value for attr, value in new_attrs.items()
    )

  only = filterlib.All(_different_vars, only)
  predicate = filterlib.to_predicate(only)

  if not allow_duplicates and (
    all_duplicates := find_duplicates(node, only=only)
  ):
    duplicates_strs = '\n  ---'
    for node_duplicates in all_duplicates:
      for path in node_duplicates:
        path_str = '/'.join(map(str, path))
        duplicates_strs += f'\n  {path_str}'
      duplicates_strs += '\n  ---'
    raise ValueError(f'Found duplicate at paths:{duplicates_strs}')

  def _to_refs(jax_path, x):
    if predicate(jax_to_nnx_path(jax_path), x):
      assert isinstance(x, Variable)
      variable = x.copy(**new_attrs)
      return variable
    return x

  node = jax.tree.map_with_path(
    _to_refs, node, is_leaf=lambda x: isinstance(x, Variable)
  )
  return node


def pure(tree: A) -> A:
  """Strip all :class:`Variable` wrappers from a pytree, keeping only the inner values.

  Useful for serialization or exporting where ``Variable`` metadata is not needed.
  When applied to a :class:`GraphState`, this produces a pytree containing only
  raw array values without ``Variable`` wrappers.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> graphdef, state = nnx.split(model)
    >>> pure_state = nnx.pure(state)  # state with Variable wrappers stripped

  Args:
    tree: A pytree, typically a :class:`GraphState`.

  Returns:
    A new pytree with all ``Variable`` objects replaced by their inner values.
  """

  def _pure_fn(x):
    if isinstance(x, Variable):
      return pure(x.get_raw_value())
    elif variablelib.is_array_ref(x):
      return x[...]
    return x

  return jax.tree.map(
    _pure_fn,
    tree,
    is_leaf=lambda x: isinstance(x, Variable),
  )


def call(
  graphdef_state: tuple[GraphDef[A], GraphState], /
) -> ApplyCaller[tuple[GraphDef[A], GraphState]]:
  """Call a method on a graph node from its ``(GraphDef, GraphState)`` representation.

  Equivalent to ``merge`` → call method → ``split``, but more convenient for
  pure JAX functions. Returns the method output along with the updated state.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Counter(nnx.Module):
    ...   def __init__(self):
    ...     self.count = nnx.Variable(jnp.array(0))
    ...   def increment(self):
    ...     self.count[...] += 1

    >>> counter = Counter()
    >>> counter_state = nnx.split(counter)

    >>> @jax.jit
    ... def step(state):
    ...   _, state = nnx.call(state).increment()
    ...   return state

    >>> counter_state = step(counter_state)
    >>> counter = nnx.merge(*counter_state)
    >>> print(counter.count[...])
    1

  Args:
    graphdef_state: A ``(GraphDef, GraphState)`` tuple.

  Returns:
    A proxy object that can be used to call methods on the underlying graph node.
    Method calls return ``(output, (GraphDef, GraphState))``.
  """

  def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
    node = merge(*graphdef_state)
    method = accessor(node)
    out = method(*args, **kwargs)
    return out, split(node)

  return CallableProxy(pure_caller)  # type: ignore


def set_metadata(
  node: tp.Any, /, *, only: filterlib.Filter = Variable, **metadata: tp.Any
) -> None:
  """Set metadata on :class:`Variable` objects in a graph node in-place.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> nnx.set_metadata(model, differentiable=False, only=nnx.Param)
    >>> assert model.kernel.get_metadata('differentiable') is False

  Args:
    node: A graph node.
    only: Filter to specify which Variables to update. Defaults to all ``Variable`` objects.
    **metadata: Key-value pairs to set as metadata on matching ``Variable`` objects.
  """

  def _set_metadata(path: PathParts, variable: V) -> None:
    del path  # unused
    if isinstance(variable, Variable):
      variable.set_metadata(**metadata)

  # inplace update of variable_state metadata
  map_state(_set_metadata, state(node, only))


def iter_graph(node: tp.Any, /) -> tp.Iterator[tuple[PathParts, tp.Any]]:
  """Iterate over all nested nodes and leaves of a graph node.

  Yields ``(path, value)`` pairs where path is a tuple of keys from the root.
  Each node is visited only once (repeated references are skipped).

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> for path, value in nnx.iter_graph(model):
    ...   if isinstance(value, nnx.Variable):
    ...     print(path, type(value).__name__)
    ('bias',) Param
    ('kernel',) Param

  Args:
    node: A graph node.

  Yields:
    Tuples of ``(path, value)`` for each node and leaf in the graph.
  """

  visited: set[int] = set()
  stack: list[tuple[PathParts, tp.Any, bool]] = [((), node, False)]
  while stack:
    # Yield if the node is either a leaf or has been traversed already.
    path_parts, node, traversed = stack.pop(-1)
    if traversed or not (is_node(node) or isinstance(node, Variable)):
      yield path_parts, node
      continue

    # Skip if the node has been visited already.
    if id(node) in visited:
      continue
    visited.add(id(node))

    # Traverse the node.
    if (node_impl := get_node_impl(node)) is None:
      yield path_parts, node
      continue

    stack.append((path_parts, node, True))
    for key, child in reversed(node_impl.node_dict(node).items()):
      stack.append(((*path_parts, key), child, False))


def recursive_map(f: tp.Callable[[PathParts, tp.Any], tp.Any], node: tp.Any, /):
  """Apply a function to all nodes and leaves of a graph node recursively.

  Creates a clone of the graph and applies ``f(path, value)`` to each node,
  replacing nodes with the function's return value.

  Example usage::

    >>> from flax import nnx

    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> def print_params(path, value):
    ...   if isinstance(value, nnx.Param):
    ...     print(path)
    ...   return value
    >>> _ = nnx.recursive_map(print_params, model)
    ('bias',)
    ('kernel',)

  Args:
    f: A function ``(path, value) -> new_value`` to apply to each node.
    node: A graph node.

  Returns:
    A new graph node with the function applied to all nodes and leaves.
  """

  node = clone(node, variables=False)
  path_parts: PathParts = ()
  visited: set[int] = set()
  results: dict[int, tp.Any] = {}
  return _recursive_map(f, node, path_parts, visited, results)


def _recursive_map(
    f: tp.Callable[[PathParts, tp.Any], tp.Any],
    node: tp.Any,
    path: PathParts,
    visited: set[int],
    results: dict[int, tp.Any],
) -> tp.Any:
  node_id = id(node)
  if node_id in visited:
    if node_id in results:
      return results[node_id]
    path_str = '/'.join(map(str, path))
    raise ValueError(
        f"Found cycle in the graph at path '{path_str}'. Node of type"
        f' {type(node)} has already been visited but has not been returned yet.'
    )
  node_impl = get_node_impl(node)
  if (
      type(node_impl) is GraphNodeImpl
      or isinstance(node, Variable)
      or is_array_ref(node)
  ):
    visited.add(node_id)
  if node_impl is not None:
    for key, value in node_impl.node_dict(node).items():
      new_value = _recursive_map(f, value, (*path, key), visited, results)
      if new_value is not value:
        if node_impl.set_key is not None and value is not new_value:
          node_impl.set_key(node, key, new_value)
        else:
          raise ValueError(
              f"Cannot update key '{key}' for node of type '{type(node)}'"
              ' because the node does not support mutation.'
          )

  new_node = f(path, node)
  results[node_id] = new_node
  return new_node


def find_duplicates(node: tp.Any, /, *, only: filterlib.Filter = ...) -> list[list[PathParts]]:
  """Find nodes or variables that appear multiple times in a graph.

  Traverses the graph and returns paths to nodes/variables that share the same
  identity (i.e., are the same object referenced from multiple locations).

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self):
    ...     self.a = nnx.Param(jnp.array(1.0))
    ...     self.b = self.a  # shared reference

    >>> model = Model()
    >>> duplicates = nnx.find_duplicates(model)
    >>> print(duplicates)
    [[('a',), ('b',)]]

  Args:
    node: A graph node.
    only: Optional filter to specify which nodes or variables to check.

  Returns:
    A list of lists, where each inner list contains the paths to a duplicated
    node or variable.
  """

  node_paths: dict[int, list[PathParts]] = {}
  duplicate_candidate = filterlib.to_predicate(only)
  _node_paths(node, node_paths, (), duplicate_candidate)
  _duplicates = [paths for paths in node_paths.values() if len(paths) > 1]
  return _duplicates


def _node_paths(
  node: tp.Any,
  node_paths: dict[int, list[PathParts]],
  path: PathParts,
  duplicate_candidate: filterlib.Predicate,
  /,
):
  _is_graph_node = is_graph_node(node)
  _is_pytree_node = is_pytree_node(node)
  _is_node_leaf = is_node_leaf(node)

  if _is_graph_node or _is_pytree_node or _is_node_leaf:
    node_id = id(node)
    if node_id in node_paths:
      if (_is_graph_node or _is_node_leaf) and duplicate_candidate(path, node):
        node_paths[node_id].append(path)
      return
    if _is_graph_node or _is_node_leaf:
      node_paths[node_id] = [path]
    node_impl = get_node_impl(node)
    if node_impl is None:
      return
    node_dict = node_impl.node_dict(node)
    for key, value in node_dict.items():
      _node_paths(value, node_paths, (*path, key), duplicate_candidate)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class Static(tp.Generic[A]):
  """An empty pytree node that treats its inner value as static.
  ``value`` must define ``__eq__`` and ``__hash__``.
  """

  value: A


# ---------------------------------------------------------
# Pytree
# ---------------------------------------------------------
class GenericPytree: ...


from jax._src.tree_util import _registry as JAX_PYTREE_REGISTRY


def is_pytree_node(x: tp.Any) -> bool:
  if type(x) in GRAPH_REGISTRY:
    return False
  elif isinstance(x, Variable):
    return False
  elif type(x) in JAX_PYTREE_REGISTRY:
    return True
  elif isinstance(x, tuple):
    return True
  else:
    return False


def _key_path_to_key(key: tp.Any) -> Key:
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx
  elif isinstance(
    key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
  ):
    if not is_key_like(key.key):  # type: ignore[not-supported-yet]
      raise ValueError(
        f'Invalid key: {key.key}. May be due to its type not being hashable or comparable.'
      )
    return key.key
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return key.name
  else:
    return str(key)


def jax_to_nnx_path(jax_path: tuple, /):
  return tuple(_key_path_to_key(part) for part in jax_path)


class IndexesPytreeDef(tp.NamedTuple):
  key_index: HashableMapping[Key, int]
  treedef: jax.tree_util.PyTreeDef


def _flatten_pytree(pytree: tp.Any):
  leaves, treedef = jax.tree_util.tree_flatten_with_path(
    pytree, is_leaf=lambda x: x is not pytree
  )
  nodes = [(_key_path_to_key(path[0]), value) for path, value in leaves]
  key_index = HashableMapping(
    {key: i for i, (key, _) in enumerate(nodes)}, copy=False
  )
  nodes.sort()  # sort by key
  return nodes, IndexesPytreeDef(key_index, treedef)


def _unflatten_pytree(
  nodes: tuple[tuple[Key, tp.Any], ...], metadata: IndexesPytreeDef
):
  # sort to original order
  sorted_nodes = sorted(nodes, key=lambda x: metadata.key_index[x[0]])
  pytree = metadata.treedef.unflatten(value for _, value in sorted_nodes)
  return pytree


PYTREE_NODE_IMPL = PytreeNodeImpl(
  type=GenericPytree,
  flatten=_flatten_pytree,
  unflatten=_unflatten_pytree,  # type: ignore
  set_key=None,
  pop_key=None,
)
def _list_set_key(x: list[tp.Any], key: int, value: tp.Any):
  x[key] = value

# common pytrees
# list
register_pytree_node_type(
  list,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: [value for _, value in nodes],  # type: ignore
  set_key=_list_set_key,  # type: ignore
)
# tuple
register_pytree_node_type(
  tuple,
  flatten=lambda x: (list(enumerate(x)), None),
  unflatten=lambda nodes, _: tuple(value for _, value in nodes),  # type: ignore
)


def _mutable_mapping_set_key(
  x: tp.MutableMapping[Key, tp.Any], key: Key, value: tp.Any
):
  x[key] = value


def _mutable_mapping_pop_key(x: tp.MutableMapping[Key, tp.Any], key: Key):
  x.pop(key)


# dict
register_pytree_node_type(
  dict,
  flatten=lambda x: (sorted(x.items()), None),
  unflatten=lambda nodes, _: dict(nodes),  # type: ignore
  set_key=_mutable_mapping_set_key,
  pop_key=_mutable_mapping_pop_key,
)
# State
register_pytree_node_type(
  State,
  flatten=lambda x: (sorted(x.raw_mapping.items()), None),
  unflatten=lambda nodes, _: State(nodes),  # type: ignore
  set_key=_mutable_mapping_set_key,
  pop_key=_mutable_mapping_pop_key,
)
# None
register_pytree_node_type(
  type(None),
  flatten=lambda x: ([], None),
  unflatten=lambda _, __: None,  # type: ignore
)

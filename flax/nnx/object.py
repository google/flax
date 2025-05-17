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
import inspect
import os
import threading
import typing as tp
from abc import ABCMeta
from copy import deepcopy

from flax.nnx import variablelib
import jax
import numpy as np
import treescope  # type: ignore[import-untyped]
from treescope import rendering_parts

from flax import errors, nnx
from flax.nnx import (
  graph,
  reprlib,
  tracers,
  visualization,
)
from flax import config
from flax.nnx.variablelib import Variable, VariableState, is_mutable_array
from flax.typing import SizeBytes

BUILDING_DOCS = 'FLAX_DOC_BUILD' in os.environ

A = tp.TypeVar('A')
O = tp.TypeVar('O', bound='Object')

DataTag = '__data__'
Data = tp.Annotated[A, DataTag]
Data.__doc__ = """Data marks attributes of a class as pytree data using type annotations.

Data annotations must be used at the class level and will apply to all instances.
The usage of Data is recommended when type annotations are used already present
or required e.g. for dataclasses.

Example::

  from flax import nnx
  import jax
  import dataclasses

  @dataclasses.dataclass
  class Foo(nnx.Object):
    a: nnx.Data[int]  # Annotates `a` as pytree data
    b: str            # `b` is not pytree data

  foo = Foo(a=42, b='hello')

  assert jax.tree.leaves(foo) == [42]
"""
DATA_REGISTRY: set[type] = set()


@dataclasses.dataclass(frozen=True, slots=True)
class DataAttr:
  value: tp.Any


def data(value: A, /) -> A:
  """Annotates a an attribute as pytree data.

  The return value from `data` must be directly assigned to an Object attribute
  which will be registered as a pytree data attribute.

  Example::

    from flax import nnx
    import jax

    class Foo(nnx.Object):
      def __init__(self):
        self.data_attr = nnx.data(42)  # pytree data
        self.static_attr = "hello"     # static attribute

    foo = Foo()

    assert jax.tree.leaves(foo) == [42]

  Args:
    value: The value to annotate as data.

  Returns:
    A value which will register the attribute as data on assignment.

  """
  return DataAttr(value)  # type: ignore[return-value]

def register_data_type(type_: type, /) -> None:
  """Registers a type as pytree data type recognized by Object.

  Custom types registered as data will be automatically recognized
  as data attributes when assigned to an Object attribute. This means
  that values of this type do not need to be wrapped in `nnx.data(...)`
  for Object to mark the attribute its being assigned to as data.

  Example::

    from flax import nnx
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MyType:
      value: int

    nnx.register_data_type(MyType)

    class Foo(nnx.Object):
      def __init__(self, a):
        self.a = MyType(a)  # Automatically registered as data
        self.b = "hello"     # str not registered as data

    foo = Foo(42)

    assert nnx.is_data_type(foo.a)  # True
    assert jax.tree.leaves(foo) == [MyType(value=42)]

  """
  DATA_REGISTRY.add(type_)


def is_data_type(value: tp.Any, /) -> bool:
  """Checks if a value is a registered data type.

  This function checks a the value is registered data type, which means it is
  automatically recognized as pytree data when assigned to an Object attribute.

  Data types are:
  - jax.Arrays
  - np.ndarrays
  - MutableArrays
  - Variables (Param, BatchStat, RngState, etc.)
  - All graph nodes (Object, Module, Rngs, etc.)
  - Any type registered with `nnx.register_data_type`

  Example::

    from flax import nnx
    import jax.numpy as jnp

    module = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    blocks = [module, module, module]

    assert nnx.is_data_type(jnp.array(42))  # Arrays are data
    assert nnx.is_data_type(nnx.Param(1))   # Variables are data
    assert nnx.is_data_type(nnx.Rngs(0))    # Objects are data
    assert nnx.is_data_type(module)         # Objects are data

    assert not nnx.is_data_type(0.)         # float is not data
    assert not nnx.is_data_type(1)          # int is not data
    assert not nnx.is_data_type("hello")    # str is not data
    assert not nnx.is_data_type(blocks)     # list is not data


  Args:
    value: The value to check.

  Returns:
    True if the value is a registered data type, False otherwise.


  """

  return (
    graph.is_node_leaf(value)
    or graph.is_graph_node(value)
    or type(value) in DATA_REGISTRY
  )


def _collect_stats(
  node: tp.Any, node_stats: dict[int, dict[type[Variable], SizeBytes]]
):
  if not graph.is_node(node) and not isinstance(node, Variable):
    raise ValueError(f'Expected a graph node or Variable, got {type(node)!r}.')

  if id(node) in node_stats:
    return

  stats: dict[type[Variable], SizeBytes] = {}
  node_stats[id(node)] = stats

  if isinstance(node, Variable):
    var_type = type(node)
    if issubclass(var_type, nnx.RngState):
      var_type = nnx.RngState
    size_bytes = SizeBytes.from_any(node.raw_value)
    if size_bytes:
      stats[var_type] = size_bytes

  else:
    node_impl = graph.get_node_impl(node)
    assert node_impl is not None
    node_dict = node_impl.node_dict(node)
    for key, value in node_dict.items():
      if id(value) in node_stats:
        continue
      if graph.is_node(value) or isinstance(value, Variable):
        _collect_stats(value, node_stats)
        child_stats = node_stats[id(value)]
        for var_type, size_bytes in child_stats.items():
          if var_type in stats:
            stats[var_type] += size_bytes
          else:
            stats[var_type] = size_bytes


@dataclasses.dataclass
class ObjectContext(threading.local):
  seen_modules_repr: set[int] | None = None
  node_stats: dict[int, dict[type[Variable], SizeBytes]] | None = None


OBJECT_CONTEXT = ObjectContext()


class ObjectState(reprlib.Representable):
  __slots__ = ('_trace_state', '_initializing', '_is_setup')

  def __init__(self, initializing: bool = False, is_setup: bool = False):
    self._trace_state = tracers.TraceState()
    self._initializing = initializing
    self._is_setup = is_setup

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  @property
  def initializing(self) -> bool:
    return self._initializing

  @property
  def is_setup(self) -> bool:
    return self._is_setup

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('trace_state', self._trace_state)

  def __treescope_repr__(self, path, subtree_renderer):
    return visualization.render_object_constructor(
      object_type=type(self),
      attributes={'trace_state': self._trace_state},
      path=path,
      subtree_renderer=subtree_renderer,
    )

def _flatten_object_state(state: ObjectState):
  return (), (state.initializing, state.is_setup)


def _unflatten_object_state(static: tuple[bool, bool], _):
  initializing, setup = static
  return ObjectState(initializing, setup)


jax.tree_util.register_pytree_node(
  ObjectState,
  _flatten_object_state,
  _unflatten_object_state,
)

class ObjectMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _graph_node_meta_call(cls, *args, **kwargs)

  def _object_meta_construct(cls, self, *args, **kwargs):
    self.__init__(*args, **kwargs)


def _graph_node_meta_call(cls: tp.Type[O], *args, **kwargs) -> O:
  node = cls.__new__(cls, *args, **kwargs)
  vars(node)['_object__state'] = ObjectState()
  vars(node)['_object__nodes'] = cls._object__nodes
  cls._object_meta_construct(node, *args, **kwargs)

  return node


@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRepr(reprlib.Representable):
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  @staticmethod
  def from_array(array: jax.Array | np.ndarray) -> ArrayRepr:
    return ArrayRepr(array.shape, array.dtype)

  def __nnx_repr__(self):
    yield reprlib.Object(type='Array', same_line=True)
    yield reprlib.Attr('shape', self.shape)
    yield reprlib.Attr('dtype', self.dtype)

@dataclasses.dataclass(frozen=True, repr=False)
class MutableArrayRepr(reprlib.Representable):
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  @staticmethod
  def from_array(array: jax.Array | np.ndarray) -> MutableArrayRepr:
    return MutableArrayRepr(array.shape, array.dtype)

  def __nnx_repr__(self):
    yield reprlib.Object(type='MutableArray', same_line=True)
    yield reprlib.Attr('shape', self.shape)
    yield reprlib.Attr('dtype', self.dtype)

class Object(reprlib.Representable, metaclass=ObjectMeta):
  """Base class for all NNX objects."""

  if tp.TYPE_CHECKING:
    _object__nodes: frozenset[str]
    _object__is_pytree: bool
    _object__state: ObjectState

  def __init_subclass__(
    cls, *, pytree: bool = config.flax_mutable_array, **kwargs
  ) -> None:
    super().__init_subclass__(**kwargs)

    graph.register_graph_node_type(
      type=cls,
      flatten=cls._graph_node_flatten,
      set_key=cls._graph_node_set_key,  # type: ignore
      pop_key=cls._graph_node_pop_key,  # type: ignore
      create_empty=cls._graph_node_create_empty,
      clear=cls._graph_node_clear,
      init=cls._graph_node_init,  # type: ignore
    )

    cls._object__is_pytree = pytree
    parent_nodes: tp.Iterable[str] = getattr(cls, '_object__nodes', ())

    all_nodes: set[str] = set(parent_nodes)

    all_nodes.add('_object__state')
    # add DataTag attributes
    type_: type
    for name, type_ in cls.__annotations__.items():
      if type_ != tp.ClassVar and DataTag in getattr(type_, '__metadata__', ()):
        all_nodes.add(name)
    cls._object__nodes = frozenset(all_nodes)

    if pytree:
      jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys=cls._object__flatten_with_paths,
        unflatten_func=cls._object__unflatten,
        flatten_func=cls._object__flatten,
      )

    if BUILDING_DOCS:
      # set correct signature for sphinx
      cls.__signature__ = inspect.signature(cls.__init__)

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    self._check_valid_context(
      lambda: f"Cannot mutate '{type(self).__name__}' from different trace level"
    )
    if type(value) is DataAttr:
      value = value.value
      if name not in self._object__nodes:
        self._object__nodes = self._object__nodes.union((name,))
    elif is_data_type(value):
      if name not in self._object__nodes:
        self._object__nodes = self._object__nodes.union((name,))
    elif type(self)._object__is_pytree and name not in self._object__nodes:
      for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array) or is_mutable_array(leaf):
          raise TypeError(
            f"Trying to set '{name}' to a value containing one or more "
            f"jax.Array, but '{name}' is not a registered as data. "
            f"Use 'obj.{name} = nnx.data(...)' to register the attribute as data "
            f"on assignment, or add '{name}: nnx.Data[{type(value).__name__}]' "
            f'to the class definition. '
            f'Got value: {value}'
          )
    object.__setattr__(self, name, value)

  def _check_valid_context(self, error_msg: tp.Callable[[], str]) -> None:
    if not self._object__state.trace_state.is_valid():
      raise errors.TraceContextError(error_msg())

  def __deepcopy__(self: O, memo=None) -> O:
    graphdef, state = graph.split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return graph.merge(graphdef, state)

  def __nnx_repr__(self):
    if OBJECT_CONTEXT.node_stats is None:
      node_stats: dict[int, dict[type[Variable], SizeBytes]] = {}
      _collect_stats(self, node_stats)
      OBJECT_CONTEXT.node_stats = node_stats
      stats = node_stats[id(self)]
      clear_node_stats = True
    else:
      if id(self) not in OBJECT_CONTEXT.node_stats:
        raise RuntimeError(
          f"Object '{type(self).__name__}' not found in node stats. This is a bug."
        )
      stats = OBJECT_CONTEXT.node_stats[id(self)]
      clear_node_stats = False

    if OBJECT_CONTEXT.seen_modules_repr is None:
      OBJECT_CONTEXT.seen_modules_repr = set()
      clear_seen = True
    else:
      clear_seen = False

    if id(self) in OBJECT_CONTEXT.seen_modules_repr:
      yield reprlib.Object(type=type(self), empty_repr='...')
      return

    try:
      if stats:
        stats_repr = ' # ' + ', '.join(
          f'{var_type.__name__}: {size_bytes}'
          for var_type, size_bytes in stats.items()
        )
        if len(stats) > 1:
          total_bytes = sum(stats.values(), SizeBytes(0, 0))
          stats_repr += f', Total: {total_bytes}'
      else:
        stats_repr = ''

      yield reprlib.Object(type=type(self), comment=stats_repr)
      OBJECT_CONTEXT.seen_modules_repr.add(id(self))

      for name, value in vars(self).items():
        if name.startswith('_'):
          continue

        def to_shape_dtype(value):
          if isinstance(value, Variable):
            return value.replace(
              raw_value=jax.tree.map(to_shape_dtype, value.raw_value)
            )
          elif variablelib.is_mutable_array(value) and np.prod(value.shape) > 1:
            return MutableArrayRepr(value.shape, value.dtype)
          elif (
            isinstance(value, (np.ndarray, jax.Array))
            and np.prod(value.shape) > 1
          ):
            return ArrayRepr(value.shape, value.dtype)
          return value

        value = jax.tree.map(to_shape_dtype, value, is_leaf=graph.is_graph_node)
        yield reprlib.Attr(name, value)
    finally:
      if clear_seen:
        OBJECT_CONTEXT.seen_modules_repr = None
      if clear_node_stats:
        OBJECT_CONTEXT.node_stats = None

  def __treescope_repr__(self, path, subtree_renderer):
    from flax import nnx

    if OBJECT_CONTEXT.node_stats is None:
      node_stats: dict[int, dict[type[Variable], SizeBytes]] = {}
      _collect_stats(self, node_stats)
      OBJECT_CONTEXT.node_stats = node_stats
      stats = node_stats[id(self)]
      clear_node_stats = True
    else:
      stats = OBJECT_CONTEXT.node_stats[id(self)]
      clear_node_stats = False

    try:
      if stats:
        stats_repr = ' # ' + ', '.join(
          f'{var_type.__name__}: {size_bytes}'
          for var_type, size_bytes in stats.items()
        )
        if len(stats) > 1:
          total_bytes = sum(stats.values(), SizeBytes(0, 0))
          stats_repr += f', Total: {total_bytes}'

        first_line_annotation = rendering_parts.comment_color(
          rendering_parts.text(f'{stats_repr}')
        )
      else:
        first_line_annotation = None
      children = {}
      for name, value in vars(self).items():
        if name.startswith('_'):
          continue
        children[name] = value

      if isinstance(self, nnx.Module):
        color = treescope.formatting_util.color_from_string(
          type(self).__qualname__
        )
      else:
        color = None
      return visualization.render_object_constructor(
        object_type=type(self),
        attributes=children,
        path=path,
        subtree_renderer=subtree_renderer,
        first_line_annotation=first_line_annotation,
        color=color,
      )
    finally:
      if clear_node_stats:
        OBJECT_CONTEXT.node_stats = None

  # pickle support
  def __getstate__(self):
    return vars(self).copy()

  def __setstate__(self, state):
    vars(self).update(state)

  # -------------------------
  # Pytree Definition
  # -------------------------
  def _object__flatten_with_paths(self):
    obj_vars = vars(self)
    type_nodes = self._object__nodes
    node_names: list[str] = []
    node_attrs: list[tuple[tp.Any, tp.Any]] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in type_nodes:
        node_names.append(name)
        node_attrs.append((jax.tree_util.GetAttrKey(name), value))
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  def _object__flatten(self):
    obj_vars = vars(self)
    type_nodes = self._object__nodes
    node_names: list[str] = []
    node_attrs: list[tp.Any] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in type_nodes:
        node_names.append(name)
        node_attrs.append(value)
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  @classmethod
  def _object__unflatten(
    cls,
    static: tuple[tuple[str, ...], tuple[tuple[str, tp.Any], ...]],
    node_attrs: tp.Iterable[tp.Any],
  ):
    node_names, static_attrs = static
    obj = object.__new__(cls)
    vars_obj = vars(obj)
    vars_obj.update(zip(node_names, node_attrs, strict=True))
    vars_obj.update(static_attrs)
    return obj

  # -------------------------
  # Graph Definition
  # -------------------------
  def _graph_node_flatten(self):
    nodes = vars(self).copy()
    nodes = sorted(nodes.items())
    return nodes, type(self)

  def _graph_node_set_key(self, key: str, value: tp.Any):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    elif (
      hasattr(self, key)
      and isinstance(variable := getattr(self, key), Variable)
      and isinstance(value, VariableState)
    ):
      variable.update_from_state(value)
    else:
      setattr(self, key, value)

  def _graph_node_pop_key(self, key: str):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    return vars(self).pop(key)

  @staticmethod
  def _graph_node_create_empty(node_type: tp.Type[O]) -> O:
    node = object.__new__(node_type)
    return node

  def _graph_node_clear(self):
    vars(self).clear()

  def _graph_node_init(self, attributes: tp.Iterable[tuple[str, tp.Any]]):
    vars(self).update(attributes)

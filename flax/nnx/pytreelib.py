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
import warnings

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
from flax.nnx.variablelib import Variable
from flax.typing import MISSING, Missing, SizeBytes

BUILDING_DOCS = 'FLAX_DOC_BUILD' in os.environ

A = tp.TypeVar('A')
P = tp.TypeVar('P', bound='Pytree')

DataAnnotation = '__data__'
Data = tp.Annotated[A, DataAnnotation]
Data.__doc__ = """Data marks attributes of a class as pytree data using type annotations.

Data annotations must be used at the class level and will apply to all instances.
The usage of Data is recommended when type annotations are used already present
or required e.g. for dataclasses.
"""
DATA_REGISTRY: set[type] = set()

@tp.overload
def data(value: A, /) -> A: ...
@tp.overload
def data(
  *,
  default: A = dataclasses.MISSING,  # type: ignore[assignment]
  default_factory: tp.Callable[[], A] | None = None,  # type: ignore[assignment]
  init: bool = True,
  repr: bool = True,
  hash: bool | None = None,
  compare: bool = True,
  metadata: tp.Mapping[str, tp.Any] | None = None,
  kw_only: bool = False,
) -> tp.Any: ...
def data(value: tp.Any = MISSING, /, **kwargs) -> tp.Any:
  """Annotates a an attribute as pytree data.

  The return value from `data` must be directly assigned to an Object attribute
  which will be registered as a pytree data attribute.

  Example::

    from flax import nnx
    import jax

    class Foo(nnx.Pytree):
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
  if not isinstance(value, Missing) and kwargs:
    raise TypeError(
      'nnx.data() accepts either a single positional argument or keyword'
      ' arguments, but not both.'
    )
  metadata = {'nnx_value': value}
  if 'metadata' in kwargs and kwargs['metadata'] is not None:
    if 'static' in kwargs['metadata']:
      raise ValueError(
        "Cannot use 'static' key in metadata argument for nnx.data."
      )
    metadata.update(kwargs.pop('metadata'))
  metadata['static'] = False
  return dataclasses.field(**kwargs, metadata=metadata)  # type: ignore[return-value]


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

    class Foo(nnx.Pytree):
      def __init__(self, a):
        self.a = MyType(a)  # Automatically registered as data
        self.b = "hello"     # str not registered as data

    foo = Foo(42)

    assert nnx.is_data(foo.a)  # True
    assert jax.tree.leaves(foo) == [MyType(value=42)]
  """
  DATA_REGISTRY.add(type_)


def is_data(value: tp.Any, /) -> bool:
  """Checks if a value is a registered data type.

  This function checks a the value is registered data type, which means it is
  automatically recognized as data when assigned a :class:`flax.nnx.Pytree` attribute.

  Data types are:

  - ``jax.Array``
  - ``np.ndarray``
  - ``ArrayRef``
  - Variables (:class:`flax.nnx.Param`, :class:`flax.nnx.BatchStat`, `nnx.RngState`, etc.)
  - All graph nodes (:class:`flax.nnx.Object`, :class:`flax.nnx.Module`, :class:`flax.nnx.Rngs`, etc.)
  - Any type registered with :func:`flax.nnx.register_data_type`
  - Any pytree that contains at least one node or leaf element of the above


  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ... # ------ DATA ------------
    >>> assert nnx.is_data( jnp.array(0) )                      # Arrays
    >>> assert nnx.is_data( nnx.Param(1) )                      # Variables
    >>> assert nnx.is_data( nnx.Rngs(2) )                       # nnx.Pytrees
    >>> assert nnx.is_data( nnx.Linear(1, 1,rngs=nnx.Rngs(0)) ) # Modules
    ... # ------ STATIC ------------
    >>> assert not nnx.is_data( 'hello' )                       # strings, arbitrary objects
    >>> assert not nnx.is_data( 42 )                            # int, float, bool, complex, etc.
    >>> assert not nnx.is_data( [1, 2.0, 3j, jnp.array(1)] )    # list, dict, tuple, pytrees


  Args:
    value: The value to check.

  Returns:
    A string representing the attribute status.
  """
  return (
    graph.is_node_leaf(value)
    or graph.is_graph_node(value)
    or type(value) in DATA_REGISTRY
  )


StaticAnnotation = '__static__'
Static = tp.Annotated[A, StaticAnnotation]
Static.__doc__ = """Static marks attributes of a class as static using type annotations.
Static annotations must be used at the class level and will apply to all instances.
The usage of Static is recommended when type annotations are used already present
or required e.g. for dataclasses.
"""

@tp.overload
def static(value: A, /) -> A: ...
@tp.overload
def static(
  *,
  default: A = dataclasses.MISSING,  # type: ignore[assignment]
  default_factory: tp.Callable[[], A] | None = None,
  init: bool = True,
  repr: bool = True,
  hash: bool | None = None,
  compare: bool = True,
  metadata: tp.Mapping[str, tp.Any] | None = None,
  kw_only: bool = False,
) -> tp.Any: ...
def static(value: tp.Any = MISSING, /, **kwargs) -> tp.Any:
  """Annotates a an attribute as static.

  The return value from `static` must be directly assigned to an Object
  attribute
  which will be registered as static attribute.

  Example::

    from flax import nnx

    class Foo(nnx.Pytree):
      def __init__(self, a, b):
        self.a = nnx.static(a)  # pytree metadata
        self.b = nnx.data(b)    # pytree data

    foo = Foo("one", "two")

    assert jax.tree.leaves(foo) == ["two"]

  By default ``nnx.Pytree`` will ...
  """
  if not isinstance(value, Missing) and kwargs:
    raise TypeError(
      'nnx.static() accepts either a single positional argument or keyword'
      ' arguments, but not both.'
    )
  metadata = {'nnx_value': value}
  if 'metadata' in kwargs and kwargs['metadata'] is not None:
    if 'static' in kwargs['metadata']:
      raise ValueError(
        "Cannot use 'static' key in metadata argument for nnx.static."
      )
    metadata.update(kwargs.pop('metadata'))
  metadata['static'] = True
  return dataclasses.field(**kwargs, metadata=metadata)  # type: ignore[return-value]

@tp.overload
def dataclass(cls: type[A], /) -> type[A]: ...
@tp.overload
def dataclass(
  *,
  init: bool = True,
  eq: bool = True,
  order: bool = False,
  unsafe_hash: bool = False,
  match_args: bool = True,
  kw_only: bool = False,
  slots: bool = False,
  weakref_slot: bool = False,
) -> tp.Callable[[type[A]], type[A]]: ...

@tp.dataclass_transform(field_specifiers=(dataclasses.field, data, static))
def dataclass(
  cls=None,
  /,
  *,
  init: bool = True,
  eq: bool = True,
  order: bool = False,
  unsafe_hash: bool = False,
  match_args: bool = True,
  kw_only: bool = False,
  slots: bool = False,
  weakref_slot: bool = False,
) -> tp.Any:
  return dataclasses.dataclass(
    cls,
    init=init,
    eq=eq,
    order=order,
    unsafe_hash=unsafe_hash,
    match_args=match_args,
    kw_only=kw_only,
    slots=slots,
    weakref_slot=weakref_slot,
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
    var_type = node.var_type
    if issubclass(var_type, nnx.RngState):
      var_type = nnx.RngState
    size_bytes = SizeBytes.from_any(node.get_raw_value())
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


class PytreeState(reprlib.Representable):
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


def _flatten_pytree_state(state: PytreeState):
  return (), (state.initializing, state.is_setup)


def _unflatten_pytree_state(static: tuple[bool, bool], _):
  initializing, setup = static
  return PytreeState(initializing, setup)


jax.tree_util.register_pytree_node(
  PytreeState,
  _flatten_pytree_state,
  _unflatten_pytree_state,
)


def check_pytree(pytree):
  """Checks if a pytree is valid."""
  if not isinstance(pytree, Pytree):
    raise TypeError(f'Expected a Pytree, got {type(pytree)}.')

  for name, value in vars(pytree).items():
    pytree._check_value(name, value, new_status=None)


class PytreeMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _graph_node_meta_call(cls, *args, **kwargs)

  def _pytree_meta_construct(cls, self, *args, **kwargs):
    self.__init__(*args, **kwargs)

ObjectMeta = PytreeMeta

def _graph_node_meta_call(cls: tp.Type[P], *args, **kwargs) -> P:
  node = cls.__new__(cls, *args, **kwargs)
  vars_obj = vars(node)
  vars_obj['_pytree__state'] = PytreeState()
  vars_obj['_pytree__nodes'] = cls._pytree__nodes
  cls._pytree_meta_construct(node, *args, **kwargs)
  if cls._pytree__is_pytree:
    missing: dict[str, bool] = {}
    for name, value in vars(node).items():
      if name not in vars_obj['_pytree__nodes']:
        missing[name] = is_data(value)
    if missing:
      vars_obj['_pytree__nodes'] = vars_obj['_pytree__nodes'].update(missing)
    check_pytree(node)

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
class VariableRepr(reprlib.Representable):
  var_type: type[Variable]
  value: tp.Any
  metadata: dict[str, tp.Any]

  def __nnx_repr__(self):
    variable = self.var_type._new(self.value, self.metadata)
    yield from variable.__nnx_repr__()


@dataclasses.dataclass(frozen=True, repr=False)
class MutableArrayRepr(reprlib.Representable):
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  @staticmethod
  def from_array(array: jax.Array | np.ndarray) -> MutableArrayRepr:
    return MutableArrayRepr(array.shape, array.dtype)

  def __nnx_repr__(self):
    yield reprlib.Object(type='ArrayRef', same_line=True)
    yield reprlib.Attr('shape', self.shape)
    yield reprlib.Attr('dtype', self.dtype)

def _to_shape_dtype(x):
  if isinstance(x, Variable):
    value = x.get_raw_value()
    metadata = x.get_metadata()
    value = jax.tree.map(_to_shape_dtype, value)
    return VariableRepr(x.var_type, value, metadata)
  elif variablelib.is_array_ref(x) and np.prod(x.shape) > 1:
    return MutableArrayRepr(x.shape, x.dtype)
  elif (
    isinstance(x, (np.ndarray, jax.Array))
    and np.prod(x.shape) > 1
  ):
    return ArrayRepr(x.shape, x.dtype)
  return x

class AttributeStatus(tp.NamedTuple):
  is_data: bool
  explicit: bool


class Pytree(reprlib.Representable, metaclass=PytreeMeta):
  """Base class for all NNX objects."""

  if tp.TYPE_CHECKING:
    _pytree__nodes: graph.HashableMapping[tp.Any, bool]
    _pytree__state: PytreeState
    _pytree__is_pytree: bool

  def __init_subclass__(
      cls,
      *,
      pytree: bool = config.flax_pytree_module,
      **kwargs,
  ) -> None:
    super().__init_subclass__(**kwargs)
    cls._pytree__is_pytree = pytree

    graph.register_graph_node_type(
      type=cls,
      flatten=cls._graph_node_flatten,
      set_key=cls._graph_node_set_key,  # type: ignore
      pop_key=cls._graph_node_pop_key,  # type: ignore
      create_empty=cls._graph_node_create_empty,
      clear=cls._graph_node_clear,
      init=cls._graph_node_init,  # type: ignore
    )

    nodes: dict[str, bool] = dict(getattr(cls, '_pytree__nodes', ()))
    nodes['_pytree__state'] = True
    try:
      type_hints = tp.get_type_hints(
          cls, globals(), {cls.__name__: cls}, include_extras=True
      )
    except NameError:
      type_hints = cls.__annotations__
    # add annotation attributes
    for name, type_ in type_hints.items():
      if isinstance(type_, str):
        if type_.startswith('nnx.Data'):
          warnings.warn(
            f"'Data' is deprecated, please replace:\n\n"
            '  some_field: nnx.Data[SomeType]\n\n'
            f'with:\n\n'
            '  some_field: SomeType = nnx.data()\n\n',
            DeprecationWarning,
            stacklevel=2,
          )
          nodes[name] = True
        elif type_.startswith('nnx.Static'):
          warnings.warn(
            f"'Static' is deprecated, please replace:\n\n"
            '  some_field: nnx.Static[SomeType]\n\n'
            f'with:\n\n'
            '  some_field: SomeType = nnx.static()\n\n',
            DeprecationWarning,
            stacklevel=2,
          )
          nodes[name] = False
      else:
        type_metadata = getattr(type_, '__metadata__', ())
        if DataAnnotation in type_metadata:
          warnings.warn(
            f"'Data' is deprecated, please replace:\n\n"
            '  some_field: nnx.Data[SomeType]\n\n'
            f'with:\n\n'
            '  some_field: SomeType = nnx.data()\n\n',
            DeprecationWarning,
            stacklevel=2,
          )
          nodes[name] = True
        elif StaticAnnotation in type_metadata:
          warnings.warn(
            f"'Static' is deprecated, please replace:\n\n"
            '  some_field: nnx.Static[SomeType]\n\n'
            f'with:\n\n'
            '  some_field: SomeType = nnx.static()\n\n',
            DeprecationWarning,
            stacklevel=2,
          )
          nodes[name] = False

    for name, value in vars(cls).items():
      if isinstance(value, dataclasses.Field) and 'static' in value.metadata:
        if not isinstance(value.metadata['static'], bool):
          raise ValueError(
            f"Invalid 'static' metadata for attribute"
            f" '{cls.__name__}.{name}': expected bool, got"
            f' {type(value.metadata["static"]).__name__}.'
          )
        is_node = not value.metadata['static']
        if name in nodes and nodes[name] != is_node:
          raise ValueError(
            f'Conflicting pytree annotation for attribute'
            f" '{cls.__name__}.{name}': previously registered as"
            f' {"data" if nodes[name] else "static"}, but found'
            f' nnx.{"data" if is_node else "static"}(...) annotation.'
          )
        nodes[name] = is_node

    cls._pytree__nodes = graph.HashableMapping(nodes, copy=False)

    if pytree:
      jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys=cls._pytree__flatten_with_paths,
        unflatten_func=cls._pytree__unflatten,
        flatten_func=cls._pytree__flatten,
      )

    if BUILDING_DOCS:
      # set correct signature for sphinx
      cls.__signature__ = inspect.signature(cls.__init__)

  # Backward compatibility with PR #4863
  @property
  def _object__nodes(self):
    warnings.warn(
      "'_object__nodes' is deprecated, use '_pytree__nodes' instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    return self._pytree__nodes

  @property
  def _object__state(self):
    warnings.warn(
      "'_object__state' is deprecated, use '_pytree__state' instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    return self._pytree__state

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name, value: tp.Any) -> None:
    self._check_valid_context(
      lambda: f"Cannot mutate '{type(self).__name__}' from different trace level"
    )
    data: bool = False
    explicit: bool = False
    if isinstance(value, dataclasses.Field) and 'nnx_value' in value.metadata:
      is_static = value.metadata['static']
      value = value.metadata['nnx_value']
      if self._pytree__is_pytree:
        data = not is_static
        explicit = True
    elif self._pytree__is_pytree:
      data = is_data(value)
    if self._pytree__is_pytree:
      self._check_value(name, value, AttributeStatus(data, explicit))
      if name not in self._pytree__nodes or (
          explicit and self._pytree__nodes[name] != data
      ):
        vars(self)['_pytree__nodes'] = self._pytree__nodes.update({name: data})
    if isinstance(name, str):
      object.__setattr__(self, name, value)
    else:
      vars(self)[name] = value

  def _check_value(self, key, value, new_status: AttributeStatus | None):
    def _has_data(leaves):
      return any(is_data(leaf) for leaf in leaves)

    def _get_annotations(leaves):
      return {
        'static' if leaf.metadata['static'] else 'data'
        for leaf in leaves
        if isinstance(leaf, dataclasses.Field) and 'nnx_value' in leaf.metadata
      }

    def _has_visited(x):
      if id(x) in visited:
        return True
      visited.add(id(x))
      return False

    visited: set[int] = set()
    leaves = jax.tree.leaves(value, is_leaf=_has_visited)
    current_is_data = (
        self._pytree__nodes[key] if key in self._pytree__nodes else False
    )
    existing_attr = key in vars(self)
    if (
        new_status is not None
        and not new_status.explicit
        and new_status.is_data
        and existing_attr
        and not current_is_data
    ):
      raise ValueError(
          f"Cannot assign data value of type '{type(value)}' to static"
          f" attribute '{key}' of Pytree type '{type(self)}'. To override the"
          ' status explicitly wrap the value with nnx.data on assignment:\n\n '
          f' _.{key} = nnx.data(...)\n\n'
      )

    if _has_data(leaves):
      # check no data in nnx.static assignments
      if new_status is not None:
        if not new_status.is_data and new_status.explicit:
          raise ValueError(
              f"Found Arrays in value of type '{type(value)}' annotated with "
              f"nnx.static(...) when setting attribute '{key}' of Pytree type "
              f"'{type(self)}'."
          )
        if not new_status.explicit and not current_is_data and existing_attr:
          base_pytree_type = Pytree
          for t in type(self).mro()[1:]:
            if issubclass(t, nnx.Pytree):
              base_pytree_type = t
              break
          raise ValueError(
            f"Found Arrays on value of type '{type(value)}' assigned to"
            f" static attribute '{key}' of Pytree type '{type(self)}'. Static"
            ' attributes should not contain Array values. Consider one of'
            ' the following options:\n\n1. If the attribute is meant to be'
            ' static, either remove the Array value or wrap it in a static'
            ' container (e.g., using `nnx.static(...)`).\n2. If the'
            ' attribute is meant to be data, wrap the value with nnx.data on'
            f' assignment:\n\n  _.{key} = nnx.data(...)\n\n3. Alternatively,'
            ' annotate the class attribute with nnx.data:\n\n  class'
            f' {type(self).__name__}({base_pytree_type.__name__}):\n   '
            f' {key}: {type(value).__name__} = nnx.data()\n\n4. Disable pytree'
            ' for this class:\n\n  class'
            f' {type(self).__name__}({base_pytree_type.__name__},'
            ' pytree=False):\n\n'
          )
      # check no data in static attributes after __init__
      elif not current_is_data:
        base_pytree_type = Pytree
        for t in type(self).mro()[1:]:
          if issubclass(t, nnx.Pytree):
            base_pytree_type = t
            break
        raise ValueError(
          f'Found unexpected Arrays on value of type {type(value)} in static'
          f" attribute '{key}' of Pytree type '{type(self)}'. This is an"
          ' error starting from Flax version 0.12.0.\nConsider one of the'
          ' following options:\n\n1. If the attribute is meant to be static,'
          ' either remove the Array value or wrap it in a static'
          ' container.\n2. Wrap the value with nnx.data on'
          f' assignment:\n\n  _.{key} = nnx.data(...)\n\n3. Annotate the'
          ' class attribute with nnx.data:\n\n  class'
          f' {type(self).__name__}({base_pytree_type.__name__}):\n    {key}:'
          f' {type(value).__name__} = nnx.data()\n\n4. If the container is a'
          ' list or dict, try using nnx.List(...) or nnx.Dict(...)'
          ' instead.\n5. Disable pytree for this class:\n\n  class'
          f' {type(self).__name__}({base_pytree_type.__name__},'
          f' pytree=False):\n\n'
        )
    if tags := _get_annotations(leaves):
      raise ValueError(
          f'Found unexpected tags {tags} on attribute'
          f" '{type(self).__name__}.{key}'. Values from nnx.data(...)"
          ' and\nnnx.static(...) should be assigned to nnx.Pytree attributes'
          ' directly, they should not be inside other structures. Got value of'
          f" type '{type(value)}' on Pytree of type '{type(self)}'."
      )

  def _check_valid_context(self, error_msg: tp.Callable[[], str]) -> None:
    if not self._pytree__state.trace_state.is_valid():
      raise errors.TraceContextError(error_msg())

  def __deepcopy__(self: P, memo=None) -> P:
    graphdef, state = graph.split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return graph.merge(graphdef, state)

  def __nnx_repr__(self):
    if OBJECT_CONTEXT.node_stats is None or id(self) not in OBJECT_CONTEXT.node_stats:
      node_stats: dict[int, dict[type[Variable], SizeBytes]] = {}
      _collect_stats(self, node_stats)
      OBJECT_CONTEXT.node_stats = node_stats
      stats = node_stats[id(self)]
      clear_node_stats = True
    else:
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
        if str(name).startswith('_'):
          continue

        value = jax.tree.map(_to_shape_dtype, value, is_leaf=graph.is_graph_node)
        yield reprlib.Attr(name, value)
    finally:
      if clear_seen:
        OBJECT_CONTEXT.seen_modules_repr = None
      if clear_node_stats:
        OBJECT_CONTEXT.node_stats = None

  def __treescope_repr__(self, path, subtree_renderer):
    from flax import nnx

    if OBJECT_CONTEXT.node_stats is None or id(self) not in OBJECT_CONTEXT.node_stats:
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
        if str(name).startswith('_'):
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
  _pytree__key_sort_fn: tp.Callable | None = None

  def _pytree__flatten_with_paths(self):
    obj_vars = vars(self)
    node_attributes = self._pytree__nodes
    node_names: list[str] = []
    node_attrs: list[tuple[tp.Any, tp.Any]] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items(), key=self._pytree__key_sort_fn):
      if name in node_attributes and node_attributes[name]:
        node_names.append(name)
        node_attrs.append((
            jax.tree_util.GetAttrKey(name)
            if isinstance(name, str)
            else jax.tree_util.SequenceKey(name),
            value,
        ))
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  def _pytree__flatten(self):
    obj_vars = vars(self)
    node_attributes = self._pytree__nodes
    node_names: list[str] = []
    node_attrs: list[tp.Any] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items(), key=self._pytree__key_sort_fn):
      if name in node_attributes and node_attributes[name]:
        node_names.append(name)
        node_attrs.append(value)
      else:
        static_attrs.append((name, value))

    return node_attrs, (tuple(node_names), tuple(static_attrs))

  @classmethod
  def _pytree__unflatten(
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
    nodes = vars(self)
    nodes = sorted(nodes.items(), key=self._pytree__key_sort_fn)
    return nodes, type(self)

  def _graph_node_set_key(self, key: str, value: tp.Any):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    elif (
      hasattr(self, key)
      and isinstance(variable := getattr(self, key), Variable)
      and isinstance(value, Variable)
    ):
      variable.update_from_state(value)
    else:
      setattr(self, key, value)

  def _graph_node_pop_key(self, key: str):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    return vars(self).pop(key)

  @staticmethod
  def _graph_node_create_empty(node_type: tp.Type[P]) -> P:
    node = object.__new__(node_type)
    return node

  def _graph_node_clear(self):
    vars(self).clear()

  def _graph_node_init(self, attributes: tp.Iterable[tuple[str, tp.Any]]):
    vars(self).update(attributes)

  if tp.TYPE_CHECKING:
    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any: ...


class Object(Pytree, pytree=False):
  """Base class for NNX objects that are not pytrees."""

  def __init_subclass__(cls, **kwargs):
    pytree = kwargs.pop('pytree', False)
    if pytree is not False:
      raise ValueError(
        "Object is not a pytree, but 'pytree' was explicitly set to "
        f'{pytree!r} for type {cls}.'
      )
    super().__init_subclass__(pytree=pytree, **kwargs)

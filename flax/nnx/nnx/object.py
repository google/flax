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
from functools import partial
import threading
import typing as tp
from abc import ABCMeta
from copy import deepcopy


import jax
import numpy as np

from flax.nnx.nnx import (
  errors,
  reprlib,
  tracers,
)
from flax.nnx.nnx import graph
from flax.nnx.nnx.state import State
from flax.nnx.nnx.variables import Variable, VariableState
from flax.typing import Key

O = tp.TypeVar('O', bound='Object')


@dataclasses.dataclass
class GraphUtilsContext(threading.local):
  seen_modules_repr: set[int] | None = None


CONTEXT = GraphUtilsContext()

@dataclasses.dataclass(frozen=True, repr=False)
class Leaf(tp.Generic[O], reprlib.Representable):
  obj: O

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('obj', self.obj)

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]

    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'obj': self.obj},
      path=path,
      subtree_renderer=subtree_renderer,
    )

class ObjectState(reprlib.Representable):
  __slots__ = ('_trace_state', '_initializing', '_is_pytree')

  def __init__(self, initializing: bool, is_pytree: bool):
    self._trace_state = tracers.TraceState()
    self._initializing = initializing
    self._is_pytree = is_pytree

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  @property
  def initializing(self) -> bool:
    return self._initializing

  @property
  def is_pytree(self) -> bool:
    return self._is_pytree

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('trace_state', self._trace_state)

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]
    return treescope.repr_lib.render_object_constructor(
        object_type=type(self),
        attributes={'trace_state': self._trace_state},
        path=path,
        subtree_renderer=subtree_renderer,
    )

class ObjectMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _graph_node_meta_call(cls, *args, **kwargs)

  def _object_meta_construct(cls, self, *args, **kwargs):
    self.__init__(*args, **kwargs)


def _graph_node_meta_call(cls: tp.Type[O], *args, **kwargs) -> O:
  node = cls.__new__(cls, *args, **kwargs)
  vars(node)['_object__state'] = ObjectState(
    initializing=False, is_pytree=cls._object__is_pytree
  )
  cls._object_meta_construct(node, *args, **kwargs)
  return node


@dataclasses.dataclass(frozen=True, repr=False)
class Array:
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  def __repr__(self):
    return f'Array(shape={self.shape}, dtype={self.dtype.name})'


class Object(reprlib.Representable, metaclass=ObjectMeta):
  _object__is_pytree: bool = False

  if tp.TYPE_CHECKING:
    _object__state: ObjectState

  def __init_subclass__(cls) -> None:
    super().__init_subclass__()

    graph.register_graph_node_type(
      type=cls,
      flatten=cls._graph_node_flatten,
      set_key=cls._graph_node_set_key,
      pop_key=cls._graph_node_pop_key,
      create_empty=cls._graph_node_create_empty,
      clear=cls._graph_node_clear,
    )

    jax.tree_util.register_pytree_with_keys(
      cls,
      partial(_flatten_object, with_keys=True),  # type: ignore
      _unflatten_object,  # type: ignore
      flatten_func=partial(_flatten_object, with_keys=False),  # type: ignore
    )

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    self.check_valid_context(
      lambda: f"Cannot mutate '{type(self).__name__}' from different trace level"
    )
    object.__setattr__(self, name, value)

  def check_valid_context(self, error_msg: tp.Callable[[], str]) -> None:
    if not self._object__state.trace_state.is_valid():
      raise errors.TraceContextError(error_msg())

  def __deepcopy__(self: O, memo=None) -> O:
    graphdef, state = graph.split(self)
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return graph.merge(graphdef, state)

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
          elif (
            isinstance(value, (np.ndarray, jax.Array))
            and np.prod(value.shape) > 1
          ):
            return Array(value.shape, value.dtype)
          return value

        value = jax.tree.map(to_shape_dtype, value)
        yield reprlib.Attr(name, repr(value))
    finally:
      if clear_seen:
        CONTEXT.seen_modules_repr = None

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]
    children = {}
    for name, value in vars(self).items():
      if name.startswith('_'):
        continue
      children[name] = value
    return treescope.repr_lib.render_object_constructor(
        object_type=type(self),
        attributes=children,
        path=path,
        subtree_renderer=subtree_renderer,
    )

  # Graph Definition
  def _graph_node_flatten(self):
    nodes = sorted(
      (key, value)
      for key, value in vars(self).items()
      if key != '_object__state'
    )
    metadata = ObjectStaticMetadata(
      type=type(self),
      initializing=self._object__state._initializing,
      is_pytree=self._object__state._is_pytree,
    )
    return nodes, metadata

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
  def _graph_node_create_empty(metadata: ObjectStaticMetadata[O]) -> O:
    node = object.__new__(metadata.type)
    vars(node).update(
      _object__state=ObjectState(
        initializing=metadata.initializing, is_pytree=metadata.is_pytree
      )
    )
    return node

  def _graph_node_clear(self):
    module_state = self._object__state
    module_vars = vars(self)
    module_vars.clear()
    module_vars['_object__state'] = module_state

@dataclasses.dataclass(frozen=True)
class ObjectStaticMetadata(tp.Generic[O]):
  type: tp.Type[O]
  initializing: bool
  is_pytree: bool


# -------------------------
# Pytree Definition
# -------------------------
def _flatten_object(obj: Object, *, with_keys: bool):
  is_pytree = obj._object__state._is_pytree
  if is_pytree:
    graphdef, state = graph.split(obj)
    key_values = sorted(state.raw_mapping.items())
    keys = tuple(key for key, _ in key_values)

    nodes: tuple[tp.Any, ...]
    if with_keys:
      nodes = tuple(
        (jax.tree_util.GetAttrKey(str(key)), value) for key, value in key_values
      )
    else:
      nodes = tuple(value for _, value in key_values)

    return nodes, (keys, graphdef)
  else:
    if with_keys:
      nodes = ((jax.tree_util.GetAttrKey('leaf'), Leaf(obj)),)
      return nodes, None


def _unflatten_object(
  metadata: tuple[tuple[Key, ...], graph.GraphDef[O]] | None,
  children: tuple[tp.Any, ...] | tuple[Leaf[O]],
) -> O:
  if metadata is None:
    if len(children) != 1:
      raise ValueError(f'Expected 1 child, got {len(children)}')
    elif not isinstance(children[0], Leaf):
      raise ValueError(f'Expected Leaf, got {type(children[0])}')
    return children[0].obj
  else:
    _children = tp.cast(tuple[tp.Any, ...], children)
    paths, graphdef = metadata
    return graph.merge(graphdef, State(zip(paths, _children)))

# -------------------------
# pytree API
# -------------------------
@tp.overload
def pytree(node_or_class: tp.Type[O]) -> tp.Type[O]: ...
@tp.overload
def pytree(node_or_class: O) -> O: ...
def pytree(node_or_class: Object | type[Object]):
  if isinstance(node_or_class, type):
    node_or_class._object__is_pytree = True
    return node_or_class
  else:
    obj = graph.clone(node_or_class)
    obj._object__state._is_pytree = True
    return obj
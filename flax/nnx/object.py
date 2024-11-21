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
import threading
import typing as tp
from abc import ABCMeta
from copy import deepcopy


import jax
import numpy as np

from flax.nnx import (
  reprlib,
  tracers,
)
from flax.nnx import graph
from flax.nnx.variablelib import Variable, VariableState
from flax.typing import Key
from flax import errors

G = tp.TypeVar('G', bound='Object')


@dataclasses.dataclass
class GraphUtilsContext(threading.local):
  seen_modules_repr: set[int] | None = None


CONTEXT = GraphUtilsContext()


class ObjectState(reprlib.Representable):
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


def _graph_node_meta_call(cls: tp.Type[G], *args, **kwargs) -> G:
  node = cls.__new__(cls, *args, **kwargs)
  vars(node)['_object__state'] = ObjectState()
  cls._object_meta_construct(node, *args, **kwargs)

  return node


@dataclasses.dataclass(frozen=True, repr=False)
class Array:
  shape: tp.Tuple[int, ...]
  dtype: tp.Any

  def __repr__(self):
    return f'Array(shape={self.shape}, dtype={self.dtype.name})'


class Object(reprlib.Representable, metaclass=ObjectMeta):
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

  def __deepcopy__(self: G, memo=None) -> G:
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
    return nodes, (type(self), self._object__state._initializing)

  def _graph_node_set_key(self, key: Key, value: tp.Any):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    elif (
      hasattr(self, key)
      and isinstance(variable := getattr(self, key), Variable)
      and isinstance(value, VariableState)
    ):
      variable.update_from_state(value)
    else:
      # setattr(self, key, value)
      vars(self)[key] = value

  def _graph_node_pop_key(self, key: Key):
    if not isinstance(key, str):
      raise KeyError(f'Invalid key: {key!r}')
    return vars(self).pop(key)

  @staticmethod
  def _graph_node_create_empty(static: tuple[tp.Type[G], bool]) -> G:
    node_type, initializing = static
    node = object.__new__(node_type)
    vars(node).update(_object__state=ObjectState(initializing))
    return node

  def _graph_node_clear(self):
    module_state = self._object__state
    module_vars = vars(self)
    module_vars.clear()
    module_vars['_object__state'] = module_state

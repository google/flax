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
import functools
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
from flax.nnx.variablelib import Variable
from flax import errors

G = tp.TypeVar('G', bound='Object')
F = tp.TypeVar('F', bound=tp.Callable)


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


def field(
  default=dataclasses.MISSING,
  default_factory=dataclasses.MISSING,
  init=True,
  repr=True,
  hash=None,
  compare=True,
  metadata=None,
  kw_only=dataclasses.MISSING,
  node: bool = False,
):
  metadata = dict(metadata or ())
  if 'node' in metadata:
    raise ValueError('"node" is a reserved metadata key')
  metadata['node'] = node
  return dataclasses.field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
    kw_only=kw_only,
  )  # type: ignore


def node_field(
  default=dataclasses.MISSING,
  default_factory=dataclasses.MISSING,
  init=True,
  repr=True,
  hash=None,
  compare=True,
  metadata=None,
  kw_only=dataclasses.MISSING,
):
  return field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
    kw_only=kw_only,
    node=True,
  )


def static_field(
  default=dataclasses.MISSING,
  default_factory=dataclasses.MISSING,
  init=True,
  repr=True,
  hash=None,
  compare=True,
  metadata=None,
  kw_only=dataclasses.MISSING,
):
  return field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
    kw_only=kw_only,
    node=False,
  )


class Object(reprlib.Representable, metaclass=ObjectMeta):
  if tp.TYPE_CHECKING:
    _object__state: ObjectState
    _object__node_attributes: set[str]

  def __init_subclass__(cls) -> None:
    super().__init_subclass__()
    node_attributes: set[str] = set()
    for name in dir(cls):
      if not name.startswith('__'):
        value = getattr(cls, name)
        if isinstance(value, dataclasses.Field):
          if value.metadata.get('node', False):
            node_attributes.add(name)

    cls._object__node_attributes = node_attributes

    jax.tree_util.register_pytree_with_keys(
      cls,
      lambda node: _object_flatten(node, with_paths=True),
      _object_unflatten,  # type: ignore
      flatten_func=lambda node: _object_flatten(node, with_paths=False),
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

class BiMap(tp.Mapping[tp.Any, int]):
  def __init__(self):
    self._forward: graph.RefMap[tp.Any, int] = graph.RefMap()
    self._backward: dict[int, tp.Any] = {}

  def __setitem__(self, obj: tp.Any, index: int):
    self._forward[obj] = index
    self._backward[index] = obj

  def __getitem__(self, obj):
    return self._forward[obj]

  def get_obj(self, index: int):
    return self._backward[index]

  def __delitem__(self, key):
    value = self._forward.pop(key)
    del self._backward[value]

  def __contains__(self, obj):
    return obj in self._forward

  def contains_index(self, index: int):
    return index in self._backward

  def __len__(self):
    return len(self._forward)

  def __repr__(self):
    return f'BiMap({self._forward})'


@dataclasses.dataclass(slots=True)
class UpdateContext:
  outer_ref_index: BiMap
  inner_ref_index: BiMap | None
  tmp_ref_index: graph.RefMap[tp.Any, int] | None
  tmp_index_ref: dict[int, tp.Any] | None

  # [1]
  def __enter__(self):
    if OBJECT_CONTEXT.update_context is not None:
      raise RuntimeError('Traversal context already set, this is a bug.')

    OBJECT_CONTEXT.update_context = self
    return self

  def __exit__(self, *args):
    ctx = OBJECT_CONTEXT.update_context
    if ctx is None:
      raise RuntimeError('Traversal context not set, this is a bug.')
    # clear references
    del ctx.outer_ref_index
    del ctx.inner_ref_index

    OBJECT_CONTEXT.update_context = None

  def __call__(self, f: F) -> F:
    @functools.wraps(f)
    def update_context_manager_wrapper(*args, **kwargs):
      with self:
        return f(*args, **kwargs)

    return update_context_manager_wrapper  # type: ignore


def update_context():
  """

                        idxmap
  (2) merge ─────────────────────────────► split (3)
        ▲                                    │
        │               inside               │
        │. . . . . . . . . . . . . . . . . . │ index_mapping
        │               outside              │
        │                                    ▼
  (1) split──────────────────────────────► merge (4)
                        refmap

  """
  return UpdateContext(BiMap(), BiMap(), None, None)


def current_update_context() -> UpdateContext:
  if OBJECT_CONTEXT.update_context is None:
    raise ValueError('Traversal context not set')
  return OBJECT_CONTEXT.update_context


@dataclasses.dataclass(slots=True)
class ObjectContext(threading.local):
  update_context: UpdateContext | None = None


OBJECT_CONTEXT = ObjectContext()


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True, slots=True)
class StaticAttribute:
  value: tp.Any


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectMetadata:
  type: type[Object]
  index: int
  index_cache: int | None


# Graph Definition
def _object_flatten(node: Object, *, with_paths: bool):
  context = OBJECT_CONTEXT.update_context
  if context is None:
    raise ValueError('Traversal context not set')

  if context.tmp_ref_index is not None:
    ref_index = context.tmp_ref_index
  else:
    ref_index = context.outer_ref_index

  children = []
  if node in ref_index:
    index = ref_index[node]
  else:
    index = ref_index[node] = len(context.outer_ref_index)

    for name, value in sorted(vars(node).items()):
      if name.startswith('_object__'):
        continue
      if name not in node._object__node_attributes:
        value = StaticAttribute(value)
      if with_paths:
        children.append((jax.tree_util.GetAttrKey(name), value))
      else:
        children.append(value)

  if context.inner_ref_index is not None:
    index_cache = context.inner_ref_index[node]
  else:
    index_cache = None

  metadata = ObjectMetadata(
    type=type(node), index=index, index_cache=index_cache
  )
  return children, metadata


def _object_unflatten(metadata: ObjectMetadata, children: tp.Sequence[tp.Any]):
  context = OBJECT_CONTEXT.update_context
  if context is None:
    raise ValueError('Traversal context not set')

  if (
    context.tmp_index_ref is not None
    and metadata.index in context.tmp_index_ref
  ):
    return context.tmp_index_ref[metadata.index]
  if (
    context.inner_ref_index is not None
    and context.inner_ref_index.contains_index(metadata.index)
  ):
    return context.inner_ref_index.get_obj(metadata.index)

  if (
    metadata.index_cache is not None
    and context.outer_ref_index.contains_index(metadata.index_cache)
  ):
    obj: Object = context.outer_ref_index.get_obj(metadata.index_cache)
    object_state = obj._object__state
    vars(obj).clear()
  else:
    # [2]
    obj = object.__new__(metadata.type)
    object_state = ObjectState()

  if context.inner_ref_index is not None:
    context.inner_ref_index[obj] = metadata.index
  elif context.tmp_index_ref is not None:
    context.tmp_index_ref[metadata.index] = obj
  else:
    raise RuntimeError(
      f'Either inner_ref_index or tmp_index_ref must be set, got {context}'
    )

  vars(obj).update(
    {
      name: value.value if type(value) is StaticAttribute else value
      for name, value in children
    },
    _object__state=object_state,
  )
  return obj
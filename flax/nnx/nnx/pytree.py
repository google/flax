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
import typing as tp

import jax
from flax.nnx.nnx import graph, reprlib
from flax.typing import Missing

A = tp.TypeVar('A')


@dataclasses.dataclass(frozen=True)
class ObjectProxy:
  _nnx__root_pytree: Pytree
  _nnx__root_obj: tp.Any
  _nnx__current_obj: tp.Any

  def __call__(self, *args, **kwargs):
    with self._nnx__root_pytree as obj:
      if obj is not self._nnx__root_obj:
        raise ValueError('ObjectProxy is no longer valid')
      return self._nnx__current_obj(*args, **kwargs)

  def __getattr__(self, name: str) -> tp.Any:
    value = getattr(self._nnx__current_obj, name)

    if graph.is_node(value) or inspect.ismethod(value):
      return ObjectProxy(
        self._nnx__root_pytree,
        self._nnx__root_obj,
        value,
      )
    else:
      return value

  def __getitem__(self, key: tp.Any) -> tp.Any:
    value = self._nnx__current_obj[key]

    if graph.is_node(value) or inspect.ismethod(value):
      return ObjectProxy(
        self._nnx__root_pytree,
        self._nnx__root_obj,
        value,
      )
    else:
      return value


class Pytree(tp.Generic[A], reprlib.Representable):
  """A a proxy object that wraps Modules and implements the pytree protocol. Its more
  convenient than ``split`` / ``merge`` when integrating with APIs that accept pytrees
  and dont have any specific semantics (all states treated equally) as ``Pytree`` can
  call any method and access any (nested) attributes from the underlaying object it wraps,
  making its usage more natural::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>> import jax
    ...
    >>> pt_linear = nnx.Pytree(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
    >>> pt_linear = jax.tree.map(lambda x: x, pt_linear)  # it's a pytree!
    ...
    >>> pt_linear.in_features  # access attributes
    2
    >>> pt_linear.out_features
    3
    >>> pt_linear.kernel.shape
    (2, 3)
    >>> y = pt_linear(jnp.ones((1, 2)))  # call methods
    >>> y.shape
    (1, 3)

  ``Pytree`` wont share its state updates with the original object since the input is
  ``split`` during construction and only the resulting `GraphDef` and `State` are stored.
  The underlyng object is only materialized (and cached) when accessing attributes. When
  methods are called the `GraphDef` and `State` are updated in place after the method
  finishes.

  ``Pytree`` instances can be used as a context manager to get access to the underlying
  object when needed for manual modifications, upon termination the pytree is updated
  with the new state.

    >>> @dataclasses.dataclass
    ... class Counter(nnx.Module):
    ...   count: nnx.BatchStat[int]
    ...
    >>> pt_counter = nnx.Pytree(Counter(nnx.BatchStat(0)))
    >>> pt_counter.count.value
    0
    >>> with pt_counter as counter:
    ...   counter.count += 1
    ...
    >>> pt_counter.count.value
    1

  The inner ``GraphDef`` and ``State`` can be accessed directly via the ``.graphdef`` and
  ``.state`` properties.

    >>> pt_counter.graphdef
    NodeDef(...)
    >>> pt_counter.state
    State(...)

  """

  __slots__ = ('_nnx__graphdef', '_nnx_state', '_nnx__obj')

  def __init__(self, obj: A):
    self._nnx__graphdef, self._nnx_state = graph.split(obj)
    self._nnx__obj: tp.Any = Missing

  @classmethod
  def from_split(
    cls, graphdef: graph.GraphDef[A], state: graph.State, *states: graph.State
  ) -> Pytree:
    if states:
      state = graph.State.merge(state, *states)
    pytree = object.__new__(cls)
    pytree._nnx__graphdef = graphdef
    pytree._nnx_state = state
    pytree._nnx__obj = Missing
    return pytree

  @property
  def graphdef(self) -> graph.GraphDef[A]:
    return self._nnx__graphdef

  @property
  def state(self):
    return self._nnx_state

  def _nnx__get_obj(self) -> A:
    if self._nnx__obj is Missing:
      self._nnx__obj = graph.merge(self._nnx__graphdef, self._nnx_state)
    return self._nnx__obj

  def __enter__(self):
    return self._nnx__get_obj()

  def __exit__(self, *args):
    self._nnx__graphdef, self._nnx_state = graph.split(self._nnx__obj)
    self._nnx__obj = Missing

  def __call__(self, *args, **kwargs):
    with self as obj:
      return obj(*args, **kwargs)  # type: ignore

  def __getattr__(self, name: str) -> tp.Any:
    obj = self._nnx__get_obj()
    proxy = ObjectProxy(self, obj, obj)
    return getattr(proxy, name)

  def __getitem__(self, key: tp.Any) -> tp.Any:
    obj = self._nnx__get_obj()
    proxy = ObjectProxy(self, obj, obj)
    return proxy[key]

  def __iter__(self) -> tp.Any:
    return iter((self._nnx__graphdef, self._nnx_state))

  def __nnx_repr__(self) -> tp.Iterator[reprlib.Object | reprlib.Attr]:
    yield reprlib.Object(type(self))
    yield reprlib.Attr('obj', self._nnx__get_obj())

  def __treescope_repr__(self, path, subtree_renderer):
    import treescope  # type: ignore[import-not-found,import-untyped]

    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'obj': self._nnx__get_obj()},
      path=path,
      subtree_renderer=subtree_renderer,
    )


def _pytree_flatten_with_keys(x: Pytree):
  items = sorted(x._nnx_state._mapping.items())
  children = tuple((jax.tree_util.DictKey(key), value) for key, value in items)
  graphdef = x._nnx__graphdef
  keys = tuple(key for key, _ in items)
  return children, (keys, graphdef)


def _pytree_unflatten(
  static: tuple[tuple[graph.Key, ...], graph.GraphDef],
  leaves: tuple[tp.Any, ...],
):
  keys, graphdef = static
  state = graph.State(zip(keys, leaves))
  return Pytree.from_split(graphdef, state)


jax.tree_util.register_pytree_with_keys(
  Pytree,
  _pytree_flatten_with_keys,
  _pytree_unflatten,  # type: ignore[arg-type]
)

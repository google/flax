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
import typing as tp
from abc import ABCMeta

import jax
from jax._src import core as jax_core
from jax._src.state.types import AbstractRef
import numpy as np
from flax.experimental.nx import variablelib
from flax.nnx import reprlib

BUILDING_DOCS = 'FLAX_DOC_BUILD' in os.environ

A = tp.TypeVar('A', bound=type)
P = tp.TypeVar('P', bound='Pytree')

class PytreeMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _pytree_meta_call(cls, *args, **kwargs)

@dataclasses.dataclass(repr=False)
class VariableRepr:
  variable: variablelib.Variable

  def __repr__(self) -> str:
    type_str = type(self.variable).__name__
    fields_strs: list[str] = []
    if np.prod(self.variable.shape) <= 1:
      fields_strs.append(f'value={self.variable[...]!r}')
    fields_strs.extend(
      [
        f'shape={self.variable.shape!r}',
        f'dtype={self.variable.dtype!r}',
        f'mutable={self.variable.mutable!r}',
      ]
    )
    fields_strs.extend(f'{k}={v!r}' for k, v in self.variable._metadata.items())
    return f'{type_str}(\n  ' + ',\n  '.join(fields_strs) + '\n)'


def _pytree_meta_call(cls: type[P], *args, **kwargs) -> P:
  pytree = cls.__new__(cls, *args, **kwargs)
  vars(pytree)['_pytree__is_mutable'] = True
  try:
    pytree.__init__(*args, **kwargs)
    pytree._pytree__check_attributes()
    return pytree
  finally:
    vars(pytree).pop('_pytree__is_mutable', None)


class Pytree(reprlib.Representable, metaclass=PytreeMeta):
  if tp.TYPE_CHECKING:
    __nodes__: tuple[str, ...]
    _pytree__nodes: frozenset[str]

  def replace(self, **kwargs):
    pytree = object.__new__(type(self))
    attributes = vars(self).copy()
    attributes.update(kwargs)
    vars(pytree).update(attributes)
    pytree._pytree__check_attributes()
    return pytree

  def _pytree__check_attributes(self):
    # check for arrays in non-node attributes
    for name, value in vars(self).items():
      if name not in self._pytree__nodes:
        for leaf in jax.tree.leaves(value):
          if isinstance(leaf, jax.Array | AbstractRef | jax_core.MutableArray):
            raise TypeError(
              f"Trying to set '{name}' to a value containing one or more "
              f"jax.Array, but '{name}' is not a registered in __nodes__. "
              f'Got value: {value}'
            )

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    for name, value in sorted(vars(self).items()):
      if name.startswith('_'):
        continue

      def to_variable_repr(x):
        if isinstance(x, variablelib.Variable):
          return VariableRepr(x)
        return x

      value = jax.tree.map(
        to_variable_repr,
        value,
        is_leaf=lambda x: isinstance(x, variablelib.Variable),
      )
      yield reprlib.Attr(name, value)

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    if '_pytree__is_mutable' not in vars(self):
      raise AttributeError(
        f"Cannot set attribute '{name}' on immutable instance of "
        f'{type(self).__name__}. This usually happens when you try to '
        f'modify an instance after it has been frozen (e.g., after calling '
        f'__init__).'
      )
    object.__setattr__(self, name, value)

  def __init_subclass__(cls, **kwargs) -> None:
    super().__init_subclass__(**kwargs)

    all_nodes: list[str] = (
      list(cls._pytree__nodes) if hasattr(cls, '_pytree__nodes') else []
    )
    current_nodes = vars(cls).get('__nodes__', ())
    all_nodes.extend(current_nodes)
    cls._pytree__nodes = frozenset(all_nodes)

    jax.tree_util.register_pytree_with_keys(
      cls,
      flatten_with_keys=cls._object__flatten_with_paths,
      unflatten_func=cls._object__unlatten,
      flatten_func=cls._object__flatten,
    )

    if BUILDING_DOCS:
      # set correct signature for sphinx
      cls.__signature__ = inspect.signature(cls.__init__)

  def _object__flatten_with_paths(self):
    obj_vars = vars(self)
    type_nodes = type(self)._pytree__nodes
    node_names: list[str] = []
    node_attrs: list[tuple[tp.Any, tp.Any]] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in type_nodes:
        node_names.append(name)
        node_attrs.append((jax.tree_util.GetAttrKey(name), value))
      else:
        static_attrs.append((name, value))

    return node_attrs, (node_names, static_attrs)

  def _object__flatten(self):
    obj_vars = vars(self)
    type_nodes = type(self)._pytree__nodes
    node_names: list[str] = []
    node_attrs: list[tp.Any] = []
    static_attrs: list[tuple[str, tp.Any]] = []
    for name, value in sorted(obj_vars.items()):
      if name in type_nodes:
        node_names.append(name)
        node_attrs.append(value)
      else:
        static_attrs.append((name, value))

    return node_attrs, (node_names, static_attrs)

  @classmethod
  def _object__unlatten(
    cls,
    static: tuple[list[str], list[tuple[str, tp.Any]]],
    node_attrs: tp.Iterable[tp.Any],
  ):
    node_names, static_attrs = static
    obj = cls.__new__(cls)
    vars_obj = vars(obj)
    vars_obj.update(zip(node_names, node_attrs))
    vars_obj.update(static_attrs)
    return obj

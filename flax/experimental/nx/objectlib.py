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

from abc import ABCMeta
import inspect
import os
import typing as tp
import jax

from flax.nnx import reprlib

BUILDING_DOCS = 'FLAX_DOC_BUILD' in os.environ

A = tp.TypeVar('A', bound=type)
O = tp.TypeVar('O', bound='Object')


class ObjectMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _object_meta_call(cls, *args, **kwargs)


def _object_meta_call(cls: tp.Type[O], *args, **kwargs) -> O:
  obj = cls.__new__(cls, *args, **kwargs)
  vars(obj)['_object__is_mutable'] = True
  try:
    obj.__init__(*args, **kwargs)
    return obj
  finally:
    vars(obj).pop('_object__is_mutable', None)


class Object(reprlib.Representable, metaclass=ObjectMeta):
  if tp.TYPE_CHECKING:
    __nodes__: tuple[str, ...]
    _object__nodes: frozenset[str]

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    if '_object__is_mutable' not in vars(self):
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
      list(cls._object__nodes) if hasattr(cls, '_object__nodes') else []
    )
    current_nodes = vars(cls).get('__nodes__', ())
    all_nodes.extend(current_nodes)
    cls._object__nodes = frozenset(all_nodes)

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
    type_nodes = type(self)._object__nodes
    node_names: list[str] = []
    node_attrs: list[tuple[jax.tree_util.GetAttrKey, tp.Any]] = []
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
    type_nodes = type(self)._object__nodes
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

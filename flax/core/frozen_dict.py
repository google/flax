# Copyright 2020 The Flax Authors.
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

"""Frozen Dictionary."""

from typing import TypeVar, Mapping, Dict

import jax
from flax import serialization

K = TypeVar('K')
V = TypeVar('V')

@jax.tree_util.register_pytree_node_class
class FrozenDict(Mapping[K, V]):
  """An immutable variant of dictionaries.
  """
  __slots__ = ('_dict', '_leaves', '_treedef', '_hash')

  def __init__(self, *args, **kwargs):
    if len(args) == 2:
      self._treedef, self._leaves = args
      self._dict = None
    else:
      self._dict = dict(*args, **kwargs)
      self._treedef = None
      self._leaves = None
    self._hash = None
    
  def __getitem__(self, key):
    self._ensure_dict()
    return self._dict[key]

  def __setitem__(self, key, value):
    raise ValueError('FrozenDict is immutable.')

  def __contains__(self, key):
    self._ensure_dict()
    return key in self._dict

  def __iter__(self):
    self._ensure_dict()
    return iter(self._dict)

  def __len__(self):
    self._ensure_dict()
    return len(self._dict)

  def __repr__(self):
    self._ensure_dict()
    return 'FrozenDict(%r)' % unfreeze(self._dict)

  def __hash__(self):
    if self._hash is None:
      self._ensure_dict()
      h = 0
      for key, value in self._dict.items():
        h ^= hash((key, value))
      self._hash = h
    return self._hash

  def copy(self, **add_or_replace):
    return type(self)(self, **add_or_replace)

  def items(self):
    self._ensure_dict()
    return self._dict.items()

  def _ensure_dict(self):
    if self._dict is None:
      self._dict = jax.tree_unflatten(self._treedef, self._leaves)

  def tree_flatten(self):
    if self._treedef is None:
      self._leaves, self._treedef = jax.tree_flatten(self._dict)
    return self._leaves, self._treedef

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(aux_data, children)


def freeze(x: Dict[K, V]) -> FrozenDict[K, V]:
  """Freeze a nested dict."""
  if not isinstance(x, dict):
    return x
  temp = {}
  for key, value in x.items():
    temp[key] = freeze(value)
  return FrozenDict(temp)


def unfreeze(x: FrozenDict[K, V]) -> Dict[K, V]:
  if not isinstance(x, FrozenDict) and not isinstance(x, dict):
    return x
  temp = {}
  for key, value in x.items():
    temp[key] = unfreeze(value)
  return temp


def _frozen_dict_state_dict(xs):
  return {key: serialization.to_state_dict(value) for key, value in xs.items()}


def _restore_frozen_dict(xs, states):
  return FrozenDict(
    {key: serialization.from_state_dict(value, states[key])
     for key, value in xs.items()})


serialization.register_serialization_state(
    FrozenDict,
    _frozen_dict_state_dict,
    _restore_frozen_dict)

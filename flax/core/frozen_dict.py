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

from typing import TypeVar, Mapping, Dict, Tuple

from flax import serialization
import jax


K = TypeVar('K')
V = TypeVar('V')


@jax.tree_util.register_pytree_node_class
class FrozenDict(Mapping[K, V]):
  """An immutable variant of the Python dict."""
  __slots__ = ('_dict', '_hash')

  def __init__(self, *args, **kwargs):
    self._dict = dict(*args, **kwargs)
    self._hash = None

  def __getitem__(self, key):
    v = self._dict[key]
    if isinstance(v, dict):
      return FrozenDict(v)
    return v

  def __setitem__(self, key, value):
    raise ValueError('FrozenDict is immutable.')

  def __contains__(self, key):
    return key in self._dict

  def __iter__(self):
    return iter(self._dict)

  def __len__(self):
    return len(self._dict)

  def __repr__(self):
    return 'FrozenDict(%r)' % self._dict

  def __hash__(self):
    if self._hash is None:
      h = 0
      for key, value in self.items():
        h ^= hash((key, value))
      self._hash = h
    return self._hash

  def copy(self, add_or_replace: Mapping[K, V]) -> 'FrozenDict[K, V]':
    """Create a new FrozenDict with additional or replaced entries."""
    return type(self)(self, **unfreeze(add_or_replace))

  def items(self):
    for key in self._dict:
      yield (key, self[key])

  def pop(self, key: K) -> Tuple['FrozenDict[K, V]', V]:
    """Create a new FrozenDict where one entry is removed.

    Example::

      state, params = variables.pop('params')

    Args:
      key: the key to remove from the dict
    Returns:
      A pair with the new FrozenDict and the removed value.
    """
    value = self[key]
    new_dict = dict(self._dict)
    new_dict.pop(key)
    new_self = type(self)(new_dict)
    return new_self, value

  def unfreeze(self) -> Dict[K, V]:
    return unfreeze(self)

  def tree_flatten(self):
    return (self._dict,), ()

  @classmethod
  def tree_unflatten(cls, _, data):
    return cls(*data)


def freeze(xs: Dict[K, V]) -> FrozenDict[K, V]:
  """Freeze a nested dict.

  Makes a nested `dict` immutable by transforming it into `FrozenDict`.
  """
  # Turn the nested FrozenDict into a dict. This way the internal data structure
  # of FrozenDict does not contain any FrozenDicts.
  # instead we create those lazily in `__getitem__`.
  # As a result tree_flatten/unflatten will be fast
  # because it operates on native dicts.
  xs = unfreeze(xs)
  return FrozenDict(xs)


def unfreeze(x: FrozenDict[K, V]) -> Dict[K, V]:
  """Unfreeze a FrozenDict.

  Makes a mutable copy of a `FrozenDict` mutable by transforming
  it into (nested) dict.
  """
  if not isinstance(x, (FrozenDict, dict)):
    return x
  ys = {}
  for key, value in x.items():
    ys[key] = unfreeze(value)
  return ys


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

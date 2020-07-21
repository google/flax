# Lint as: python3
"""Frozen dict.
"""

from typing import TypeVar, Mapping, Dict

import jax
from flax import serialization

K = TypeVar('K')
V = TypeVar('V')


class FrozenDict(Mapping[K, V]):
  """An immutable variant of dictionaries.
  """
  __slots__ = ('_dict', '_hash')

  def __init__(self, *args, **kwargs):
    self._dict = dict(*args, **kwargs)
    self._hash = None

  def __getitem__(self, key):
    return self._dict[key]

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
      for key, value in self._dict.items():
        h ^= hash((key, value))
      self._hash = h
    return self._hash

  def copy(self, **add_or_replace):
    return type(self)(self, **add_or_replace)

  def items(self):
    return self._dict.items()

jax.tree_util.register_pytree_node(
    FrozenDict,
    lambda x: ((dict(x),), ()),
    lambda _, data: FrozenDict(data[0]))


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


serialization.register_serialization_state(
    FrozenDict, unfreeze, lambda _, x: freeze(x))

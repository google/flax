"""Simple syntactic wrapper for nested dictionaries to allow dot traversal."""

from jax import tree_util
from flax.core.frozen_dict import FrozenDict


def is_leaf(x):
  return tree_util.treedef_is_leaf(tree_util.tree_flatten(x)[1])


class DotGetter:
  """Dot-notation helper for interactive access of variable trees."""
  def __init__(self, data):
    object.__setattr__(self, '_data', data)

  def __getattr__(self, key):
    """Returns leaves unwrapped."""
    if is_leaf(self._data[key]):
      return self._data[key]
    else:
      return DotGetter(self._data[key])

  def __getitem__(self, key):
    if is_leaf(self._data[key]):
      return self._data[key]
    else:
      return DotGetter(self._data[key])

  def __setitem__(self, key, val):
    self._data[key] = val

  def __setattr__(self, key, val):
    self._data[key] = val

  def __dir__(self):
    if isinstance(self._data, dict):
      return list(self._data.keys())
    elif isinstance(self._data, FrozenDict):
      return list(self._data._dict.keys())
    else:
      return []

  def __repr__(self):
    return f'{self._data}'

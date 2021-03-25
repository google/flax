# Copyright 2021 The Flax Authors.
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

from typing import Any, TypeVar, Mapping, Dict, Tuple

from flax import serialization
import jax


K = TypeVar('K')
V = TypeVar('V')


def _indent(x, num_spaces):
  indent_str = ' ' * num_spaces
  lines = x.split('\n')
  assert lines[-1] == ''
  # skip the final line because it's empty and should not be indented.
  return '\n'.join(indent_str + line for line in lines[:-1]) + '\n'


@jax.tree_util.register_pytree_node_class
class FrozenDict(Mapping[K, V]):
  """An immutable variant of the Python dict."""
  __slots__ = ('_dict', '_hash')

  def __init__(self, *args, __unsafe_skip_copy__=False, **kwargs):
    # make sure the dict is as
    xs = dict(*args, **kwargs)
    if __unsafe_skip_copy__:
      self._dict = xs
    else:
      self._dict = _prepare_freeze(xs)

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
    return self.pretty_repr()

  def __reduce__(self):
    return FrozenDict, (self.unfreeze(),)

  def pretty_repr(self, num_spaces=4):
    """Returns an indented representation of the nested dictionary."""
    def pretty_dict(x):
      if not isinstance(x, dict):
        return repr(x)
      rep = ''
      for key, val in x.items():
        rep += f'{key}: {pretty_dict(val)},\n'
      if rep:
        return '{\n' + _indent(rep, num_spaces) + '}'
      else:
        return '{}'
    return f'FrozenDict({pretty_dict(self._dict)})'

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
    """Unfreeze this FrozenDict.

    Returns:
      An unfrozen version of this FrozenDict instance.
    """
    return unfreeze(self)

  def tree_flatten(self) -> Tuple[Tuple[Dict[Any, Any]], Tuple[()]]:
    """Flattens this FrozenDict.

    Returns:
      A flattened version of this FrozenDict instance.
    """
    return (self._dict,), ()

  @classmethod
  def tree_unflatten(cls, _, data):
    # data is already deep copied due to tree map mechanism
    # we can skip the deep copy in the constructor
    return cls(*data, __unsafe_skip_copy__=True)


def _prepare_freeze(xs: Any) -> Any:
  """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
  if isinstance(xs, FrozenDict):
    # we can safely ref share the internal state of a FrozenDict
    # because it is immutable.
    return xs._dict  # pylint: disable=protected-access
  if not isinstance(xs, dict):
    # return a leaf as is.
    return xs
  # recursively copy dictionary to avoid ref sharing
  return {key: _prepare_freeze(val) for key, val in xs.items()}


def freeze(xs: Mapping[Any, Any]) -> FrozenDict[Any, Any]:
  """Freeze a nested dict.

  Makes a nested `dict` immutable by transforming it into `FrozenDict`.
  """
  return FrozenDict(xs)


def unfreeze(x: FrozenDict[Any, Any]) -> Dict[Any, Any]:
  """Unfreeze a FrozenDict.

  Makes a mutable copy of a `FrozenDict` mutable by transforming
  it into (nested) dict.
  """
  if isinstance(x, FrozenDict):
    # deep copy internal state of a FrozenDict
    # the dict branch would also work here but
    # it is much less performant because jax.tree_map
    # uses an optimized C implementation.
    return jax.tree_map(lambda y: y, x._dict)
  elif isinstance(x, dict):
    ys = {}
    for key, value in x.items():
      ys[key] = unfreeze(value)
    return ys
  else:
    return x


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

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


# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A utility for traversing immutable datastructures.

A Traversal can be used to iterate and update complex data structures.
Traversals take in an object and return a subset of its contents.
For example, a Traversal could select an attribute of an object::

  x = Foo(foo=1)
  traverse_util.TraverseAttr('foo').iterate(x) # [1]


More complex traversals can be constructed using composition.
It is often useful to start from the identity traversal and use a method chain
to construct the intended Traversal::

  data = [{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]
  traversal = traverse_util.t_identity.each()['foo']
  traversal.iterate(data) # [1, 3]

Traversals can also be used to make changes using the `update` method::

  data = {'foo': Foo(bar=2)}
  traversal = traverse_util.t_identity['foo'].bar
  traversal.update(lambda x: x + x, data) # {'foo': Foo(bar=4)}

Traversals never mutate the original data. Therefore, an update essentially
returns a copy of the data including the provided updates.
"""

import abc
import copy
import dataclasses

import jax

from . import struct


# the empty node is a struct.dataclass to 
# be compatible with JAX.
@struct.dataclass
class _EmptyNode:
  pass

empty_node = _EmptyNode()


def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None):
  """Flatten a nested dictionary.

  The nested keys are flattened to a tuple.
  See `unflatten_dict` on how to restore the
  nested dictionary structure.

  Example::

    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = flatten_dict(xs)
    print(flat_xs)
    # {
    #   ('foo',): 1,
    #   ('bar', 'a'): 2,
    # }

  Note that empty dictionaries are ignored and
  will not be restored by `unflatten_dict`.

  Args:
    xs: a nested dictionary
    keep_empty_nodes: replaces empty dictionaries
      with `traverse_util.empty_node`. This must
      be set to `True` for `unflatten_dict` to
      correctly restore empty dictionaries.
    is_leaf: an optional function that takes the
      next nested dictionary and nested keys and
      returns True if the nested dictionary is a
      leaf (i.e., should not be flattened further).
  Returns:
    The flattened dictionary.
  """
  assert isinstance(xs, dict), 'input is not a dict'

  def _flatten(xs, prefix):
    if not isinstance(xs, dict) or (is_leaf and is_leaf(prefix, xs)):
      return {prefix: xs}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    if keep_empty_nodes and is_empty:
      return {prefix: empty_node}
    return result
  return _flatten(xs, ())


def unflatten_dict(xs):
  """Unflatten a dictionary.

  See `flatten_dict`

  Example::

    flat_xs = {
      ('foo',): 1,
      ('bar', 'a'): 2,
    }
    xs = unflatten_dict(flat_xs)
    print(xs)
    # {
    #   'foo': 1
    #   'bar': {'a': 2}
    # }

  Args:
    xs: a flattened dictionary
  Returns:
    The nested dictionary.
  """
  assert isinstance(xs, dict), 'input is not a dict'
  result = {}
  for path, value in xs.items():
    if value is empty_node:
      value = {}
    cursor = result
    for key in path[:-1]:
      if key not in cursor:
        cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return result


class Traversal(abc.ABC):
  """Base class for all traversals."""

  @abc.abstractmethod
  def update(self, fn, inputs):
    """Update the focused items.

    Args:
      fn: the callback function that maps each traversed item
        to its updated value.
      inputs: the object that should be traversed.
    Returns:
      A new object with the updated values.
    """
    pass

  @abc.abstractmethod
  def iterate(self, inputs):
    """Iterate over the values selected by this `Traversal`.

    Args:
      inputs: the object that should be traversed.
    Returns:
      An iterator over the traversed values.
    """
    pass

  def set(self, values, inputs):
    """Overrides the values selected by the `Traversal`.

    Args:
      values: a list containing the new values.
      inputs: the object that should be traversed.
    Returns:
      A new object with the updated values.
    """
    def update_fn(_):
      if not values:
        raise ValueError('Not enough values provided')
      return values.pop(0)

    y = self.update(update_fn, inputs)
    if values:
      raise ValueError('Too many values provided')
    return y

  def compose(self, other):
    """Compose two traversals."""
    return TraverseCompose(self, other)

  def merge(self, *traversals):
    """Compose an arbitrary number of traversals and merge the results."""
    return self.compose(TraverseMerge(*traversals))

  def each(self):
    """Traverse each item in the selected containers."""
    return self.compose(TraverseEach())

  def tree(self):
    """Traverse each item in a pytree."""
    return self.compose(TraverseTree())

  def filter(self, fn):
    """Filter the selected values."""
    return self.compose(TraverseFilter(fn))

  def __getattr__(self, attr):
    return self.compose(TraverseAttr(attr))

  def __getitem__(self, key):
    return self.compose(TraverseItem(key))


class TraverseId(Traversal):
  """The identity Traversal."""

  def update(self, fn, inputs):
    return fn(inputs)

  def iterate(self, inputs):
    yield inputs

t_identity = TraverseId()


class TraverseMerge(Traversal):
  """Merges the selection from a set of traversals."""

  def __init__(self, *traversals):
    self._traversals = traversals

  def update(self, fn, inputs):
    for traversal in self._traversals:
      inputs = traversal.update(fn, inputs)
    return inputs

  def iterate(self, inputs):
    for traversal in self._traversals:
      yield from traversal.iterate(inputs)


class TraverseCompose(Traversal):
  """Compose two traversals."""

  def __init__(self, x, y):
    self._x = x
    self._y = y

  def update(self, fn, inputs):
    def update_fn(x):
      return self._y.update(fn, x)

    return self._x.update(update_fn, inputs)

  def iterate(self, inputs):
    for x in self._x.iterate(inputs):
      yield from self._y.iterate(x)


class TraverseFilter(Traversal):
  """Filter selected values based on a predicate."""

  def __init__(self, fn):
    self._fn = fn

  def update(self, fn, inputs):
    if self._fn(inputs):
      return fn(inputs)
    else:
      return inputs

  def iterate(self, inputs):
    if self._fn(inputs):
      yield inputs


def _is_namedtuple(t):
  return issubclass(t, tuple) and hasattr(t, '_fields')


class TraverseAttr(Traversal):
  """Traverse the attribute of an object."""

  def __init__(self, attr):
    self._attr = attr

  def update(self, fn, inputs):
    value = fn(getattr(inputs, self._attr))
    if _is_namedtuple(type(inputs)):
      return inputs._replace(**{self._attr: value})
    elif dataclasses.is_dataclass(inputs):
      return dataclasses.replace(inputs, **{self._attr: value})
    else:
      inputs = copy.copy(inputs)
      setattr(inputs, self._attr, value)
      return inputs

  def iterate(self, inputs):
    yield getattr(inputs, self._attr)


class TraverseItem(Traversal):
  """Traverse the item of an object."""

  def __init__(self, key):
    self._key = key

  def update(self, fn, inputs):
    if isinstance(inputs, tuple):
      ty = type(inputs)
      if isinstance(self._key, slice):
        sl = self._key
      else:
        sl = slice(self._key, self._key + 1)
      indices = set(range(*sl.indices(len(inputs))))

      args = [fn(inputs[i]) if i in indices else inputs[i]
              for i in range(len(inputs))]
      if _is_namedtuple(ty):
        return ty(*args)
      else:
        return ty(args)
    else:
      xs = copy.copy(inputs)
      xs[self._key] = fn(xs[self._key])
      return xs

  def iterate(self, inputs):
    if isinstance(self._key, slice):
      yield from inputs[self._key]
    else:
      yield inputs[self._key]


class TraverseEach(Traversal):
  """Traverse each item of a container."""

  def update(self, fn, inputs):
    ty = type(inputs)
    if ty is dict:
      return {key: fn(val) for key, val in inputs.items()}
    if ty not in {list, tuple}:
      raise ValueError('Only the entries of a list or tuple can be traversed.')
    return ty(fn(x) for x in inputs)

  def iterate(self, inputs):
    if isinstance(inputs, dict):
      yield from inputs.values()
    else:
      yield from inputs


class TraverseTree(Traversal):
  """Traverse every item in a pytree.
  """

  def update(self, fn, inputs):
    return jax.tree_map(fn, inputs)

  def iterate(self, inputs):
    yield from jax.tree_leaves(inputs)

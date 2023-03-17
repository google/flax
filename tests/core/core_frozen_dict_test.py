# Copyright 2022 The Flax Authors.
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

from absl.testing import absltest, parameterized
from flax.core import FrozenDict, copy, freeze, pop, unfreeze
import jax


class FrozenDictTest(parameterized.TestCase):

  def test_frozen_dict_copies(self):
    xs = {'a': 1, 'b': {'c': 2}}
    frozen = freeze(xs)
    xs['a'] += 1
    xs['b']['c'] += 1
    self.assertEqual(unfreeze(frozen), {'a': 1, 'b': {'c': 2}})

  def test_frozen_dict_maps(self):
    xs = {'a': 1, 'b': {'c': 2}}
    frozen = FrozenDict(xs)
    frozen2 = jax.tree_util.tree_map(lambda x: x + x, frozen)
    self.assertEqual(unfreeze(frozen2), {'a': 2, 'b': {'c': 4}})

  def test_frozen_dict_pop(self):
    xs = {'a': 1, 'b': {'c': 2}}
    b, a = FrozenDict(xs).pop('a')
    self.assertEqual(a, 1)
    self.assertEqual(unfreeze(b), {'b': {'c': 2}})

  def test_frozen_dict_partially_maps(self):
    x = jax.tree_util.tree_map(
        lambda a, b: (a, b),
        freeze({'a': 2}), freeze({'a': {'b': 1}}))
    self.assertEqual(unfreeze(x), {'a': (2, {'b': 1})})

  def test_frozen_dict_hash(self):
    xs = {'a': 1, 'b': {'c': 2}}
    ys = {'a': 1, 'b': {'c': 3}}
    self.assertNotEqual(hash(freeze(xs)), hash(freeze(ys)))

  def test_frozen_items(self):
    xs = {'a': 1, 'b': {'c': 2}}
    items = list(freeze(xs).items())

    self.assertEqual(items, [('a', 1), ('b', freeze(xs['b']))])

  def test_frozen_dict_repr(self):
    expected = """FrozenDict({
    a: 1,
    b: {
        c: 2,
        d: {},
    },
})"""

    xs = FrozenDict({'a': 1, 'b': {'c': 2, 'd': {}}})
    self.assertEqual(repr(xs), expected)
    self.assertEqual(repr(FrozenDict()), 'FrozenDict({})')

  def test_frozen_dict_reduce(self):
    before = FrozenDict(a=FrozenDict(b=1, c=2))
    cl, data = before.__reduce__()
    after = cl(*data)
    self.assertIsNot(before, after)
    self.assertEqual(before, after)
    self.assertEqual(after, {'a': {'b': 1, 'c': 2}})

  def test_frozen_dict_copy_reserved_name(self):
    result = FrozenDict({'a': 1}).copy({'cls': 2})
    self.assertEqual(result, {'a': 1, 'cls': 2})

  @parameterized.parameters(
      {
          'x': {'a': 1, 'b': {'c': 2}},
          'key': 'b',
          'actual_new_x': {'a': 1},
          'actual_value': {'c': 2},
      },
      {
          'x': FrozenDict({'a': 1, 'b': {'c': 2}}),
          'key': 'b',
          'actual_new_x': FrozenDict({'a': 1}),
          'actual_value': FrozenDict({'c': 2}),
      },
  )
  def test_utility_pop(self, x, key, actual_new_x, actual_value):
    new_x, value = pop(x, key)
    self.assertTrue(
        new_x == actual_new_x and isinstance(new_x, type(actual_new_x))
    )
    self.assertTrue(
        value == actual_value and isinstance(value, type(actual_value))
    )

  @parameterized.parameters(
      {
          'x': {'a': 1, 'b': {'c': 2}},
          'add_or_replace': {'b': {'c': -1, 'd': 3}},
          'actual_new_x': {'a': 1, 'b': {'c': -1, 'd': 3}},
      },
      {
          'x': FrozenDict({'a': 1, 'b': {'c': 2}}),
          'add_or_replace': FrozenDict({'b': {'c': -1, 'd': 3}}),
          'actual_new_x': FrozenDict({'a': 1, 'b': {'c': -1, 'd': 3}}),
      },
  )
  def test_utility_copy(self, x, add_or_replace, actual_new_x):
    new_x = copy(x, add_or_replace=add_or_replace)
    self.assertTrue(
        new_x == actual_new_x and isinstance(new_x, type(actual_new_x))
    )

  def test_flatten(self):
    frozen = freeze({'c': 1, 'b': {'a': 2}})
    flat_leaves, tdef = jax.tree_util.tree_flatten(frozen)
    self.assertEqual(flat_leaves, [2, 1])
    self.assertEqual(
        jax.tree_util.tree_unflatten(tdef, flat_leaves),
        frozen,
    )
    flat_path_leaves, tdef = jax.tree_util.tree_flatten_with_path(frozen)
    self.assertEqual(
        flat_path_leaves,
        [((jax.tree_util.DictKey('b'), jax.tree_util.DictKey('a')), 2),
         ((jax.tree_util.DictKey('c'),), 1)],
    )
    self.assertEqual(
        jax.tree_util.tree_unflatten(tdef, [l for _, l in flat_path_leaves]),
        frozen,
    )

if __name__ == '__main__':
  absltest.main()

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

from flax.core import FrozenDict, unfreeze, freeze

import jax


from absl.testing import absltest


class FrozenDictTest(absltest.TestCase):

  def test_frozen_dict_copies(self):
    xs = {'a': 1, 'b': {'c': 2}}
    frozen = freeze(xs)
    xs['a'] += 1
    xs['b']['c'] += 1
    self.assertEqual(unfreeze(frozen), {'a': 1, 'b': {'c': 2}})

  def test_frozen_dict_maps(self):
    xs = {'a': 1, 'b': {'c': 2}}
    frozen = FrozenDict(xs)
    frozen2 = jax.tree_map(lambda x: x + x, frozen)
    self.assertEqual(unfreeze(frozen2), {'a': 2, 'b': {'c': 4}})

  def test_frozen_dict_pop(self):
    xs = {'a': 1, 'b': {'c': 2}}
    b, a = FrozenDict(xs).pop('a')
    self.assertEqual(a, 1)
    self.assertEqual(unfreeze(b), {'b': {'c': 2}})

  def test_frozen_dict_partially_maps(self):
    x = jax.tree_multimap(
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
    expected = (
"""FrozenDict({
    a: 1,
    b: {
        c: 2,
        d: {},
    },
})""")

    xs = FrozenDict({'a': 1, 'b': {'c': 2, 'd': {}}})
    self.assertEqual(repr(xs), expected)
    self.assertEqual(repr(FrozenDict()), 'FrozenDict({})')


if __name__ == '__main__':
  absltest.main()

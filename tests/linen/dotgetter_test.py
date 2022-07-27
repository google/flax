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

from absl.testing import absltest

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.core import Scope, FrozenDict, freeze, unfreeze
from flax.linen.dotgetter import DotGetter, is_leaf
from flax import serialization

# Parse absl flags test_srcdir and test_tmpdir.
#jax.config.parse_flags_with_absl()

class DotGetterTest(absltest.TestCase):

  def test_simple(self):
    dg = DotGetter({'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}})
    self.assertEqual(dg.a, 1)
    self.assertEqual(dg.b.c, 2)
    self.assertEqual(dg.d.e.f, 3)
    self.assertEqual(dg['a'], 1)
    self.assertEqual(dg['b'].c, 2)
    self.assertEqual(dg['b']['c'], 2)
    self.assertEqual(dg.b['c'], 2)
    self.assertEqual(dg.d.e.f, 3)

  def test_simple_frozen(self):
    dg = DotGetter(freeze({'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}))
    self.assertEqual(dg.a, 1)
    self.assertEqual(dg.b.c, 2)
    self.assertEqual(dg.d.e.f, 3)
    self.assertEqual(dg['a'], 1)
    self.assertEqual(dg['b'].c, 2)
    self.assertEqual(dg['b']['c'], 2)
    self.assertEqual(dg.b['c'], 2)
    self.assertEqual(dg.d.e.f, 3)

  def test_eq(self):
    dg1 = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    dg2 = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    self.assertEqual(dg1, dg2)
    self.assertEqual(freeze(dg1), dg2)
    self.assertEqual(freeze(dg1), freeze(dg2))

  def test_dir(self):
    dg = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    self.assertEqual(dir(dg), ['a', 'b'])
    self.assertEqual(dir(dg.b), ['c', 'd'])

  def test_freeze(self):
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    dg = DotGetter(d)
    self.assertEqual(freeze(dg), freeze(d))
    fd = freeze({'a': 1, 'b': {'c': 2, 'd': 3}})
    fdg = DotGetter(d)
    self.assertEqual(unfreeze(fdg), unfreeze(fd))

  def test_hash(self):
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    dg = DotGetter(d)
    fd = freeze(d)
    fdg = DotGetter(fd)
    self.assertEqual(hash(fdg), hash(fd))
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      hash(dg)

  def test_pytree(self):
    dg1 = DotGetter({'a': jnp.array([1.0]),
                     'b': {'c': jnp.array([2.0]),
                           'd': jnp.array([3.0])}})
    dg2 = DotGetter({'a': jnp.array([2.0]),
                     'b': {'c': jnp.array([4.0]),
                           'd': jnp.array([6.0])}})
    self.assertEqual(jax.tree_util.tree_map(lambda x: 2 * x, dg1), dg2)

  def test_statedict(self):
    d = {'a': jnp.array([1.0]),
         'b': {'c': jnp.array([2.0]),
               'd': jnp.array([3.0])}}
    dg = DotGetter(d)
    ser = serialization.to_state_dict(dg)
    deser = serialization.from_state_dict(dg, ser)
    self.assertEqual(d, deser)

  def test_is_leaf(self):
    for x in [0, 'foo', jnp.array([0.]), {}, [], (), {1, 2}]:
      self.assertTrue(is_leaf(x))
    self.assertFalse(is_leaf({'a': 1}))
    self.assertFalse(is_leaf([1,2,3]))
    self.assertFalse(is_leaf((1,2,3)))


if __name__ == '__main__':
  absltest.main()

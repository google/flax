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

  def simple_test(self):
    dg = DotGetter({'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}})
    self.assertEqual(dg.a, 1)
    self.assertEqual(dg.b.c, 2)
    self.assertEqual(dg.d.e.f, 3)
    self.assertEqual(dg['a'], 1)
    self.assertEqual(dg['b'].c, 2)
    self.assertEqual(dg['b']['c'], 2)
    self.assertEqual(dg.b['c'], 2)
    self.assertEqual(dg.d.e.f, 3)

  def simple_frozen_test(self):
    dg = DotGetter(freeze({'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}))
    self.assertEqual(dg.a, 1)
    self.assertEqual(dg.b.c, 2)
    self.assertEqual(dg.d.e.f, 3)
    self.assertEqual(dg['a'], 1)
    self.assertEqual(dg['b'].c, 2)
    self.assertEqual(dg['b']['c'], 2)
    self.assertEqual(dg.b['c'], 2)
    self.assertEqual(dg.d.e.f, 3)

  def eq_test(self):
    dg1 = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    dg2 = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    self.assertEqual(dg1, dg2)
    self.assertEqual(freeze(dg1), dg2)
    self.assertEqual(freeze(dg1), freeze(dg2))

  def dir_test(self):
    dg = DotGetter({'a': 1, 'b': {'c': 2, 'd': 3}})
    self.assertEqual(dir(dg), ['a', 'b'])
    self.assertEqual(dir(dg.b), ['c', 'd'])

  def freeze_test(self):
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    dg = DotGetter(d)
    self.assertEqual(freeze(dg), freeze(d))
    fd = freeze({'a': 1, 'b': {'c': 2, 'd': 3}})
    fdg = DotGetter(d)
    self.assertEqual(unfreeze(fdg), unfreeze(fd))

  def hash_test(self):
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    dg = DotGetter(d)
    fd = freeze(d)
    fdg = DotGetter(fd)
    self.assertEqual(hash(fdg), hash(fd))
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      hash(dg)

  def pytree_test(self):
    dg1 = DotGetter({'a': jnp.array([1.0]),
                     'b': {'c': jnp.array([2.0]),
                           'd': jnp.array([3.0])}})
    dg2 = DotGetter({'a': jnp.array([2.0]),
                     'b': {'c': jnp.array([4.0]),
                           'd': jnp.array([6.0])}})
    self.assertEqual(jax.tree_map(lambda x: 2 * x, dg1), dg2)

  def statedict_test(self):
    d = {'a': jnp.array([1.0]),
         'b': {'c': jnp.array([2.0]),
               'd': jnp.array([3.0])}}
    dg = DotGetter(d)
    ser = serialization.to_state_dict(dg)
    deser = serialization.from_state_dict(dg, ser)
    self.assertEqual(d, deser)

  def is_leaf_test(self):
    for x in [0, 'foo', jnp.array([0.]), {}, [], (), {1, 2}]:
      self.assertTrue(is_leaf(x))
    self.assertFalse(is_leaf({'a': 1}))
    self.assertFalse(is_leaf([1,2,3]))
    self.assertFalse(is_leaf((1,2,3)))


if __name__ == '__main__':
  absltest.main()
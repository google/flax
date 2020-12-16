from flax.core import unfreeze
from flax.core.non_final_variables_dict import NonFinalVariablesDict, make_nonfinal

import jax


from absl.testing import absltest


class NonFinalVariablesTest(absltest.TestCase):

  def test_non_final_dict_maps(self):
    xs = {'a': 1, 'b': {'c': 2}}
    frozen = NonFinalVariablesDict(xs)
    frozen2 = jax.tree_map(lambda x: x + x, frozen)
    self.assertEqual(unfreeze(frozen2), {'a': 2, 'b': {'c': 4}})

  def test_non_final_dict_pop(self):
    xs = {'a': 1, 'b': {'c': 2}}
    b, a = NonFinalVariablesDict(xs).pop('a')
    self.assertEqual(a, 1)
    self.assertEqual(unfreeze(b), {'b': {'c': 2}})

  def test_non_final_dict_partially_maps(self):
    x = jax.tree_multimap(
        lambda a, b: (a, b),
        make_nonfinal({'a': 2}), make_nonfinal({'a': {'b': 1}}))
    self.assertEqual(unfreeze(x), {'a': (2, {'b': 1})})

  def test_non_final_dict_hash(self):
    xs = {'a': 1, 'b': {'c': 2}}
    ys = {'a': 1, 'b': {'c': 3}}
    self.assertNotEqual(hash(make_nonfinal(xs)), hash(make_nonfinal(ys)))

  def test_non_final_items(self):
    xs = {'a': 1, 'b': {'c': 2}}
    items = list(make_nonfinal(xs).items())

    self.assertEqual(items, [('a', 1), ('b', make_nonfinal(xs['b']))])


  def test_frozen_dict_repr(self):
    expected = (
"""NonFinalVariablesDict({
    a: 1,
    b: {
        c: 2,
        d: {},
    },
})""")

    xs = NonFinalVariablesDict({'a': 1, 'b': {'c': 2, 'd': {}}})
    self.assertEqual(repr(xs), expected)
    self.assertEqual(repr(NonFinalVariablesDict()), 'NonFinalVariablesDict({})')


if __name__ == '__main__':
  absltest.main()

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

"""Tests for flax.traverse_util."""


import collections
from absl.testing import absltest
import numpy as np
import optax
import flax
from flax.core import freeze
from flax import traverse_util
import jax
import jax.numpy as jnp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class Foo(object):

  def __init__(self, foo, bar=None):
    self.foo = foo
    self.bar = bar

  def __eq__(self, other):
    return self.foo == other.foo and self.bar == other.bar


Point = collections.namedtuple('Point', ['x', 'y'])


class TraversalTest(absltest.TestCase):

  def test_traversal_id(self):
    x = 1
    traversal = traverse_util.t_identity
    self.assertEqual(list(traversal.iterate(x)), [1])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, 2)

  def test_traverse_item(self):
    x = {'foo': 1}
    traversal = traverse_util.t_identity['foo']
    self.assertEqual(list(traversal.iterate(x)), [1])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, {'foo': 2})

  def test_traverse_tuple_item(self):
    x = (1, 2, 3)
    traversal = traverse_util.t_identity[1]
    self.assertEqual(list(traversal.iterate(x)), [2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, (1, 4, 3))

  def test_traverse_tuple_items(self):
    x = (1, 2, 3, 4)
    traversal = traverse_util.t_identity[1:3]
    self.assertEqual(list(traversal.iterate(x)), [2, 3])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, (1, 4, 6, 4))

  def test_traverse_namedtuple_item(self):
    x = Point(x=1, y=2)
    traversal = traverse_util.t_identity[1]
    self.assertEqual(list(traversal.iterate(x)), [2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, Point(x=1, y=4))

  def test_traverse_attr(self):
    x = Foo(foo=1)
    traversal = traverse_util.t_identity.foo
    self.assertEqual(list(traversal.iterate(x)), [1])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, Foo(foo=2))

  def test_traverse_namedtuple_attr(self):
    x = Point(x=1, y=2)
    traversal = traverse_util.t_identity.y
    self.assertEqual(list(traversal.iterate(x)), [2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, Point(x=1, y=4))

  def test_traverse_dataclass_attr(self):
    x = Point(x=1, y=2)
    traversal = traverse_util.t_identity.y
    self.assertEqual(list(traversal.iterate(x)), [2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, Point(x=1, y=4))

  def test_traverse_merge(self):
    x = [{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]
    traversal_base = traverse_util.t_identity.each()
    traversal = traversal_base.merge(traverse_util.TraverseItem('foo'),
                                     traverse_util.TraverseItem('bar'))
    self.assertEqual(list(traversal.iterate(x)), [1, 2, 3, 4])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, [{'foo': 2, 'bar': 4}, {'foo': 6, 'bar': 8}])

  def test_traverse_each(self):
    x = [{'foo': 1}, {'foo': 2}]
    traversal = traverse_util.t_identity.each()['foo']
    self.assertEqual(list(traversal.iterate(x)), [1, 2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, [{'foo': 2}, {'foo': 4}])

  def test_traverse_each_dict(self):
    x = {'foo': 1, 'bar': 2}
    traversal = traverse_util.t_identity.each()
    self.assertEqual(list(traversal.iterate(x)), [1, 2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, {'foo': 2, 'bar': 4})

  def test_traverse_tree(self):
    x = [{'foo': 1}, {'bar': 2}]
    traversal = traverse_util.t_identity.tree()
    self.assertEqual(list(traversal.iterate(x)), [1, 2])
    y = traversal.update(lambda x: x + x, x)
    self.assertEqual(y, [{'foo': 2}, {'bar': 4}])

  def test_traverse_filter(self):
    x = [1, -2, 3, -4]
    traversal = traverse_util.t_identity.each().filter(lambda x: x < 0)
    self.assertEqual(list(traversal.iterate(x)), [-2, -4])
    y = traversal.update(lambda x: -x, x)
    self.assertEqual(y, [1, 2, 3, 4])

  def test_traversal_set(self):
    x = {'foo': [1, 2]}
    traversal = traverse_util.t_identity['foo'].each()
    y = traversal.set([3, 4], x)
    self.assertEqual(y, {'foo': [3, 4]})
    with self.assertRaises(ValueError):
      traversal.set([3], x)  # too few values
    with self.assertRaises(ValueError):
      traversal.set([3, 4, 5], x)  # too many values

  def test_flatten_dict(self):
    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = traverse_util.flatten_dict(xs)
    self.assertEqual(flat_xs, {
        ('foo',): 1,
        ('bar', 'a'): 2,
    })
    flat_xs = traverse_util.flatten_dict(freeze(xs))
    self.assertEqual(flat_xs, {
      ('foo',): 1,
      ('bar', 'a'): 2,
    })
    flat_xs = traverse_util.flatten_dict(xs, sep='/')
    self.assertEqual(flat_xs, {
      'foo': 1,
      'bar/a': 2,
    })

  def test_unflatten_dict(self):
    expected_xs = {
      'foo': 1,
      'bar': {'a': 2}
    }
    xs = traverse_util.unflatten_dict({
      ('foo',): 1,
      ('bar', 'a'): 2,
    })
    self.assertEqual(xs, expected_xs)
    xs = traverse_util.unflatten_dict({
      'foo': 1,
      'bar/a': 2,
    }, sep='/')
    self.assertEqual(xs, expected_xs)

  def test_flatten_dict_keep_empty(self):
    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = traverse_util.flatten_dict(xs, keep_empty_nodes=True)
    self.assertEqual(flat_xs, {
        ('foo',): 1,
        ('bar', 'a'): 2,
        ('bar', 'b'): traverse_util.empty_node,
    })
    xs_restore = traverse_util.unflatten_dict(flat_xs)
    self.assertEqual(xs, xs_restore)

  def test_flatten_dict_is_leaf(self):
    xs = {'foo': {'c': 4}, 'bar': {'a': 2, 'b': {}}}
    flat_xs = traverse_util.flatten_dict(
        xs,
        is_leaf=lambda k, x: len(k) == 1 and len(x) == 2)
    self.assertEqual(flat_xs, {
        ('foo', 'c'): 4,
        ('bar',): {
            'a': 2,
            'b': {}
        },
    })
    xs_restore = traverse_util.unflatten_dict(flat_xs)
    self.assertEqual(xs, xs_restore)


class ModelParamTraversalTest(absltest.TestCase):

  def test_only_works_on_model_params(self):
    traversal = traverse_util.ModelParamTraversal(lambda *_: True)
    with self.assertRaises(ValueError):
      list(traversal.iterate([]))

  def test_param_selection(self):
    params = {
        'x': {
            'kernel': 1,
            'bias': 2,
            'y': {
                'kernel': 3,
                'bias': 4,
            },
            'z': {},
        },
    }
    expected_params = {
        'x': {
            'kernel': 2,
            'bias': 2,
            'y': {
                'kernel': 6,
                'bias': 4,
            },
            'z': {}
        },
    }
    names = []
    def filter_fn(name, _):
      names.append(name)  # track names passed to filter_fn for testing
      return 'kernel' in name
    traversal = traverse_util.ModelParamTraversal(filter_fn)

    values = list(traversal.iterate(params))
    configs = [
        (params, expected_params),
        (flax.core.FrozenDict(params), flax.core.FrozenDict(expected_params)),
    ]
    for model, expected_model in configs:
      self.assertEqual(values, [1, 3])
      self.assertEqual(set(names), set([
          '/x/kernel', '/x/bias', '/x/y/kernel', '/x/y/bias']))
      new_model = traversal.update(lambda x: x + x, model)
      self.assertEqual(new_model, expected_model)

  def test_path_value(self):
    params_in = {'a': {'b': 10, 'c': 2}}
    params_out = traverse_util.path_aware_map(
      lambda path, x: x + 1 if 'b' in path else -x, params_in)

    self.assertEqual(params_out, {'a': {'b': 11, 'c': -2}})

  def test_path_aware_map_with_multi_transform(self):
    params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
            'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)  # dummy gradients

    param_labels = traverse_util.path_aware_map(
      lambda path, x: 'kernel' if 'w' in path else 'bias', params)
    tx = optax.multi_transform(
      {'kernel': optax.sgd(1.0), 'bias': optax.set_to_zero()}, param_labels)
    state = tx.init(params)
    updates, new_state = tx.update(gradients, state, params)
    new_params = optax.apply_updates(params, updates)


    self.assertTrue(np.allclose(new_params['linear_1']['b'], params['linear_1']['b']))
    self.assertTrue(np.allclose(new_params['linear_2']['b'], params['linear_2']['b']))
    self.assertFalse(np.allclose(new_params['linear_1']['w'], params['linear_1']['w']))
    self.assertFalse(np.allclose(new_params['linear_2']['w'], params['linear_2']['w']))

  def test_path_aware_map_with_masked(self):
    params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
            'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)  # dummy gradients

    params_mask = traverse_util.path_aware_map(
      lambda path, x: 'w' in path, params)
    tx = optax.masked(optax.sgd(1.0), params_mask)
    state = tx.init(params)
    updates, new_state = tx.update(gradients, state, params)
    new_params = optax.apply_updates(params, updates)


    self.assertTrue(np.allclose(new_params['linear_1']['b'], gradients['linear_1']['b']))
    self.assertTrue(np.allclose(new_params['linear_2']['b'], gradients['linear_2']['b']))
    self.assertTrue(np.allclose(new_params['linear_1']['w'], -gradients['linear_1']['w']))
    self.assertTrue(np.allclose(new_params['linear_2']['w'], -gradients['linear_2']['w']))

  def test_path_aware_map_with_empty_nodes(self):
    params_in = {'a': {'b': 10, 'c': 2}, 'b': {}}
    params_out = traverse_util.path_aware_map(
      lambda path, x: x + 1 if 'b' in path else -x, params_in)

    self.assertEqual(params_out, {'a': {'b': 11, 'c': -2}, 'b': {}})


if __name__ == '__main__':
  absltest.main()

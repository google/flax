# Copyright 2023 The Flax Authors.
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

"""Tests for flax.struct."""


from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax
from typing import Any, Generic, NamedTuple

import flax.linen as nn
from flax.core import freeze
from flax.cursor import cursor
from flax.training import train_state

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class GenericTuple(NamedTuple):
  x: Any
  y: Any = None


class CursorTest(absltest.TestCase):

  def test_set_and_build(self):
    # test regular dict and FrozenDict
    dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    for d, freeze_wrap in ((dict_obj, lambda x: x), (freeze(dict_obj), freeze)):
      # set API
      self.assertEqual(
          cursor(d)['b'][0].set(10),
          freeze_wrap({'a': 1, 'b': (10, 3), 'c': [4, 5]}),
      )
      # build API
      c = cursor(d)
      c['b'][0] = 20
      c['a'] = (100, 200)
      d2 = c.build()
      self.assertEqual(
          d2, freeze_wrap({'a': (100, 200), 'b': (20, 3), 'c': [4, 5]})
      )
    self.assertEqual(
        dict_obj, {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    )  # make sure original object is unchanged

    # test list and tuple
    list_obj = [0, dict_obj, (1, 2), [3, 4, 5]]
    for l, tuple_wrap in ((list_obj, lambda x: x), (tuple(list_obj), tuple)):
      # set API
      self.assertEqual(
          cursor(l)[1]['b'][0].set(10),
          tuple_wrap(
              [0, {'a': 1, 'b': (10, 3), 'c': [4, 5]}, (1, 2), [3, 4, 5]]
          ),
      )
      # build API
      c = cursor(l)
      c[1]['b'][0] = 20
      c[2] = (100, 200)
      l2 = c.build()
      self.assertEqual(
          l2,
          tuple_wrap(
              [0, {'a': 1, 'b': (20, 3), 'c': [4, 5]}, (100, 200), [3, 4, 5]]
          ),
      )
    self.assertEqual(
        list_obj, [0, {'a': 1, 'b': (2, 3), 'c': [4, 5]}, (1, 2), [3, 4, 5]]
    )  # make sure original object is unchanged

    # test TrainState
    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params=dict_obj,
        tx=optax.adam(1e-3),
    )
    # set API
    self.assertEqual(
        cursor(state).params['b'][0].set(10).params,
        {'a': 1, 'b': (10, 3), 'c': [4, 5]},
    )
    # build API
    new_fn = lambda x: x + 1
    c = cursor(state)
    c.apply_fn = new_fn
    c.params['b'][0] = 20
    c.params['a'] = (100, 200)
    state2 = c.build()
    self.assertEqual(state2.apply_fn, new_fn)
    self.assertEqual(
        state2.params, {'a': (100, 200), 'b': (20, 3), 'c': [4, 5]}
    )

    self.assertEqual(
        dict_obj, {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    )  # make sure original object is unchanged

    # test NamedTuple
    # set API
    t = GenericTuple(GenericTuple(0))
    self.assertEqual(cursor(t).x.x.set(1), GenericTuple(GenericTuple(1)))

    # build API
    c = cursor(t)
    c.x.x = 2
    c.x.y = 3
    c.y = 4
    t2 = c.build()
    self.assertEqual(t2, GenericTuple(GenericTuple(2, 3), 4))

    self.assertEqual(
        t, GenericTuple(GenericTuple(0))
    )  # make sure original object is unchanged

  def test_apply_update(self):
    # test list and tuple
    def update_fn(path, value):
      """Multiply the first element of all leaf nodes of the pytree by -1."""
      if path[-1] == '0' and isinstance(value, int):
        return value * -1
      return value

    for tuple_wrap in (lambda x: x, tuple):
      l = tuple_wrap([tuple_wrap([1, 2]), tuple_wrap([3, 4])])
      c = cursor(l)
      l2 = c.apply_update(update_fn).build()
      self.assertEqual(
          l2, tuple_wrap([tuple_wrap([-1, 2]), tuple_wrap([-3, 4])])
      )
      self.assertEqual(
          l, tuple_wrap([tuple_wrap([1, 2]), tuple_wrap([3, 4])])
      )  # make sure the original object is unchanged

    # test regular dict and FrozenDict
    def update_fn(path, value):
      """Multiply all dense kernel params by 2 and add 1.
      Subtract the Dense_1 bias param by 1."""
      if 'kernel' in path:
        return value * 2 + 1
      elif 'Dense_1' in path and 'bias' in path:
        return value - 1
      return value

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        return x

    for freeze_wrap in (lambda x: x, freeze):
      params = freeze_wrap(
          Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']
      )

      c = cursor(params)
      params2 = c.apply_update(update_fn).build()
      for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
        self.assertTrue(
            (params2[layer]['kernel'] == 2 * params[layer]['kernel'] + 1).all()
        )
        if layer == 'Dense_1':
          self.assertTrue(
              (params2[layer]['bias'] == jnp.array([-1, -1, -1])).all()
          )
        else:
          self.assertTrue(
              (params2[layer]['bias'] == params[layer]['bias']).all()
          )
      self.assertTrue(
          jax.tree_util.tree_all(
              jax.tree_util.tree_map(
                  lambda x, y: (x == y).all(),
                  params,
                  freeze_wrap(
                      Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))[
                          'params'
                      ]
                  ),
              )
          )
      )  # make sure original params are unchanged

    # test TrainState
    def update_fn(path, value):
      """Replace params with empty dictionary."""
      if 'params' in path:
        return {}
      return value

    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params={'a': 1, 'b': 2},
        tx=optax.adam(1e-3),
    )
    c = cursor(state)
    state2 = c.apply_update(update_fn).build()
    self.assertEqual(state2.params, {})
    self.assertEqual(
        state.params, {'a': 1, 'b': 2}
    )  # make sure original params are unchanged

    # test NamedTuple
    def update_fn(path, value):
      """Add 5 to all x-attribute values that are ints"""
      if path[-1] == 'x' and isinstance(value, int):
        return value + 5
      return value

    t = GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    c = cursor(t)
    t2 = c.apply_update(update_fn).build()
    self.assertEqual(t2, GenericTuple(GenericTuple(5, 1), GenericTuple(7, 3)))
    self.assertEqual(
        t, GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    )  # make sure original object is unchanged

  def test_apply_update_root_node_unmodified(self):
    def update_fn(path, value):
      if isinstance(value, list):
        value = value.copy()
        value.append(-1)
      return value

    l = [[1, 2], [3, 4], 5]
    l2 = cursor(l).apply_update(update_fn).build()
    self.assertEqual(l2, [[1, 2, -1], [3, 4, -1], 5])

  def test_multi_modify(self):
    d = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    c = cursor(d)
    # test multiple changes on same element
    c['b'][0] = 6
    c['b'][0] = 7
    d2 = c.build()
    self.assertEqual(d2, {'a': 1, 'b': (7, 3), 'c': [4, 5]})
    # test nested changes
    c['a'] = (100, 200)
    c['a'][1] = -1
    d3 = c.build()
    self.assertEqual(d3, {'a': (100, -1), 'b': (7, 3), 'c': [4, 5]})

  def test_hidden_change(self):
    # test list
    l = [1, 2]
    c = cursor(l)
    c[0] = 100
    l[1] = -1
    l2 = c.build()
    self.assertEqual(
        l2, [100, -1]
    )  # change in l affects l2 (this is expected behavior)
    self.assertEqual(l, [1, -1])

    # test regular dict
    d = {'a': 1, 'b': 2}
    c = cursor(d)
    c['a'] = 100
    d['b'] = -1
    d2 = c.build()
    self.assertEqual(
        d2, {'a': 100, 'b': -1}
    )  # change in d affects d2 (this is expected behavior)
    self.assertEqual(d, {'a': 1, 'b': -1})

    # test TrainState
    params = {'a': 1, 'b': 2}
    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params=params,
        tx=optax.adam(1e-3),
    )
    c = cursor(state)
    c.params['a'] = 100
    params['b'] = -1
    state2 = c.build()
    self.assertEqual(
        state2.params, {'a': 100, 'b': -1}
    )  # change in state affects state2 (this is expected behavior)
    self.assertEqual(state.params, {'a': 1, 'b': -1})

  def test_named_tuple_multi_access(self):
    t = GenericTuple(GenericTuple(0, 1), GenericTuple(2, 3))
    c = cursor(t)
    c.x.x = 4
    c[0].y = 5
    c.y.x = 6
    c.y[1] = 7
    self.assertEqual(
        c.build(), GenericTuple(GenericTuple(4, 5), GenericTuple(6, 7))
    )

    c[0][1] = -5
    self.assertEqual(
        c.build(), GenericTuple(GenericTuple(4, -5), GenericTuple(6, 7))
    )
    c.x[1] = -6
    self.assertEqual(
        c.build(), GenericTuple(GenericTuple(4, -6), GenericTuple(6, 7))
    )
    c.x.y = -7
    self.assertEqual(
        c.build(), GenericTuple(GenericTuple(4, -7), GenericTuple(6, 7))
    )
    c[0].y = -8
    self.assertEqual(
        c.build(), GenericTuple(GenericTuple(4, -8), GenericTuple(6, 7))
    )


if __name__ == '__main__':
  absltest.main()

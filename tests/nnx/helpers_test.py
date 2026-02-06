# Copyright 2024 The Flax Authors.
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

import jax
import jax.numpy as jnp
import optax

from absl.testing import absltest
import numpy as np

from flax import linen
from flax import nnx


class TrainState(nnx.TrainState):
  batch_stats: nnx.State


class TestHelpers(absltest.TestCase):
  def test_train_state(self):
    m = nnx.Dict(a=nnx.Param(1), b=nnx.BatchStat(2))

    graphdef, params, batch_stats = nnx.split(m, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    leaves = jax.tree_util.tree_leaves(state)

  def test_train_state_methods(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(4, rngs=rngs)

      def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.linear(x)
        x = self.batch_norm(x, use_running_average=not train)
        return x

    module = Foo(rngs=nnx.Rngs(0))
    graphdef, params, batch_stats = nnx.split(module, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    x = jax.numpy.ones((1, 2))
    y, _updates = state.apply('params', 'batch_stats')(x, train=True)

    assert y.shape == (1, 4)

    # fake gradient
    grads = jax.tree.map(jnp.ones_like, state.params)
    # test apply_gradients
    state = state.apply_gradients(grads)

  def test_nnx_linen_sequential_equivalence(self):
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    rngs = nnx.Rngs(0)
    x = jax.random.uniform(key1, (3, 1, 5))

    model_nnx = nnx.Sequential(
      nnx.Linear(5, 4, rngs=rngs), nnx.Linear(4, 2, rngs=rngs)
    )
    model = linen.Sequential([linen.Dense(4), linen.Dense(2)])

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        variables['params'][f'layers_{layer_index}'][param] = getattr(
          model_nnx.layers[layer_index], param
        )[...]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        getattr(model_nnx.layers[layer_index], param)[...] = variables[
          'params'
        ][f'layers_{layer_index}'][param]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)

  def test_nnx_empty_sequential_is_identity(self):
    iden = nnx.Sequential()
    assert iden(12) == 12
    assert iden(12, 23) == (12, 23)
    assert iden() is None
    assert iden(k=2) == {'k': 2}

  def test_dict_mutable_mapping(self):
    d = nnx.Dict({'a': 1, 'b': 2})
    self.assertEqual(d['a'], 1)
    self.assertEqual(d['b'], 2)
    self.assertEqual(len(d), 2)

    d['c'] = 3
    self.assertEqual(d['c'], 3)
    self.assertEqual(len(d), 3)

    del d['a']
    self.assertEqual(len(d), 2)
    with self.assertRaises(AttributeError):
      _ = d['a']

    self.assertSetEqual(set(d), {'b', 'c'})

  def test_list_mutable_sequence(self):
    l = nnx.List([1, 2, 3])
    self.assertEqual(len(l), 3)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 2)
    self.assertEqual(l[2], 3)

    l.append(4)
    self.assertEqual(len(l), 4)
    self.assertEqual(l[3], 4)

    l.insert(1, 5)
    self.assertEqual(len(l), 5)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 5)
    self.assertEqual(l[2], 2)
    self.assertEqual(l[3], 3)
    self.assertEqual(l[4], 4)

    del l[2]
    self.assertEqual(len(l), 4)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 5)
    self.assertEqual(l[2], 3)
    self.assertEqual(l[3], 4)

    l[1:3] = [6, 7]
    self.assertEqual(l[1], 6)
    self.assertEqual(l[2], 7)

    self.assertEqual(l[1:3], [6, 7])

  def test_list_fori_loop(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.layers = nnx.List([
            nnx.Linear(1, 1, rngs=nnx.Rngs(0)),
            nnx.Linear(1, 1, rngs=nnx.Rngs(0)),
        ])

    def batch_loop_body(i, carry):
      return carry

    net = Foo()
    jax.lax.fori_loop(0, 2, batch_loop_body, net)

  def test_list_pytree_default_behavior(self):
    ls = nnx.List([jnp.array(1), jnp.array(2), jnp.array(3)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 3)
    np.testing.assert_array_equal(leaves[0], jnp.array(1))
    np.testing.assert_array_equal(leaves[1], jnp.array(2))
    np.testing.assert_array_equal(leaves[2], jnp.array(3))

  def test_list_pytree_static_elements(self):
    ls = nnx.List([nnx.static(10), nnx.static(20), nnx.static(30)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertEmpty(leaves)

  def test_list_pytree_data_elements(self):
    ls = nnx.List([nnx.data(1), nnx.data(2), nnx.data(3)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 3)
    self.assertEqual(leaves[0], 1)
    self.assertEqual(leaves[1], 2)
    self.assertEqual(leaves[2], 3)

  def test_list_pytree_mixed_static_data(self):
    ls = nnx.List([
        nnx.data(jnp.array(1)),
        nnx.static(100),
        nnx.data(jnp.array(2)),
        nnx.static(200),
    ])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 2)
    np.testing.assert_array_equal(leaves[0], jnp.array(1))
    np.testing.assert_array_equal(leaves[1], jnp.array(2))

  def test_list_pytree_flatten_unflatten(self):
    ls = nnx.List([nnx.data(10), nnx.static('hello'), nnx.data(20)])
    leaves, treedef = jax.tree_util.tree_flatten(ls)
    self.assertLen(leaves, 2)
    self.assertEqual(leaves[0], 10)
    self.assertEqual(leaves[1], 20)

    new_leaves = [x * 2 for x in leaves]
    new_ls = jax.tree_util.tree_unflatten(treedef, new_leaves)
    self.assertEqual(new_ls[0], 20)
    self.assertEqual(new_ls[1], 'hello')
    self.assertEqual(new_ls[2], 40)

  def test_list_pytree_jit(self):
    ls = nnx.List([nnx.data(jnp.array(1.0)), nnx.static(999)])

    @jax.jit
    def double(ls):
      return jax.tree.map(lambda x: x * 2, ls)

    result = double(ls)
    np.testing.assert_array_equal(result[0], jnp.array(2.0))
    self.assertEqual(result[1], 999)


if __name__ == '__main__':
  absltest.main()


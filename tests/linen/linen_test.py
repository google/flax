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

"""Tests for flax.linen."""

import copy
from absl.testing import absltest, parameterized

from flax import ids
from flax import linen as nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def check_eq(xs, ys):
  return jax.tree_util.tree_all(
      jax.tree_util.tree_map(np.testing.assert_allclose, xs, ys))


class PoolTest(parameterized.TestCase):

  def test_pool_custom_reduce(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    mul_reduce = lambda x, y: x * y
    y = nn.pooling.pool(x, 1., mul_reduce, (2, 2), (1, 1), 'VALID')
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2. ** 4))

  @parameterized.parameters(
      {'count_include_pad': True},
      {'count_include_pad': False})
  def test_avg_pool(self, count_include_pad):
    x = jnp.full((1, 3, 3, 1), 2.)
    pool = lambda x: nn.avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0.25, 0.5, 0.25],
        [0.5, 1., 0.5],
        [0.25, 0.5, 0.25],
    ]).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
      {'count_include_pad': True},
      {'count_include_pad': False})
  def test_avg_pool_no_batch(self, count_include_pad):
    x = jnp.full((3, 3, 1), 2.)
    pool = lambda x: nn.avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((2, 2, 1), 2.))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0.25, 0.5, 0.25],
        [0.5, 1., 0.5],
        [0.25, 0.5, 0.25],
    ]).reshape((3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
      {'count_include_pad': True},
      {'count_include_pad': False})
  def test_avg_pool_padding_same(self, count_include_pad):
    x = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    pool = lambda x: nn.avg_pool(x, (2, 2), padding="SAME", count_include_pad=count_include_pad)
    y = pool(x)
    if count_include_pad:
      expected_y = jnp.array([10.0 / 4, 6.0 / 4, 7.0 / 4, 4.0 / 4]).reshape((1, 2, 2, 1))
    else:
      expected_y = jnp.array([10.0 / 4, 6.0 / 2, 7.0 / 2, 4.0 / 1]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_max_pool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    pool = lambda x: nn.max_pool(x, (2, 2))
    expected_y = jnp.array([
        [4., 5.],
        [7., 8.],
    ]).reshape((1, 2, 2, 1))
    y = pool(x)
    np.testing.assert_allclose(y, expected_y)
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.],
    ]).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
      {'count_include_pad': True},
      {'count_include_pad': False})
  def test_avg_pool_padding_same(self, count_include_pad):
    x = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    pool = lambda x: nn.avg_pool(x, (2, 2), padding="SAME", count_include_pad=count_include_pad)
    y = pool(x)
    if count_include_pad:
      expected_y = jnp.array([10.0 / 4, 6.0 / 4, 7.0 / 4, 4.0 / 4]).reshape((1, 2, 2, 1))
    else:
      expected_y = jnp.array([10.0 / 4, 6.0 / 2, 7.0 / 2, 4.0 / 1]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_pooling_variable_batch_dims(self):
    x = jnp.zeros((1, 8, 32, 32, 3), dtype=jnp.float32)
    y = nn.max_pool(x, (2, 2), (2, 2))

    assert y.shape == (1, 8, 16, 16, 3)

  def test_pooling_no_batch_dims(self):
    x = jnp.zeros((32, 32, 3), dtype=jnp.float32)
    y = nn.max_pool(x, (2, 2), (2, 2))

    assert y.shape == (16, 16, 3)

class NormalizationTest(parameterized.TestCase):

  def test_batch_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (4, 3, 2))
    model_cls = nn.BatchNorm(momentum=0.9, use_running_average=False)
    y, initial_params = model_cls.init_with_output(key2, x)

    mean = y.mean((0, 1))
    var = y.var((0, 1))
    np.testing.assert_allclose(mean, np.array([0., 0.]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1., 1.]), rtol=1e-4)

    y, vars_out = model_cls.apply(initial_params, x, mutable=['batch_stats'])

    ema = vars_out['batch_stats']
    np.testing.assert_allclose(
        ema['mean'], 0.1 * x.mean((0, 1), keepdims=False), atol=1e-4)
    np.testing.assert_allclose(
        ema['var'], 0.9 + 0.1 * x.var((0, 1), keepdims=False), rtol=1e-4)

  def test_batch_norm_complex(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (4, 3, 2), dtype=jnp.complex64)
    model_cls = nn.BatchNorm(momentum=0.9, use_running_average=False, dtype=jnp.complex64)
    y, initial_params = model_cls.init_with_output(key2, x)

    mean = y.mean((0, 1))
    var = y.var((0, 1))
    np.testing.assert_allclose(mean, np.array([0., 0.]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1., 1.]), rtol=1e-4)
    self.assertEqual(mean.dtype, jnp.complex64)

    y, vars_out = model_cls.apply(initial_params, x, mutable=['batch_stats'])

    ema = vars_out['batch_stats']
    np.testing.assert_allclose(
        ema['mean'], 0.1 * x.mean((0, 1), keepdims=False), atol=1e-4)
    np.testing.assert_allclose(
        ema['var'], 0.9 + 0.1 * x.var((0, 1), keepdims=False), rtol=1e-4)

  @parameterized.parameters(
      {'reduction_axes': -1},
      {'reduction_axes': 1},
      {'reduction_axes': (1, 2)})
  def test_layer_norm(self, reduction_axes):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    model_cls = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=e,
                             reduction_axes=reduction_axes)
    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)
    y_one_liner = ((x - x.mean(axis=reduction_axes, keepdims=True)) *
                   jax.lax.rsqrt(x.var(axis=reduction_axes, keepdims=True) + e))
    np.testing.assert_allclose(y_one_liner, y, atol=1e-4)

  @parameterized.parameters(
      {'reduction_axes': -1},
      {'reduction_axes': 1},
      {'reduction_axes': (1, 2)})
  def test_rms_norm(self, reduction_axes):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    model_cls = nn.RMSNorm(use_scale=False, epsilon=e,
                           reduction_axes=reduction_axes)
    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)
    y_one_liner = (x * jax.lax.rsqrt(jnp.mean(jax.lax.square(x), axis=reduction_axes, keepdims=True) + e))
    np.testing.assert_allclose(y_one_liner, y, atol=1e-4)

  def test_group_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(num_groups=2, use_bias=False, use_scale=False, epsilon=e)

    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)

    x_gr = x.reshape([2, 5, 4, 4, 2, 16])
    y_test = ((x_gr - x_gr.mean(axis=[1, 2, 3, 5], keepdims=True)) *
              jax.lax.rsqrt(x_gr.var(axis=[1, 2, 3, 5], keepdims=True) + e))
    y_test = y_test.reshape([2, 5, 4, 4, 32])

    np.testing.assert_allclose(y_test, y, atol=1e-4)

  def test_group_norm_raises(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(num_groups=3, use_bias=False, use_scale=False, epsilon=e)

    with self.assertRaises(ValueError):
      model_cls.init_with_output(key2, x)

  def test_batch_norm_multi_init(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        norm = nn.BatchNorm(
            name="norm",
            use_running_average=False,
            axis_name="batch",
        )
        x = norm(x)
        return x, norm(x)

    key = random.PRNGKey(0)
    model = Foo()
    x = random.normal(random.PRNGKey(1), (2, 4))
    (y1, y2), variables = model.init_with_output(key, x)
    np.testing.assert_allclose(y1, y2, rtol=1e-4)


class StochasticTest(absltest.TestCase):

  def test_dropout(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    module = nn.Dropout(rate=0.5)
    y1 = module.apply({},
                      jnp.ones((20, 20)),
                      deterministic=False,
                      rngs={'dropout': key1})
    y2 = module.apply({},
                      jnp.ones((20, 20)),
                      deterministic=False,
                      rngs={'dropout': key2})
    self.assertFalse(np.all(y1 == y2))

    y1 = module.apply({},
                      jnp.ones((20, 20)),
                      deterministic=True,
                      rngs={'dropout': key1})
    y2 = module.apply({},
                      jnp.ones((20, 20)),
                      deterministic=True,
                      rngs={'dropout': key2})
    self.assertTrue(np.all(y1 == y2))

  def test_dropout_rate_stats(self):
    rootkey = random.PRNGKey(0)
    for rate in np.arange(0.1, 1.0, 0.1):
      rootkey, subkey = random.split(rootkey)
      module = nn.Dropout(rate=rate)
      n_trials = 10
      nonzero_counts = 0
      for key in random.split(subkey, n_trials):
        y = module.apply({},
                         jnp.ones((100, 100)),
                         deterministic=False,
                         rngs={'dropout': key})
        nonzero_counts += np.sum(y > 0.0)
      all_counts = np.prod((100, 100, n_trials))
      frac = np.sum(nonzero_counts) / all_counts
      keep_rate = 1.0 - rate
      # just check within 3 sigma.
      delta = 3 * np.sqrt(rate * keep_rate) / np.sqrt(all_counts)
      self.assertTrue(keep_rate - delta < frac < keep_rate + delta)

  def test_dropout_rate_limits(self):
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    inputs = jnp.ones((20, 20))
    d0 = nn.Dropout(rate=0.0)
    y1 = d0.apply({}, inputs,
                  deterministic=False,
                  rngs={'dropout': key1})
    np.testing.assert_array_equal(y1, inputs)
    d1 = nn.Dropout(rate=1.0)
    y2 = d1.apply({}, inputs,
                  deterministic=False,
                  rngs={'dropout': key2})
    np.testing.assert_array_equal(y2, np.zeros_like(inputs))
    # ensure gradient of rate==1.0 case is non-NaN
    fn = lambda x, k: d1.apply({}, x,
                               rngs={'dropout': k},
                               deterministic=False)
    res = jax.grad(lambda x, k: jnp.sum(fn(x, k)))(inputs, key3)
    self.assertFalse(np.isnan(res).any())


# TODO(flax-dev): add integration tests for RNN cells
class RecurrentTest(absltest.TestCase):

  def test_lstm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    c0, h0 = nn.LSTMCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    lstm = nn.LSTMCell()
    (carry, y), initial_params = lstm.init_with_output(key2, (c0, h0), x)
    self.assertEqual(carry[0].shape, (2, 4))
    self.assertEqual(carry[1].shape, (2, 4))
    np.testing.assert_allclose(y, carry[1])
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(param_shapes, {
        'ii': {'kernel': (3, 4)},
        'if': {'kernel': (3, 4)},
        'ig': {'kernel': (3, 4)},
        'io': {'kernel': (3, 4)},
        'hi': {'kernel': (4, 4), 'bias': (4,)},
        'hf': {'kernel': (4, 4), 'bias': (4,)},
        'hg': {'kernel': (4, 4), 'bias': (4,)},
        'ho': {'kernel': (4, 4), 'bias': (4,)},
    })

  def test_gru(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    carry0 = nn.GRUCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(carry0.shape, (2, 4))
    gru = nn.GRUCell()
    (carry, y), initial_params = gru.init_with_output(key2, carry0, x)
    self.assertEqual(carry.shape, (2, 4))
    np.testing.assert_allclose(y, carry)
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(param_shapes, {
        'ir': {'kernel': (3, 4), 'bias': (4,)},
        'iz': {'kernel': (3, 4), 'bias': (4,)},
        'in': {'kernel': (3, 4), 'bias': (4,)},
        'hr': {'kernel': (4, 4)},
        'hz': {'kernel': (4, 4)},
        'hn': {'kernel': (4, 4), 'bias': (4,)},
    })

  def test_complex_input_gru(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3), dtype=jnp.complex64)
    carry0 = nn.GRUCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(carry0.shape, (2, 4))
    gru = nn.GRUCell()
    (carry, y), _ = gru.init_with_output(key2, carry0, x)
    self.assertEqual(carry.dtype, jnp.complex64)
    self.assertEqual(y.dtype, jnp.complex64)

  def test_convlstm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 4, 4, 3))
    c0, h0 = nn.ConvLSTMCell.initialize_carry(rng, (2,), (4, 4, 6))
    self.assertEqual(c0.shape, (2, 4, 4, 6))
    self.assertEqual(h0.shape, (2, 4, 4, 6))
    lstm = nn.ConvLSTMCell(features=6, kernel_size=(3, 3))
    (carry, y), initial_params = lstm.init_with_output(key2, (c0, h0), x)
    self.assertEqual(carry[0].shape, (2, 4, 4, 6))
    self.assertEqual(carry[1].shape, (2, 4, 4, 6))
    np.testing.assert_allclose(y, carry[1])
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(param_shapes, {
        'hh': {'bias': (6*4,), 'kernel': (3, 3, 6, 6*4)},
        'ih': {'bias': (6*4,), 'kernel': (3, 3, 3, 6*4)},
    })

  def test_optimized_lstm_cell_matches_regular(self):

    # Create regular LSTMCell.
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    c0, h0 = nn.LSTMCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    lstm = nn.LSTMCell()
    (_, y), lstm_params = lstm.init_with_output(key2, (c0, h0), x)

    # Create OptimizedLSTMCell.
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (2, 3))
    c0, h0 = nn.OptimizedLSTMCell.initialize_carry(rng, (2,), 4)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    lstm_opt = nn.OptimizedLSTMCell()
    (_, y_opt), lstm_opt_params = lstm_opt.init_with_output(key2, (c0, h0), x)

    np.testing.assert_allclose(y, y_opt, rtol=1e-6)
    check_eq(lstm_params, lstm_opt_params)


class IdsTest(absltest.TestCase):

  def test_hashable(self):
    id1 = ids.uuid()
    id2 = ids.uuid()
    self.assertEqual(id1, id1)
    self.assertNotEqual(id1, id2)
    self.assertNotEqual(hash(id1), hash(id2))
    id1c = copy.copy(id1)
    id1dc = copy.deepcopy(id1)
    self.assertNotEqual(hash(id1), hash(id1c))
    self.assertNotEqual(hash(id1), hash(id1dc))


if __name__ == '__main__':
  absltest.main()

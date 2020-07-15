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

"""Tests for flax.nn."""

from absl.testing import absltest

from flax import linen as nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class PoolTest(absltest.TestCase):

  def test_pool_custom_reduce(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    mul_reduce = lambda x, y: x * y
    y = nn.pooling.pool(x, 1., mul_reduce, (2, 2), (1, 1), 'VALID')
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2. ** 4))

  def test_avg_pool(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    pool = lambda x: nn.avg_pool(x, (2, 2))
    y = pool(x)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array([
        [0.25, 0.5, 0.25],
        [0.5, 1., 0.5],
        [0.25, 0.5, 0.25],
    ]).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

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



class NormalizationTest(absltest.TestCase):

  def test_batch_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (4, 3, 2))
    model_cls = nn.BatchNorm(None, momentum=0.9)
    y, initial_params = model_cls.init_with_output(key2, x)

    mean = y.mean((0, 1))
    var = y.var((0, 1))
    np.testing.assert_allclose(mean, np.array([0., 0.]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1., 1.]), rtol=1e-4)

    y, vars_out = model_cls.apply(initial_params, x, mutable=['batchstats'])

    ema = vars_out['batchstats']
    np.testing.assert_allclose(
        ema['mean'], 0.1 * x.mean((0, 1), keepdims=False), atol=1e-4)
    np.testing.assert_allclose(
        ema['var'], 0.9 + 0.1 * x.var((0, 1), keepdims=False), rtol=1e-4)

  def test_layer_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    model_cls = nn.LayerNorm(None, use_bias=False, use_scale=False, epsilon=e)
    y, _ = model_cls.init_with_output(key2, x)
    assert x.shape == y.shape
    input_type = type(x)
    assert isinstance(y, input_type)
    y_one_liner = ((x - x.mean(axis=-1, keepdims=True)) *
                   jax.lax.rsqrt(x.var(axis=-1, keepdims=True) + e))
    np.testing.assert_allclose(y_one_liner, y, atol=1e-4)

  def test_group_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(None, num_groups=2, use_bias=False, use_scale=False, epsilon=e)

    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.shape, y.shape)
    self.assertIsInstance(y, type(x))

    x_gr = x.reshape([2, 5, 4, 4, 2, 16])
    y_test = ((x_gr - x_gr.mean(axis=[1, 2, 3, 5], keepdims=True)) *
              jax.lax.rsqrt(x_gr.var(axis=[1, 2, 3, 5], keepdims=True) + e))
    y_test = y_test.reshape([2, 5, 4, 4, 32])

    np.testing.assert_allclose(y_test, y, atol=1e-4)

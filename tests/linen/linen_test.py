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

"""Tests for flax.linen."""

import copy
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest, parameterized
from jax import random

from flax import ids
from flax import linen as nn
from flax.linen import fp8_ops
from flax.training import train_state

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def check_eq(xs, ys):
  return jax.tree_util.tree_all(
    jax.tree_util.tree_map(np.testing.assert_allclose, xs, ys)
  )


class PoolTest(parameterized.TestCase):
  def test_pool_custom_reduce(self):
    x = jnp.full((1, 3, 3, 1), 2.0)
    mul_reduce = lambda x, y: x * y
    y = nn.pooling.pool(x, 1.0, mul_reduce, (2, 2), (1, 1), 'VALID')
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.0**4))

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool(self, count_include_pad):
    x = jnp.full((1, 3, 3, 1), 2.0)
    pool = lambda x: nn.avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.0))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.25, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 0.25],
      ]
    ).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool_no_batch(self, count_include_pad):
    x = jnp.full((3, 3, 1), 2.0)
    pool = lambda x: nn.avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((2, 2, 1), 2.0))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.25, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 0.25],
      ]
    ).reshape((3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  def test_max_pool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    pool = lambda x: nn.max_pool(x, (2, 2))
    expected_y = jnp.array(
      [
        [4.0, 5.0],
        [7.0, 8.0],
      ]
    ).reshape((1, 2, 2, 1))
    y = pool(x)
    np.testing.assert_allclose(y, expected_y)
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
      ]
    ).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool_padding_same(self, count_include_pad):
    x = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    pool = lambda x: nn.avg_pool(
      x, (2, 2), padding='SAME', count_include_pad=count_include_pad
    )
    y = pool(x)
    if count_include_pad:
      expected_y = jnp.array([10.0 / 4, 6.0 / 4, 7.0 / 4, 4.0 / 4]).reshape(
        (1, 2, 2, 1)
      )
    else:
      expected_y = jnp.array([10.0 / 4, 6.0 / 2, 7.0 / 2, 4.0 / 1]).reshape(
        (1, 2, 2, 1)
      )
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
  def test_layer_norm_mask(self):
    key = random.key(0)
    keys = random.split(key)
    x = random.normal(keys[0], (3, 4, 5))
    m = random.choice(keys[1], 2, x.shape).astype(bool)
    m = m.at[..., :2].set(True)  # guarantee at least 2 elements
    x = jnp.where(m, x, jnp.nan)

    module = nn.LayerNorm()
    y, w = module.init_with_output(key, x, mask=m)

    z = y.mean(-1, where=m)
    np.testing.assert_allclose(z, 0, atol=1e-4)

    z = y.var(-1, where=m)
    np.testing.assert_allclose(z, 1, atol=1e-4)

  def test_rms_norm_mask(self):
    key = random.key(0)
    keys = random.split(key)
    x = random.normal(keys[0], (3, 4, 5))
    m = random.choice(keys[1], 2, x.shape).astype(bool)
    m = m.at[..., :1].set(True)  # guarantee at least 1 element
    x = jnp.where(m, x, jnp.nan)

    module = nn.RMSNorm()
    y, w = module.init_with_output(key, x, mask=m)

    z = np.square(y).mean(-1, where=m)
    np.testing.assert_allclose(z, 1, atol=1e-4)

  def test_group_norm_mask(self):
    key = random.key(0)
    keys = random.split(key)
    x = random.normal(keys[0], (13, 3, 5, 7 * 11))
    m = random.choice(keys[1], 2, x.shape).astype(bool)
    m = m.at[..., :2].set(True)  # guarantee at least 2 elements
    x = jnp.where(m, x, jnp.nan)

    module = nn.GroupNorm(7, use_bias=False, use_scale=False)
    y, w = module.init_with_output(key, x, mask=m)

    yr = y.reshape((13, 3, 5, 7, 11))
    mr = m.reshape((13, 3, 5, 7, 11))

    axes = list(range(1, x.ndim - 1)) + [-1]

    z = yr.mean(axes, where=mr)
    np.testing.assert_allclose(z, 0, atol=1e-4)

    z = yr.var(axes, where=mr)
    np.testing.assert_allclose(z, 1, atol=1e-4)

  @parameterized.parameters({'test_mask': True}, {'test_mask': False})
  def test_batch_norm(self, test_mask):
    rng = random.key(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (4, 3, 2))
    if test_mask:
      m = random.randint(
        key2, (4, 3, 1), minval=0, maxval=2, dtype=jnp.int32
      ).astype(jnp.bool_)
      x = jnp.where(m, x, jnp.ones_like(x) * jnp.nan)
    else:
      m = None
    model_cls = nn.BatchNorm(momentum=0.9, use_running_average=False)
    y, initial_params = model_cls.init_with_output(key3, x, mask=m)

    mean = y.mean((0, 1), where=m)
    var = y.var((0, 1), where=m)
    np.testing.assert_allclose(mean, np.array([0.0, 0.0]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1.0, 1.0]), rtol=1e-4)
    _, vars_out = model_cls.apply(
      initial_params, x, mutable=['batch_stats'], mask=m
    )

    ema = vars_out['batch_stats']
    np.testing.assert_allclose(
      ema['mean'], 0.1 * x.mean((0, 1), keepdims=False, where=m), atol=1e-4
    )
    np.testing.assert_allclose(
      ema['var'],
      0.9 + 0.1 * x.var((0, 1), keepdims=False, where=m),
      rtol=1e-4,
    )

  @parameterized.parameters({'test_mask': True}, {'test_mask': False})
  def test_batch_norm_complex(self, test_mask):
    rng = random.key(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (4, 3, 2), dtype=jnp.complex64)
    if test_mask:
      m = random.randint(
        key2, (4, 3, 1), minval=0, maxval=2, dtype=jnp.int32
      ).astype(jnp.bool_)
      x = jnp.where(m, x, jnp.ones_like(x) * jnp.nan)
    else:
      m = None
    model_cls = nn.BatchNorm(
      momentum=0.9, use_running_average=False, dtype=jnp.complex64
    )
    y, initial_params = model_cls.init_with_output(key3, x, mask=m)

    mean = y.mean((0, 1), where=m)
    var = y.var((0, 1), where=m)
    np.testing.assert_allclose(mean, np.array([0.0, 0.0]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1.0, 1.0]), rtol=1e-4)
    self.assertEqual(mean.dtype, jnp.complex64)

    _, vars_out = model_cls.apply(
      initial_params, x, mutable=['batch_stats'], mask=m
    )

    ema = vars_out['batch_stats']
    np.testing.assert_allclose(
      ema['mean'], 0.1 * x.mean((0, 1), keepdims=False, where=m), atol=1e-4
    )
    np.testing.assert_allclose(
      ema['var'],
      0.9 + 0.1 * x.var((0, 1), keepdims=False, where=m),
      rtol=1e-4,
    )

  @parameterized.parameters(
    {'reduction_axes': -1},
    {'reduction_axes': 1},
    {'reduction_axes': (1, 2)},
    {'reduction_axes': (0, 1, 2)},
    {'reduction_axes': -1, 'use_fast_variance': False},
  )
  def test_layer_norm(self, reduction_axes, use_fast_variance=True):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    if not use_fast_variance:
      x += 1e6  # This blows up fast variance, but should work otherwise.
    model_cls = nn.LayerNorm(
      use_bias=False,
      use_scale=False,
      epsilon=e,
      reduction_axes=reduction_axes,
      use_fast_variance=use_fast_variance,
    )
    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)
    y_one_liner = (
      x - x.mean(axis=reduction_axes, keepdims=True)
    ) * jax.lax.rsqrt(x.var(axis=reduction_axes, keepdims=True) + e)
    np.testing.assert_allclose(y_one_liner, y, atol=1e-3, rtol=1e-3)

  @parameterized.parameters(
    {'reduction_axes': -1}, {'reduction_axes': 1}, {'reduction_axes': (1, 2)}
  )
  def test_rms_norm(self, reduction_axes):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4))
    model_cls = nn.RMSNorm(
      use_scale=False, epsilon=e, reduction_axes=reduction_axes
    )
    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)
    y_one_liner = x * jax.lax.rsqrt(
      jnp.mean(jax.lax.square(x), axis=reduction_axes, keepdims=True) + e
    )
    np.testing.assert_allclose(y_one_liner, y, atol=1e-4)

  def test_group_norm(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(
      num_groups=2, use_bias=False, use_scale=False, epsilon=e
    )

    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)

    x_gr = x.reshape([2, 5, 4, 4, 2, 16])
    y_test = (
      x_gr - x_gr.mean(axis=[1, 2, 3, 5], keepdims=True)
    ) * jax.lax.rsqrt(x_gr.var(axis=[1, 2, 3, 5], keepdims=True) + e)
    y_test = y_test.reshape([2, 5, 4, 4, 32])

    np.testing.assert_allclose(y_test, y, atol=1e-4)

  def test_group_norm_unbatched(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(
        num_groups=2,
        use_bias=False,
        use_scale=False,
        epsilon=e,
        reduction_axes=(0, 1, 3, 4),
    )

    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)

    x_gr = x.reshape([2, 5, 4, 4, 2, 16])
    y_test = (
        x_gr - x_gr.mean(axis=[0, 1, 3, 5], keepdims=True)
    ) * jax.lax.rsqrt(x_gr.var(axis=[0, 1, 3, 5], keepdims=True) + e)
    y_test = y_test.reshape([2, 5, 4, 4, 32])

    np.testing.assert_allclose(y_test, y, atol=1e-4)

  def test_group_norm_batched(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (3, 4, 32))
    model_cls = nn.GroupNorm(
        num_groups=2,
        use_bias=False,
        use_scale=False,
        epsilon=e,
        reduction_axes=(-3, -2, -1),
    )

    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)

    x_stacked = jnp.stack([x] * 5)
    y_stacked = model_cls.apply({}, x_stacked)

    np.testing.assert_allclose(y, y_stacked[0, ...], atol=1e-4)

    x_gr = x_stacked.reshape([5, 3, 4, 2, 16])
    y_test = (x_gr - x_gr.mean(axis=[1, 2, 4], keepdims=True)) * jax.lax.rsqrt(
        x_gr.var(axis=[1, 2, 4], keepdims=True) + e
    )
    y_test = y_test.reshape([5, 3, 4, 32])

    np.testing.assert_allclose(y_test, y_stacked, atol=1e-4)

  def test_group_norm_raises(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(
        num_groups=3, use_bias=False, use_scale=False, epsilon=e
    )

    with self.assertRaises(ValueError):
      model_cls.init_with_output(key2, x)

  def test_group_norm_raises_incorrect_reduction_axes(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 5, 4, 4, 32))
    model_cls = nn.GroupNorm(
        num_groups=3,
        use_bias=False,
        use_scale=False,
        epsilon=e,
        reduction_axes=(0, 1, 2, 3),
    )

    with self.assertRaises(ValueError):
      model_cls.init_with_output(key2, x)

  def test_batch_norm_multi_init(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        norm = nn.BatchNorm(
          name='norm',
          use_running_average=False,
          axis_name='batch',
        )
        x = norm(x)
        return x, norm(x)

    key = random.key(0)
    model = Foo()
    x = random.normal(random.key(1), (2, 4))
    (y1, y2), _ = model.init_with_output(key, x)
    np.testing.assert_allclose(y1, y2, rtol=0.005)

  @parameterized.parameters(
    {'feature_axes': -1},
    {'feature_axes': (1, 2)},
    {'feature_axes': (1, 2, 3)},
    {'feature_axes': -1, 'use_fast_variance': False},
  )
  def test_instance_norm(self, feature_axes, use_fast_variance=True):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    e = 1e-5
    x = random.normal(key1, (2, 3, 4, 5))
    if not use_fast_variance:
      x += 1e4  # This blows up fast variance, but should work otherwise.
    model_cls = nn.InstanceNorm(
      use_bias=False,
      use_scale=False,
      epsilon=e,
      feature_axes=feature_axes,
      use_fast_variance=use_fast_variance,
    )
    y, _ = model_cls.init_with_output(key2, x)
    self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.shape, y.shape)

    canonicalized_feature_axes = [
      i if i >= 0 else (x.ndim + i)
      for i in (
        feature_axes if isinstance(feature_axes, tuple) else (feature_axes,)
      )
    ]
    reduction_axes = [
      i for i in range(1, x.ndim) if i not in canonicalized_feature_axes
    ]
    y_one_liner = (
      x - x.mean(axis=reduction_axes, keepdims=True)
    ) * jax.lax.rsqrt(x.var(axis=reduction_axes, keepdims=True) + e)

    np.testing.assert_allclose(y_one_liner, y, atol=1e-6)

  @parameterized.parameters(
    {'feature_axes': 0},
    {'feature_axes': -4},
    {'feature_axes': (0, 3)},
    {'feature_axes': (2, -4)},
  )
  def test_instance_norm_raise_error(self, feature_axes):
    with self.assertRaisesRegex(
      ValueError,
      'The channel axes cannot include the leading dimension '
      'as this is assumed to be the batch axis.',
    ):
      x = jax.random.normal(jax.random.key(0), (2, 3, 4, 5))
      layer = nn.InstanceNorm(feature_axes=feature_axes)
      _ = layer.init(jax.random.key(1), x)

  @parameterized.parameters(
    {
      'layer1': nn.LayerNorm(feature_axes=(1, 2)),
      'layer2': nn.InstanceNorm(feature_axes=(1, 2)),
    },
    {
      'layer1': nn.LayerNorm(reduction_axes=(1, 2), feature_axes=-1),
      'layer2': nn.InstanceNorm(feature_axes=-1),
    },
    {
      'layer1': nn.LayerNorm(
        reduction_axes=-2,
        feature_axes=(1, 3),
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform(),
      ),
      'layer2': nn.InstanceNorm(
        feature_axes=(1, -1),
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform(),
      ),
    },
    {
      'layer1': nn.LayerNorm(
        reduction_axes=(1, 2, 3),
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform(),
      ),
      'layer2': nn.GroupNorm(
        num_groups=1,
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform()
      ),
    },
    {
      'layer1': nn.InstanceNorm(
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform(),
      ),
      'layer2': nn.GroupNorm(
        num_groups=None,
        group_size=1,
        bias_init=nn.initializers.uniform(),
        scale_init=nn.initializers.uniform(),
      ),
    },
  )
  def test_normalization_equivalence(self, layer1, layer2):
    x = jax.random.normal(jax.random.key(0), (2, 3, 4, 5))
    layer1_variables = layer1.init(jax.random.key(1), x)
    layer2_variables = layer2.init(jax.random.key(1), x)
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda v1, v2: (v1 == v2).all(),
                layer1_variables,
                layer2_variables,
            )
        )
    )

    layer1_y = layer1.apply(layer1_variables, x)
    layer2_y = layer2.apply(layer2_variables, x)
    np.testing.assert_allclose(layer1_y, layer2_y, atol=1e-7)

  @parameterized.parameters(
    {
      'model_index': 0,
      'key_paths': {'Dense_1/kernel/u', 'Dense_1/kernel/sigma'},
    },
    {
      'model_index': 1,
      'key_paths': {'Conv_0/kernel/u', 'Conv_0/kernel/sigma'},
    },
    {
      'model_index': 2,
      'key_paths': {
        'MultiHeadDotProductAttention_0/key/bias/u',
        'MultiHeadDotProductAttention_0/key/kernel/u',
        'MultiHeadDotProductAttention_0/out/kernel/u',
        'MultiHeadDotProductAttention_0/query/bias/u',
        'MultiHeadDotProductAttention_0/query/kernel/u',
        'MultiHeadDotProductAttention_0/value/bias/u',
        'MultiHeadDotProductAttention_0/value/kernel/u',
        'MultiHeadDotProductAttention_0/key/bias/sigma',
        'MultiHeadDotProductAttention_0/key/kernel/sigma',
        'MultiHeadDotProductAttention_0/out/kernel/sigma',
        'MultiHeadDotProductAttention_0/query/bias/sigma',
        'MultiHeadDotProductAttention_0/query/kernel/sigma',
        'MultiHeadDotProductAttention_0/value/bias/sigma',
        'MultiHeadDotProductAttention_0/value/kernel/sigma',
      },
    },
  )
  def test_spectral_norm_train(self, model_index, key_paths):
    class FooDense(nn.Module):
      @nn.compact
      def __call__(self, x, train):
        x = nn.Dense(8)(x)
        x = nn.SpectralNorm(nn.Dense(6))(x, update_stats=train)
        x = nn.Dense(4)(x)
        return x

    class FooConv(nn.Module):
      @nn.compact
      def __call__(self, x, train):
        x = nn.Dense(9)(x)
        x = x.reshape((1, 3, 3))
        x = nn.SpectralNorm(nn.Conv(2, kernel_size=(2, 2)))(
          x, update_stats=train
        )
        x = x.reshape(1, -1)
        x = nn.Dense(4)(x)
        return x

    class FooAttention(nn.Module):
      @nn.compact
      def __call__(self, x, train):
        a = nn.Dense(4)(x)
        b = nn.Dense(4)(x)
        x = nn.SpectralNorm(nn.attention.MultiHeadDotProductAttention(4))(
          a, b, update_stats=train
        )
        x = nn.Dense(4)(x)
        return x

    key1, key2, key3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(key1, (1, 4))
    y = random.normal(key2, (1, 4))

    model_cls = (FooDense, FooConv, FooAttention)[model_index]
    variables = model_cls().init(key3, x, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    self.assertEqual(key_paths, batch_stats['SpectralNorm_0'].keys())

    class TrainState(train_state.TrainState):
      batch_stats: Any

    state = TrainState.create(
      apply_fn=model_cls().apply,
      params=params,
      batch_stats=batch_stats,
      tx=optax.adam(1e-3),
    )

    @jax.jit
    def train_step(state, batch):
      def loss_fn(params):
        logits, updates = state.apply_fn(
          {'params': params, 'batch_stats': state.batch_stats},
          x=batch['image'],
          train=True,
          mutable=['batch_stats'],
        )
        loss = jnp.mean(
          optax.l2_loss(predictions=logits, targets=batch['label'])
        )
        return loss, updates

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, updates), grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(batch_stats=updates['batch_stats'])
      return state, loss

    prev_loss = float('inf')
    for _ in range(10):
      state, loss = train_step(state, {'image': x, 'label': y})
      self.assertLess(loss, prev_loss)
      prev_loss = loss

  @parameterized.parameters(
    {'n_steps': 1, 'update_stats': True, 'result': 4.0},
    {'n_steps': 3, 'update_stats': True, 'result': 4.0},
    {'n_steps': 10, 'update_stats': True, 'result': 4.0},
    {'n_steps': 1, 'update_stats': False, 'result': 1.0},
  )
  def test_spectral_norm_sigma(self, n_steps, update_stats, result):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x, train):
        x = nn.SpectralNorm(nn.Dense(8, use_bias=False), n_steps=n_steps)(
          x, update_stats=train
        )
        return x

    x = jnp.ones((1, 8))
    model_cls = Foo()
    variables = model_cls.init(random.PRNGKey(0), x, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    params = jax.tree_util.tree_map(lambda x: 4 * jnp.eye(*x.shape), params)
    _, updates = model_cls.apply(
      {'params': params, 'batch_stats': batch_stats},
      x=x,
      train=update_stats,
      mutable=True,
    )
    np.testing.assert_allclose(
      updates['batch_stats']['SpectralNorm_0']['Dense_0/kernel/sigma'],
      result,
      atol=1e-3,
    )

  @parameterized.parameters(
    {'error_on_non_matrix': True}, {'error_on_non_matrix': False}
  )
  def test_spectral_norm_3d_tensor(self, error_on_non_matrix):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x, train):
        x = nn.SpectralNorm(
          nn.DenseGeneral((3, 4), use_bias=False),
          error_on_non_matrix=error_on_non_matrix,
        )(x, update_stats=train)
        return x

    x = jnp.ones((1, 2))
    model_cls = Foo()

    if error_on_non_matrix:
      with self.assertRaisesRegex(
        ValueError, 'Input is 3D but error_on_non_matrix is True'
      ):
        _ = model_cls.init(random.PRNGKey(0), x, train=False)
    else:
      _ = model_cls.init(random.PRNGKey(0), x, train=False)

  @parameterized.parameters(
    {'feature_axes': -1, 'reduction_axes': 0, 'variable_filter': {'kernel'}},
    {'feature_axes': 0, 'reduction_axes': 1, 'variable_filter': {'kernel'}},
    {
      'feature_axes': (0, 1),
      'reduction_axes': (),
      'variable_filter': {'kernel'},
    },
    {
      'feature_axes': (),
      'reduction_axes': (0, 1),
      'variable_filter': {'kernel'},
    },
    {
      'feature_axes': None,
      'reduction_axes': (0, 1),
      'variable_filter': {'kernel'},
    },
    {'feature_axes': 0, 'reduction_axes': (), 'variable_filter': {'bias'}},
    {'feature_axes': (), 'reduction_axes': -1, 'variable_filter': {'bias'}},
  )
  def test_manual_weight_norm(
    self, feature_axes, reduction_axes, variable_filter
  ):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.WeightNorm(
          nn.Dense(2, bias_init=nn.initializers.normal()),
          feature_axes=feature_axes,
          variable_filter=variable_filter,
        )(x)

    key1, key2 = jax.random.split(jax.random.key(1))
    x = jax.random.normal(key1, (1, 3))
    module = Foo()
    v = module.init(key2, x)
    v = jax.tree_util.tree_map(lambda x: x + 0.5, v)
    out = module.apply(v, x)

    kernel = v['params']['Dense_0']['kernel']
    if 'kernel' in variable_filter:
      kernel /= jnp.sqrt(jnp.sum(kernel**2, axis=reduction_axes, keepdims=True))
      kernel_scale = jnp.expand_dims(
        v['params']['WeightNorm_0']['Dense_0/kernel/scale'],
        axis=reduction_axes,
      )
    else:
      kernel_scale = 1
    bias = v['params']['Dense_0']['bias']
    if 'bias' in variable_filter:
      bias /= jnp.sqrt(jnp.sum(bias**2, axis=reduction_axes, keepdims=True))
      bias_scale = jnp.expand_dims(
        v['params']['WeightNorm_0']['Dense_0/bias/scale'], axis=reduction_axes
      )
    else:
      bias_scale = 1
    manual_out = jnp.dot(x, kernel_scale * kernel) + (
      bias_scale * bias
    ).reshape(1, -1)

    self.assertTrue(jnp.allclose(out, manual_out))

  @parameterized.parameters(
    {
      'variable_filters': ({}, None, {'kernel', 'bias'}, {'Bar'}),
      'key_paths': {
        'Bar_0/Baz_0/Dense_0/kernel/scale',
        'Bar_0/Baz_0/Dense_0/bias/scale',
        'Bar_0/Dense_0/kernel/scale',
        'Bar_0/Dense_0/bias/scale',
        'Bar_0/Baz_1/Dense_0/kernel/scale',
        'Bar_0/Baz_1/Dense_0/bias/scale',
        'Bar_0/Dense_1/kernel/scale',
        'Bar_0/Dense_1/bias/scale',
      },
    },
    {
      'variable_filters': ({'kernel'},),
      'key_paths': {
        'Bar_0/Baz_0/Dense_0/kernel/scale',
        'Bar_0/Dense_0/kernel/scale',
        'Bar_0/Baz_1/Dense_0/kernel/scale',
        'Bar_0/Dense_1/kernel/scale',
      },
    },
    {
      'variable_filters': ({'Baz', 'kernel'},),
      'key_paths': {
        'Bar_0/Baz_0/Dense_0/kernel/scale',
        'Bar_0/Baz_0/Dense_0/bias/scale',
        'Bar_0/Dense_0/kernel/scale',
        'Bar_0/Baz_1/Dense_0/kernel/scale',
        'Bar_0/Baz_1/Dense_0/bias/scale',
        'Bar_0/Dense_1/kernel/scale',
      },
    },
  )
  def test_weight_norm_variable_filter(self, variable_filters, key_paths):
    class Baz(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(2)(x)

    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = Baz()(x)
        x = nn.Dense(3)(x)
        x = Baz()(x)
        x = nn.Dense(3)(x)
        return x

    for variable_filter in variable_filters:

      class Foo(nn.Module):
        @nn.compact
        def __call__(self, x):
          return nn.WeightNorm(Bar(), variable_filter=variable_filter)(x)

      v = Foo().init(jax.random.key(0), jnp.ones((1, 4)))
      self.assertEqual(key_paths, v['params']['WeightNorm_0'].keys())

  @parameterized.parameters(
    {'model_index': 0, 'key_paths': {'Dense_1/kernel/scale'}},
    {'model_index': 1, 'key_paths': {'Conv_0/kernel/scale'}},
    {
      'model_index': 2,
      'key_paths': {
        'MultiHeadDotProductAttention_0/key/kernel/scale',
        'MultiHeadDotProductAttention_0/out/kernel/scale',
        'MultiHeadDotProductAttention_0/query/kernel/scale',
        'MultiHeadDotProductAttention_0/value/kernel/scale',
      },
    },
  )
  def test_weight_norm_train(self, model_index, key_paths):
    class FooDense(nn.Module):
      @nn.compact
      def __call__(
        self,
        x,
      ):
        x = nn.Dense(8)(x)
        x = nn.WeightNorm(nn.Dense(6))(x)
        x = nn.Dense(4)(x)
        return x

    class FooConv(nn.Module):
      @nn.compact
      def __call__(
        self,
        x,
      ):
        x = nn.Dense(9)(x)
        x = x.reshape((1, 3, 3))
        x = nn.WeightNorm(nn.Conv(2, kernel_size=(2, 2)))(x)
        x = x.reshape(1, -1)
        x = nn.Dense(4)(x)
        return x

    class FooAttention(nn.Module):
      @nn.compact
      def __call__(self, x):
        a = nn.Dense(4)(x)
        b = nn.Dense(4)(x)
        x = nn.WeightNorm(nn.attention.MultiHeadDotProductAttention(4))(a, b)
        x = nn.Dense(4)(x)
        return x

    key1, key2, key3 = random.split(random.PRNGKey(0), 3)
    x = random.normal(key1, (1, 4))
    y = random.normal(key2, (1, 4))

    model_cls = (FooDense, FooConv, FooAttention)[model_index]
    params = model_cls().init(key3, x)['params']
    self.assertEqual(key_paths, params['WeightNorm_0'].keys())

    state = train_state.TrainState.create(
      apply_fn=model_cls().apply,
      params=params,
      tx=optax.adam(1e-3),
    )

    @jax.jit
    def train_step(state, batch):
      def loss_fn(params):
        logits = state.apply_fn(
          {'params': params},
          x=batch['image'],
        )
        loss = jnp.mean(
          optax.l2_loss(predictions=logits, targets=batch['label'])
        )
        return loss

      grad_fn = jax.value_and_grad(loss_fn)
      loss, grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)
      return state, loss

    prev_loss = float('inf')
    for _ in range(10):
      state, loss = train_step(state, {'image': x, 'label': y})
      self.assertLess(loss, prev_loss)
      prev_loss = loss


class StochasticTest(parameterized.TestCase):
  def test_dropout(self):
    rng = random.key(0)
    key1, key2 = random.split(rng)
    module = nn.Dropout(rate=0.5)
    y1 = module.apply(
      {}, jnp.ones((20, 20)), deterministic=False, rngs={'dropout': key1}
    )
    y2 = module.apply(
      {}, jnp.ones((20, 20)), deterministic=False, rngs={'dropout': key2}
    )
    self.assertFalse(np.all(y1 == y2))

    y1 = module.apply(
      {}, jnp.ones((20, 20)), deterministic=True, rngs={'dropout': key1}
    )
    y2 = module.apply(
      {}, jnp.ones((20, 20)), deterministic=True, rngs={'dropout': key2}
    )
    self.assertTrue(np.all(y1 == y2))

  def test_dropout_rate_stats(self):
    rootkey = random.key(0)
    for rate in np.arange(0.1, 1.0, 0.1):
      rootkey, subkey = random.split(rootkey)
      module = nn.Dropout(rate=rate)
      n_trials = 10
      nonzero_counts = 0
      for key in random.split(subkey, n_trials):
        y = module.apply(
          {}, jnp.ones((100, 100)), deterministic=False, rngs={'dropout': key}
        )
        nonzero_counts += np.sum(y > 0.0)
      all_counts = np.prod((100, 100, n_trials))
      frac = np.sum(nonzero_counts) / all_counts
      keep_rate = 1.0 - rate
      # just check within 4 sigma.
      delta = 4 * np.sqrt(rate * keep_rate) / np.sqrt(all_counts)
      self.assertTrue(keep_rate - delta < frac < keep_rate + delta)

  def test_dropout_rate_limits(self):
    rng = random.key(0)
    key1, key2, key3 = random.split(rng, 3)
    inputs = jnp.ones((20, 20))
    d0 = nn.Dropout(rate=0.0)
    y1 = d0.apply({}, inputs, deterministic=False, rngs={'dropout': key1})
    np.testing.assert_array_equal(y1, inputs)
    d1 = nn.Dropout(rate=1.0)
    y2 = d1.apply({}, inputs, deterministic=False, rngs={'dropout': key2})
    np.testing.assert_array_equal(y2, np.zeros_like(inputs))
    # ensure gradient of rate==1.0 case is non-NaN
    fn = lambda x, k: d1.apply({}, x, rngs={'dropout': k}, deterministic=False)
    res = jax.grad(lambda x, k: jnp.sum(fn(x, k)))(inputs, key3)
    self.assertFalse(np.isnan(res).any())

  @parameterized.parameters(
    {
      'num_dims': 2,
      'broadcast_dims': (1,),
      'slice_fn': lambda out, i: out[i, :],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 2,
      'broadcast_dims': (0,),
      'slice_fn': lambda out, i: out[:, i],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 3,
      'broadcast_dims': (1, 2),
      'slice_fn': lambda out, i: out[i, :, :],
      'summed_total': 2 * 10 * 10,
    },
    {
      'num_dims': 3,
      'broadcast_dims': (1,),
      'slice_fn': lambda out, i, j: out[i, :, j],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (0, 2, 3),
      'slice_fn': lambda out, i: out[:, i, :, :],
      'summed_total': 2 * 10 * 10 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (0, 1),
      'slice_fn': lambda out, i, j: out[:, :, i, j],
      'summed_total': 2 * 10 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (3,),
      'slice_fn': lambda out, i, j, k: out[i, j, k, :],
      'summed_total': 2 * 10,
    },
  )
  def test_dropout_broadcast(
    self, num_dims, broadcast_dims, slice_fn, summed_total
  ):
    module = nn.Dropout(
      rate=0.5, broadcast_dims=broadcast_dims, deterministic=False
    )
    x = jnp.ones((10,) * num_dims)
    out = module.apply({}, x, rngs={'dropout': random.key(0)})

    for i in range(10):
      if num_dims - len(broadcast_dims) >= 2:
        for j in range(10):
          if num_dims - len(broadcast_dims) >= 3:
            for k in range(10):
              self.assertTrue(slice_fn(out, i, j, k).sum() in (0, summed_total))
          else:
            self.assertTrue(slice_fn(out, i, j).sum() in (0, summed_total))
      else:
        self.assertTrue(slice_fn(out, i).sum() in (0, summed_total))

  def test_dropout_manual_rng(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        key = self.make_rng('dropout')
        x1 = nn.Dropout(rate=0.5, deterministic=False)(x, rng=key)
        x2 = nn.Dropout(rate=0.5, deterministic=False)(x, rng=jax.random.clone(key))
        return x1, x2

    module = Foo()
    x1, x2 = module.apply(
      {}, jnp.ones((20, 20)), rngs={'dropout': random.key(0)}
    )

    np.testing.assert_array_equal(x1, x2)


# TODO(flax-dev): add integration tests for RNN cells
class RecurrentTest(parameterized.TestCase):
  def test_lstm(self):
    lstm = nn.LSTMCell(features=4)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3))
    c0, h0 = lstm.initialize_carry(rng, x.shape)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    (carry, y), initial_params = lstm.init_with_output(key2, (c0, h0), x)
    self.assertEqual(carry[0].shape, (2, 4))
    self.assertEqual(carry[1].shape, (2, 4))
    np.testing.assert_allclose(y, carry[1])
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(
      param_shapes,
      {
        'ii': {'kernel': (3, 4)},
        'if': {'kernel': (3, 4)},
        'ig': {'kernel': (3, 4)},
        'io': {'kernel': (3, 4)},
        'hi': {'kernel': (4, 4), 'bias': (4,)},
        'hf': {'kernel': (4, 4), 'bias': (4,)},
        'hg': {'kernel': (4, 4), 'bias': (4,)},
        'ho': {'kernel': (4, 4), 'bias': (4,)},
      },
    )

  @parameterized.parameters(
    {
      'module_cls': nn.SimpleCell,
      'expected_param_shapes': {
        'i': {'kernel': (3, 4), 'bias': (4,)},
        'h': {'kernel': (4, 4)},
      },
    },
    {
      'module_cls': nn.GRUCell,
      'expected_param_shapes': {
        'ir': {'kernel': (3, 4), 'bias': (4,)},
        'iz': {'kernel': (3, 4), 'bias': (4,)},
        'in': {'kernel': (3, 4), 'bias': (4,)},
        'hr': {'kernel': (4, 4)},
        'hz': {'kernel': (4, 4)},
        'hn': {'kernel': (4, 4), 'bias': (4,)},
      },
    },
    {
      'module_cls': nn.MGUCell,
      'expected_param_shapes': {
        'if': {'kernel': (3, 4), 'bias': (4,)},
        'in': {'kernel': (3, 4), 'bias': (4,)},
        'hf': {'kernel': (4, 4)},
        'hn': {'kernel': (4, 4), 'bias': (4,)},
      },
    },
  )
  def test_gated_units(self, module_cls, expected_param_shapes):
    module = module_cls(features=4)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3))
    carry0 = module.initialize_carry(rng, x.shape)
    self.assertEqual(carry0.shape, (2, 4))
    (carry, y), initial_params = module.init_with_output(key2, carry0, x)
    self.assertEqual(carry.shape, (2, 4))
    np.testing.assert_allclose(y, carry)
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(
      param_shapes,
      expected_param_shapes,
    )
    if module_cls == nn.MGUCell:
      self.assertTrue(
        (initial_params['params']['if']['bias'] == jnp.ones((4,))).all()
      )
      self.assertTrue(
        (initial_params['params']['in']['bias'] == jnp.zeros((4,))).all()
      )
      self.assertTrue(
        (initial_params['params']['hn']['bias'] == jnp.zeros((4,))).all()
      )

  @parameterized.parameters(
    {'module_cls': nn.SimpleCell},
    {'module_cls': nn.GRUCell},
    {'module_cls': nn.MGUCell},
  )
  def test_complex_input_gated_units(self, module_cls):
    module_instance = module_cls(features=4)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3), dtype=jnp.complex64)
    carry0 = module_instance.initialize_carry(rng, x.shape)
    self.assertEqual(carry0.shape, (2, 4))
    (carry, y), _ = module_instance.init_with_output(key2, carry0, x)
    self.assertEqual(carry.dtype, jnp.complex64)
    self.assertEqual(y.dtype, jnp.complex64)

  def test_convlstm(self):
    lstm = nn.ConvLSTMCell(features=6, kernel_size=(3, 3))
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 4, 4, 3))
    c0, h0 = lstm.initialize_carry(rng, x.shape)
    self.assertEqual(c0.shape, (2, 4, 4, 6))
    self.assertEqual(h0.shape, (2, 4, 4, 6))
    (carry, y), initial_params = lstm.init_with_output(key2, (c0, h0), x)
    self.assertEqual(carry[0].shape, (2, 4, 4, 6))
    self.assertEqual(carry[1].shape, (2, 4, 4, 6))
    np.testing.assert_allclose(y, carry[1])
    param_shapes = jax.tree_util.tree_map(np.shape, initial_params['params'])
    self.assertEqual(
      param_shapes,
      {
        'hh': {'bias': (6 * 4,), 'kernel': (3, 3, 6, 6 * 4)},
        'ih': {'bias': (6 * 4,), 'kernel': (3, 3, 3, 6 * 4)},
      },
    )

  def test_optimized_lstm_cell_matches_regular(self):
    # Create regular LSTMCell.
    lstm = nn.LSTMCell(features=4)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3))
    c0, h0 = lstm.initialize_carry(rng, x.shape)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    (_, y), lstm_params = lstm.init_with_output(key2, (c0, h0), x)

    # Create OptimizedLSTMCell.
    lstm_opt = nn.OptimizedLSTMCell(features=4)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3))
    c0, h0 = lstm_opt.initialize_carry(rng, x.shape)
    self.assertEqual(c0.shape, (2, 4))
    self.assertEqual(h0.shape, (2, 4))
    (_, y_opt), lstm_opt_params = lstm_opt.init_with_output(key2, (c0, h0), x)

    np.testing.assert_allclose(y, y_opt, rtol=1e-6)
    check_eq(lstm_params, lstm_opt_params)

  def test_mgu_reset_gate(self):
    module = nn.MGUCell(features=4, reset_gate=False)
    rng = random.key(0)
    rng, key1, key2 = random.split(rng, 3)
    x = random.normal(key1, (2, 3))
    carry0 = module.initialize_carry(rng, x.shape)
    (carry, y), v = module.init_with_output(key2, carry0, x)

    self.assertIn('kernel', v['params']['hn'])
    self.assertNotIn('bias', v['params']['hn'])

    f = jax.nn.sigmoid(
      jnp.dot(x, v['params']['if']['kernel'])
      + v['params']['if']['bias'].reshape(1, -1)
      + jnp.dot(carry0, v['params']['hf']['kernel'])
    )
    n = jax.nn.tanh(
      jnp.dot(x, v['params']['in']['kernel'])
      + v['params']['in']['bias'].reshape(1, -1)
      + jnp.dot(carry0, v['params']['hn']['kernel'])
    )
    expected_out = (1 - f) * n + f * carry0
    np.testing.assert_allclose(y, expected_out)


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

def get_fp8_dtypes(fp8_genre):
    assert fp8_genre in ('OCP', 'NANOO')
    if fp8_genre == 'OCP':
      e4m3_dtype = jnp.float8_e4m3fn
      e5m2_dtype = jnp.float8_e5m2
    else: # fp8_genre == 'NANOO'
      e4m3_dtype = jnp.float8_e4m3fnuz
      e5m2_dtype = jnp.float8_e5m2fnuz
    return e4m3_dtype, e5m2_dtype

class Fp8Test(parameterized.TestCase):
  @parameterized.parameters(
    {'x_shape': (16, 32), 'y_shape': (32, 64), 'g_shape': (16, 64), 'eqn': 'mk,kn->mn'},
    {'x_shape': (2, 3, 32), 'y_shape': (64, 32), 'g_shape': (2, 3, 64), 'eqn': '...k,nk->...n'},
    {'x_shape': (2, 3, 64), 'y_shape': (64, 32), 'g_shape': (2, 3, 32), 'eqn': '...k,kn->...n'},
  )
  def test_fp8_einsum(self, x_shape, y_shape, g_shape, eqn):
    rng, key1, key2, key3 = random.split(random.key(42), 4)
    x = random.normal(key1, x_shape)
    y = random.normal(key2, y_shape)
    g = random.normal(key3, g_shape)
    e4m3_dtype = jnp.float8_e4m3fn
    e5m2_dtype = jnp.float8_e5m2
    cast_to_representable = functools.partial(
        fp8_ops.qdq,
        scale=jnp.ones((1,)),
        compute_dtype=jnp.float32,
    )

    x = cast_to_representable(x, e4m3_dtype)
    y = cast_to_representable(y, e4m3_dtype)
    g = cast_to_representable(g, e5m2_dtype)

    p = nn.Fp8Einsum()
    vars = p.init(rng, eqn, x, y)
    def loss_fn(vars, x, y):
      out = p.apply(vars, eqn, x, y)
      return jnp.sum(out * g.astype(out.dtype))
    step_fn = jax.value_and_grad(loss_fn, argnums=[1, 2])
    out, grads = jax.jit(step_fn)(vars, x, y)

    def loss_fn_ref(x, y):
      out = jnp.einsum(eqn, x, y)
      return jnp.sum(out * g.astype(out.dtype))
    step_fn_ref = jax.value_and_grad(loss_fn_ref, argnums=[0, 1])
    out_ref, grads_ref = jax.jit(step_fn_ref)(x, y)

    np.testing.assert_allclose(out, out_ref, atol=1e-02, rtol=1e-02)
    np.testing.assert_allclose(grads[0], grads_ref[0], atol=1e-02, rtol=1e-02)
    np.testing.assert_allclose(grads[1], grads_ref[1], atol=1e-02, rtol=1e-02)


  @parameterized.parameters(
    {'fp8_genre': 'OCP'}, {'fp8_genre': 'NANOO'}
  )
  def test_fp8_dot_general_injection(self, fp8_genre):
    # Used to cast the inputs to be representable in FP8, so that the difference
    # of the results from the original gemm and fp8 gemm is small.
    cast_to_representable = functools.partial(
      fp8_ops.qdq,
      scale=jnp.ones((1,)),
      compute_dtype=jnp.float32,
    )

    e4m3_dtype, e5m2_dtype = get_fp8_dtypes(fp8_genre)

    init_key, random_key = random.split(random.PRNGKey(seed=123), 2)
    x = cast_to_representable(
      random.uniform(random_key, (16, 32)), e4m3_dtype
    )
    dy = cast_to_representable(
      random.uniform(random_key, (16, 64)), e5m2_dtype
    )

    quant_cls = nn.Fp8DotGeneral if fp8_genre == 'OCP' else nn.NANOOFp8DotGeneralOp

    def run(fp8_injection, expected_shapes):
      p = nn.DenseGeneral(features=64, name='dense')

      if fp8_injection:
        p.dot_general_cls = quant_cls

      init_fn = jax.jit(p.init_with_output)
      y, initial_vars = init_fn(init_key, x)
      var_shapes = jax.tree_util.tree_map(jnp.shape, initial_vars)
      self.assertEqual(var_shapes, expected_shapes)

      def _train(variables, x):
        y = p.apply(variables, x)
        loss = y * dy
        return jnp.mean(loss)

      train_fn = jax.jit(jax.value_and_grad(_train, argnums=[0, 1]))
      outputs, grads = train_fn(initial_vars, x)
      return outputs, grads

    expected_shapes_original = {
      'params': {'kernel': (32, 64), 'bias': (64,)},
    }

    expected_shapes_new = {
      'params': {'kernel': (32, 64), 'bias': (64,)},
      fp8_ops.OVERWRITE_WITH_GRADIENT: {
        f'{quant_cls.__name__}_0': {
          'input_amax_history': (1024,),
          'kernel_amax_history': (1024,),
          'output_grad_amax_history': (1024,),
          'input_scale': (1,),
          'kernel_scale': (1,),
          'output_grad_scale': (1,),
        }
      },
    }
    output1a, output1b = run(False, expected_shapes_original)
    output2a, output2b = run(True, expected_shapes_new)
    dw1, dw2 = output1b[0]['params']['kernel'], output2b[0]['params']['kernel']
    dx1, dx2 = output1b[1], output2b[1]

    np.testing.assert_allclose(output1a, output2a, atol=1e-02)
    np.testing.assert_allclose(dw1, dw2, atol=1e-04)
    np.testing.assert_allclose(dx1, dx2, atol=1e-04)

  @parameterized.parameters(
    {'fp8_genre': 'OCP'}, {'fp8_genre': 'NANOO'}
  )
  def test_fp8_train_state(self, fp8_genre):
    key, init_key, random_key = random.split(random.PRNGKey(seed=123), 3)
    x = random.uniform(random_key, (16, 16), dtype=jnp.float32)

    quant_cls = nn.Fp8DotGeneral if fp8_genre == 'OCP' else nn.NANOOFp8DotGeneralOp
    dense = nn.DenseGeneral(
      features=32, use_bias=True, dot_general_cls=quant_cls
    )

    init_fn = jax.jit(dense.init)
    variables = init_fn(init_key, x)
    opt = optax.adam(learning_rate=0.1)
    state = train_state.TrainState.create(
      params=variables, tx=opt, apply_fn=dense.apply
    )

    def _roll_and_update(amax_h, update):
      return jnp.roll(amax_h, shift=-1, axis=0).at[0].set(update)

    def _train_loss(state, x, dy):
      def loss_fn(vars):
        y = state.apply_fn(vars, x)
        loss = y * dy.astype(y.dtype)
        return jnp.sum(loss)

      grad_fn = jax.grad(loss_fn)
      grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)
      return state

    train_fn = jax.jit(_train_loss)

    scale_x, amax_history_x = jnp.ones(()), jnp.zeros((1024,))
    scale_k, amax_history_k = jnp.ones(()), jnp.zeros((1024,))
    scale_g, amax_history_g = jnp.ones(()), jnp.zeros((1024,))
    e4m3_dtype, e5m2_dtype = get_fp8_dtypes(fp8_genre)
    e4m3_max = jnp.finfo(e4m3_dtype).max.astype(jnp.float32)
    e5m2_max = jnp.finfo(e5m2_dtype).max.astype(jnp.float32)

    for _ in range(5):
      key, random_key = random.split(key, 2)
      x = random.normal(random_key, (16, 16), dtype=jnp.float32)
      g = random.normal(random_key, (16, 32), dtype=jnp.float32)
      k = state.params['params']['kernel']

      # Manually compute the expected amax history and scaling factors.
      amax_from_history_x = jnp.max(amax_history_x, axis=0)
      amax_from_history_k = jnp.max(amax_history_k, axis=0)
      amax_from_history_g = jnp.max(amax_history_g, axis=0)
      scale_x = fp8_ops.compute_scale(amax_from_history_x, scale_x, e4m3_max)
      scale_k = fp8_ops.compute_scale(amax_from_history_k, scale_k, e4m3_max)
      scale_g = fp8_ops.compute_scale(amax_from_history_g, scale_g, e5m2_max)
      amax_history_x = _roll_and_update(amax_history_x, jnp.max(jnp.abs(x)))
      amax_history_k = _roll_and_update(amax_history_k, jnp.max(jnp.abs(k)))
      amax_history_g = _roll_and_update(amax_history_g, jnp.max(jnp.abs(g)))

      state = train_fn(state, x, g)

      rtol, atol = 0.001, 0.001
      fp8_vars = state.params[fp8_ops.OVERWRITE_WITH_GRADIENT][
        f'{quant_cls.__name__}_0'
      ]
      np.testing.assert_allclose(
        fp8_vars['input_amax_history'],
        amax_history_x,
        rtol=rtol,
        atol=atol,
      )
      np.testing.assert_allclose(
        fp8_vars['kernel_amax_history'],
        amax_history_k,
        rtol=rtol,
        atol=atol,
      )
      np.testing.assert_allclose(
        fp8_vars['output_grad_amax_history'],
        amax_history_g,
        rtol=rtol,
        atol=atol,
      )

      np.testing.assert_allclose(fp8_vars['input_scale'][0], scale_x)
      np.testing.assert_allclose(fp8_vars['kernel_scale'][0], scale_k)
      np.testing.assert_allclose(fp8_vars['output_grad_scale'][0], scale_g)

  @parameterized.parameters(
          {'fp8_genre': 'OCP', 'use_jit': True},
          {'fp8_genre': 'OCP', 'use_jit': False},
          {'fp8_genre': 'NANOO', 'use_jit': True},
          {'fp8_genre': 'NANOO', 'use_jit': False}
  )
  def test_fp8_meta_dtype(self, fp8_genre, use_jit):
    if not use_jit and not fp8_ops.CAN_USE_EARRAY:
      self.skipTest("TODO: requires newer jax that has earray")
    f32 = jnp.dtype('float32')
    fmax32 = fp8_ops.fp32_max_grad
    e4m3_dtype, _ = get_fp8_dtypes(fp8_genre)
    e4m3_max = 448 if fp8_genre == 'OCP' else 240

    # Create a scan loop with reused ah_f32 and sf_f32. So, the autograd will
    # accumulate the grads of them. We expect the max op (rather than add op)
    # for the accumulation by converting them to fmax32 dtype.
    def outer(x, ah_f32, sf_f32):
      ah_fmax32 = jax.lax.convert_element_type(ah_f32, fmax32)
      sf_fmax32 = jax.lax.convert_element_type(sf_f32, fmax32)
      array_x = jnp.array([x], f32)
      def body_fun(carry, _):
        carry = fp8_ops.in_qdq(f32, e4m3_dtype, carry, sf_fmax32, ah_fmax32)
        return carry, None
      array_x, _ = jax.lax.scan(body_fun, array_x, None, length=3)
      return array_x[0]

    outer_fn = jax.grad(outer, (0, 1, 2))
    if use_jit:
      outer_fn = jax.jit(outer_fn)
    ah = jnp.array([0., 0., 0.], f32)
    sf = jnp.array([1.], f32)
    # 1st iteration
    grads, new_ah, new_sf = outer_fn(2.0, ah, sf)
    np.testing.assert_allclose(new_ah, [2., 0., 0.])
    np.testing.assert_allclose(new_sf, [1.])
    # 2nd iteration
    grads, new_ah, new_sf = outer_fn(3., new_ah, new_sf)
    np.testing.assert_allclose(new_ah, [3., 0., 2.])
    np.testing.assert_allclose(new_sf, [2. / e4m3_max])
    # 3rd iteration
    grads, new_ah, new_sf = outer_fn(4., new_ah, new_sf)
    np.testing.assert_allclose(new_ah, [4., 2., 3.])
    np.testing.assert_allclose(new_sf, [3. / e4m3_max])

if __name__ == '__main__':
  absltest.main()

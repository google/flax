# Copyright 2021 The Flax Authors.
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

"""Tests for flax.nn.linear."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LinearTest(parameterized.TestCase):

  def test_dense(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 3))
    dense_module = nn.Dense(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    np.testing.assert_allclose(y, np.full((1, 4), 4.))

  def test_dense_extra_batch_dims(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 2, 3))
    dense_module = nn.Dense(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 4), 4.))

  def test_dense_no_bias(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 3))
    dense_module = nn.Dense(
        features=4,
        use_bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 4), 3.))

  def test_dense_is_dense_general(self):
    x = jax.random.normal(random.PRNGKey(0), (5, 3))
    dense_module = nn.Dense(
        features=4,
        use_bias=True,
        bias_init=initializers.normal(),
    )
    y1, _ = dense_module.init_with_output(dict(params=random.PRNGKey(1)), x)
    dg_module = nn.DenseGeneral(
        features=4,
        use_bias=True,
        bias_init=initializers.normal(),
    )
    y2, _ = dg_module.init_with_output(dict(params=random.PRNGKey(1)), x)

    np.testing.assert_allclose(y1, y2)

  def test_dense_general_batch_dim_raises(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 3, 2, 5))
    with self.assertRaises(ValueError):
      dg_module = nn.DenseGeneral(
          features=4,
          batch_dims=(0, 2),
          kernel_init=initializers.ones,
          bias_init=initializers.ones,
      )
      dg_module.init_with_output(rng, x)

  def test_dense_general_two_out(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 3))
    dg_module = nn.DenseGeneral(
        features=(2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 2), 4.))

  def test_dense_general_two_in(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 2, 2))
    dg_module = nn.DenseGeneral(
        features=3,
        axis=(-2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 3), 5.))

  def test_dense_general_batch_dim(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((2, 1, 3, 5))

    state = {'counter': 0.}
    def _counter_init(rng, shape, dtype, state):
      del rng, dtype
      state['counter'] += 1.
      return jnp.full(shape, state['counter'])
    counter_init = functools.partial(_counter_init, state=state)

    dg_module = nn.DenseGeneral(
        features=7,
        axis=(3, -2),
        batch_dims=0,
        bias_init=initializers.ones,
        kernel_init=counter_init,
    )
    y, _ = dg_module.init_with_output(rng, x)
    target = np.concatenate(
        [np.full((1, 1, 7), 16.), np.full((1, 1, 7), 31.)], axis=0)
    np.testing.assert_allclose(y, target)

  @parameterized.parameters([((-2, 3), (), 'bijk,jklm->bilm'),
                             ((3, -2), (), 'bijk,jklm->bilm'),
                             ((-2, 3), (0,), 'bijk,bjklm->bilm')])
  def test_dense_general_vs_numpy(self, axis, batch_dims, einsum_expr):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((16, 8, 9, 10))

    dg_module = nn.DenseGeneral(
        features=(11, 12),
        axis=axis,
        batch_dims=batch_dims,
        bias_init=initializers.ones,
        kernel_init=initializers.normal(),
    )
    y, initial_params = dg_module.init_with_output(rng, x)
    target = np.einsum(einsum_expr, x, initial_params['params']['kernel']) + 1.
    np.testing.assert_allclose(y, target, atol=1e-6)

  def test_conv(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 8, 3))
    conv_module = nn.Conv(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 10.))

  def test_single_input_conv(self):
      rng = dict(params=random.PRNGKey(0))
      x = jnp.ones((8, 3))
      conv_module = nn.Conv(
          features=4,
          kernel_size=(3,),
          padding='VALID',
          kernel_init=initializers.ones,
          bias_init=initializers.ones,
      )
      y, initial_params = conv_module.init_with_output(rng, x)
      self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
      np.testing.assert_allclose(y, np.full((6, 4), 10.))

  def test_group_conv(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 8, 4))
    conv_module = nn.Conv(
        features=4,
        kernel_size=(3,),
        feature_group_count=2,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.))

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2),
      kernel_size=(1, 2, 3, 9),
      n_input_features=(1, 3),
      input_size=(1, 8),
  )
  def test_circular_conv_1d_constant(
          self, n_batch, n_features, kernel_size, n_input_features, input_size
  ):
      """
      1D filter with all elements equal to 1 applied on 1D input with all elements equal to 1.
      Result should have the same shape as input (except for the feature dimension) and
      have all elements equal to n_input_features * kernel_lin_size
      """
      rng = dict(params=random.PRNGKey(0))
      x = jnp.ones((n_batch, input_size, n_input_features))
      conv_module = nn.Conv(
          features=n_features,
          kernel_size=(kernel_size,),
          padding='CIRCULAR',
          kernel_init=initializers.ones,
          bias_init=initializers.zeros,
      )
      y, initial_params = conv_module.init_with_output(rng, x)

      self.assertEqual(
          initial_params['params']['kernel'].shape,
          (kernel_size, n_input_features, n_features),
      )
      correct_ans = np.full(
          (n_batch, input_size, n_features), kernel_size * n_input_features
      )
      np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
      n_batch=(1, 3),
      n_features=(1, 2, 10),
      kernel_lin_size=(1, 2, 3, 9),
      n_input_features=(1, 5),
      input_x_size=(14,),
      input_y_size=(10,),
  )
  def test_circular_conv_2d_constant(
          self,
          n_batch,
          n_features,
          kernel_lin_size,
          n_input_features,
          input_x_size,
          input_y_size,
  ):
      """
      Square filter with all elements equal to 1 applied on a 2D input with all elements equal to 1.
      Result should have the same shape as input (except for the feature dimension) and
      have all elements equal to n_input_features * kernel_lin_size^2
      """
      rng = dict(params=random.PRNGKey(0))
      x = jnp.ones((n_batch, input_x_size, input_y_size, n_input_features))
      conv_module = nn.Conv(
          features=n_features,
          kernel_size=(kernel_lin_size, kernel_lin_size),
          padding='CIRCULAR',
          kernel_init=initializers.ones,
          bias_init=initializers.zeros,
      )
      y, initial_params = conv_module.init_with_output(rng, x)

      self.assertEqual(
          initial_params['params']['kernel'].shape,
          (kernel_lin_size, kernel_lin_size, n_input_features, n_features),
      )
      correct_ans = np.full(
          (n_batch, input_x_size, input_y_size, n_features),
          kernel_lin_size * kernel_lin_size * n_input_features,
      )
      np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_1d_custom(self):
      """
      `Circular 1d convolution with a stride
      """
      rng = dict(params=random.PRNGKey(0))
      x = np.arange(1, 6)
      x = np.expand_dims(x, (0, 2))
      kernel = np.array((1, 2, 1))
      kernel = np.expand_dims(kernel, (1, 2))

      conv_module = nn.Conv(
          features=1,
          kernel_size=(3,),
          strides=(3,),
          padding='CIRCULAR',
          kernel_init=lambda *_: kernel,
          bias_init=initializers.zeros,
      )
      y, initial_params = conv_module.init_with_output(rng, x)

      self.assertEqual(initial_params['params']['kernel'].shape, (3, 1, 1))
      # Compare with manually computed convolution
      correct_ans = np.array((5 + 2 * 1 + 2, 3 + 2 * 4 + 5))
      correct_ans = np.expand_dims(correct_ans, (0, 2))
      np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_2d_custom(self):
      """
      `Circular 2d convolution on a 3x3 example
      """
      rng = dict(params=random.PRNGKey(0))
      x = np.array(((1, 2, 3),
                    (4, 5, 6),
                    (7, 8, 9)))
      x = np.expand_dims(x, (0, 3))
      kernel = np.array(((0, 1, 0),
                         (1, 2, 1),
                         (0, 1, 0)))
      kernel = np.expand_dims(kernel, (2, 3))

      conv_module = nn.Conv(
          features=1,
          kernel_size=(3, 3),
          padding='CIRCULAR',
          kernel_init=lambda *_: kernel,
          bias_init=initializers.zeros,
      )
      y, initial_params = conv_module.init_with_output(rng, x)

      self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 1, 1))
      # Compare with manually computed convolution
      correct_ans = np.array(
          (
              (2 * 1 + 7 + 2 + 4 + 3, 2 * 2 + 8 + 3 + 5 + 1, 2 * 3 + 9 + 1 + 6 + 2),
              (2 * 4 + 1 + 5 + 7 + 6, 2 * 5 + 2 + 6 + 8 + 4, 2 * 6 + 3 + 4 + 9 + 5),
              (2 * 7 + 4 + 8 + 1 + 9, 2 * 8 + 5 + 9 + 2 + 7, 2 * 9 + 6 + 7 + 3 + 8),
          )
      )
      correct_ans = np.expand_dims(correct_ans, (0, 3))
      np.testing.assert_allclose(y, correct_ans)

  def test_conv_transpose(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((1, 8, 3))
    conv_transpose_module = nn.ConvTranspose(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array([[[ 4.,  4.,  4.,  4.],
                              [ 7.,  7.,  7.,  7.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [ 7.,  7.,  7.,  7.],
                              [ 4.,  4.,  4.,  4.]]])
    np.testing.assert_allclose(y, correct_ans)

  def test_single_input_conv_transpose(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.ones((8, 3))
    conv_transpose_module = nn.ConvTranspose(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array([[ 4.,  4.,  4.,  4.],
                              [ 7.,  7.,  7.,  7.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [ 7.,  7.,  7.,  7.],
                              [ 4.,  4.,  4.,  4.]])
    np.testing.assert_allclose(y, correct_ans)

  def test_int_kernel_size(self):
    conv = nn.Conv(features=4, kernel_size=3)
    x = jnp.ones((8, 3))
    with self.assertRaises(TypeError):
      conv.init(random.PRNGKey(0), x)

  def test_embed(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(
        jnp.arange(4)[..., None], (4, 3)).astype(jnp.float32)
    embed_module = nn.Embed(
        num_embeddings=4,
        features=3,
        embedding_init=lambda rng, shape, dtype: dummy_embedding,
    )
    y, initial_params = embed_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, dummy_embedding[None])
    z = embed_module.apply(initial_params, jnp.ones((3,)), method=embed_module.attend)
    np.testing.assert_allclose(z, 3. * jnp.arange(4))
  
  def test_embed_numpy(self):
    rng = dict(params=random.PRNGKey(0))
    x = jnp.arange(4)[None]
    dummy_embedding = np.broadcast_to(
        np.arange(4)[..., None], (4, 3)).astype(np.float32)
    embed_module = nn.Embed(
        num_embeddings=4,
        features=3,
        embedding_init=lambda rng, shape, dtype: dummy_embedding,
    )
    y, initial_params = embed_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, dummy_embedding[None])
    z = embed_module.apply(initial_params, jnp.ones((3,)), method=embed_module.attend)
    np.testing.assert_allclose(z, 3. * jnp.arange(4))

  def test_embed_hash(self):
    self.assertEqual(hash(nn.Embed(2, 3)), hash(nn.Embed(2, 3)))
    self.assertNotEqual(hash(nn.Embed(3, 4)), hash(nn.Embed(2, 3)))
  
  def test_non_final_axis(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.DenseGeneral(features=6, axis=1, name='dense')(x)

    x = jnp.ones((2, 4, 8))
    y, variables = Foo().init_with_output(random.PRNGKey(0), x)
    self.assertEqual(jax.tree_map(jnp.shape, variables['params']), {
      'dense': {'kernel': (4, 6), 'bias': (6,)}
    })
    self.assertEqual(y.shape, (2, 8, 6))
  
  def test_non_final_axes(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.DenseGeneral(features=6, axis=(0, 1), name='dense')(x)

    x = jnp.ones((2, 4, 8))
    y, variables = Foo().init_with_output(random.PRNGKey(0), x)
    self.assertEqual(jax.tree_map(jnp.shape, variables['params']), {
      'dense': {'kernel': (2, 4, 6), 'bias': (6,)}
    })
    self.assertEqual(y.shape, (8, 6))


if __name__ == '__main__':
  absltest.main()

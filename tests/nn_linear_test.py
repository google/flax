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

from flax.deprecated import nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LinearTest(parameterized.TestCase):

  def test_dense(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = nn.Dense.partial(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.init(rng, x)
    self.assertEqual(y.shape, (1, 4))
    np.testing.assert_allclose(y, np.full((1, 4), 4.))

  def test_dense_extra_batch_dims(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 3))
    dense_module = nn.Dense.partial(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.init(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 4), 4.))

  def test_dense_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = nn.Dense.partial(
        features=4,
        bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = dense_module.init(rng, x)
    np.testing.assert_allclose(y, np.full((1, 4), 3.))

  def test_dense_is_dense_general(self):
    x = jax.random.normal(random.PRNGKey(0), (5, 3))
    dense_module = nn.Dense.partial(
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y1, _ = dense_module.init(random.PRNGKey(1), x)
    dg_module = nn.DenseGeneral.partial(
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y2, _ = dg_module.init(random.PRNGKey(1), x)

    np.testing.assert_allclose(y1, y2)

  def test_dense_general_batch_dim_raises(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3, 2, 5))
    with self.assertRaises(ValueError):
      dg_module = nn.DenseGeneral.partial(
          features=4,
          batch_dims=(0, 2),
          kernel_init=initializers.ones,
          bias_init=initializers.ones,
      )
      dg_module.init(rng, x)

  def test_dense_general_two_out(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dg_module = nn.DenseGeneral.partial(
        features=(2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.init(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 2), 4.))

  def test_dense_general_two_in(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    dg_module = nn.DenseGeneral.partial(
        features=3,
        axis=(-2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.init(rng, x)
    np.testing.assert_allclose(y, np.full((1, 3), 5.))

  def test_dense_general_batch_dim(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((2, 1, 3, 5))

    state = {'counter': 0.}
    def _counter_init(rng, shape, dtype, state):
      del rng, dtype
      state['counter'] += 1.
      return jnp.full(shape, state['counter'])
    counter_init = functools.partial(_counter_init, state=state)

    dg_module = nn.DenseGeneral.partial(
        features=7,
        axis=(3, -2),
        batch_dims=0,
        bias_init=initializers.ones,
        kernel_init=counter_init,
    )
    y, _ = dg_module.init(rng, x)
    target = np.concatenate(
        [np.full((1, 1, 7), 16.), np.full((1, 1, 7), 31.)], axis=0)
    np.testing.assert_allclose(y, target)

  @parameterized.parameters([((-2, 3), (), 'bijk,jklm->bilm'),
                             ((3, -2), (), 'bijk,kjlm->bilm'),
                             ((-2, 3), (0,), 'bijk,bjklm->bilm')])
  def test_dense_general_vs_numpy(self, axis, batch_dims, einsum_expr):
    rng = random.PRNGKey(0)
    x = jnp.ones((16, 8, 9, 10))

    dg_module = nn.DenseGeneral.partial(
        features=(11, 12),
        axis=axis,
        batch_dims=batch_dims,
        bias_init=initializers.ones,
        kernel_init=initializers.normal(),
    )
    y, initial_params = dg_module.init(rng, x)
    dg_module = nn.Model(dg_module, initial_params)
    target = np.einsum(einsum_expr, x, dg_module.params['kernel']) + 1.
    np.testing.assert_allclose(y, target, atol=1e-6)

  @parameterized.parameters([((3,),), (3,)])
  def test_conv(self, kernel_size):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 3))
    conv_module = nn.Conv.partial(
        features=4,
        kernel_size=kernel_size,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    self.assertEqual(model.params['kernel'].shape, (3, 3, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 10.))

  @parameterized.parameters([((3,),), (3,)])
  def test_single_input_conv(self, kernel_size):
    rng = random.PRNGKey(0)
    x = jnp.ones((8, 3))
    conv_module = nn.Conv.partial(
        features=4,
        kernel_size=kernel_size,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    self.assertEqual(model.params['kernel'].shape, (3, 3, 4))
    np.testing.assert_allclose(y, np.full((6, 4), 10.))

  @parameterized.parameters([((3,),), (3,)])
  def test_group_conv(self, kernel_size):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 4))
    conv_module = nn.Conv.partial(
        features=4,
        kernel_size=kernel_size,
        feature_group_count=2,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    self.assertEqual(model.params['kernel'].shape, (3, 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.))

  @parameterized.parameters([((3,),), (3,)])
  def test_conv_transpose(self, kernel_size):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 3))
    conv_transpose_module = nn.ConvTranspose.partial(
        features=4,
        kernel_size=kernel_size,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init(rng, x)
    model = nn.Model(conv_transpose_module, initial_params)
    self.assertEqual(model.params['kernel'].shape, (3, 3, 4))
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

  @parameterized.parameters([((3,),), (3,)])
  def test_single_input_conv_transpose(self, kernel_size):
    rng = random.PRNGKey(0)
    x = jnp.ones((8, 3))
    conv_transpose_module = nn.ConvTranspose.partial(
        features=4,
        kernel_size=kernel_size,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init(rng, x)
    model = nn.Model(conv_transpose_module, initial_params)
    self.assertEqual(model.params['kernel'].shape, (3, 3, 4))
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

  def test_embed(self):
    rng = random.PRNGKey(0)
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(
        jnp.arange(4)[..., None], (4, 3)).astype(jnp.float32)
    embed_module = nn.Embed.partial(
        num_embeddings=4,
        features=3,
        embedding_init=lambda rng, shape: dummy_embedding,
    )
    y, initial_params = embed_module.init(rng, x)
    model = nn.Model(embed_module, initial_params)
    np.testing.assert_allclose(y, dummy_embedding[None])

    z = model.attend(jnp.ones((3,)))
    np.testing.assert_allclose(z, 3. * jnp.arange(4))


if __name__ == '__main__':
  absltest.main()

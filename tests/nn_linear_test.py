# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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

from flax import nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp


class LinearTest(parameterized.TestCase):

  def test_dense(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = nn.Dense.partial(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.create(rng, x)
    self.assertEqual(y.shape, (1, 4))
    onp.testing.assert_allclose(y, onp.full((1, 4), 4.))

  def test_dense_extra_batch_dims(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 3))
    dense_module = nn.Dense.partial(
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dense_module.create(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 2, 4), 4.))

  def test_dense_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = nn.Dense.partial(
        features=4,
        bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = dense_module.create(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 4), 3.))

  def test_dense_is_dense_general(self):
    x = jax.random.normal(random.PRNGKey(0), (5, 3))
    dense_module = nn.Dense.partial(
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y1, _ = dense_module.create(random.PRNGKey(1), x)
    dg_module = nn.DenseGeneral.partial(
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y2, _ = dg_module.create(random.PRNGKey(1), x)

    onp.testing.assert_allclose(y1, y2)

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
      dg_module.create(rng, x)

  def test_dense_general_two_out(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dg_module = nn.DenseGeneral.partial(
        features=(2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.create(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 2, 2), 4.))

  def test_dense_general_two_in(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    dg_module = nn.DenseGeneral.partial(
        features=3,
        axis=(-2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = dg_module.create(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 3), 5.))

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
    y, _ = dg_module.create(rng, x)
    target = onp.concatenate(
        [onp.full((1, 1, 7), 16.), onp.full((1, 1, 7), 31.)], axis=0)
    onp.testing.assert_allclose(y, target)

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
    y, dg_module = dg_module.create(rng, x)
    target = onp.einsum(einsum_expr, x, dg_module.params['kernel']) + 1.
    onp.testing.assert_allclose(y, target, atol=1e-6)

  def test_conv(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 3))
    conv_module = nn.Conv.partial(
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, model = conv_module.create(rng, x)
    self.assertEqual(model.params['kernel'].shape, (3, 3, 4))
    onp.testing.assert_allclose(y, onp.full((1, 6, 4), 10.))

  def test_group_conv(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 4))
    conv_module = nn.Conv.partial(
        features=4,
        kernel_size=(3,),
        feature_group_count=2,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, model = conv_module.create(rng, x)
    self.assertEqual(model.params['kernel'].shape, (3, 2, 4))
    onp.testing.assert_allclose(y, onp.full((1, 6, 4), 7.))

  def test_embed(self):
    rng = random.PRNGKey(0)
    x = jnp.arange(4)[None, ..., None]
    dummy_embedding = jnp.broadcast_to(jnp.arange(4)[..., None], (4, 3))
    embed_module = nn.Embed.partial(
        num_embeddings=4,
        features=3,
        embedding_init=lambda rng, shape: dummy_embedding,
    )
    y, _ = embed_module.create(rng, x)
    onp.testing.assert_allclose(y, dummy_embedding[None])


if __name__ == '__main__':
  absltest.main()

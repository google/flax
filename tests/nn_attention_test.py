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

"""Tests for flax.nn.attention."""

from absl.testing import absltest
from absl.testing import parameterized

from flax import nn

import jax
from jax import lax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp


class AttentionTest(parameterized.TestCase):

  def test_multihead_self_attention(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module = nn.SelfAttention.partial(
        num_heads=8,
        attention_axis=(1, 2),
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, _ = sa_module.create(rng, x)
    self.assertEqual(y.shape, x.shape)

  def test_causal_mask_1d(self):
    """Tests autoregresive masking for 1d attention."""
    key = jnp.ones((4, 5, 2, 16))  # (bs, dim1, dim2, heads, channel)
    att_axis = (1,)
    mask_1d = nn.attention._make_causal_mask(
        key, attention_axis=att_axis, self_mask=False)

    ts = onp.arange(key.shape[1])
    mask_1d_simple = (ts[:, None] >= ts[None, :])[None, None, :, :]
    onp.testing.assert_allclose(mask_1d, mask_1d_simple,)

  def test_causal_mask_2d(self):
    """Tests autoregresive masking for 2d attention."""
    key = jnp.ones((4, 5, 5, 2, 16))  # (bs, dim1, dim2, heads, channel)

    # masking when dealing with nd attention weights
    # w_nd_shape = (4, 5, 5, 5, 5, 2)
    att_axis = (1, 2)
    mask_nd = nn.attention._make_causal_mask(
        key, attention_axis=att_axis, self_mask=False)

    # masking when dealing with 1d attention weights
    # w_1d_shape = (4, 5*5, 5*5, 2)
    ts = onp.arange(25)
    mask_1d = (ts[:, None] >= ts[None, :])[None, None, :, :]

    onp.testing.assert_allclose(mask_nd.reshape(mask_1d.shape), mask_1d,
                                atol=1e-9)

  @parameterized.parameters([((5,), (1,)),
                             ((5, 6), (1,)),
                             ((5, 6), (2,)),
                             ((5, 6), (1, 2)),])
  def test_decoding(self, spatial_shape, attn_dims):
    bs = 2
    num_heads = 3
    num_features = 4
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    inputs = random.normal(
        key1, (bs,) + spatial_shape + (num_heads * num_features,))
    model_def = nn.SelfAttention.partial(
        num_heads=num_heads,
        qkv_features=num_heads * num_features,
        attention_axis=attn_dims,
        causal_mask=True,
        precision=lax.Precision.HIGHEST)

    with nn.attention.Cache().mutate() as cache_def:
      _, model = model_def.create_by_shape(
          key2, [(inputs.shape, inputs.dtype)], cache=cache_def)
    y_ref = jax.jit(lambda f, x: f(x))(model, inputs)

    # feed the inputs sequentially to simulate decoding
    cache0 = cache_def.initialize_cache((bs,) + spatial_shape)
    def body_fn(cache, x):
      with cache.mutate() as new_cache:
        y = model(x, cache=new_cache)
      return new_cache, y
    # attention.scan_in_dim supports scanning multiple dims
    _, y = nn.attention.scan_in_dim(body_fn, cache0, inputs,
                                    axis=attn_dims, keepdims=True)

    onp.testing.assert_allclose(y_ref, y, atol=1e-5)

  def test_autoregresive_receptive_field_1d(self):
    """Tests the autoregresive self-attention receptive field."""
    rng = random.PRNGKey(0)
    rng1, rng2 = random.split(rng, num=2)

    def model_loss(inputs, pos):
      out = model(inputs)
      assert out.shape == input_shape
      assert len(out.shape) == 3
      return out[0, pos, :].sum()

    grad_fn = jax.jit(jax.grad(model_loss))

    def get_receptive_field_1d(pos):
      g = grad_fn(inputs, pos)[0, :, :]
      return jnp.any((jnp.abs(g) > 1e-5).astype(int), axis=-1)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    inputs = random.normal(rng2, input_shape)

    model_def = nn.attention.SelfAttention.partial(
        num_heads=num_heads,
        causal_mask=True,
        kernel_init=jax.nn.initializers.ones)
    _, model = model_def.create_by_shape(
        rng1, [((1,) + (length, dim), jnp.float32)])

    for i in range(length):
      deps = get_receptive_field_1d(i)
      assert (deps[:i] == 1).all(), ('Receptive Field Error: Some of the '
                                     'previous postions are not reachable '
                                     'in autoregressive self-attention.')
      if i != length - 1:
        k = i + 1
        assert (deps[k:] == 0).all(), ('Receptive Field Error: Some of the '
                                       'future postions are reachable in '
                                       'autoregressive self-attention.')


if __name__ == '__main__':
  absltest.main()

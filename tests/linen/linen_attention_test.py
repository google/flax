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

"""Tests for flax.nn.attention."""

from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax import jax_utils

import jax
from jax import lax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class AttentionTest(parameterized.TestCase):

  def test_multihead_self_attention(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module = nn.SelfAttention(
        None,
        num_heads=8,
        attention_axis=(1, 2),
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, _ = sa_module.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)

  def test_multihead_encoder_decoder_attention(self):
    rng = random.PRNGKey(0)
    q = jnp.ones((4, 2, 3, 5))
    kv = jnp.ones((4, 2, 3, 5))
    sa_module = nn.MultiHeadDotProductAttention(
        None,
        num_heads=8,
        attention_axis=(1, 2),
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
    )
    y, _ = sa_module.init_with_output(rng, q, kv)
    self.assertEqual(y.shape, q.shape)

  def test_multihead_self_attention_w_dropout(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module = nn.MultiHeadDotProductAttention(
        None,
        num_heads=8,
        attention_axis=(1, 2),
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.1,
    )
    rng1, rng2 = random.split(rng)
    rngs = {'param': rng1, 'dropout': rng2}
    y, _ = sa_module.init_with_output(rngs, x, x)
    self.assertEqual(y.shape, x.shape)

  def test_causal_mask_1d(self):
    """Tests autoregresive masking for 1d attention."""
    key = jnp.ones((4, 5, 2, 16))  # (bs, dim1, dim2, heads, channel)
    att_axis = (1,)
    mask_1d = nn.attention._make_causal_mask(
        key, attention_axis=att_axis, self_mask=False)

    ts = np.arange(key.shape[1])
    mask_1d_simple = (ts[:, None] >= ts[None, :])[None, None, :, :]
    np.testing.assert_allclose(mask_1d, mask_1d_simple,)

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
    ts = np.arange(25)
    mask_1d = (ts[:, None] >= ts[None, :])[None, None, :, :]

    np.testing.assert_allclose(mask_nd.reshape(mask_1d.shape), mask_1d,
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
    module = nn.MultiHeadDotProductAttention(
        None,
        num_heads=num_heads,
        qkv_features=num_heads * num_features,
        attention_axis=attn_dims,
        causal_mask=True,
        precision=lax.Precision.HIGHEST,
        decode=False)
    decode_module = module.clone(decode=True)

    initial_vars = decode_module.init(key2, inputs, inputs)
    y_ref = jax.jit(lambda x: module.apply(initial_vars, x, x))(inputs)
    # feed the inputs sequentially to simulate decoding
    def body_fn(vars_in, x):
      y, vars_out = decode_module.apply(vars_in, x, x,
                                        decode=True, mutable=['cache'])
      return vars_out, y
    # scan_in_dim supports scanning multiple dims
    _, y = jax_utils.scan_in_dim(body_fn, initial_vars, inputs,
                                    axis=attn_dims, keepdims=True)

    np.testing.assert_allclose(y_ref, y, atol=1e-5)

  def test_autoregresive_receptive_field_1d(self):
    """Tests the autoregresive self-attention receptive field."""
    rng = random.PRNGKey(0)
    rng1, rng2 = random.split(rng, num=2)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    inputs = random.normal(rng2, input_shape)

    module = nn.MultiHeadDotProductAttention(
        None,
        num_heads=num_heads,
        causal_mask=True,
        kernel_init=jax.nn.initializers.ones)

    initial_vars = module.init(rng1, inputs, inputs, decode=False)

    def model_loss(inputs, pos):
      out = module.apply(initial_vars, inputs, inputs, decode=False)
      assert out.shape == input_shape
      assert len(out.shape) == 3
      return out[0, pos, :].sum()

    grad_fn = jax.jit(jax.grad(model_loss))

    def get_receptive_field_1d(pos):
      g = grad_fn(inputs, pos)[0, :, :]
      return jnp.any((jnp.abs(g) > 1e-5).astype(jnp.uint32), axis=-1)

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

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

"""Tests for flax.linen.attention."""

from absl.testing import absltest
from absl.testing import parameterized

from flax import errors
from flax import linen as nn
from flax import jax_utils
from flax.core import pop

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
    rng = random.key(0)
    x = jnp.ones((4, 6, 5))
    sa_module = nn.SelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        deterministic=False,
    )
    y, _ = sa_module.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, jnp.float32)

  def test_dtype_infer(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5), jnp.complex64)
    sa_module = nn.SelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        deterministic=False,
    )
    y, _ = sa_module.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, jnp.complex64)

  def test_multihead_encoder_decoder_attention(self):
    rng = random.key(0)
    q = jnp.ones((4, 2, 3, 5))
    sa_module = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        deterministic=False,
    )
    y, _ = sa_module.init_with_output(rng, q)
    self.assertEqual(y.shape, q.shape)

  def test_multihead_self_attention_w_dropout(self):
    rng = random.key(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.1,
        deterministic=False,
    )
    rng1, rng2 = random.split(rng)
    rngs = {'params': rng1, 'dropout': rng2}
    y, _ = sa_module.init_with_output(rngs, x)
    self.assertEqual(y.shape, x.shape)

  def test_multihead_self_attention_explicit_dropout(self):
    class Foo(nn.Module):
      attention_kwargs: dict
      @nn.compact
      def __call__(self, x, dropout_rng=None):
        a = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, x, dropout_rng=dropout_rng)
        b = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, x, dropout_rng=dropout_rng)
        return a, b

    module = Foo(
      dict(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.5,
        deterministic=False,
        )
      )
    rng1, rng2, rng3 = random.split(random.key(0), 3)
    x = jnp.ones((4, 2, 3, 5))
    rngs = {'params': rng1, 'dropout': rng2}
    v = module.init(rngs, x)
    a, b = module.apply(v, x, rngs=rngs)
    self.assertTrue(not (a == b).all())
    a, b = module.apply(v, x, rngs=rngs, dropout_rng=rng3)
    self.assertTrue((a == b).all())
    a, b = module.apply(v, x, dropout_rng=rng3)
    self.assertTrue((a == b).all())
    self.assertTrue(a.shape == b.shape == x.shape)

  def test_multihead_self_attention_w_dropout_disabled(self):
    rng = random.key(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module0 = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.0,
        deterministic=True,
    )
    rng1, rng2, rng3, rng4 = random.split(rng, 4)
    rngs1 = {'params': rng1, 'dropout': rng2}
    rngs2 = {'params': rng3, 'dropout': rng4}
    y1, vs = sa_module0.init_with_output(rngs1, x)
    y2, _ = sa_module0.init_with_output(rngs2, x)
    np.testing.assert_allclose(y1, y2)
    y3 = sa_module0.apply(vs, x, rngs=rngs1)
    y4 = sa_module0.apply(vs, x, rngs=rngs2)
    np.testing.assert_allclose(y3, y4)
    sa_module1 = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.0,
    )
    y5 = sa_module1.apply(vs, x, deterministic=True, rngs=rngs1)
    y6 = sa_module1.apply(vs, x, deterministic=True, rngs=rngs2)
    np.testing.assert_allclose(y5, y6)
    sa_module2 = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.5,
    )
    y7 = sa_module2.apply(vs, x, deterministic=True, rngs=rngs1)
    y8 = sa_module2.apply(vs, x, deterministic=True, rngs=rngs2)
    np.testing.assert_allclose(y7, y8)

  def test_causal_mask_1d(self):
    """Tests autoregresive masking for 1d attention."""
    x = jnp.ones((3, 16))  # (bs1, length)
    mask_1d = nn.attention.make_causal_mask(x)
    ts = np.arange(16)
    mask_1d_simple = (ts[:, None] >= ts[None, :])[None, None, :, :]
    mask_1d_simple = jnp.broadcast_to(mask_1d_simple, (3, 1, 16, 16))
    np.testing.assert_allclose(
        mask_1d,
        mask_1d_simple,
    )

  @parameterized.parameters([((5,), (1,)), ((6, 5), (2,))])
  def test_decoding(self, spatial_shape, attn_dims):
    bs = 2
    num_heads = 3
    num_features = 4
    rng = random.key(0)
    key1, key2 = random.split(rng)
    inputs = random.normal(
        key1, (bs,) + spatial_shape + (num_heads * num_features,)
    )
    module = nn.SelfAttention(
        num_heads=num_heads,
        qkv_features=num_heads * num_features,
        precision=lax.Precision.HIGHEST,
        deterministic=False,
        decode=False,
    )
    decode_module = module.clone(decode=True)

    initial_vars = decode_module.init(key2, inputs)
    state, params = pop(initial_vars, 'params')
    causal_mask = nn.attention.make_causal_mask(jnp.ones((bs,) + spatial_shape))
    y_ref = jax.jit(lambda x, y: module.apply(initial_vars, x, y))(
        inputs, causal_mask
    )

    # feed the inputs sequentially to simulate decoding
    def body_fn(state, x):
      y, state = decode_module.apply(
          {'params': params, **state}, x, mutable=['cache']
      )
      return state, y

    # scan_in_dim supports scanning multiple dims
    _, y = jax_utils.scan_in_dim(
        body_fn, state, inputs, axis=attn_dims, keepdims=True
    )

    np.testing.assert_allclose(y_ref, y, atol=1e-5)

  def test_autoregresive_receptive_field_1d(self):
    """Tests the autoregresive self-attention receptive field."""
    rng = random.key(0)
    rng1, rng2 = random.split(rng, num=2)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    inputs = random.normal(rng2, input_shape)

    module = nn.MultiHeadDotProductAttention(
        num_heads=num_heads,
        kernel_init=jax.nn.initializers.ones,
        deterministic=False,
    )

    initial_vars = module.init(rng1, inputs)
    causal_mask = nn.attention.make_causal_mask(jnp.ones(input_shape[:-1]))

    def model_loss(inputs, pos):
      out = module.apply(initial_vars, inputs, mask=causal_mask)
      assert out.shape == input_shape
      assert len(out.shape) == 3
      return out[0, pos, :].sum()

    grad_fn = jax.jit(jax.grad(model_loss))

    def get_receptive_field_1d(pos):
      g = grad_fn(inputs, pos)[0, :, :]
      return jnp.any((jnp.abs(g) > 1e-5).astype(jnp.uint32), axis=-1)

    for i in range(length):
      deps = get_receptive_field_1d(i)
      assert (deps[:i] == 1).all(), (
          'Receptive Field Error: Some of the '
          'previous postions are not reachable '
          'in autoregressive self-attention.'
      )
      if i != length - 1:
        k = i + 1
        assert (deps[k:] == 0).all(), (
            'Receptive Field Error: Some of the '
            'future postions are reachable in '
            'autoregressive self-attention.'
        )

  def test_multihead_self_attention_equality(self):
    rng = random.key(0)
    q = jnp.ones((4, 2, 3, 5))
    module_kwargs = {'num_heads': 8,
                     'qkv_features': 16,
                     'kernel_init': initializers.ones,
                     'bias_init': initializers.zeros,
                     'deterministic': False}
    sa_module0 = nn.MultiHeadDotProductAttention(**module_kwargs)
    sa_module1 = nn.SelfAttention(**module_kwargs)
    y0, v0 = sa_module0.init_with_output(rng, q)
    with self.assertWarnsRegex(DeprecationWarning, 'SelfAttention will be deprecated soon.'):
      y1, v1 = sa_module1.init_with_output(rng, q)
    self.assertTrue((y0 == y1).all())
    self.assertTrue(jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), v0, v1)))

  def test_multihead_kv_args(self):
    key1, key2 = random.split(random.key(0), 2)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key1, (9, 5))
    module = nn.MultiHeadDotProductAttention(
      num_heads=8,
      qkv_features=16,
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
      deterministic=False,
    )
    y0, v0 = module.init_with_output(key2, query, inputs_k=key_value, inputs_v=key_value)
    y1, v1 = module.init_with_output(key2, query, inputs_k=key_value)
    with self.assertWarnsRegex(DeprecationWarning, 'The inputs_kv arg will be deprecated soon.'):
      y2, v2 = module.init_with_output(key2, query, inputs_kv=key_value)
    self.assertTrue((y0 == y1).all() and (y1 == y2).all())
    self.assertTrue(
      jax.tree_util.tree_all(
        jax.tree_map(lambda x, y, z: (x == y).all() and (y == z).all(),
                     v0, v1, v2)))

    with self.assertRaisesRegex(ValueError, '`inputs_k` cannot be None if `inputs_v` is not None.'):
      y3, v3 = module.init_with_output(key2, query, inputs_v=key_value)
    with self.assertRaisesRegex(ValueError, 'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must be None.'):
      y3, v3 = module.init_with_output(key2, query, inputs_kv=key_value, inputs_v=key_value)
    with self.assertRaisesRegex(ValueError, 'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must be None.'):
      y3, v3 = module.init_with_output(key2, query, key_value, key_value, inputs_kv=key_value)

  def test_multihead_mask_warning(self):
    rng = random.key(0)
    rng1, rng2 = random.split(rng, num=2)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    query = key = random.normal(rng2, input_shape)

    module = nn.MultiHeadDotProductAttention(
        num_heads=num_heads,
        kernel_init=jax.nn.initializers.ones,
        deterministic=False,
    )

    initial_vars = module.init(rng1, query, key)
    causal_mask = nn.attention.make_causal_mask(jnp.ones(input_shape[:-1]))

    module.apply(initial_vars, query, key, mask=causal_mask)
    with self.assertWarnsRegex(DeprecationWarning,
                               "the function signature of MultiHeadDotProductAttention's `__call__` method has changed"):
      with self.assertRaises(errors.ScopeParamShapeError):
        module.apply(initial_vars, query, key, causal_mask)


if __name__ == '__main__':
  absltest.main()

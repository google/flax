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

"""Tests for flax.linen.attention."""

from absl.testing import absltest, parameterized
from flax import errors, jax_utils
from flax import linen as nn
from flax.core import pop
import jax
from jax import lax, random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class AttentionTest(parameterized.TestCase):
  def test_multihead_self_attention(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5))
    sa_module = nn.MultiHeadDotProductAttention(
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
    sa_module = nn.MultiHeadDotProductAttention(
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

  def test_mha_out_initializers(self):
    rng = random.key(0)
    q = jnp.ones((4, 2, 3, 5))
    sa_module = nn.MultiHeadDotProductAttention(
      num_heads=8,
      qkv_features=16,
      kernel_init=initializers.ones,
      out_kernel_init=initializers.zeros,
      bias_init=initializers.zeros,
      out_bias_init=initializers.ones,
      deterministic=False,
    )
    variables = sa_module.init(rng, q)
    params = variables['params']
    # test kernels
    np.testing.assert_allclose(params['query']['kernel'], 1.0)
    np.testing.assert_allclose(params['key']['kernel'], 1.0)
    np.testing.assert_allclose(params['value']['kernel'], 1.0)
    np.testing.assert_allclose(params['out']['kernel'], 0.0)
    # test biases
    np.testing.assert_allclose(params['query']['bias'], 0.0)
    np.testing.assert_allclose(params['key']['bias'], 0.0)
    np.testing.assert_allclose(params['value']['bias'], 0.0)
    np.testing.assert_allclose(params['out']['bias'], 1.0)

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
    def clone(key):
      return jax.tree.map(jax.random.clone, key)

    class Foo(nn.Module):
      attention_kwargs: dict

      @nn.compact
      def __call__(self, x, dropout_rng=None):
        a = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(
          x, x, dropout_rng=dropout_rng
        )
        if dropout_rng is not None:
          dropout_rng = clone(dropout_rng)
        b = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(
          x, x, dropout_rng=dropout_rng
        )
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
    rng1, rng2, rng3, rng4 = random.split(random.key(0), 4)
    x = jnp.ones((4, 2, 3, 5))
    rngs = {'params': rng1, 'dropout': rng2}
    v = module.init(rngs, x)
    a, b = module.apply(v, x, rngs=clone(rngs))
    c, d = module.apply(v, x, rngs={'dropout': clone(rng2)})
    e, f = module.apply(v, x, rngs={'dropout': rng3})
    self.assertFalse((a == b).all())
    self.assertTrue((a == c).all())
    self.assertTrue((b == d).all())
    self.assertFalse((a == e).all())
    self.assertFalse((b == f).all())
    a, b = module.apply(v, x, rngs=clone(rngs), dropout_rng=rng4)
    self.assertTrue((a == b).all())
    a, b = module.apply(v, x, dropout_rng=clone(rng4))
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
    module = nn.MultiHeadDotProductAttention(
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
    y_ref = jax.jit(lambda x, y: module.apply(initial_vars, x, mask=y))(
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

  def test_multihead_kv_args(self):
    key1, key2 = random.split(random.key(0), 2)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key2, (9, 5))
    module = nn.MultiHeadDotProductAttention(
      num_heads=8,
      qkv_features=16,
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
      deterministic=False,
    )
    key = lambda: random.key(43279)
    y0, v0 = module.init_with_output(
      key(), query, inputs_k=key_value, inputs_v=key_value
    )
    y1, v1 = module.init_with_output(key(), query, inputs_k=key_value)
    with self.assertWarnsRegex(
      DeprecationWarning, 'The inputs_kv arg will be deprecated soon.'
    ):
      y2, v2 = module.init_with_output(key(), query, inputs_kv=key_value)
    self.assertTrue((y0 == y1).all() and (y1 == y2).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x, y, z: (x == y).all() and (y == z).all(), v0, v1, v2
            )
        )
    )

    with self.assertRaisesRegex(
      ValueError, '`inputs_k` cannot be None if `inputs_v` is not None.'
    ):
      y3, v3 = module.init_with_output(key(), query, inputs_v=key_value)
    with self.assertRaisesRegex(
      ValueError,
      'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must be None.',
    ):
      y3, v3 = module.init_with_output(
        key(), query, inputs_kv=key_value, inputs_v=key_value
      )
    with self.assertRaisesRegex(
      ValueError,
      'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must be None.',
    ):
      y3, v3 = module.init_with_output(
        key(), query, key_value, key_value, inputs_kv=key_value
      )

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
    with self.assertWarnsRegex(
      DeprecationWarning,
      "the function signature of MultiHeadDotProductAttention's `__call__` method has changed",
    ):
      with self.assertRaises(errors.ScopeParamShapeError):
        module.apply(initial_vars, query, key, causal_mask)

  def test_multihead_sow_attention_weights(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5))

    class Model(nn.Module):
      attention_kwargs: dict

      @nn.compact
      def __call__(self, x, sow_weights=False):
        x = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(
          x, sow_weights=sow_weights
        )
        x = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x)
        x = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(
          x, sow_weights=sow_weights
        )
        return x

    module = Model(
      dict(
        num_heads=8,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        deterministic=False,
      )
    )
    v = module.init(rng, x)
    _, intermediates = module.apply(
      v, x, mutable=['intermediates'], sow_weights=True
    )
    self.assertEqual(
      intermediates['intermediates']['MultiHeadDotProductAttention_0'][
        'attention_weights'
      ][0].shape,
      (4, 8, 6, 6),
    )
    self.assertNotIn(
      'MultiHeadDotProductAttention_1', intermediates['intermediates']
    )
    self.assertEqual(
      intermediates['intermediates']['MultiHeadDotProductAttention_2'][
        'attention_weights'
      ][0].shape,
      (4, 8, 6, 6),
    )
    _, intermediates = module.apply(
      v, x, mutable=['intermediates'], sow_weights=False
    )
    self.assertNotIn('intermediates', intermediates)

  def test_autoregressive_decode_with_x64(self):
    with jax.experimental.enable_x64():
      x = jnp.ones((1, 4, 4))
      module = nn.MultiHeadDotProductAttention(
          num_heads=2,
          qkv_features=4,
          decode=True
        )

      rng = random.PRNGKey(0)
      variables = module.init(rng, x, x, x)
      params, cache = variables['params'], variables['cache']
      y1, updates = module.apply(
          { 'params': params, 'cache': cache },
          x[:, :1, :],
          mutable=['cache']
        )
      cache = updates['cache']
      y2, updates = module.apply(
          { 'params': params, 'cache': cache },
          x[:, 1:2, :],
          mutable=['cache']
        )
      assert y1.shape == (1, 1, 4)
      assert y2.shape == (1, 1, 4)

  def test_attention_alias_equivalence(self):
    key1, key2 = random.split(random.key(0), 2)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key2, (9, 5))
    attention_kwargs = dict(
      num_heads=8,
      qkv_features=16,
      kernel_init=initializers.lecun_normal(),
      bias_init=initializers.uniform(),
      deterministic=False,
    )
    module1 = nn.MultiHeadDotProductAttention(**attention_kwargs)
    module2 = nn.MultiHeadAttention(**attention_kwargs)
    key = lambda: random.key(43279)
    out1, v1 = module1.init_with_output(key(), query, key_value)
    out2, v2 = module2.init_with_output(key(), query, key_value, key_value)
    self.assertTrue((out1 == out2).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: (x == y).all(), v1, v2)
        )
    )

  def test_attention_alias_submodule(self):
    key1, key2 = random.split(random.key(0), 2)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key2, (9, 5))
    attention_kwargs = dict(
      num_heads=8,
      qkv_features=16,
      kernel_init=initializers.lecun_normal(),
      bias_init=initializers.uniform(),
      deterministic=False,
    )

    class Foo1(nn.Module):
      attention_kwargs: dict

      @nn.compact
      def __call__(self, query, key):
        return nn.MultiHeadDotProductAttention(**self.attention_kwargs)(
          query, key
        )

    class Foo2(nn.Module):
      attention_kwargs: dict

      @nn.compact
      def __call__(self, query, key, value):
        return nn.MultiHeadAttention(**self.attention_kwargs)(query, key, value)

    key = lambda: random.key(5478392)
    module1 = Foo1(attention_kwargs)
    module2 = Foo2(attention_kwargs)
    out1, v1 = module1.init_with_output(key(), query, key_value)
    out2, v2 = module2.init_with_output(key(), query, key_value, key_value)

    # test different output and variables if layer names are different
    self.assertTrue((out1 != out2).all())
    v2['params']['MultiHeadDotProductAttention_0'] = v2['params'][
      'MultiHeadAttention_0'
    ]
    del v2['params']['MultiHeadAttention_0']
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: (x != y).all(), v1, v2)
        )
    )

    # test same output if variables are the same
    v2 = jax.tree_util.tree_map(lambda x: x, v1)
    v2['params']['MultiHeadAttention_0'] = v2['params'][
      'MultiHeadDotProductAttention_0'
    ]
    del v2['params']['MultiHeadDotProductAttention_0']
    out2 = module2.apply(v2, query, key_value, key_value)
    self.assertTrue((out1 == out2).all())

    # test same output and variables if names are the same
    class Foo2(nn.Module):
      attention_kwargs: dict

      @nn.compact
      def __call__(self, query, key, value):
        return nn.MultiHeadAttention(
          **self.attention_kwargs, name='MultiHeadDotProductAttention_0'
        )(query, key, value)

    module2 = Foo2(attention_kwargs)
    out2, v2 = module2.init_with_output(key(), query, key_value, key_value)
    self.assertTrue((out1 == out2).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: (x == y).all(), v1, v2)
        )
    )

  @parameterized.parameters(
      {'force_fp32': True, 'attn_weights_dtype': jnp.float32},
      {'force_fp32': False, 'attn_weights_dtype': jnp.bfloat16},
  )
  def test_mixed_precision_multihead_attention(
      self, force_fp32, attn_weights_dtype
  ):
    input_key, params_key, dropout_key = random.split(random.key(0), 3)
    x = random.uniform(input_key, (2, 4))
    attention_kwargs = dict(
        num_heads=2,
        qkv_features=4,
        kernel_init=initializers.lecun_normal(),
        bias_init=initializers.uniform(),
        force_fp32_for_softmax=force_fp32,
        deterministic=False,
        dtype=jnp.bfloat16,
    )
    mha = nn.MultiHeadDotProductAttention(**attention_kwargs)
    init_vars = mha.init({'params': params_key, 'dropout': dropout_key}, x)
    _, updated_vars = mha.apply(
        init_vars, x, mutable=['intermediates'], sow_weights=True
    )
    self.assertEqual(
        updated_vars['intermediates']['attention_weights'][0].dtype,
        attn_weights_dtype,
    )

  @parameterized.parameters(
      (lax.Precision.DEFAULT, None),
      (None, jax.lax.dot_general),
  )
  def test_dot_product_attention_precision_and_einsum_override(
      self, precision, einsum_dot_general
  ):
    # Test that we raise a ValueError if the user specifies both
    # `precision` and/or `einsum_dot_general` and `qk_attn_weights_einsum`.
    einsum_cls = lambda: jnp.einsum
    self.assertRaises(
        ValueError,
        nn.dot_product_attention,
        query=jnp.ones((1, 4, 2)),
        key=jnp.ones((1, 4, 2)),
        value=jnp.ones((1, 4, 2)),
        precision=precision,
        einsum_dot_general=einsum_dot_general,
        qk_attn_weights_einsum=einsum_cls,
        attn_weights_value_einsum=einsum_cls,
    )

  @parameterized.parameters(
      (lambda: jax.lax.dot_general, None),
      (None, lambda: jax.lax.dot_general),
  )
  def test_dot_product_attention_specify_einsums_together(
      self, qk_attn_weights_einsum, attn_weights_value_einsum
  ):
    # Test that we raise a ValueError if the user specifies only one of
    # `qk_attn_weights_einsum` and `attn_weights_value_einsum`.
    self.assertRaises(
        ValueError,
        nn.dot_product_attention,
        query=jnp.ones((1, 4, 2)),
        key=jnp.ones((1, 4, 2)),
        value=jnp.ones((1, 4, 2)),
        qk_attn_weights_einsum=qk_attn_weights_einsum,
        attn_weights_value_einsum=attn_weights_value_einsum,
    )


if __name__ == '__main__':
  absltest.main()

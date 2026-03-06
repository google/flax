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

import jax, jax.numpy as jnp
from jax.lax import Precision

from flax import linen
from flax import nnx
from flax.nnx.nn.attention import combine_masks
from flax.typing import Dtype, PrecisionLike

import numpy as np

import typing as tp
from absl.testing import parameterized
from absl.testing import absltest

try:
  # JAX v0.8.0 and newer
  from jax import enable_x64
except ImportError:
  from jax.experimental import enable_x64


class TestMultiHeadAttention(parameterized.TestCase):
  def test_basic(self):
    module = nnx.MultiHeadAttention(
      num_heads=2,
      in_features=3,
      qkv_features=6,
      out_features=6,
      rngs=nnx.Rngs(0),
    )
    y = module(jnp.ones((1, 7, 3)), decode=False)
    assert y.shape == (1, 7, 6)


  def test_multihead_sow_attention_weights(self):
    class Model(nnx.Module):
      attention_kwargs: dict

      def __init__(self, attention_kwargs, rng):
        self.attention_layers = nnx.data([
          nnx.MultiHeadAttention(**attention_kwargs, rngs=rng) for i in range(3)
        ])

      def __call__(self, x, sow_weights=False):
        x = self.attention_layers[0](x, sow_weights=sow_weights)
        x = self.attention_layers[1](x)
        x = self.attention_layers[2](x, sow_weights=sow_weights)
        return x

    rng = nnx.Rngs(0)
    x = jnp.ones((4, 6, 8))

    module = Model(
      dict(
        in_features=8,
        num_heads=8,
        kernel_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        deterministic=False,
      ),
      rng,
    )
    module.set_attributes(decode=False)

    _ = module(x, True)
    intermediates = nnx.pop(module, nnx.Intermediate)
    assert intermediates['attention_layers'][0]['attention_weights'][
      0
    ].shape == (4, 8, 6, 6)
    assert 1 not in intermediates['attention_layers']
    assert intermediates['attention_layers'][2]['attention_weights'][
      0
    ].shape == (4, 8, 6, 6)

    _ = module(x)
    intermediates = nnx.pop(module, nnx.Intermediate)
    assert not intermediates  # empty

  def test_autoregressive_decode_with_x64(self):
    with enable_x64():
      x = jnp.ones((1, 4, 4))
      module = nnx.MultiHeadAttention(
        in_features=4,
        num_heads=2,
        qkv_features=4,
        decode=True,
        rngs=nnx.Rngs(0),
      )
      module.init_cache(x.shape, dtype=x.dtype)
      assert module.cached_key.shape == (1, 4, 2, 2)
      assert module.cached_value.shape == (1, 4, 2, 2)

      y1 = module(x[:, :1, :])
      y2 = module(x[:, 1:2, :])

      assert y1.shape == (1, 1, 4)
      assert y2.shape == (1, 1, 4)

  @parameterized.product(keep_rngs=[True, False])
  def test_keep_rngs(self, keep_rngs):
    rngs = nnx.Rngs(42)
    module = nnx.MultiHeadAttention(
      in_features=4,
      num_heads=2,
      qkv_features=4,
      decode=True,
      rngs=rngs,
      dropout_rate=0.5,
      keep_rngs=keep_rngs
    )
    if keep_rngs:
      assert module.rngs is not None
    else:
      assert module.rngs is None
    if keep_rngs:
      _, _, nondiff = nnx.split(module, nnx.Param, ...)
      assert isinstance(nondiff['rngs']['count'], nnx.RngCount)
      assert isinstance(nondiff['rngs']['key'], nnx.RngKey)
    else:
      nnx.split(module, nnx.Param)

  @parameterized.product(use_padding=[True, False], is_cross_attention=[True, False])
  def test_causal_mask_equivalence(
    self,
    use_padding: bool,
    is_cross_attention: bool
  ):
    batch_size = 1
    num_heads = 2
    q_len = 2
    kv_len = 4 if is_cross_attention else q_len
    head_dim = 4

    q = jax.random.normal(
      key=jax.random.key(0),
      shape=(batch_size, 1, q_len, num_heads, head_dim)
    )
    k = jax.random.normal(
      key=jax.random.key(1),
      shape=(batch_size, 1, kv_len, num_heads, head_dim)
    )
    v = jax.random.normal(
      key=jax.random.key(2),
      shape=(batch_size, 1, kv_len, num_heads, head_dim)
    )

    causal_mask = jnp.tril(jnp.ones(
        shape=(q_len, kv_len),
        dtype=jnp.bool_
      )
    )
    causal_mask = jnp.broadcast_to(
      array=causal_mask,
      shape=(batch_size, 1, num_heads, q_len, kv_len)
    )

    padding_mask = None

    if use_padding:
      padding_mask = jnp.ones(
        shape=(batch_size, 1, 1, q_len, kv_len),
        dtype=jnp.bool_,
      )
      padding_mask = padding_mask.at[..., -2:].set(False)

    manual_mask = combine_masks(padding_mask, causal_mask, dtype=q.dtype)

    # Jax.nn path with precombined mask and is_causal = False
    attn_jax = nnx.dot_product_attention(
      query=q,
      key=k,
      value=v,
      mask=manual_mask,
      is_causal=False,
      deterministic=True,
      module=None,
    )

    class DummyModule(nnx.Module):
      pass

    # nnx path with padding mask and is_causal = True (internally combines them)
    attn_manual = nnx.dot_product_attention(
      query=q,
      key=k,
      value=v,
      mask=padding_mask,
      is_causal=True,
      deterministic=True,
      module=DummyModule(),
    )

    np.testing.assert_allclose(attn_jax, attn_manual, atol=1e-6)


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    decode=[True, False],
    normalize_qk=[True, False],
    qkv_features=[None, 8],
    out_features=[None, 6],
  )
  def test_nnx_attention_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    decode: bool,
    normalize_qk: bool,
    qkv_features: tp.Optional[int],
    out_features: tp.Optional[int],
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 4

    x = jnp.ones((1, in_features))
    model_nnx = nnx.MultiHeadAttention(
      num_heads=num_heads,
      in_features=in_features,
      qkv_features=qkv_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      decode=decode,
      normalize_qk=normalize_qk,
      rngs=rngs,
    )
    model = linen.MultiHeadDotProductAttention(
      num_heads=num_heads,
      qkv_features=qkv_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      decode=decode,
      normalize_qk=normalize_qk,
    )
    variables = model.init(key, x)

    for qkvo in ('query', 'key', 'value', 'out'):
      getattr(model_nnx, qkvo).kernel[...] = variables['params'][qkvo]['kernel']
      if use_bias:
        getattr(model_nnx, qkvo).bias[...] = variables['params'][qkvo]['bias']
    if normalize_qk:
      model_nnx.query_ln.scale[...] = variables['params']['query_ln']['scale']
      model_nnx.key_ln.scale[...] = variables['params']['key_ln']['scale']

    # Guard: verify params were copied correctly
    for name in ('query', 'key', 'value', 'out'):
      np.testing.assert_array_equal(
        variables['params'][name]['kernel'],
        getattr(model_nnx, name).kernel[...],
      )
      if use_bias:
        np.testing.assert_array_equal(
          variables['params'][name]['bias'],
          getattr(model_nnx, name).bias[...],
        )
    if normalize_qk:
      np.testing.assert_array_equal(
        variables['params']['query_ln']['scale'],
        model_nnx.query_ln.scale[...],
      )
      np.testing.assert_array_equal(
        variables['params']['key_ln']['scale'],
        model_nnx.key_ln.scale[...],
      )
    if decode:
      model_nnx.init_cache(x.shape, dtype=dtype)

    out_nnx = model_nnx(x)
    out, _ = model.apply(variables, x, mutable=['cache'])
    rtol = 1e-3 if dtype == jnp.float16 or param_dtype == jnp.float16 else 1e-6
    np.testing.assert_allclose(out, out_nnx, rtol=rtol)


class TestKVFeatures(parameterized.TestCase):

  def test_varying_num_features(self):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 3
    in_kv_features = 4
    qkv_features = 6
    out_features = 6

    x = jnp.ones((1, in_features))
    y = jax.random.normal(key, (1, in_kv_features))
    layer = nnx.MultiHeadAttention(
      num_heads=num_heads,
      in_features=in_features,
      qkv_features=qkv_features,
      out_features=out_features,
      in_kv_features=in_kv_features,
      rngs=rngs,
      decode=False
    )

    self.assertIsNotNone(layer(x, y))

class TestGQADotProductAttention(parameterized.TestCase):

  def test_gqa_shapes(self):
    B, T, S = 2, 4, 5
    D = 8
    num_heads_q = 6
    num_heads_kv = 3

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(k1, (B, T, num_heads_q, D))
    key   = jax.random.normal(k2, (B, S, num_heads_kv, D))
    value = jax.random.normal(k3, (B, S, num_heads_kv, D))

    output = nnx.dot_product_attention(query, key, value)
    expected_shape = (B, T, num_heads_q, D)
    self.assertEqual(output.shape, expected_shape)

  def test_gqa_invalid_heads(self):
    B, T, D = 1, 4, 8
    query = jnp.ones((B, T, 5, D))
    key   = jnp.ones((B, T, 2, D))
    value = key

    with self.assertRaisesRegex(ValueError, "must be a multiple"):
        nnx.dot_product_attention(query, key, value)

  def test_gqa_parity_with_jax(self):
    class DummyModule(nnx.Module):
      pass

    dummy_module = DummyModule()

    B, T, S, D = 2, 8, 8, 16
    num_heads_q = 4
    num_heads_kv = 2

    rng = jax.random.key(42)
    k1, k2, k3 = jax.random.split(rng, 3)

    query = jax.random.normal(k1, (B, T, num_heads_q, D))
    key   = jax.random.normal(k2, (B, S, num_heads_kv, D))
    value = jax.random.normal(k3, (B, S, num_heads_kv, D))

    jax_out = jax.nn.dot_product_attention(query, key, value)

    # NNX should handle broadcasting internally
    nnx_out = nnx.dot_product_attention(
      query, key, value,
      module=dummy_module
    )

    np.testing.assert_allclose(nnx_out, jax_out, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()

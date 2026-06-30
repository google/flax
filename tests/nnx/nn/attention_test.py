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

import functools

import jax
import jax, jax.numpy as jnp
from jax.lax import Precision

import torch
import torch.onnx.ops

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

    _, intermediates = nnx.capture(module, nnx.Intermediate)(x, True)
    assert intermediates['attention_layers'][0]['attention_weights'][
      0
    ].shape == (4, 8, 6, 6)
    assert 1 not in intermediates['attention_layers']
    assert intermediates['attention_layers'][2]['attention_weights'][
      0
    ].shape == (4, 8, 6, 6)

    _, intermediates = nnx.capture(module, nnx.Intermediate)(x)
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
    dummy = DummyModule()
    def _run(m):
      return nnx.dot_product_attention(
        query=q,
        key=k,
        value=v,
        mask=padding_mask,
        is_causal=True,
        deterministic=True,
        module=m,
      )
    attn_manual, _ = nnx.capture(_run, nnx.Intermediate)(dummy)

    np.testing.assert_allclose(attn_jax, attn_manual, atol=1e-6)

# TODO: add all possible constructor argument values to parameterized.product
class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    decode=[True, False],
    normalize_qk=[True, False],
  )
  def test_nnx_attention_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    decode: bool,
    normalize_qk: bool,
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 3
    qkv_features = 6
    out_features = 6

    x = jax.numpy.ones((1, in_features))
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
    if decode:
      model_nnx.init_cache(x.shape, dtype=dtype)

    out_nnx = model_nnx(x)
    out, cache = model.apply(variables, x, mutable=['cache'])
    np.testing.assert_array_equal(out, out_nnx)


class TestKVFeatures(parameterized.TestCase):

  def test_varying_num_features(self):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 3
    in_kv_features = 4
    qkv_features = 6
    out_features = 6

    x = jax.numpy.ones((1, in_features))
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

  def test_gqa_multihead_attention(self):
    in_feat = 128
    n_heads = 32
    n_kv_heads = 8
    qkv_feat = 2048
    head_dim = qkv_feat // n_heads

    model = nnx.MultiHeadAttention(
      num_heads=n_heads,
      in_features=in_feat,
      qkv_features=qkv_feat,
      num_kv_heads=n_kv_heads,
      rngs=nnx.Rngs(0),
    )

    assert model.query.kernel.shape == (in_feat, n_heads, head_dim)
    assert model.key.kernel.shape == (in_feat, n_kv_heads, head_dim)
    assert model.value.kernel.shape == (in_feat, n_kv_heads, head_dim)

    x = jnp.ones((1, 10, in_feat))
    y = model(x, decode=False)
    assert y.shape == (1, 10, in_feat)

    model.init_cache((1, 10, in_feat))
    assert model.cached_key.shape == (1, 10, n_kv_heads, head_dim)

    x_token = jnp.ones((1, 1, in_feat))
    y_token = model(x_token, decode=True)
    assert y_token.shape == (1, 1, in_feat)
    assert model.cache_index == 1

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
    def _run(m):
      return nnx.dot_product_attention(query, key, value, module=m)
    nnx_out, _ = nnx.capture(_run, nnx.Intermediate)(dummy_module)

    np.testing.assert_allclose(nnx_out, jax_out, atol=1e-3, rtol=1e-3)


class TestDotProductAttentionValidation(parameterized.TestCase):

  @parameterized.parameters(
    ((2, 16, 4, 8), (16, 4, 8), (16, 4, 8), 'same rank'),
    ((2, 16, 4, 8), (3, 16, 4, 8), (3, 16, 4, 8), 'batch dims'),
    ((2, 16, 4, 8), (2, 16, 4, 8), (2, 15, 4, 8), 'lengths must match'),
    ((2, 16, 4, 8), (2, 16, 4, 7), (2, 16, 4, 8), 'depths must match'),
  )
  def test_invalid_shapes(self, q_shape, k_shape, v_shape, match):
    q = jnp.ones(q_shape)
    k = jnp.ones(k_shape)
    v = jnp.ones(v_shape)
    with self.assertRaisesRegex(ValueError, match):
      nnx.dot_product_attention(q, k, v, dropout_rate=0.1,
                                dropout_rng=jax.random.key(0))
class TestRoPE(absltest.TestCase):

  def _torch_rope(self, x_np, theta, seq_len, head_dim):
    """Apply RoPE using torch.onnx.ops.rotary_embedding as reference."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    # torch expects (batch, heads, seq, dim)
    x_torch = torch.from_numpy(np.array(x_np))
    batch_size = x_torch.shape[0]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    out = torch.onnx.ops.rotary_embedding(
      x_torch, cos_cache, sin_cache, position_ids, interleaved=False
    )
    return out.numpy()

  def test_matches_torch_single_head(self):
    """RoPE.__call__ matches torch.onnx.ops.rotary_embedding on a single (seq, dim) input."""

    seq_len, head_dim = 32, 64
    x = jax.random.normal(jax.random.key(0), (seq_len, head_dim))

    flax_rope = nnx.RoPE(theta=10000.0)
    out_flax = flax_rope(x)

    # torch 4D: (batch=1, heads=1, seq, dim)
    out_torch = self._torch_rope(
      x[None, None], theta=10000.0, seq_len=seq_len, head_dim=head_dim
    )[0, 0]

    np.testing.assert_allclose(out_flax, out_torch, atol=1e-5)

  def test_matches_torch_multi_head(self):
    """RoPE applied per-head via vmap matches torch.onnx.ops.rotary_embedding with heads."""
    seq_len, num_heads, head_dim = 16, 4, 32
    x = jax.random.normal(jax.random.key(1), (seq_len, num_heads, head_dim))

    flax_rope = nnx.RoPE(theta=10000.0)
    out_flax = jax.vmap(flax_rope, in_axes=1, out_axes=1)(x)

    # torch 4D: (batch=1, heads, seq, dim) — need to transpose from (seq, heads, dim)
    x_torch_4d = np.array(x).transpose(1, 0, 2)[None]  # (1, heads, seq, dim)
    out_torch = self._torch_rope(
      x_torch_4d, theta=10000.0, seq_len=seq_len, head_dim=head_dim
    )[0].transpose(1, 0, 2)  # back to (seq, heads, dim)

    np.testing.assert_allclose(out_flax, out_torch, atol=1e-5)

  def test_dot_product_attention_with_rope_matches_torch(self):
    """Full attention with RoPE matches manual torch-RoPE + standard attention."""
    batch, seq_len, num_heads, head_dim = 2, 16, 4, 32

    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    query = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim))
    kv = jax.random.normal(k2, (batch, seq_len, num_heads, head_dim))
    value = jax.random.normal(k3, (batch, seq_len, num_heads, head_dim))

    # torch expects (batch, heads, seq, dim)
    def apply_torch_rope(x):
      x_np = np.array(x).transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
      out = self._torch_rope(x_np, theta=10000.0, seq_len=seq_len, head_dim=head_dim)
      return jnp.array(out.transpose(0, 2, 1, 3))  # back to (batch, seq, heads, dim)

    q_torch = apply_torch_rope(query)
    k_torch = apply_torch_rope(kv)
    out_torch = nnx.dot_product_attention(q_torch, k_torch, value)

    flax_rope = nnx.RoPE(theta=10000.0)
    out_flax = nnx.dot_product_attention_with_rope(
      query, kv, value, rope=flax_rope
    )

    np.testing.assert_allclose(out_flax, out_torch, atol=1e-5)

  def test_with_mha(self):
    """RoPE integrates correctly as attention_fn in MultiHeadAttention."""
    with jax.numpy_rank_promotion("raise"):
      batch, seq_len, in_features = 2, 8, 64
      num_heads = 4

      rope = nnx.RoPE()
      layer = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=in_features,
        qkv_features=in_features,
        attention_fn=functools.partial(
          nnx.dot_product_attention_with_rope, rope=rope
        ),
        decode=False,
        rngs=nnx.Rngs(0),
      )

      x = jax.random.normal(jax.random.key(1), (batch, seq_len, in_features))
      out = layer(x)
      self.assertEqual(out.shape, (batch, seq_len, in_features))

  def test_different_theta(self):
    """Different theta values produce different outputs."""
    seq_len, head_dim = 16, 32
    x = jax.random.normal(jax.random.key(0), (seq_len, head_dim))

    rope_a = nnx.RoPE(theta=10000.0)
    rope_b = nnx.RoPE(theta=500.0)

    out_a = rope_a(x)
    out_b = rope_b(x)
    self.assertFalse(jnp.allclose(out_a, out_b, atol=1e-3))

  def test_relative_position_invariance(self):
    """Dot product of RoPE-rotated vectors depends only on relative position.

    For any offset d, <RoPE(q, pos=i), RoPE(k, pos=j)> should equal
    <RoPE(q, pos=i+d), RoPE(k, pos=j+d)>.
    """
    head_dim = 64
    max_len = 128
    rope = nnx.RoPE()

    k1, k2 = jax.random.split(jax.random.key(7))
    q_vec = jax.random.normal(k1, (head_dim,))
    k_vec = jax.random.normal(k2, (head_dim,))

    # Place q at position i and k at position j, then shift both by d.
    i, j = 5, 12
    d = 37

    def rope_at(vec, pos):
      # Build a dummy sequence long enough, apply RoPE, extract one position.
      seq = jnp.zeros((pos + 1, head_dim)).at[pos].set(vec)
      return rope(seq)[pos]

    q_rot = rope_at(q_vec, i)
    k_rot = rope_at(k_vec, j)
    dot_original = jnp.dot(q_rot, k_rot)

    q_rot_shifted = rope_at(q_vec, i + d)
    k_rot_shifted = rope_at(k_vec, j + d)
    dot_shifted = jnp.dot(q_rot_shifted, k_rot_shifted)

    np.testing.assert_allclose(dot_original, dot_shifted, atol=1e-5)


if __name__ == '__main__':
  absltest.main()

# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for transformer modules."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import modules
import jax
import jax.numpy as jnp
import numpy as np


class EmbedderTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          vocab_size=10,
          embed_dim=4,
          inputs=[2, 3],
          expected=[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
      ),
  )
  def test_encode(self, vocab_size, embed_dim, inputs, expected):
    embedder = modules.Embedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        rngs=nnx.Rngs(params=0),
    )
    embedder.input_embedding.value = jnp.ones((vocab_size, embed_dim))
    output = embedder.encode(inputs)
    np.testing.assert_array_equal(output, jnp.array(expected))

  @parameterized.parameters(
      dict(
          vocab_size=5,
          embed_dim=2,
          inputs=[[1, 2]],
          expected=[[3.0, 3.0, 3.0, 3.0, 3.0]],
      ),
  )
  def test_decode(self, vocab_size, embed_dim, inputs, expected):
    embedder = modules.Embedder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        rngs=nnx.Rngs(params=0),
    )
    embedder.input_embedding.value = jnp.ones((vocab_size, embed_dim))
    output = embedder.decode(jnp.array(inputs))
    np.testing.assert_array_equal(output, jnp.array(expected))


class AttentionTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          head_dim=2,
      ),
      dict(
          head_dim=20,
      ),
  )
  def test_head_dim(self, head_dim):
    attn = modules.Attention(
        num_heads=2,
        num_kv_heads=4,
        features=5,
        head_dim=head_dim,
        attn_type=modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )

    self.assertEqual(attn.head_dim, head_dim)

  @parameterized.parameters(
      dict(
          num_heads=2,
          num_kv_heads=4,
          expected_use_qkv_einsum=False,
      ),
      dict(
          num_heads=3,
          num_kv_heads=3,
          expected_use_qkv_einsum=True,
      ),
  )
  def test_use_qkv_einsum(
      self,
      num_heads,
      num_kv_heads,
      expected_use_qkv_einsum,
  ):
    attn = modules.Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=5,
        head_dim=8,
        attn_type=modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )

    self.assertEqual(attn.use_qkv_einsum, expected_use_qkv_einsum)
    self.assertEqual(hasattr(attn, 'q_einsum'), not expected_use_qkv_einsum)
    self.assertEqual(hasattr(attn, 'kv_einsum'), not expected_use_qkv_einsum)

  @parameterized.parameters(
      dict(
          num_heads=2,
          head_dim=4,
          features=8,
          segment_pos=0,
          cache_size=3,
          batch_size=2,
          expected_cache_shape=(2, 3, 2, 4),
          expected_output_shape=(2, 1, 8),
      ),
  )
  def test_attention(
      self,
      num_heads,
      head_dim,
      features,
      segment_pos,
      cache_size,
      batch_size,
      expected_cache_shape,
      expected_output_shape,
  ):
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    attn = modules.Attention(
        num_heads,
        num_heads,
        features,
        head_dim,
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )
    cache = attn.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    x = jnp.ones((batch_size, 1, features))
    cache, output = attn(x, jnp.array([[segment_pos]]), cache, attn_mask)

    self.assertEqual(cache['k'].shape, expected_cache_shape)
    self.assertEqual(output.shape, expected_output_shape)

  @parameterized.parameters(
      dict(
          sliding_window_size=2,
      ),
  )
  def test_sliding_window(self, sliding_window_size):
    num_heads = 2
    head_dim = 4
    features = 8
    segment_pos = 0
    cache_size = 3
    batch_size = 2
    attn_mask = jnp.ones((batch_size, 1, cache_size))
    x = jnp.ones((batch_size, 1, features))
    attn = modules.Attention(
        num_heads,
        num_heads,
        features,
        head_dim,
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )
    cache = attn.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    _, output = attn(x, jnp.array([[segment_pos]]), cache, attn_mask)
    sliding_attn = modules.Attention(
        num_heads,
        num_heads,
        features,
        head_dim,
        modules.AttentionType.LOCAL_SLIDING,
        sliding_window_size=sliding_window_size,
        rngs=nnx.Rngs(params=0),
    )
    _, sliding_output = sliding_attn(
        x, jnp.array([[segment_pos]]), cache, attn_mask
    )

    self.assertFalse((output == sliding_output).all())


class FeedForwardTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          features=2,
          hidden_dim=3,
          batch_size=2,
          expected_val=[11.72758674, 47.99916],
          expected_shape=(2, 1, 2),
      ),
  )
  def test_ffw(
      self, features, hidden_dim, batch_size, expected_val, expected_shape
  ):
    inputs = jnp.arange(1, batch_size+1)[:, None, None]
    inputs = jnp.repeat(inputs, features, axis=-1)
    ffw = modules.FeedForward(
        features=features,
        hidden_dim=hidden_dim,
        rngs=nnx.Rngs(params=0),
    )
    ffw.gate_proj.kernel.value = jnp.ones((features, hidden_dim))
    ffw.up_proj.kernel.value = jnp.ones((features, hidden_dim))
    ffw.down_proj.kernel.value = jnp.ones((hidden_dim, features))

    with jax.default_matmul_precision('float32'):
      outputs = ffw(inputs)

    np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
    self.assertEqual(outputs.shape, expected_shape)


class BlockTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_heads=2,
          embed_dim=4,
          head_dim=6,
          cache_size=3,
          batch_size=2,
          use_post_attn_norm=False,
          use_post_ffw_norm=False,
          expected_cache_shape=(2, 3, 2, 6),
          expected_output_shape=(2, 1, 4),
      ),
  )
  def test_block(
      self,
      num_heads,
      embed_dim,
      head_dim,
      cache_size,
      batch_size,
      use_post_attn_norm,
      use_post_ffw_norm,
      expected_cache_shape,
      expected_output_shape,
  ):
    inputs = jnp.ones((batch_size, 1, embed_dim))
    attn_mask = jnp.ones((batch_size, 1, cache_size))

    block = modules.Block(
        num_heads,
        num_heads,
        embed_dim,
        head_dim,
        1,
        use_post_attn_norm,
        use_post_ffw_norm,
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )
    cache = block.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    new_cache, outputs = block(inputs, jnp.array([[0]]), cache, attn_mask)

    self.assertEqual(block.use_post_attn_norm, use_post_attn_norm)
    self.assertEqual(new_cache['k'].shape, expected_cache_shape)
    self.assertEqual(outputs.shape, expected_output_shape)

  @parameterized.parameters(
      dict(
          num_heads=1,
          embed_dim=1,
          head_dim=2,
          cache_size=1,
          batch_size=1,
      ),
  )
  def test_post_attention_norm(
      self,
      num_heads,
      embed_dim,
      head_dim,
      cache_size,
      batch_size,
  ):
    inputs = jnp.ones((batch_size, 1, embed_dim))
    attn_mask = jnp.ones((batch_size, 1, cache_size))

    normed_block = modules.Block(
        num_heads,
        num_heads,
        embed_dim,
        head_dim,
        1,
        True,
        False,  # use_post_ffw_norm
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )
    unnormed_block = modules.Block(
        num_heads,
        num_heads,
        embed_dim,
        head_dim,
        1,
        False,
        False,  # use_post_ffw_norm
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )

    # Ok to use the same cache for both blocks.
    cache = normed_block.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    all_outputs = []
    for block in (normed_block, unnormed_block):
      _, outputs = block(inputs, jnp.array([[0]]), cache, attn_mask)
      all_outputs.append(outputs)

    normed_output, unnormed_output = all_outputs  # pylint: disable=unbalanced-tuple-unpacking
    self.assertFalse(jnp.not_equal(normed_output, unnormed_output).all())

  @parameterized.parameters(
      dict(
          num_heads=1,
          embed_dim=1,
          head_dim=2,
          cache_size=1,
          batch_size=1,
      ),
  )
  def test_post_ffw_norm(
      self,
      num_heads,
      embed_dim,
      head_dim,
      cache_size,
      batch_size,
  ):
    inputs = jnp.ones((batch_size, 1, embed_dim))
    attn_mask = jnp.ones((batch_size, 1, cache_size))

    normed_block = modules.Block(
        num_heads,
        num_heads,
        embed_dim,
        head_dim,
        1,
        True,
        True,  # use_post_ffw_norm
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )
    unnormed_block = modules.Block(
        num_heads,
        num_heads,
        embed_dim,
        head_dim,
        1,
        False,
        False,  # use_post_ffw_norm
        modules.AttentionType.GLOBAL,
        rngs=nnx.Rngs(params=0),
    )

    # Ok to use the same cache for both blocks.
    cache = normed_block.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    all_outputs = []
    for block in (normed_block, unnormed_block):
      _, outputs = block(inputs, jnp.array([[0]]), cache, attn_mask)
      all_outputs.append(outputs)

    normed_output, unnormed_output = all_outputs  # pylint: disable=unbalanced-tuple-unpacking
    self.assertFalse(jnp.not_equal(normed_output, unnormed_output).all())


if __name__ == '__main__':
  absltest.main()

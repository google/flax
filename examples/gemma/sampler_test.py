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
"""Minimal test for sampler."""

from collections.abc import Iterable

from absl.testing import absltest
from flax import nnx
import modules
import sampler as sampler_lib
import sow_lib
import transformer as transformer_lib
import jax
import jax.numpy as jnp
import numpy as np

import sentencepiece as spm


class MockVocab(spm.SentencePieceProcessor):

  def __init__(self):
    super().__init__()
    self._start_id = 3
    self._mapping_text_to_id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        'input': 3,
        'string': 4,
        'hello': 5,
        'world': 6,
        'Hello': 7,
        'there': 8,
        '!': 9,
        'My': 10,
        'name': 11,
        'is': 12,
        'Morgane': 13,
    }
    self._vocab_size = len(self._mapping_text_to_id)

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return ' '.join(reverse_mapping[e] for e in ids)

  def EncodeAsIds(self, text: str) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(' ')
    return [self._mapping_text_to_id[word] for word in words]


class SamplerTest(absltest.TestCase):

  def assertReasonableTensor(self, array, expected_shape=None):
    self.assertIsNotNone(array)
    if expected_shape is not None:
      self.assertEqual(array.shape, expected_shape)

  def test_samples(self):
    vocab = MockVocab()
    num_layers = 6
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=num_layers,
        num_embed=vocab.GetPieceSize(),
        embed_dim=768,
        hidden_dim=6144,
        num_heads=4,
        num_kv_heads=4,
        head_dim=256,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        attn_logits_soft_cap=None,
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )
    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=1024,
    )

    result = sampler(['input string', 'hello world'], total_generation_steps=10)
    self.assertIsNotNone(result)

    top_p_result = sampler(
        ['input string', 'hello world'],
        total_generation_steps=10,
        temperature=9,
        top_p=0.95,
    )
    self.assertIsNotNone(top_p_result)
    self.assertNotEqual(result.text, top_p_result.text)

    top_p_result_2 = sampler(
        ['input string', 'hello world'],
        total_generation_steps=10,
        temperature=9,
        top_p=0.95,
        seed=jax.random.PRNGKey(42),
    )
    self.assertIsNotNone(top_p_result_2)
    self.assertNotEqual(top_p_result.text, top_p_result_2.text)

  def test_state_update(self):
    vocab = MockVocab()
    num_layers = 6
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=num_layers,
        num_embed=vocab.GetPieceSize(),
        embed_dim=768,
        hidden_dim=6144,
        num_heads=4,
        num_kv_heads=4,
        head_dim=256,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        attn_logits_soft_cap=None,
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )
    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=1024,
    )
    input_strings = ['input string', 'hello world']
    original_logits = sampler(input_strings, total_generation_steps=10).logits

    new_transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=42)
    )
    sampler.transformer_state = nnx.state(new_transformer, nnx.Param)
    new_logits = sampler(input_strings, total_generation_steps=10).logits
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(
          original_logits, new_logits, atol=1e-1, rtol=1e-1
      )

  def test_invalid_state_update(self):
    vocab = MockVocab()

    def make_config(num_layers, embed_dim):
      return transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
          num_layers=num_layers,
          num_embed=vocab.GetPieceSize(),
          embed_dim=embed_dim,
          hidden_dim=6144,
          num_heads=4,
          num_kv_heads=4,
          head_dim=256,
          final_logit_softcap=None,
          attention_types=[modules.AttentionType.GLOBAL] * num_layers,
          attn_logits_soft_cap=None,
          use_post_attn_norm=None,
          use_post_ffw_norm=None,
      )

    transformer = transformer_lib.Transformer(
        make_config(num_layers=6, embed_dim=768), rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=1024,
    )

    new_transformer = transformer_lib.Transformer(
        make_config(num_layers=3, embed_dim=768), rngs=nnx.Rngs(params=42)
    )
    with self.assertRaisesRegex(
        ValueError, '.*must have the same structure.*'
    ):
      sampler.transformer_state = nnx.state(new_transformer, nnx.Param)

    new_transformer = transformer_lib.Transformer(
        make_config(num_layers=6, embed_dim=1024), rngs=nnx.Rngs(params=42)
    )
    with self.assertRaisesRegex(
        ValueError, '.*must have the same shape and dtype.*'
    ):
      sampler.transformer_state = nnx.state(new_transformer, nnx.Param)

  def test_forbidden_tokens(self):
    vocab = MockVocab()
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=0,
        num_embed=vocab.GetPieceSize(),
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        final_logit_softcap=None,
        attention_types=[],
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )
    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    # Pre-cook the embedding matrix so that the output is deterministic.
    transformer.embedder.input_embedding.value = jnp.eye(
        vocab.GetPieceSize(), 32
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=8,
    )

    # First, we check that the sampler would produce the tokens that we are
    # trying to forbid.
    result1 = sampler(
        ['input string', 'hello world'],
        total_generation_steps=10,
        forbidden_tokens=None,
    )
    self.assertIn('string', result1.text[0])
    self.assertIn('world', result1.text[1])

    # Then, we check that the sampler does not produce the forbidden tokens.
    result2 = sampler(
        ['input string', 'hello world'],
        total_generation_steps=10,
        forbidden_tokens=['string', 'world'],
    )
    for output in result2.text:
      self.assertNotIn('string', output)
      self.assertNotIn('world', output)

  def test_forward_equivalence(self):
    vocab = MockVocab()
    num_layers = 2
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=num_layers,
        num_embed=vocab.GetPieceSize(),
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )

    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    raw_input = 'Hello there ! My name is Morgane <pad>'
    token_input = jnp.asarray(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))
    batch_size = 1
    cache_size = 9
    cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    input_mask = token_input != vocab.pad_id()
    positions = transformer_lib.build_positions_from_mask(input_mask)
    attention_mask = transformer_lib.make_causal_attn_mask(input_mask)

    n_input_tokens = token_input.shape[1]

    output_forward, _ = transformer(
        last_tokens=token_input,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
    )
    output_forward = output_forward[0, :n_input_tokens]

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=cache_size,
    )

    output_transformer = sampler(
        [raw_input],
        total_generation_steps=10,
        echo=True,
    )
    out_logits = np.array(output_transformer.logits)[0, 1 : n_input_tokens + 1]

    np.testing.assert_almost_equal(output_forward, out_logits)

  def test_sampler_init_sample_state(self):
    vocab = MockVocab()
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=0,
        num_embed=vocab.GetPieceSize(),
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        final_logit_softcap=None,
        attention_types=[],
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )
    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=8,
    )

    input_strings = ['<pad> hello world', 'input string <pad>']
    all_input_ids = [sampler.tokenize(x) for x in input_strings]
    total_sampling_steps = 5
    sample_state = sampler.init_sample_state(
        all_input_ids,
        total_sampling_steps=total_sampling_steps,
        include_logits=True,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=0.95,
        seed=jax.random.PRNGKey(0),
    )

    # Check that the position indices correctly ignore padding
    self.assertListEqual(list(sample_state.positions[0]), [0, 0, 1, 2, 3, 4])
    self.assertListEqual(list(sample_state.positions[1]), [0, 1, 2, 2, 3, 4])

  def test_sampler_mask_tokens_after_eos_ids(self):
    vocab = MockVocab()
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=0,
        num_embed=vocab.GetPieceSize(),
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        final_logit_softcap=None,
        attention_types=[],
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
    )
    transformer = transformer_lib.Transformer(
        transformer_config, rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=8,
    )

    input_strings = ['hello world </s> hello world', 'input string </s> hello']
    all_input_ids = [sampler.tokenize(x) for x in input_strings]
    total_sampling_steps = 5
    sample_state = sampler.init_sample_state(
        all_input_ids,
        total_sampling_steps=total_sampling_steps,
        include_logits=True,
        forbidden_token_ids=None,
        temperature=0.0,
        top_p=0.95,
        seed=jax.random.PRNGKey(0),
    )

    masked_token_buffer = sampler.mask_tokens_after_eos_ids(
        sample_state.token_buffer
    )

    self.assertListEqual(list(masked_token_buffer[0]), [1, 5, 6, 2, 0, 0])
    self.assertListEqual(list(masked_token_buffer[1]), [1, 3, 4, 2, 0, 0])

  def test_sampler_sows_intermediates(self):
    vocab = MockVocab()
    num_layers = 3
    config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=num_layers,
        num_embed=vocab.GetPieceSize(),
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=None,
        attn_logits_soft_cap=None,
        use_post_ffw_norm=None,
    )
    sow_config = sow_lib.SowConfig(
        embeddings=True,
        rs_after_attention=False,  # This should results in a None value.
        rs_after_ffw=True,
        attn_logits_topk=5,
        mlp_hidden_topk=11,
    )
    transformer = transformer_lib.Transformer(
        config, rngs=nnx.Rngs(params=0), sow_config=sow_config
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
    )
    raw_input = ['input string', 'hello world']

    result = sampler(raw_input, total_generation_steps=10)
    input_length = max([len(vocab.EncodeAsIds(i)) for i in raw_input])
    input_length += 1  # +1 for BOS token
    output_length = max(len(tokens) for tokens in result.tokens)
    length = input_length + output_length
    self.assertIsNotNone(result)
    intermediates = result.intermediates
    self.assertIsNotNone(intermediates)
    self.assertReasonableTensor(
        intermediates.embeddings,
        expected_shape=(2, length, config.embed_dim),
    )
    # Verify that the intermediates are different for two different steps.
    self.assertNotAlmostEqual(
        jnp.sum(intermediates.embeddings[:, 1, ...]),
        jnp.sum(intermediates.embeddings[:, 2, ...]),
    )
    # Verify that the intermediates are filled in for each layer.
    self.assertLen(intermediates.layers, config.num_layers)
    for layer in intermediates.layers:
      # For the requested intermediates we check the shape and that values are
      # not all zeros, which was the initial value.
      self.assertReasonableTensor(
          layer.rs_after_ffw,
          expected_shape=(2, length, config.embed_dim),
      )
      self.assertReasonableTensor(
          layer.attn_logits_topk_values,
          expected_shape=(
              2,
              length,
              config.num_heads,
              sow_config.attn_logits_topk,
          ),
      )
      self.assertReasonableTensor(
          layer.attn_logits_topk_indices,
          expected_shape=(
              2,
              length,
              config.num_heads,
              sow_config.attn_logits_topk,
          ),
      )
      self.assertReasonableTensor(
          layer.mlp_hidden_topk_values,
          expected_shape=(2, length, sow_config.mlp_hidden_topk),
      )
      self.assertReasonableTensor(
          layer.mlp_hidden_topk_indices,
          expected_shape=(2, length, sow_config.mlp_hidden_topk),
      )
      # For the none requested intermediates we want to have None values.
      self.assertIsNone(layer.rs_after_attention)

  def test_compute_attention_mask(self):
    # Check that the input mask is correctly applied when total sampling steps
    # is lower than the max cache length.
    input_mask = jnp.array([[1, 1, 0, 0, 0], [1, 1, 0, 1, 0]], dtype=jnp.bool_)
    seq_len = 8
    time_step = jnp.asarray(4, dtype=jnp.int32)
    attn_mask = sampler_lib._compute_attention_masks(
        time_step, seq_len, input_mask
    )
    expected_attn_mask = jnp.array(
        [[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())

    # Check that the input mask is correctly applied when total sampling steps
    # is *longer* than the max cache length.
    seq_len = 4
    time_step = jnp.asarray(4, dtype=jnp.int32)
    attn_mask = sampler_lib._compute_attention_masks(
        time_step, seq_len, input_mask
    )
    expected_attn_mask = jnp.array(
        [[0, 1, 1, 1], [0, 1, 0, 1]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())


if __name__ == '__main__':
  absltest.main()

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
"""Tests for the Gemma transformer."""

from collections import defaultdict
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import modules
import transformer as transformer_lib
import jax.numpy as jnp
import numpy as np


def create_fake_params(config: transformer_lib.TransformerConfig):
  def nested_defaultdict():
    return defaultdict(nested_defaultdict)

  res = nested_defaultdict()
  res['transformer'] = nested_defaultdict()
  params = res['transformer']
  # 1. embedding params
  params['embedder']['input_embedding'] = jnp.ones(
      (config.num_embed, config.embed_dim)
  )
  # 2. final norm params
  params['final_norm'] = {'scale': jnp.ones((config.embed_dim,))}

  # 3. attention block params
  for layer_idx in range(config.num_layers):
    params[f'layer_{layer_idx}']['attn']['attn_vec_einsum']['w'] = jnp.ones(
        (config.num_heads, config.head_dim, config.embed_dim)
    )
    if config.num_heads == config.num_kv_heads:
      params[f'layer_{layer_idx}']['attn']['qkv_einsum']['w'] = jnp.ones(
          (3, config.num_heads, config.embed_dim, config.head_dim)
      )
    else:
      params[f'layer_{layer_idx}']['attn']['q_einsum']['w'] = jnp.ones(
          (config.num_heads, config.embed_dim, config.head_dim)
      )
      params[f'layer_{layer_idx}']['attn']['kv_einsum']['w'] = jnp.ones(
          (config.num_kv_heads, config.embed_dim, config.head_dim)
      )

    # 4. feedforward block params
    params[f'layer_{layer_idx}']['mlp']['gating_einsum'] = jnp.ones(
        (2, config.embed_dim, config.hidden_dim)
    )
    params[f'layer_{layer_idx}']['mlp']['linear'] = jnp.ones(
        (config.hidden_dim, config.embed_dim)
    )

    # 5. layer norm params
    params[f'layer_{layer_idx}']['pre_attention_norm']['scale'] = jnp.ones((
        config.embed_dim,
    ))
    params[f'layer_{layer_idx}']['pre_ffw_norm']['scale'] = jnp.ones((
        config.embed_dim,
    ))

    if config.use_post_attn_norm:
      params[f'layer_{layer_idx}']['post_attn_norm']['scale'] = jnp.ones((
          config.embed_dim,
      ))
    if config.use_post_ffw_norm:
      params[f'layer_{layer_idx}']['post_ffw_norm']['scale'] = jnp.ones((
          config.embed_dim,
      ))
  return res


class TransformerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Prime number to ease shape tracing
      dict(
          num_layers=3,
          num_embed=17,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=2,
          hidden_dim=11,
          head_dim=8,
          cache_size=29,
          batch_size=7,
          sequence_length=17,
          expected_outputs_shape=(7, 17, 17),
          expected_cache_shape=(7, 29, 2, 8),
      ),
      dict(
          num_layers=3,
          num_embed=4,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=1,
          hidden_dim=4,
          head_dim=4,
          cache_size=2,
          batch_size=1,
          sequence_length=1,
          expected_outputs_shape=(1, 1, 4),
          expected_cache_shape=(1, 2, 1, 4),
      ),
  )
  def test_transformer(
      self,
      num_layers,
      num_embed,
      embed_dim,
      num_heads,
      num_kv_heads,
      hidden_dim,
      head_dim,
      cache_size,
      batch_size,
      sequence_length,
      expected_outputs_shape,
      expected_cache_shape,
  ):

    config = transformer_lib.TransformerConfig(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    transformer = transformer_lib.Transformer(
        config=config, rngs=nnx.Rngs(params=0)
    )
    cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    outputs, cache = transformer(
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        cache,
        attention_mask,
    )

    self.assertEqual(outputs.shape, expected_outputs_shape)
    self.assertEqual(cache['layer_0']['v'].shape, expected_cache_shape)

  @parameterized.parameters(
      ('final_logit_softcap',),
      ('attn_logits_soft_cap',),
  )
  def test_logit_softcap(
      self,
      soft_cap_arg,
  ):
    cache_size = 2
    batch_size = 1
    soft_cap_val = 0.001

    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)

    params = dict(
        num_layers=3,
        num_embed=4,
        embed_dim=2,
        num_heads=2,
        num_kv_heads=1,
        hidden_dim=4,
        head_dim=4,
        attention_types=[modules.AttentionType.GLOBAL] * 3,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

    no_soft_cap_args = {
        'final_logit_softcap': None,
        'attn_logits_soft_cap': None,
    }

    soft_cap_args = no_soft_cap_args.copy()
    soft_cap_args[soft_cap_arg] = soft_cap_val

    config_soft_cap = transformer_lib.TransformerConfig(
        **(params | soft_cap_args)
    )
    config_no_soft_cap = transformer_lib.TransformerConfig(
        **(params | no_soft_cap_args)
    )

    all_outputs = []
    for config in [config_soft_cap, config_no_soft_cap]:
      transformer = transformer_lib.Transformer(
          config=config, rngs=nnx.Rngs(params=1)
      )
      cache = transformer.init_cache(
          cache_size=cache_size,
          batch_size=batch_size,
          dtype=jnp.float32,
      )

      outputs, _ = transformer(
          jnp.array([[1]]), jnp.array([[1]]), cache, attention_mask
      )
      all_outputs.append(outputs)

    soft_cap_outputs, no_soft_cap_outputs = all_outputs  # pylint: disable=unbalanced-tuple-unpacking

    # Ensure that values aren't equal coming out of computation
    self.assertFalse((soft_cap_outputs == no_soft_cap_outputs).all())

    # Run soft capping manually
    manual_soft_cap_logits = no_soft_cap_outputs / soft_cap_val
    manual_soft_cap_logits = jnp.tanh(manual_soft_cap_logits) * soft_cap_val

    np.testing.assert_array_almost_equal(
        manual_soft_cap_logits, soft_cap_outputs, 1e-5
    )

  @parameterized.parameters([
      dict(
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=0,  # unused
              embed_dim=0,  # unused
              hidden_dim=0,  # unused
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=False,
              use_post_ffw_norm=False,
          ),
          cache_size=2,
          keys=['layer_0', 'layer_1'],
          k_shape=(1, 2, 3, 4),
          v_shape=(1, 2, 3, 4),
      )
  ])
  def test_creates_cache(self, config, cache_size, keys, k_shape, v_shape):
    transformer = transformer_lib.Transformer(
        config=config, rngs=nnx.Rngs(params=0)
    )
    cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=1,
        dtype=jnp.float32,
    )
    self.assertEqual(list(cache.keys()), keys)
    self.assertEqual(cache['layer_0']['k'].shape, k_shape)
    self.assertEqual(cache['layer_0']['v'].shape, v_shape)

  @parameterized.parameters([
      dict(
          batch_size=1,
          seq_size=4,
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=4,  # unused
              embed_dim=2,
              hidden_dim=12,  # unused
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=False,
              use_post_ffw_norm=False,
          ),
      )
  ])
  def test_forward_no_cache(
      self,
      batch_size: int,
      seq_size: int,
      config: transformer_lib.TransformerConfig,
  ):
    cache_size = 6

    token_input = jnp.ones((batch_size, seq_size), dtype=jnp.int32)
    transformer = transformer_lib.Transformer(
        config=config, rngs=nnx.Rngs(params=0)
    )
    empty_cache = transformer.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )
    attention_mask = jnp.ones(
        (batch_size, seq_size, cache_size), dtype=jnp.bool
    )
    positions = transformer_lib.build_positions_from_mask(token_input != 0)

    output_cache, _ = transformer(
        token_input, positions, empty_cache, attention_mask
    )

    attention_mask = jnp.ones((batch_size, seq_size, seq_size), dtype=jnp.bool)
    output_none, cache_none = transformer(
        token_input, positions, None, attention_mask
    )

    self.assertIsNone(cache_none)
    np.testing.assert_array_almost_equal(output_cache, output_none, 1e-5)

  def test_attention_types(
      self,
  ):

    config = transformer_lib.TransformerConfig(
        num_layers=2,
        num_embed=4,
        embed_dim=2,
        hidden_dim=12,
        num_heads=3,
        head_dim=4,
        num_kv_heads=3,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * 2,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    transformer = transformer_lib.Transformer(
        config=config, rngs=nnx.Rngs(params=0)
    )
    cache = transformer.init_cache(
        cache_size=6,
        batch_size=1,
        dtype=jnp.float32,
    )
    self.assertTrue(cache)

  @parameterized.parameters(
      dict(
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=4,
              embed_dim=2,
              hidden_dim=12,
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=False,
              use_post_ffw_norm=False,
          ),
      ),
      dict(
          config=transformer_lib.TransformerConfig(
              num_layers=2,
              num_embed=4,
              embed_dim=2,
              hidden_dim=12,
              num_heads=3,
              head_dim=4,
              num_kv_heads=3,
              final_logit_softcap=None,
              attention_types=[modules.AttentionType.GLOBAL] * 2,
              use_post_attn_norm=True,
              use_post_ffw_norm=True,
          ),
      ),
  )
  def test_load_from_params(self, config):
    params = create_fake_params(config)
    transformer = transformer_lib.Transformer.from_params(params, config)
    logits, _ = transformer(
        last_tokens=jnp.tile(jnp.arange(3), (2, 1)),
        positions=jnp.tile(jnp.arange(3), (2, 1)),
        cache=None,
        attention_mask=jnp.ones((2, 1, 3), dtype=jnp.bool),
    )
    self.assertEqual(logits.shape, (2, 3, 4))


if __name__ == '__main__':
  absltest.main()

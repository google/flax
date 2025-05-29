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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


import dataclasses

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import random

from flax import nnx
from configs import default
from models import TransformerConfig, TransformerLM
from utils import HasCache

jax.config.update('jax_disable_most_optimizations', True)

# add project_root to import lm1b Linen model
# "/path/to/flax/examples/lm1b_nnx/models_test.py" -> "/path/to/flax"
project_root = str(Path(__file__).absolute().parents[2])
sys.path.append(project_root)
from examples.lm1b.models import TransformerLM as TransformerLinen  # type: ignore[import-error]

sys.path.pop()


@dataclasses.dataclass(unsafe_hash=True)
class CompatTransformerConfig(TransformerConfig):
  decode: bool | None = None
  deterministic: bool | None = None


def get_transformer_config(**kwargs):
  base_config = default.get_config()
  config = CompatTransformerConfig(
    vocab_size=base_config.vocab_size,
    output_vocab_size=base_config.vocab_size,
    logits_via_embedding=base_config.logits_via_embedding,
    dtype=jnp.bfloat16 if base_config.use_bfloat16 else jnp.float32,
    emb_dim=base_config.emb_dim,
    num_heads=base_config.num_heads,
    num_layers=base_config.num_layers,
    qkv_dim=base_config.qkv_dim,
    mlp_dim=base_config.mlp_dim,
    max_len=max(
      base_config.max_target_length, base_config.max_eval_target_length
    ),
    dropout_rate=base_config.dropout_rate,
    attention_dropout_rate=base_config.attention_dropout_rate,
    kernel_init=nnx.initializers.xavier_uniform(),
    bias_init=nnx.initializers.normal(stddev=1e-6),
    **kwargs,
  )
  return base_config, config


class ModelTest(absltest.TestCase):
  def transfer_params(
    self,
    config: TransformerConfig,
    params_nnx: nnx.State,
    params_linen: dict[str, Any],
  ):
    rules = dataclasses.asdict(config.axis_rules)
    flat_params_nnx = dict(nnx.to_flat_state(params_nnx))
    flat_params_linen = nnx.traversals.flatten_mapping(params_linen, sep='/')

    def apply_rules(names: tuple[str, ...]):
      return tuple(rules[name] for name in names)

    def copy_var(nnx_name: str, linen_name: str):
      nnx_path = tuple(nnx_name.split('/'))
      assert (
        flat_params_nnx[nnx_path].value.shape
        == flat_params_linen[linen_name].value.shape
      )
      flat_params_nnx[nnx_path].value = flat_params_linen[linen_name].value
      assert flat_params_nnx[nnx_path].sharding == apply_rules(
        flat_params_linen[linen_name].names
      )

    copy_var('decoder/output_embed/embedding', 'decoder/Embed_0/embedding')
    copy_var(
      'decoder/encoderdecoder_norm/bias', 'decoder/encoderdecoder_norm/bias'
    )
    copy_var(
      'decoder/encoderdecoder_norm/scale', 'decoder/encoderdecoder_norm/scale'
    )

    for idx in range(config.num_layers):
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/ln1/bias',
        f'decoder/encoderdecoderblock_{idx}/LayerNorm_0/bias',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/ln1/scale',
        f'decoder/encoderdecoderblock_{idx}/LayerNorm_0/scale',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/ln2/bias',
        f'decoder/encoderdecoderblock_{idx}/LayerNorm_1/bias',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/ln2/scale',
        f'decoder/encoderdecoderblock_{idx}/LayerNorm_1/scale',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/query/kernel',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/query/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/key/kernel',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/key/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/value/kernel',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/value/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/out/kernel',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/out/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/mlp/linear1/kernel',
        f'decoder/encoderdecoderblock_{idx}/MlpBlock_0/Dense_0/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/mlp/linear1/bias',
        f'decoder/encoderdecoderblock_{idx}/MlpBlock_0/Dense_0/bias',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/mlp/linear2/kernel',
        f'decoder/encoderdecoderblock_{idx}/MlpBlock_0/Dense_1/kernel',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/mlp/linear2/bias',
        f'decoder/encoderdecoderblock_{idx}/MlpBlock_0/Dense_1/bias',
      )

    copy_var('decoder/logitdense/kernel', 'decoder/logitdense/kernel')
    copy_var('decoder/logitdense/bias', 'decoder/logitdense/bias')

  def transfer_cache(
    self,
    config: TransformerConfig,
    cache_nnx: nnx.State,
    cache_linen: dict[str, Any],
  ):
    flat_cache_nnx = dict(nnx.to_flat_state(cache_nnx))
    flat_cache_linen = nnx.traversals.flatten_mapping(cache_linen, sep='/')

    def copy_var(nnx_name: str, linen_name: str):
      nnx_path = tuple(nnx_name.split('/'))
      assert (
        flat_cache_nnx[nnx_path].value.shape
        == flat_cache_linen[linen_name].shape
      )
      flat_cache_nnx[nnx_path].value = flat_cache_linen[linen_name]

    for idx in range(config.num_layers):
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/cache_index',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/cache_index',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/cached_key',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/cached_key',
      )
      copy_var(
        f'decoder/encoderdecoderblock_{idx}/attention/cached_value',
        f'decoder/encoderdecoderblock_{idx}/MultiHeadDotProductAttention_0/cached_value',
      )

    copy_var(
      'decoder/posembed_output/cache_index',
      'decoder/posembed_output/cache_index',
    )

  def test_forward_eval(self):
    _, config = get_transformer_config(
      axis_rules=default.MeshRules(
        embed='model',
        mlp='data',
        kv=None,
        vocab=None,
      ),
      deterministic=True,
      decode=False,
    )
    # Set dropout rates to avoid create dropout states
    config.dropout_rate = 0.0
    config.attention_dropout_rate = 0.0

    model_nnx = nnx.eval_shape(lambda: TransformerLM(config, rngs=nnx.Rngs(0)))
    _, params_nnx = nnx.split(model_nnx, nnx.Param)

    model_linen = TransformerLinen(config)

    sample_inputs = random.randint(random.PRNGKey(0), (1, 3), 0, 20)
    params_linen = model_linen.init(random.key(0), sample_inputs)['params']

    self.transfer_params(config, params_nnx, params_linen)
    nnx.update(model_nnx, params_nnx)

    model_nnx.set_attributes(deterministic=True, decode=False)
    output_nnx = model_nnx(sample_inputs)

    output_linen: jax.Array = model_linen.apply(
      {'params': params_linen}, sample_inputs
    )

    assert jnp.allclose(output_nnx, output_linen, atol=1e-5)

  def test_forward_decode(self):
    batch_size = 2

    _, config = get_transformer_config(
      axis_rules=default.MeshRules(
        embed='model',
        mlp='data',
        kv=None,
        vocab=None,
      ),
      deterministic=True,
      decode=True,
    )
    # Set dropout rates to avoid create dropout states
    config.dropout_rate = 0.0
    config.attention_dropout_rate = 0.0

    model_nnx = nnx.eval_shape(lambda: TransformerLM(config, rngs=nnx.Rngs(0)))
    for _path, m in model_nnx.iter_modules():
      if isinstance(m, HasCache):
        input_shape = (batch_size, config.max_len, config.emb_dim)
        m.init_cache(input_shape, dtype=config.dtype)

    _, params_nnx, cache_nnx = nnx.split(model_nnx, nnx.Param, nnx.Cache)

    model_linen = TransformerLinen(config)

    flax_init_inputs = random.randint(
      random.PRNGKey(0), (batch_size, config.max_len), 0, config.vocab_size
    )
    ar_decode_inputs = random.randint(
      random.PRNGKey(0), (3, batch_size, 1), 0, config.vocab_size
    )
    variables = model_linen.init(random.key(0), flax_init_inputs)
    params_linen = variables['params']
    cache_linen = variables['cache']

    self.transfer_params(config, params_nnx, params_linen)
    self.transfer_cache(config, cache_nnx, cache_linen)
    nnx.update(model_nnx, params_nnx, cache_nnx)
    model_nnx.set_attributes(deterministic=True, decode=True)

    outputs_nnx = []
    outputs_linen = []

    for inputs in ar_decode_inputs:
      output_nnx = model_nnx(inputs)
      outputs_nnx.append(output_nnx)

    output_linen: jax.Array
    for inputs in ar_decode_inputs:
      output_linen, updates = model_linen.apply(
        {'params': params_linen, 'cache': cache_linen},
        inputs,
        mutable=['cache'],
      )
      cache_linen = updates['cache']
      outputs_linen.append(output_linen)

    for output_nnx, output_linen in zip(outputs_nnx, outputs_linen):
      assert jnp.allclose(output_nnx, output_linen, atol=1e-5)


if __name__ == '__main__':
  absltest.main()

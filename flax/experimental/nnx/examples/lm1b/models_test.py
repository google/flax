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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# add project_root to import lm1b Linen model
project_root = str(Path(__file__).parents[6])
sys.path.append(project_root)
from examples.lm1b.models import TransformerLM as TransformerLinen

sys.path.pop()

import dataclasses

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import random

from flax import traverse_util
from flax.experimental import nnx
from flax.experimental.nnx.examples.lm1b.models import (
  MeshRules,
  TransformerConfig,
  TransformerLM,
)

jax.config.update('jax_disable_most_optimizations', True)


@dataclasses.dataclass(unsafe_hash=True)
class CompatTransformerConfig(TransformerConfig):
  decode: bool | None = None
  deterministic: bool | None = None


class ModelTest(absltest.TestCase):
  def transfer_params(
    self,
    config: TransformerConfig,
    params_nnx: nnx.State,
    params_linen: dict[str, Any],
  ):
    rules = dataclasses.asdict(config.rules)
    flat_params_nnx = params_nnx.flat_state()
    flat_params_linen = traverse_util.flatten_dict(params_linen, sep='/')

    def apply_rules(names: tuple[str, ...]):
      return tuple(rules[name] for name in names)

    def copy_var(nnx_name, linen_name):
      assert (
        flat_params_nnx[nnx_name].value.shape
        == flat_params_linen[linen_name].value.shape
      )
      flat_params_nnx[nnx_name].value = flat_params_linen[linen_name].value
      assert flat_params_nnx[nnx_name].sharding == apply_rules(
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
    flat_cache_nnx = cache_nnx.flat_state()
    flat_cache_linen = traverse_util.flatten_dict(cache_linen, sep='/')

    def copy_var(nnx_name, linen_name):
      assert (
        flat_cache_nnx[nnx_name].value.shape
        == flat_cache_linen[linen_name].shape
      )
      flat_cache_nnx[nnx_name].value = flat_cache_linen[linen_name]

    # cache nnx
    # {
    #   'decoder/encoderdecoderblock_0/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_0/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_0/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_1/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_1/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_1/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_2/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_2/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_2/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_3/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_3/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_3/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_4/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_4/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_4/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_5/attention/cache_index': Cache(value=()),
    #   'decoder/encoderdecoderblock_5/attention/cached_key': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/encoderdecoderblock_5/attention/cached_value': Cache(value=(2, 2048, 8, 64)),
    #   'decoder/posembed_output/cache_index': Cache(value=())
    # }

    # cache linen
    # {
    #   'decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_2/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_2/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_2/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_3/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_3/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_3/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_4/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_4/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_4/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_5/MultiHeadDotProductAttention_0/cache_index': (),
    #   'decoder/encoderdecoderblock_5/MultiHeadDotProductAttention_0/cached_key': (1, 3, 8, 64),
    #   'decoder/encoderdecoderblock_5/MultiHeadDotProductAttention_0/cached_value': (1, 3, 8, 64),
    #   'decoder/posembed_output/cache_index': ()
    # }

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
    config = CompatTransformerConfig(
      vocab_size=20,
      output_vocab_size=20,
      deterministic=True,
      rules=MeshRules(
        embed='model',
        mlp='data',
        kv=None,
        vocab=None,
      ),
    )

    model_nnx = TransformerLM.create_abstract(config, rngs=nnx.Rngs(0))
    params_nnx, _ = model_nnx.split(nnx.Param)

    model_linen = TransformerLinen(config)

    sample_inputs = random.randint(random.PRNGKey(0), (1, 3), 0, 20)
    params_linen = model_linen.init(random.key(0), sample_inputs)['params']

    self.transfer_params(config, params_nnx, params_linen)
    model_nnx.update(params_nnx)

    with nnx.flags(deterministic=True):
      output_nnx = model_nnx(sample_inputs)

    output_linen: jax.Array = model_linen.apply(
      {'params': params_linen}, sample_inputs
    )

    assert jnp.allclose(output_nnx, output_linen, atol=1e-5)

  def test_forward_decode(self):
    batch_size = 2

    config = CompatTransformerConfig(
      vocab_size=20,
      output_vocab_size=20,
      max_len=3,
      emb_dim=16,
      qkv_dim=16,
      num_heads=2,
      deterministic=True,
      decode=True,
      rules=MeshRules(
        embed='model',
        mlp='data',
        kv=None,
        vocab=None,
      ),
    )

    model_nnx = TransformerLM.create_abstract(config, rngs=nnx.Rngs(0))
    for _path, m in model_nnx.module():
      if isinstance(m, nnx.HasCacheInitializer):
        input_shape = (batch_size, config.max_len, config.emb_dim)
        m.init_cache(input_shape, dtype=config.dtype)

    params_nnx, cache_nnx, _ = model_nnx.split(nnx.Param, nnx.Cache)

    model_linen = TransformerLinen(config)

    flax_init_inputs = random.randint(
      random.PRNGKey(0), (batch_size, config.max_len), 0, config.vocab_size
    )
    ar_decode_inputs = random.randint(
      random.PRNGKey(0), (batch_size, 1), 0, config.vocab_size
    )
    variables = model_linen.init(random.key(0), flax_init_inputs)
    params_linen = variables['params']
    cache_linen = variables['cache']

    self.transfer_params(config, params_nnx, params_linen)
    self.transfer_cache(config, cache_nnx, cache_linen)
    model_nnx.update(params_nnx, cache_nnx)

    with nnx.flags(deterministic=True, decode=True):
      output_nnx = model_nnx(ar_decode_inputs)

    output_linen: jax.Array
    output_linen, updates = model_linen.apply(
      {'params': params_linen, 'cache': cache_linen},
      ar_decode_inputs,
      mutable=['cache'],
    )

    assert jnp.allclose(output_nnx, output_linen, atol=1e-5)


if __name__ == '__main__':
  absltest.main()

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
"""Gemma transformer."""

from __future__ import annotations

import functools
from typing import Any

from flax import nnx
import helpers
import layers
import modules
import params as params_lib
import sow_lib
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-importing-member,g-multiple-import
from transformer_cfg import TransformerConfig

Cache = dict[str, modules.LayerCache]


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  """Maps linen variable names to nnx variable names."""
  new_key = []
  for k in key:
    if k.startswith('layer_'):
      prefix, suffix = k.split('layer_')
      assert not prefix, prefix
      new_key.append('layers')
      new_key.append(int(suffix))
    elif k == 'gating_einsum':
      new_key.append('gate_proj')
      new_key.append('kernel')
    elif k == 'linear':
      new_key.append('down_proj')
      new_key.append('kernel')
    else:
      new_key.append(k)

  return tuple(new_key)


def _assign_linen_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
    transpose_gating_einsum: bool,
) -> dict[tuple[str, ...], Any]:
  """Splits and maybe transposes gate_proj."""
  if 'gate_proj' in mapped_path:
    if transpose_gating_einsum:
      val = jnp.swapaxes(val, 1, 2)
    state[mapped_path].set_value(val[0])
    state[mapped_path[:-2] + ('up_proj', 'kernel')].set_value(val[1])
  else:
    state[mapped_path].set_value(val)
  return state


class Transformer(nnx.Module):
  """Gemma text-only transformer."""

  @classmethod
  def from_params(
      cls,
      params: params_lib.Params,
      config: None | TransformerConfig = None,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ) -> Transformer:
    if config is None:
      config = TransformerConfig.from_params(params)
    assign_val_fn = functools.partial(
        _assign_linen_params_to_nnx_state,
        transpose_gating_einsum=config.transpose_gating_einsum,
    )
    return helpers.module_from_linen_variables(
        module_factory=lambda: cls(
            config, rngs=nnx.Rngs(params=0), sow_config=sow_config
        ),
        variables=params['transformer'],
        map_key_fn=_map_linen_var_names,
        assign_val_fn=assign_val_fn,
    )

  def __init__(
      self,
      config: TransformerConfig,
      *,
      rngs: nnx.Rngs,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ):
    self.embedder = modules.ScaledEmbed(
        config.num_embed,
        config.embed_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        rngs=rngs,
        embedding_metadata={"out_sharding": config.shd_config.emb_vd},
    )
    self.layers = nnx.List([
        modules.Block(
          config=config,
          attn_type=attn_type,
          sow_config=sow_config,
          rngs=rngs,
        )
        for _, attn_type in zip(
            range(config.num_layers), config.attention_types
        )
    ])
    self.final_norm = layers.RMSNorm(
      config.embed_dim,
      scale_metadata={"out_sharding": config.shd_config.rms_norm_weight},
      rngs=rngs,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
    )
    self.final_dropout = nnx.Dropout(config.dropout_rate, deterministic=False)
    self.final_logits_softcap = config.final_logit_softcap
    self.sow_config = sow_config
    self.shd_config = config.shd_config

  def __call__(
      self,
      last_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: Array,  # [B, L, L']
      rngs: nnx.Rngs | None = None,
  ) -> tuple[Array, Cache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      rngs: optional rngs to pass in stochastic layers

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder(last_tokens, out_sharding=self.shd_config.act_btd)
    self.sow_config.maybe_sow_embeddings(x, self)

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          rngs=rngs,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    x = self.final_dropout(x, rngs=rngs)
    logits = self.embedder.attend(x, out_sharding=self.shd_config.act_btd)

    if self.final_logits_softcap is not None:
      logits /= self.final_logits_softcap
      logits = jnp.tanh(logits) * self.final_logits_softcap

    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def embed_dim(self) -> int:
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    return len(self.layers)

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.float32,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    return {
        f'layer_{i}': self.layers[i].init_cache(
            cache_size=cache_size,
            batch_size=batch_size,
            dtype=dtype,
        )
        for i in range(self.num_layers)
    }

  def init_intermediates(
      self,
      batch_size: int,
      buffer_size: int,
      sow_config: sow_lib.SowConfig,
      dtype: jnp.dtype = jnp.float32,
  ) -> sow_lib.TransformerIntermediates:
    """Initializes the intermediate activations that will be filled."""
    intermediates = sow_lib.TransformerIntermediates()
    residual_stream_dummy = jnp.zeros(
        (batch_size, buffer_size, self.embed_dim),
        dtype=dtype,
    )
    if sow_config.embeddings:
      intermediates.embeddings = residual_stream_dummy
    for layer in self.layers:
      layer_intermediates = sow_lib.LayerIntermediates()
      if sow_config.rs_after_attention:
        layer_intermediates.rs_after_attention = residual_stream_dummy
      if sow_config.rs_after_ffw:
        layer_intermediates.rs_after_ffw = residual_stream_dummy
      if sow_config.attn_logits_topk:
        shape = (
            batch_size,
            buffer_size,
            layer.attn.num_heads,
            sow_config.attn_logits_topk,
        )
        layer_intermediates.attn_logits_topk_values = jnp.zeros(
            shape,
            dtype=dtype,
        )
        layer_intermediates.attn_logits_topk_indices = jnp.zeros(
            shape,
            dtype=jnp.int32,
        )
      if sow_config.mlp_hidden_topk:
        shape = (
            batch_size,
            buffer_size,
            sow_config.mlp_hidden_topk,
        )
        layer_intermediates.mlp_hidden_topk_values = jnp.zeros(
            shape,
            dtype=dtype,
        )
        layer_intermediates.mlp_hidden_topk_indices = jnp.zeros(
            shape,
            dtype=jnp.int32,
        )
      intermediates.layers.append(layer_intermediates)
    return intermediates


def make_causal_attn_mask(
    input_mask: Array,
) -> Array:
  """Attention mask in batch mode.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def build_positions_from_mask(input_mask: Array) -> Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)

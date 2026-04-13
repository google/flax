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
"""Transformer sub-modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from flax import nnx
import layers
import positional_embeddings
import sow_lib
from transformer_cfg import (
    AttentionType,
    DEFAULT_ROPE_BASE_FREQUENCY,
    DEFAULT_ROPE_SCALE_FACTOR,
    ShardingConfig,
    TransformerConfig,
)
import jax
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-importing-member,g-multiple-import

LayerCache = dict[str, Array]
Shape = Sequence[Union[int, Any]]

K_MASK = -2.3819763e38  # Set to a large negative number.


class ScaledEmbed(nnx.Embed):

  def __call__(self, x: Array, out_sharding=None) -> Array:
    x = super().__call__(x, out_sharding)
    x *= jnp.sqrt(x.shape[-1])
    return x

  @property
  def embed_dim(self):
    return self.features

  @property
  def num_embed(self):
    return self.num_embeddings


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      query_pre_attn_scalar: float,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY,
      rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR,
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_qk_norm: bool = False,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
      shd_config: ShardingConfig,
      dtype: jnp.dtype = jnp.float32,
      weight_dtype: jnp.dtype = jnp.float32,
      dropout_rate: float = 0.0,
  ):
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.shd_config = shd_config
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.attn_type = attn_type
    self.sliding_window_size = sliding_window_size
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.attn_vec_einsum = nnx.Einsum(
        einsum_str='BTNH,NHD->BTD',
        kernel_shape=(num_heads, head_dim, features),
        kernel_metadata={
            'out_sharding': shd_config.o_weight_nhd,
        },
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.use_qk_norm = use_qk_norm
    self.sow_config = sow_config
    self.shd_config = shd_config

    self.act_cbtkh_shd = None
    if num_heads == num_kv_heads:
      self.qkv_einsum = nnx.Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          kernel_shape=(3, num_heads, features, head_dim),
          kernel_metadata={
              'out_sharding': shd_config.qkv_weight_cndh,
          },
          dtype=dtype,
          param_dtype=weight_dtype,
          rngs=rngs,
      )
    else:
      if num_heads % num_kv_heads != 0:
        raise ValueError(
          f"Number of query heads ({num_heads}) must be divisible by "
          f"number of key/value heads ({num_kv_heads})."
        )
      self.q_einsum = nnx.Einsum(
          einsum_str='BTD,NDH->BTNH',
          kernel_shape=(num_heads, features, head_dim),
          kernel_metadata={
              'out_sharding': shd_config.q_weight_ndh,
          },
          dtype=dtype,
          param_dtype=weight_dtype,
          rngs=rngs,
      )
      self.kv_einsum = nnx.Einsum(
          einsum_str='BTD,CKDH->CBTKH',
          kernel_shape=(2, num_kv_heads, features, head_dim),
          kernel_metadata={
              'out_sharding': (
                  shd_config.kv_weight_cndh
                  if num_kv_heads > 1
                  else jax.P(None, None, shd_config.fsdp_axis_name, None)
              ),
          },
          dtype=dtype,
          param_dtype=weight_dtype,
          rngs=rngs,
      )
      if shd_config.kv_weight_cndh:
        self.act_cbtkh_shd = (
            self.shd_config.act_sbtnh
            if num_kv_heads > 1
            else jax.P(None, shd_config.fsdp_axis_name, None, None, None)
        )

    if self.use_qk_norm:
      self._query_norm = layers.RMSNorm(
          head_dim,
          dtype=dtype,
          weight_dtype=weight_dtype,
          rngs=rngs,
      )
      self._key_norm = layers.RMSNorm(
          head_dim,
          dtype=dtype,
          weight_dtype=weight_dtype,
          rngs=rngs,
      )

    self.dropout = nnx.Dropout(dropout_rate)

  def __call__(
      self,
      x: Array,
      segment_pos: Array,
      cache: LayerCache | None,
      attn_mask: Array,
      rngs: nnx.Rngs | None = None,
  ) -> tuple[LayerCache | None, Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum(
          x, out_sharding=self.shd_config.act_sbtnh
      )
    else:
      query_proj = self.q_einsum(x, out_sharding=self.shd_config.act_btnh)
      key_proj, value_proj = self.kv_einsum(x, out_sharding=self.act_cbtkh_shd)

    if self.use_qk_norm:
      query_proj = self._query_norm(query_proj)
      key_proj = self._key_norm(key_proj)

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
        max_wavelength=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
        max_wavelength=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    # Cache is left aligned.
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    use_gqa = self.num_heads > self.num_kv_heads and self.num_kv_heads > 1
    if use_gqa:
      # Reshape matrices to enable einsums over groups.
      num_groups = self.num_heads // self.num_kv_heads
      batch_size, seq_size, _, head_dim = query_scaled.shape
      query_scaled = query_scaled.reshape(
        (batch_size, seq_size, self.num_kv_heads, num_groups, head_dim)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      logits = logits.reshape(
        (batch_size, seq_size, self.num_heads, -1)
      )
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'sliding_window_size must be set if attn_type is Local Sliding.'
        )

      # workaround for a bug with triu/tril if all_ones is explicitly sharded
      # all_ones = jnp.ones_like(attn_mask)
      all_ones = jnp.ones(attn_mask.shape, dtype=attn_mask.dtype)
      sliding_mask = jnp.triu(
          all_ones, -1 * self.sliding_window_size + 1
      ) * jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask = sliding_mask * attn_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    self.sow_config.maybe_sow_attn_logits_topk(padded_logits, self)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

    probs = self.dropout(probs, rngs=rngs)

    if use_gqa:
      # Reshape matrices to enable einsums over groups.
      num_groups = self.num_heads // self.num_kv_heads
      batch_size, seq_size1, _, _ = probs.shape
      probs = probs.reshape(
        (batch_size, seq_size1, self.num_kv_heads, num_groups, -1)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      encoded = encoded.reshape(
        (batch_size, seq_size, self.num_heads, head_dim)
      )
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    attn_output = self.attn_vec_einsum(
        encoded, out_sharding=self.shd_config.act_btd
    )

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @property
  def head_dim(self):
    return self.attn_vec_einsum.kernel.shape[1]

  @property
  def num_heads(self):
    return (
        self.qkv_einsum.kernel.shape[1]
        if self.use_qkv_einsum
        else self.q_einsum.kernel.shape[0]
    )

  @property
  def num_kv_heads(self):
    return (
        self.qkv_einsum.kernel.shape[1]
        if self.use_qkv_einsum
        else self.kv_einsum.kernel.shape[1]
    )

  @property
  def use_qkv_einsum(self):
    return hasattr(self, 'qkv_einsum') and self.qkv_einsum is not None

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.float32,
  ) -> LayerCache:
    out_sharding = (
        None if self.act_cbtkh_shd is None else jax.P(*self.act_cbtkh_shd[1:])
    )
    return {
        'v': jnp.zeros(
            (batch_size, cache_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            out_sharding=out_sharding,
        ),
        'k': jnp.zeros(
            (batch_size, cache_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            out_sharding=out_sharding,
        ),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
    }


class FeedForward(nnx.Module):
  """Feed forward module."""

  def __init__(
      self,
      features: int,
      hidden_dim: int,
      *,
      kernel_init: nnx.Initializer = nnx.initializers.normal(),
      rngs: nnx.Rngs,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
      shd_config: ShardingConfig,
      dtype: jnp.dtype = jnp.float32,
      weight_dtype: jnp.dtype = jnp.float32,
      dropout_rate: float = 0.0,
  ):
    self.gate_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=kernel_init,
        kernel_metadata={'out_sharding': shd_config.ffw_weight_df},
        dtype=dtype,
        param_dtype=weight_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=kernel_init,
        kernel_metadata={'out_sharding': shd_config.ffw_weight_df},
        dtype=dtype,
        param_dtype=weight_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=hidden_dim,
        out_features=features,
        use_bias=False,
        rngs=rngs,
        kernel_init=kernel_init,
        kernel_metadata={'out_sharding': shd_config.ffw_weight_fd},
        dtype=dtype,
        param_dtype=weight_dtype,
    )
    self.dropout = nnx.Dropout(dropout_rate)
    self.sow_config = sow_config
    self.shd_config = shd_config

  def __call__(self, x: Array, rngs: nnx.Rngs | None = None) -> Array:
    ff_gate = self.gate_proj(x, out_sharding=self.shd_config.act_btf)
    gate_value = nnx.gelu(ff_gate)

    ff1 = self.up_proj(x, out_sharding=self.shd_config.act_btf)
    activations = gate_value * ff1
    self.sow_config.maybe_sow_mlp_hidden_topk(activations, self)

    activations = self.dropout(activations, rngs=rngs)
    outputs = self.down_proj(activations, out_sharding=self.shd_config.act_btd)
    return outputs


class Block(nnx.Module):
  """Transformer block."""

  def __init__(
      self,
      config: TransformerConfig,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ):
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    embed_dim = config.embed_dim
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    sliding_window_size = config.sliding_window_size
    use_post_attn_norm = config.use_post_attn_norm
    use_post_ffw_norm = config.use_post_ffw_norm
    query_pre_attn_scalar = config.query_pre_attn_scalar()
    if attn_type == AttentionType.LOCAL_SLIDING:
      rope_base_frequency = config.local_base_frequency
      rope_scale_factor = config.local_scale_factor
    else:
      rope_base_frequency = config.global_base_frequency
      rope_scale_factor = config.global_scale_factor

    attn_logits_soft_cap = config.attn_logits_soft_cap
    use_qk_norm = config.use_qk_norm
    dtype = config.dtype
    weight_dtype = config.weight_dtype

    self.pre_attention_norm = layers.RMSNorm(
        embed_dim,
        scale_metadata={'out_sharding': config.shd_config.rms_norm_weight},
        rngs=rngs,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )
    self.attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=embed_dim,
        head_dim=head_dim,
        query_pre_attn_scalar=query_pre_attn_scalar,
        attn_type=attn_type,
        rope_base_frequency=rope_base_frequency,
        rope_scale_factor=rope_scale_factor,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        rngs=rngs,
        use_qk_norm=use_qk_norm,
        sow_config=sow_config,
        shd_config=config.shd_config,
        dtype=dtype,
        weight_dtype=weight_dtype,
        dropout_rate=config.dropout_rate,
    )
    if use_post_attn_norm:
      self.post_attention_norm = layers.RMSNorm(
          embed_dim,
          scale_metadata={'out_sharding': config.shd_config.rms_norm_weight},
          rngs=rngs,
          dtype=dtype,
          weight_dtype=weight_dtype,
      )
    else:
      self.post_attention_norm = None

    self.pre_ffw_norm = layers.RMSNorm(
        embed_dim,
        scale_metadata={'out_sharding': config.shd_config.rms_norm_weight},
        rngs=rngs,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )
    self.mlp = FeedForward(
        features=embed_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
        sow_config=sow_config,
        shd_config=config.shd_config,
        dtype=dtype,
        weight_dtype=weight_dtype,
        dropout_rate=config.dropout_rate,
    )
    if use_post_ffw_norm:
      self.post_ffw_norm = layers.RMSNorm(
          embed_dim,
          scale_metadata={'out_sharding': config.shd_config.rms_norm_weight},
          rngs=rngs,
          dtype=dtype,
          weight_dtype=weight_dtype,
      )
    else:
      self.post_ffw_norm = None

    self.dropout = nnx.Dropout(config.dropout_rate)
    self.sow_config = sow_config

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
      rngs: nnx.Rngs | None = None,
  ) -> tuple[LayerCache | None, jax.Array]:

    # Attention.
    attn_inputs = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        attn_inputs,
        segment_pos,
        cache,
        attn_mask,
        rngs=rngs,
    )
    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)
    x += attn_output
    self.sow_config.maybe_sow_rs_after_attention(x, self)

    # Feed forward.
    ffw_inputs = self.pre_ffw_norm(x)
    ffw_outputs = self.mlp(ffw_inputs, rngs=rngs)
    if self.post_ffw_norm is not None:
      ffw_outputs = self.post_ffw_norm(ffw_outputs)
    x += ffw_outputs
    x = self.dropout(x, rngs=rngs)
    self.sow_config.maybe_sow_rs_after_ffw(x, self)

    return cache, x

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.float32,
  ) -> LayerCache:
    return self.attn.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=dtype,
    )

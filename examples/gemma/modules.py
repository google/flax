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
import enum
from typing import Any, Union

from flax import nnx
import layers
import positional_embeddings
import sow_lib
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike  # pylint: disable=g-importing-member,g-multiple-import

LayerCache = dict[str, Array]
Shape = Sequence[Union[int, Any]]

K_MASK = -2.3819763e38  # Set to a large negative number.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.input_embedding = nnx.Param(
        nn.initializers.normal()(rngs.params(), (vocab_size, embed_dim))
    )

  def encode(self, x: ArrayLike) -> Array:
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    return x

  def decode(self, x: ArrayLike) -> Array:
    return jnp.dot(x, self.input_embedding.value.T)

  @property
  def embed_dim(self):
    return self.input_embedding.value.shape[1]

  @property
  def num_embed(self):
    return self.input_embedding.value.shape[0]


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
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig()
  ):
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.attn_type = attn_type
    self.sliding_window_size = sliding_window_size
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.attn_vec_einsum = layers.Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(num_heads, head_dim, features),
        rngs=rngs,
    )
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.use_qk_norm = use_qk_norm
    self.sow_config = sow_config

    if num_heads == num_kv_heads:
      self.qkv_einsum = layers.Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
      )
    else:
      self.q_einsum = layers.Einsum(
          einsum_str='BTD,NDH->BTNH',
          shape=(num_heads, features, head_dim),
          rngs=rngs,
      )
      self.kv_einsum = layers.Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
      )
    if self.use_qk_norm:
      self._query_norm = layers.RMSNorm(head_dim, rngs=rngs)
      self._key_norm = layers.RMSNorm(head_dim, rngs=rngs)

  def __call__(
      self,
      x: Array,
      segment_pos: Array,
      cache: LayerCache | None,
      attn_mask: Array,
  ) -> tuple[LayerCache | None, Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum(x)
    else:
      query_proj = self.q_einsum(x)
      key_proj, value_proj = self.kv_einsum(x)

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

    logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'sliding_window_size must be set if attn_type is Local Sliding.'
        )

      all_ones = jnp.ones_like(attn_mask)
      sliding_mask = jnp.triu(
          all_ones, -1 * self.sliding_window_size + 1
      ) * jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask = sliding_mask * attn_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    self.sow_config.maybe_sow_attn_logits_topk(padded_logits, self)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    attn_output = self.attn_vec_einsum(encoded)

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
    return self.attn_vec_einsum.shape[1]

  @property
  def num_heads(self):
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.q_einsum.shape[0]
    )

  @property
  def num_kv_heads(self):
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.kv_einsum.shape[1]
    )

  @property
  def use_qkv_einsum(self):
    return hasattr(self, 'qkv_einsum') and self.qkv_einsum is not None

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    return {
        'v': jnp.zeros(
            (batch_size, cache_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
        ),
        'k': jnp.zeros(
            (batch_size, cache_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
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
      rngs: nnx.Rngs,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig()
  ):
    self.gate_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nn.initializers.normal(),
    )
    self.up_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nn.initializers.normal(),
    )
    self.down_proj = nnx.Linear(
        in_features=hidden_dim,
        out_features=features,
        use_bias=False,
        rngs=rngs,
        kernel_init=nn.initializers.normal(),
    )
    self.sow_config = sow_config

  def __call__(self, x: ArrayLike) -> Array:
    ff_gate = self.gate_proj(x)
    gate_value = nnx.gelu(ff_gate)

    ff1 = self.up_proj(x)
    activations = gate_value * ff1
    self.sow_config.maybe_sow_mlp_hidden_topk(activations, self)

    outputs = self.down_proj(activations)
    return outputs


class Block(nnx.Module):
  """Transformer block."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      embed_dim: int,
      head_dim: int,
      hidden_dim: int,
      use_post_attn_norm: bool,
      use_post_ffw_norm: bool,
      query_pre_attn_scalar: float,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY,
      rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR,
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_qk_norm: bool = False,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig()
  ):
    self.pre_attention_norm = layers.RMSNorm(embed_dim, rngs=rngs)
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
    )
    if use_post_attn_norm:
      self.post_attention_norm = layers.RMSNorm(embed_dim, rngs=rngs)
    else:
      self.post_attention_norm = None

    self.pre_ffw_norm = layers.RMSNorm(embed_dim, rngs=rngs)
    self.mlp = FeedForward(
        features=embed_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
        sow_config=sow_config,
    )
    if use_post_ffw_norm:
      self.post_ffw_norm = layers.RMSNorm(embed_dim, rngs=rngs)
    else:
      self.post_ffw_norm = None
    self.sow_config = sow_config

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:

    # Attention.
    attn_inputs = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        attn_inputs,
        segment_pos,
        cache,
        attn_mask,
    )
    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)
    x += attn_output
    self.sow_config.maybe_sow_rs_after_attention(x, self)

    # Feed forward.
    ffw_inputs = self.pre_ffw_norm(x)
    ffw_outputs = self.mlp(ffw_inputs)
    if self.post_ffw_norm is not None:
      ffw_outputs = self.post_ffw_norm(ffw_outputs)
    x += ffw_outputs
    self.sow_config.maybe_sow_rs_after_ffw(x, self)

    return cache, x

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    return self.attn.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=dtype,
    )

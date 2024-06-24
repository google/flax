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

import enum
from typing import Any, Sequence, Union

from flax import nnx
import flax.linen as nn
import layers
import positional_embeddings
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike  # pylint: disable=g-importing-member,g-multiple-import

LayerCache = dict[str, Array]
Shape = Sequence[Union[int, Any]]

K_MASK = -2.3819763e38  # Set to a large negative number.


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
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      attn_logits_soft_cap: Union[float, None] = None,
      sliding_window_size: Union[int, None] = None,
  ):
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.attn_type = attn_type
    self.sliding_window_size = sliding_window_size
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.attn_vec_einsum = layers.Einsum(
        shape=(num_heads, head_dim, features),
        rngs=rngs,
    )

    if num_heads == num_kv_heads:
      self.qkv_einsum = layers.Einsum(
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
      )
    else:
      self.q_einsum = layers.Einsum(
          shape=(num_heads, features, head_dim),
          rngs=rngs,
      )
      self.kv_einsum = layers.Einsum(
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
      )

  def __call__(
      self,
      x: Array,
      segment_pos: Array,
      cache: Union[LayerCache, None],
      attn_mask: Array,
  ) -> tuple[Union[LayerCache, None], Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.head_dim**-0.5
    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
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
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

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
  ):
    self.gating_einsum = nnx.Param(
        nn.initializers.zeros_init()(
            rngs.params(),
            ((2, features, hidden_dim)),
        )
    )
    self.linear = nnx.Param(
        nn.initializers.zeros_init()(
            rngs.params(),
            (hidden_dim, features),
        )
    )

  def __call__(self, x: ArrayLike) -> Array:
    ff_gate = jnp.dot(x, self.gating_einsum.value[0])
    gate_value = nnx.gelu(ff_gate)

    ff1 = jnp.dot(x, self.gating_einsum.value[1])
    activations = gate_value * ff1

    outputs = jnp.dot(activations, self.linear.value)
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
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      attn_logits_soft_cap: Union[float, None] = None,
      sliding_window_size: Union[int, None] = None,
  ):
    self.pre_attention_norm = layers.RMSNorm(embed_dim, rngs=rngs)
    self.attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=embed_dim,
        head_dim=head_dim,
        attn_type=attn_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        rngs=rngs,
    )
    if use_post_attn_norm:
      self.post_attn_norm = layers.RMSNorm(embed_dim, rngs=rngs)

    self.pre_ffw_norm = layers.RMSNorm(embed_dim, rngs=rngs)
    self.mlp = FeedForward(
        features=embed_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
    )
    if use_post_ffw_norm:
      self.post_ffw_norm = layers.RMSNorm(embed_dim, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: Union[LayerCache, None],
      attn_mask: jax.Array,
  ) -> tuple[Union[LayerCache, None], jax.Array]:
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.pre_ffw_norm(attn_output)

    if self.use_post_attn_norm:
      attn_output = self.post_attn_norm(attn_output)

    outputs = self.mlp(attn_output)
    if self.use_post_ffw_norm:
      outputs = self.post_ffw_norm(outputs)
    outputs = residual + outputs
    return cache, outputs

  @property
  def use_post_attn_norm(self):
    return hasattr(self, 'post_attn_norm') and self.post_attn_norm is not None

  @property
  def use_post_ffw_norm(self):
    return hasattr(self, 'post_ffw_norm') and self.post_ffw_norm is not None

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

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
"""JAX DeepSeek model."""

import math
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
  _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
  BaseModelOutputWithPast,
  CausalLMOutputWithPast,
  SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
  add_start_docstrings,
  add_start_docstrings_to_model_forward,
  logging,
  replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from config import Config
from einop import einop
import einx



# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
  if not is_torch_greater_or_equal_than_1_13:
    import torch.fx

  _prepare_4d_causal_attention_mask = torch.fx.wrap(
    _prepare_4d_causal_attention_mask
  )


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = 'Config'


class RMSNorm(nnx.Module):
  def __init__(self, hidden_size, eps=1e-6):
    """
    DeepseekV3RMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.weight = nnx.Param(jnp.ones((hidden_size,)))
    self.variance_epsilon = eps

  def __call__(self, hidden_states: jax.Array):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.pow(hidden_states, 2).mean(-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(
      variance + self.variance_epsilon
    )
    return self.weight.value * hidden_states.astype(input_dtype)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
  num_rotations: int,
  dim: int,
  base: float = 10000,
  max_position_embeddings: int = 2048,
):
  return (
    dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
  ) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(
  low_rot: int,
  high_rot: int,
  dim: int,
  base: float = 10000,
  max_position_embeddings: int = 2048,
):
  low = math.floor(
    yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
  )
  high = math.ceil(
    yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
  )
  return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale: float = 1, mscale: float = 1):
  if scale <= 1:
    return 1.0
  return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
  if min == max:
    max += 0.001  # Prevent singularity

  linear_func: jax.Array = (jnp.arange(dim, dtype=jnp.float32) - min) / (
    max - min
  )
  ramp_func = jnp.clip(linear_func, 0, 1)
  return ramp_func

class Buffer(nnx.Variable[nnx.A]):
  pass

class YarnRotaryEmbedding(nnx.Module):
  def __init__(
    self,
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    mscale: float = 1,
    mscale_all_dim: int = 0,
  ):
    self.scaling_factor = scaling_factor
    self.original_max_position_embeddings = original_max_position_embeddings
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    self.mscale = mscale
    self.mscale_all_dim = mscale_all_dim
    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq: jax.Array = 1.0 / (
      self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
    )
    self.inv_freq = Buffer(inv_freq)

    # Build here to make `torch.jit.trace` work.
    seq_len = max_position_embeddings
    dtype = jnp.float32
    self.max_seq_len_cached = seq_len
    dim = self.dim

    freq_extra = 1.0 / (
      self.base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    freq_inter = 1.0 / (
      self.scaling_factor
      * self.base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )

    low, high = yarn_find_correction_range(
      self.beta_fast,
      self.beta_slow,
      dim,
      self.base,
      self.original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).astype(
      dtype=jnp.float32
    )
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    self.inv_freq = Buffer(inv_freq)

    t = jnp.arange(seq_len, dtype=jnp.float32)

    freqs = jnp.outer(t, inv_freq)

    _mscale = float(
      yarn_get_mscale(self.scaling_factor, self.mscale)
      / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
    )

    emb = jnp.concatenate((freqs, freqs), axis=-1)
    self.cos_cached = Buffer((jnp.cos(emb) * _mscale).astype(dtype))
    self.sin_cached = Buffer((jnp.sin(emb) * _mscale).astype(dtype))
    self.max_seq_len_cached = None

  def __call__(self, x: jax.Array, seq_len: int | None = None):
    return (
      self.cos_cached.value[:seq_len].astype(dtype=x.dtype),
      self.sin_cached.value[:seq_len].astype(dtype=x.dtype),
    )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
  """Applies Rotary Position Embedding to the query and key tensors.
  Args:
      q (`torch.Tensor`): The query tensor.
      k (`torch.Tensor`): The key tensor.
      cos (`torch.Tensor`): The cosine part of the rotary embedding.
      sin (`torch.Tensor`): The sine part of the rotary embedding.
      position_ids (`torch.Tensor`):
          The position indices of the tokens corresponding to the query and key tensors. For example, this can be
          used to pass offsetted position ids when working with a KV-cache.
      unsqueeze_dim (`int`, *optional*, defaults to 1):
          The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
          sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
          that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
          k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
          cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
          the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
  Returns:
      `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
  """
  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
  sin = sin[position_ids].unsqueeze(unsqueeze_dim)

  b, h, s, d = q.shape
  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  b, h, s, d = k.shape
  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class MLP(nnx.Module):
  def __init__(
    self,
    config: Config,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.config = config
    self.hidden_size = (
      config.hidden_size if hidden_size is None else hidden_size
    )
    self.intermediate_size = (
      config.intermediate_size
      if intermediate_size is None
      else intermediate_size
    )

    self.gate_proj = nnx.Linear(
      self.hidden_size, self.intermediate_size, use_bias=False, rngs=rngs
    )
    self.up_proj = nnx.Linear(
      self.hidden_size, self.intermediate_size, use_bias=False, rngs=rngs
    )
    self.down_proj = nnx.Linear(
      self.intermediate_size, self.hidden_size, use_bias=False, rngs=rngs
    )
    self.act_fn: Callable[[jax.Array], jax.Array] = (
      getattr(nnx, config.hidden_act)
      if isinstance(config.hidden_act, str)
      else config.hidden_act
    )

  def __call__(self, x: jax.Array):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array | None = None):
  # y = jnp.dot(x, weight)
  y = einx.dot('... j, j k -> ... k', x, weight)
  if bias is not None:
    y = einx.add('... k, k -> ... k', y, bias)
  return y


class MoEGate(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    super().__init__()
    self.config = config
    self.top_k = config.num_experts_per_tok
    self.n_routed_experts = config.n_routed_experts
    self.routed_scaling_factor = config.routed_scaling_factor
    self.scoring_func = config.scoring_func
    self.seq_aux = config.seq_aux
    self.topk_method = config.topk_method
    self.n_group = config.n_group
    self.topk_group = config.topk_group

    # topk selection algorithm
    self.norm_topk_prob = config.norm_topk_prob
    self.gating_dim = config.hidden_size
    self.weight = nnx.Param(jnp.empty((self.gating_dim, self.n_routed_experts)))
    if self.topk_method != 'noaux_tc':
      raise NotImplementedError(
        f'insupportable TopK function for MoE gating: {self.topk_method}'
      )
    if self.scoring_func != 'sigmoid':
      raise NotImplementedError(
        f'insupportable scoring function for MoE gating: {self.scoring_func}'
      )
    self.e_score_correction_bias = nnx.Param(
      jnp.empty((self.n_routed_experts,))
    )
    self.weight.value = nnx.initializers.kaiming_uniform()(
      rngs.params(), self.weight.value.shape, self.weight.value.dtype
    )

  def __call__(self, hidden_states: jax.Array):
    # [b, n, h]
    bsz, seq_len, h = hidden_states.shape
    ### compute gating score
    # [b * n, h]
    # hidden_states = hidden_states.reshape(-1, h)
    hidden_states = einop(hidden_states, '... h -> (...) h')
    # [b * n, e]
    logits = linear(
      hidden_states.astype(jnp.float32),
      self.weight.value.astype(jnp.float32),
      None,
    )
    # [b * n, e]
    scores: jax.Array = nnx.sigmoid(logits)
    ### select top-k experts
    # [b * n, e]
    scores_for_choice: jax.Array = scores + self.e_score_correction_bias[None]
    # [b * n, n_group, e_g]
    # group_features = scores_for_choice.reshape(bsz * seq_len, self.n_group, -1)
    group_features = einop(
      scores_for_choice, 'bn (g e_g) -> bn g e_g', g=self.n_group
    )
    # [b * n, n_group]
    group_scores = jax.lax.top_k(group_features, 2)[0].sum(axis=-1)
    # [n, top_k_group]
    _, group_idx = jax.lax.top_k(group_scores, k=self.topk_group)
    # [b * n, n_group]
    group_mask = jnp.zeros_like(group_scores)
    # [b * n, n_group]
    group_mask = jnp.put_along_axis(
      group_mask, group_idx, values=1, axis=1, inplace=False
    )
    # [b * n, e]
    score_mask = jnp.broadcast_to(
      group_mask[..., None],
      (bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group),
    ).reshape(bsz * seq_len, -1)
    # [b * n, e]
    tmp_scores = jnp.where(score_mask.astype(jnp.bool), scores_for_choice, 0.0)
    # [b * n, top_k]
    _, topk_idx = jax.lax.top_k(tmp_scores, k=self.top_k)
    # [b * n, top_k]
    topk_weight = jnp.take_along_axis(scores_for_choice, topk_idx, axis=-1)
    # [b * n, e]
    ### norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
      denominator = topk_weight.sum(axis=-1, keepdims=True) + 1e-20
      topk_weight = topk_weight / denominator
    topk_weight = (
      topk_weight * self.routed_scaling_factor
    )  # must multiply the scaling factor
    return topk_idx, topk_weight


class MoE(nnx.Module):
  """
  A mixed expert module containing shared experts.
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    super().__init__()
    self.config = config
    self.num_experts_per_tok = config.num_experts_per_tok
    self.experts = [
      MLP(config, intermediate_size=config.moe_intermediate_size, rngs=rngs)
      for i in range(config.n_routed_experts)
    ]

    self.gate = MoEGate(config, rngs=rngs)
    if config.n_shared_experts is not None:
      intermediate_size = config.moe_intermediate_size * config.n_shared_experts
      self.shared_experts = MLP(
        config=config, intermediate_size=intermediate_size, rngs=rngs
      )
    else:
      self.shared_experts = None

  def __call__(
    self,
    x: jax.Array,  # [b, n, h]
  ):
    b, n, h = x.shape
    x0 = x
    # [b * n, top_k], [b * n, top_k]
    topk_idx, topk_weight = self.gate(x)
    # [b * n, h]
    # x = x.reshape(-1, x.shape[-1])
    x = einop(x, 'b n h -> (b n) h')
    # [b * n, e, h]
    experts_y = jnp.stack([expert(x) for expert in self.experts], axis=1)
    # [b * n, top_k, h]
    masked_y = jnp.take_along_axis(experts_y, topk_idx[:, :, None], axis=1)
    # [b, n, h]
    y = einx.dot(
      '(b n) top_k h, (b n) top_k -> b n h', masked_y, topk_weight, b=b
    )
    if self.shared_experts is not None:
      y = y + self.shared_experts(x0)
    return y

  def moe_infer(
    self,
    x: jax.Array,  # [b * n, h]
    topk_ids: jax.Array,  # [b * n, top_k]
    topk_weight: jax.Array,  # [b * n, top_k]
  ):
    # [b * n, e, h]
    experts_y = jnp.stack([expert(x) for expert in self.experts], axis=1)
    # [b * n, top_k, h]
    masked_y = jnp.take_along_axis(experts_y, topk_ids[:, :, None], axis=1)
    # [b * n, h]
    y = jnp.sum(masked_y * topk_weight[:, :, None], axis=1)
    return y


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class Attention(nnx.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  def __init__(self, config: Config, layer_idx: int, *, rngs: nnx.Rngs):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx

    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads

    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.q_lora_rank = config.q_lora_rank
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.kv_lora_rank = config.kv_lora_rank
    self.v_head_dim = config.v_head_dim
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.is_causal = True

    if self.q_lora_rank is None:
      self.q_proj = nnx.Linear(
        self.hidden_size,
        self.num_heads * self.q_head_dim,
        use_bias=False,
        rngs=rngs,
      )
    else:
      self.q_a_proj = nnx.Linear(
        self.hidden_size,
        self.q_lora_rank,
        use_bias=config.attention_bias,
        rngs=rngs,
      )
      self.q_a_layernorm = RMSNorm(config.q_lora_rank)
      self.q_b_proj = nnx.Linear(
        self.q_lora_rank,
        self.num_heads * self.q_head_dim,
        use_bias=False,
        rngs=rngs,
      )

    self.kv_a_proj_with_mqa = nnx.Linear(
      self.hidden_size,
      config.kv_lora_rank + config.qk_rope_head_dim,
      use_bias=config.attention_bias,
      rngs=rngs,
    )
    self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
    self.kv_b_proj = nnx.Linear(
      config.kv_lora_rank,
      self.num_heads
      * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
      use_bias=False,
      rngs=rngs,
    )

    self.o_proj = nnx.Linear(
      self.num_heads * self.v_head_dim,
      self.hidden_size,
      use_bias=config.attention_bias,
      rngs=rngs,
    )
    if (
      self.config.rope_scaling is None
      or self.config.rope_scaling['type'] != 'yarn'
    ):
      raise ValueError('Only "yarn" scaling is supported for "rope_scaling".')

    scaling_factor = self.config.rope_scaling['factor']
    kwargs = {
      key: self.config.rope_scaling[key]
      for key in [
        'original_max_position_embeddings',
        'beta_fast',
        'beta_slow',
        'mscale',
        'mscale_all_dim',
      ]
      if key in self.config.rope_scaling
    }
    self.rotary_emb = YarnRotaryEmbedding(
      self.qk_rope_head_dim,
      max_position_embeddings=self.max_position_embeddings,
      scaling_factor=scaling_factor,
      base=self.rope_theta,
      **kwargs,
    )

    self.softmax_scale = self.q_head_dim ** (-0.5)
    mscale_all_dim = self.config.rope_scaling.get('mscale_all_dim', 0)
    scaling_factor = self.config.rope_scaling['factor']
    if mscale_all_dim:
      mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
      self.softmax_scale = self.softmax_scale * mscale * mscale

  def __call__(
    self,
    hidden_states: jax.Array,
    attention_mask: jax.Array | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
  ) -> tuple[jax.Array, jax.Array | None, tuple[jax.Array] | None]:
    if 'padding_mask' in kwargs:
      warnings.warn(
        'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
      )
    bsz, q_len, _ = hidden_states.shape

    if self.q_lora_rank is None:
      q = self.q_proj(hidden_states)
    else:
      q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # q = einop(q, 'b n (head q_h) -> b head n q_h', num_heads=self.num_heads)
    q_nope, q_pe = jnp.split(
      q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = jnp.split(
      compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1
    )
    k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # k_pe = einop(k_pe, 'b n (head q_h) -> b head n q_h', num_heads=1)
    kv = (
      self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
      .reshape(
        bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
      )
      .transpose(1, 2)
    )

    k_nope, value_states = jnp.split(
      kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1
    )
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
      if self.layer_idx is None:
        raise ValueError(
          f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
          'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
          'with a layer index.'
        )
      kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
      cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
      key_states, value_states = past_key_value.update(
        key_states, value_states, self.layer_idx, cache_kwargs
      )

    attn_weights = (
      torch.matmul(query_states, key_states.transpose(2, 3))
      * self.softmax_scale
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
      raise ValueError(
        f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
        f' {attn_weights.size()}'
      )
    assert attention_mask is not None
    if attention_mask is not None:
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        raise ValueError(
          f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
        )
      attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
      attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
      attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
      raise ValueError(
        f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is'
        f' {attn_output.size()}'
      )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(
      bsz, q_len, self.num_heads * self.v_head_dim
    )

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
      attn_weights = None

    return attn_output, attn_weights, past_key_value



class DeepseekV3DecoderLayer(nnx.Module):
  def __init__(self, config: Config, layer_idx: int, *, rngs: nnx.Rngs):
    super().__init__()
    self.hidden_size = config.hidden_size

    self.self_attn = Attention(config=config, layer_idx=layer_idx, rngs=rngs)

    self.mlp = (
      MoE(config)
      if (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
      )
      else MLP(config)
    )
    self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )

  def __call__(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: tuple[torch.Tensor] | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    **kwargs,
  ) -> tuple[
    torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
  ]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if 'padding_mask' in kwargs:
      warnings.warn(
        'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
      )
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_value=past_key_value,
      output_attentions=output_attentions,
      use_cache=use_cache,
      **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (self_attn_weights,)

    if use_cache:
      outputs += (present_key_value,)

    return outputs


DeepseekV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
  'The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.',
  DeepseekV3_START_DOCSTRING,
)
class DeepseekV3PreTrainedModel(PreTrainedModel):
  config_class = Config
  base_model_prefix = 'model'
  supports_gradient_checkpointing = True
  _no_split_modules = ['DeepseekV3DecoderLayer']
  _skip_keys_device_placement = 'past_key_values'
  _supports_flash_attn_2 = True
  _supports_cache_class = True

  def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=std)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()


DeepseekV3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
  'The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.',
  DeepseekV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
  """
  Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]
  Args:
      config: Config
  """

  def __init__(self, config: Config):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(
      config.vocab_size, config.hidden_size, self.padding_idx
    )
    self.layers = [
      DeepseekV3DecoderLayer(config, layer_idx)
      for layer_idx in range(config.num_hidden_layers)
    ]
    self._use_flash_attention_2 = (
      config._attn_implementation == 'flash_attention_2'
    )
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, value):
    self.embed_tokens = value

  @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
  ) -> tuple | BaseModelOutputWithPast:
    output_attentions = (
      output_attentions
      if output_attentions is not None
      else self.config.output_attentions
    )
    output_hidden_states = (
      output_hidden_states
      if output_hidden_states is not None
      else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
      return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
      raise ValueError(
        'You cannot specify both input_ids and inputs_embeds at the same time'
      )
    elif input_ids is not None:
      batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
      batch_size, seq_length = inputs_embeds.shape[:2]
    else:
      raise ValueError('You have to specify either input_ids or inputs_embeds')

    past_key_values_length = 0
    if use_cache:
      use_legacy_cache = not isinstance(past_key_values, Cache)
      if use_legacy_cache:
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
      past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
      device = (
        input_ids.device if input_ids is not None else inputs_embeds.device
      )
      position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
      )
      position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
      inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
      # 2d mask is passed through the layers
      attention_mask = (
        attention_mask
        if (attention_mask is not None and 0 in attention_mask)
        else None
      )
    else:
      # 4d mask is passed through the layers
      attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
      )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
      if output_hidden_states:
        all_hidden_states += (hidden_states,)

      layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
      )

      hidden_states = layer_outputs[0]

      if use_cache:
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

      if output_attentions:
        all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
      all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
      next_cache = (
        next_decoder_cache.to_legacy_cache()
        if use_legacy_cache
        else next_decoder_cache
      )
    if not return_dict:
      return tuple(
        v
        for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
        if v is not None
      )
    return BaseModelOutputWithPast(
      last_hidden_state=hidden_states,
      past_key_values=next_cache,
      hidden_states=all_hidden_states,
      attentions=all_self_attns,
    )


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
  _tied_weights_keys = ['lm_head.weight']

  def __init__(self, config):
    super().__init__(config)
    self.model = DeepseekV3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def set_decoder(self, decoder):
    self.model = decoder

  def get_decoder(self):
    return self.model

  @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
  @replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
  )
  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
  ) -> tuple | CausalLMOutputWithPast:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.
    Returns:
    Example:
    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM
    >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")
    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = (
      output_attentions
      if output_attentions is not None
      else self.config.output_attentions
    )
    output_hidden_states = (
      output_hidden_states
      if output_hidden_states is not None
      else self.config.output_hidden_states
    )
    return_dict = (
      return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      shift_logits = shift_logits.view(-1, self.config.vocab_size)
      shift_labels = shift_labels.view(-1)
      # Enable model parallelism
      shift_labels = shift_labels.to(shift_logits.device)
      loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
      output = (logits,) + outputs[1:]
      return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
      loss=loss,
      logits=logits,
      past_key_values=outputs.past_key_values,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )

  def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
  ):
    if past_key_values is not None:
      if isinstance(past_key_values, Cache):
        cache_length = past_key_values.get_seq_length()
        past_length = past_key_values.seen_tokens
        max_cache_length = past_key_values.get_max_length()
      else:
        cache_length = past_length = past_key_values[0][0].shape[2]
        max_cache_length = None

      # Keep only the unprocessed tokens:
      # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
      # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
      # input)
      if (
        attention_mask is not None
        and attention_mask.shape[1] > input_ids.shape[1]
      ):
        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
      # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
      # input_ids based on the past_length.
      elif past_length < input_ids.shape[1]:
        input_ids = input_ids[:, past_length:]
      # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

      # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
      if (
        max_cache_length is not None
        and attention_mask is not None
        and cache_length + input_ids.shape[1] > max_cache_length
      ):
        attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      if past_key_values:
        position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
      model_inputs = {'inputs_embeds': inputs_embeds}
    else:
      model_inputs = {'input_ids': input_ids}

    model_inputs.update(
      {
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'use_cache': kwargs.get('use_cache'),
        'attention_mask': attention_mask,
      }
    )
    return model_inputs

  @staticmethod
  def _reorder_cache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
      reordered_past += (
        tuple(
          past_state.index_select(0, beam_idx.to(past_state.device))
          for past_state in layer_past
        ),
      )
    return reordered_past


@add_start_docstrings(
  """
    The DeepseekV3 Model transformer with a sequence classification head on top (linear layer).
    [`DeepseekV3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
  DeepseekV3_START_DOCSTRING,
)
class DeepseekV3ForSequenceClassification(DeepseekV3PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.model = DeepseekV3Model(config)
    self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    # Initialize weights and apply final processing
    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
  ) -> tuple | SequenceClassifierOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = (
      return_dict if return_dict is not None else self.config.use_return_dict
    )

    transformer_outputs = self.model(
      input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      inputs_embeds=inputs_embeds,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    if input_ids is not None:
      batch_size = input_ids.shape[0]
    else:
      batch_size = inputs_embeds.shape[0]

    if self.config.pad_token_id is None and batch_size != 1:
      raise ValueError(
        'Cannot handle batch sizes > 1 if no padding token is defined.'
      )
    if self.config.pad_token_id is None:
      sequence_lengths = -1
    else:
      if input_ids is not None:
        sequence_lengths = (
          torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        ).to(logits.device)
      else:
        sequence_lengths = -1

    pooled_logits = logits[
      torch.arange(batch_size, device=logits.device), sequence_lengths
    ]

    loss = None
    if labels is not None:
      labels = labels.to(logits.device)
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = 'regression'
        elif self.num_labels > 1 and (
          labels.dtype == torch.long or labels.dtype == torch.int
        ):
          self.config.problem_type = 'single_label_classification'
        else:
          self.config.problem_type = 'multi_label_classification'

      if self.config.problem_type == 'regression':
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(pooled_logits, labels)
      elif self.config.problem_type == 'single_label_classification':
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
          pooled_logits.view(-1, self.num_labels), labels.view(-1)
        )
      elif self.config.problem_type == 'multi_label_classification':
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    if not return_dict:
      output = (pooled_logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
      loss=loss,
      logits=pooled_logits,
      past_key_values=transformer_outputs.past_key_values,
      hidden_states=transformer_outputs.hidden_states,
      attentions=transformer_outputs.attentions,
    )

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
"""Model in PyTorch."""
import math
from typing import Literal, Optional, Tuple
from absl import app
from absl import flags
from configs.base import ModelArgs
from kernel_pytorch import act_quant, fp8_gemm, weight_dequant
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


class ParallelEmbedding(nn.Module):
  """Embedding layer with parallelism support across distributed processes.

  Args:
      vocab_size (int): Vocabulary size.
      dim (int): Embedding dimension.
  """

  def __init__(self, vocab_size: int, dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    assert vocab_size % world_size == 0
    self.part_vocab_size = vocab_size // world_size
    self.vocab_start_idx = rank * self.part_vocab_size
    self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
    self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for parallel embedding layer.

    Args:
        x (torch.Tensor): Input tensor containing token indices.

    Returns:
        torch.Tensor: Embedded representations.

    Raises:
        ValueError: If `world_size` is not defined.
    """
    if world_size > 1:
      mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
      x = x - self.vocab_start_idx
      x[mask] = 0
    y = F.embedding(x, self.weight)
    if world_size > 1:
      y[mask] = 0
      dist.all_reduce(y)
    return y


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
  """Applies a linear transformation to the incoming data: y = xA^T + b.

  This function supports specialized implementations based on quantization and
  tensor formats.

  Args:
      x (torch.Tensor): The input tensor.
      weight (torch.Tensor): The weight tensor. It may be quantized and requires
        dequantization for certain cases.
      bias (Optional[torch.Tensor]): The bias tensor to be added. Default is
        None.

  Returns:
      torch.Tensor: The result of the linear transformation, which may involve
      quantization-aware computations depending on the input parameters.

  Notes:
      - If `weight` is quantized (e.g., `element_size() > 1`), a dequantized
      version
        is used for computation.
      - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are
      applied.
      - For other cases, the function applies quantization to `x` and uses
      `fp8_gemm` for computation.
  """
  if weight.element_size() > 1:
    return F.linear(x, weight, bias)
  elif gemm_impl == "bf16":
    weight = weight_dequant(weight, weight.scale)
    return F.linear(x, weight, bias)
  else:
    x, scale = act_quant(x, block_size)
    y = fp8_gemm(x, scale, weight, weight.scale)
    if bias is not None:
      y += bias
    return y


class Linear(nn.Module):
  """Custom linear layer with support for quantized weights and optional bias.

  Args:
      in_features (int): Number of input features.
      out_features (int): Number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
  """

  dtype = torch.bfloat16

  def __init__(
      self, in_features: int, out_features: int, bias: bool = False, dtype=None
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(
        torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
    )
    if self.weight.element_size() == 1:
      scale_out_features = (out_features + block_size - 1) // block_size
      scale_in_features = (in_features + block_size - 1) // block_size
      self.weight.scale = self.scale = nn.Parameter(
          torch.empty(
              scale_out_features, scale_in_features, dtype=torch.float32
          )
      )
    else:
      self.register_parameter("scale", None)
    if bias:
      self.bias = nn.Parameter(torch.empty(self.part_out_features))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the custom linear layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor after linear computation.
    """
    return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
  """Linear layer with column parallelism, splitting output features across distributed processes.

  Args:
      in_features (int): Number of input features.
      out_features (int): Total number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
  """

  def __init__(
      self, in_features: int, out_features: int, bias: bool = False, dtype=None
  ):
    assert out_features % world_size == 0
    self.part_out_features = out_features // world_size
    super().__init__(in_features, self.part_out_features, bias, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for column parallel linear layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor with column-parallel computation.
    """
    y = linear(x, self.weight, self.bias)
    return y


class RowParallelLinear(Linear):
  """Linear layer with row parallelism, splitting input features across distributed processes.

  Args:
      in_features (int): Total number of input features.
      out_features (int): Number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
  """

  def __init__(
      self, in_features: int, out_features: int, bias: bool = False, dtype=None
  ):
    assert in_features % world_size == 0
    self.part_in_features = in_features // world_size
    super().__init__(self.part_in_features, out_features, bias, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for row parallel linear layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor with row-parallel computation.
    """
    y = linear(x, self.weight)
    if world_size > 1:
      dist.all_reduce(y)
    if self.bias is not None:
      y += self.bias
    return y


class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization (RMSNorm).

  Args:
      dim (int): Dimension of the input tensor.
      eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor):
    """Forward pass for RMSNorm.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """
    return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
  """Precomputes frequency-based complex exponential values for rotary positional embeddings.

  Args:
      args (ModelArgs): Model arguments containing positional embedding
        parameters.

  Returns:
      torch.Tensor: Precomputed complex exponential values for positional
      embeddings.
  """
  dim = args.qk_rope_head_dim
  seqlen = args.max_seq_len
  beta_fast = args.beta_fast
  beta_slow = args.beta_slow
  base = args.rope_theta
  factor = args.rope_factor

  def find_correction_dim(num_rotations, dim, base, max_seq_len):
    """Computes the correction dimension for a given number of rotations in the rotary positional embedding.

    Args:
        num_rotations (float): Number of rotations to compute the correction
          for.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.

    Returns:
        float: The correction dimension based on the input parameters.
    """
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )

  def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    """Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.

    Returns:
        Tuple[int, int]: The range of correction dimensions (low, high), clamped
        to valid indices.
    """
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min, max, dim):
    """Computes a linear ramp function used to smooth values between a minimum and maximum range.

    Args:
        min (float): Minimum value for the ramp function.
        max (float): Maximum value for the ramp function.
        dim (int): Dimensionality of the ramp tensor.

    Returns:
        torch.Tensor: A tensor of shape (dim,) with values linearly interpolated
        between 0 and 1,
            clamped to the range [0, 1].
    """
    if min == max:
      max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

  freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  if seqlen > args.original_seq_len:
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, args.original_seq_len
    )
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth

  t = torch.arange(seqlen)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """Applies rotary positional embeddings to the input tensor.

  Args:
      x (torch.Tensor): Input tensor with positional embeddings to be applied.
      freqs_cis (torch.Tensor): Precomputed complex exponential values for
        positional embeddings.

  Returns:
      torch.Tensor: Tensor with rotary embeddings applied.
  """
  dtype = x.dtype
  x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
  freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
  y = torch.view_as_real(x * freqs_cis).flatten(3)
  return y.to(dtype)


class MLA(nn.Module):
  """Multi-Headed Attention Layer (MLA).

  Attributes:
      dim (int): Dimensionality of the input features.
      n_heads (int): Number of attention heads.
      n_local_heads (int): Number of local attention heads for distributed
        systems.
      q_lora_rank (int): Rank for low-rank query projection.
      kv_lora_rank (int): Rank for low-rank key/value projection.
      qk_nope_head_dim (int): Dimensionality of non-positional query/key
        projections.
      qk_rope_head_dim (int): Dimensionality of rotary-positional query/key
        projections.
      qk_head_dim (int): Total dimensionality of query/key projections.
      v_head_dim (int): Dimensionality of value projections.
      softmax_scale (float): Scaling factor for softmax in attention
        computation.
  """

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.dim = args.dim
    self.n_heads = args.n_heads
    self.n_local_heads = args.n_heads // world_size
    self.q_lora_rank = args.q_lora_rank
    self.kv_lora_rank = args.kv_lora_rank
    self.qk_nope_head_dim = args.qk_nope_head_dim
    self.qk_rope_head_dim = args.qk_rope_head_dim
    self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    self.v_head_dim = args.v_head_dim

    if self.q_lora_rank == 0:
      self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
    else:
      self.wq_a = Linear(self.dim, self.q_lora_rank)
      self.q_norm = RMSNorm(self.q_lora_rank)
      self.wq_b = ColumnParallelLinear(
          self.q_lora_rank, self.n_heads * self.qk_head_dim
      )
    self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
    self.kv_norm = RMSNorm(self.kv_lora_rank)
    self.wkv_b = ColumnParallelLinear(
        self.kv_lora_rank,
        self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
    )
    self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
    self.softmax_scale = self.qk_head_dim**-0.5
    if args.max_seq_len > args.original_seq_len:
      mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    if attn_impl == "naive":
      self.register_buffer(
          "k_cache",
          torch.zeros(
              args.max_batch_size,
              args.max_seq_len,
              self.n_local_heads,
              self.qk_head_dim,
          ),
          persistent=False,
      )
      self.register_buffer(
          "v_cache",
          torch.zeros(
              args.max_batch_size,
              args.max_seq_len,
              self.n_local_heads,
              self.v_head_dim,
          ),
          persistent=False,
      )
    else:
      self.register_buffer(
          "kv_cache",
          torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
          persistent=False,
      )
      self.register_buffer(
          "pe_cache",
          torch.zeros(
              args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim
          ),
          persistent=False,
      )

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
  ):
    """Forward pass for the Multi-Headed Attention Layer (MLA).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        start_pos (int): Starting position in the sequence for caching.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for
          rotary embeddings.
        mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions
          from attention.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input.
    """
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    if self.q_lora_rank == 0:
      q = self.wq(x)
    else:
      q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    kv = self.wkv_a(x)
    kv, k_pe = torch.split(
        kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    if attn_impl == "naive":
      q = torch.cat([q_nope, q_pe], dim=-1)
      kv = self.wkv_b(self.kv_norm(kv))
      kv = kv.view(
          bsz,
          seqlen,
          self.n_local_heads,
          self.qk_nope_head_dim + self.v_head_dim,
      )
      k_nope, v = torch.split(
          kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
      )
      k = torch.cat(
          [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1
      )
      self.k_cache[:bsz, start_pos:end_pos] = k
      self.v_cache[:bsz, start_pos:end_pos] = v
      scores = (
          torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
          * self.softmax_scale
      )
    else:
      wkv_b = (
          self.wkv_b.weight
          if self.wkv_b.scale is None
          else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
      )
      wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
      q_nope = torch.einsum(
          "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
      )
      self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
      self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
      scores = (
          torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
          + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
      ) * self.softmax_scale
    if mask is not None:
      scores += mask.unsqueeze(1)
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
    if attn_impl == "naive":
      x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
    else:
      x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
      x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
    x = self.wo(x.flatten(2))
    return x


class MLP(nn.Module):
  """Multi-Layer Perceptron (MLP) used as a feed-forward layer.

  Attributes:
      w1 (nn.Module): Linear layer for input-to-hidden transformation.
      w2 (nn.Module): Linear layer for hidden-to-output transformation.
      w3 (nn.Module): Additional linear layer for feature transformation.
  """

  def __init__(self, dim: int, inter_dim: int):
    """Initializes the MLP layer.

    Args:
        dim (int): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
    """
    super().__init__()
    self.w1 = ColumnParallelLinear(dim, inter_dim)
    self.w2 = RowParallelLinear(inter_dim, dim)
    self.w3 = ColumnParallelLinear(dim, inter_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the MLP layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after MLP computation.
    """
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
  """Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

  Attributes:
      dim (int): Dimensionality of input features.
      topk (int): Number of top experts activated for each input.
      n_groups (int): Number of groups for routing.
      topk_groups (int): Number of groups to route inputs to.
      score_func (str): Scoring function ('softmax' or 'sigmoid').
      route_scale (float): Scaling factor for routing weights.
      weight (torch.nn.Parameter): Learnable weights for the gate.
      bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
  """

  def __init__(self, args: ModelArgs):
    """Initializes the Gate module.

    Args:
        args (ModelArgs): Model arguments containing gating parameters.
    """
    super().__init__()
    self.dim = args.dim
    self.topk = args.n_activated_experts
    self.n_groups = args.n_expert_groups
    self.topk_groups = args.n_limited_groups
    self.score_func = args.score_func
    self.route_scale = args.route_scale
    self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
    self.bias = (
        nn.Parameter(torch.empty(args.n_routed_experts))
        if self.dim == 7168
        else None
    )

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for the gating mechanism.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert
        indices.
    """
    scores = linear(x, self.weight)
    if self.score_func == "softmax":
      scores = scores.softmax(dim=-1, dtype=torch.float32)
    else:
      scores = scores.sigmoid()
    original_scores = scores
    if self.bias is not None:
      scores = scores + self.bias
    if self.n_groups > 1:
      scores = scores.view(x.size(0), self.n_groups, -1)
      if self.bias is None:
        group_scores = scores.amax(dim=-1)
      else:
        group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
      indices = group_scores.topk(self.topk_groups, dim=-1)[1]
      mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
      scores = (scores * mask.unsqueeze(-1)).flatten(1)
    indices = torch.topk(scores, self.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if self.score_func == "sigmoid":
      weights /= weights.sum(dim=-1, keepdim=True)
    weights *= self.route_scale
    return weights.type_as(x), indices


class Expert(nn.Module):
  """Expert layer for Mixture-of-Experts (MoE) models.

  Attributes:
      w1 (nn.Module): Linear layer for input-to-hidden transformation.
      w2 (nn.Module): Linear layer for hidden-to-output transformation.
      w3 (nn.Module): Additional linear layer for feature transformation.
  """

  def __init__(self, dim: int, inter_dim: int):
    """Initializes the Expert layer.

    Args:
        dim (int): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
    """
    super().__init__()
    self.w1 = Linear(dim, inter_dim)
    self.w2 = Linear(inter_dim, dim)
    self.w3 = Linear(dim, inter_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the Expert layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after expert computation.
    """
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
  """Mixture-of-Experts (MoE) module.

  Attributes:
      dim (int): Dimensionality of input features.
      n_routed_experts (int): Total number of experts in the model.
      n_local_experts (int): Number of experts handled locally in distributed
        systems.
      n_activated_experts (int): Number of experts activated for each input.
      gate (nn.Module): Gating mechanism to route inputs to experts.
      experts (nn.ModuleList): List of expert modules.
      shared_experts (nn.Module): Shared experts applied to all inputs.
  """

  def __init__(self, args: ModelArgs):
    """Initializes the MoE module.

    Args:
        args (ModelArgs): Model arguments containing MoE parameters.
    """
    super().__init__()
    self.dim = args.dim
    assert args.n_routed_experts % world_size == 0
    self.n_routed_experts = args.n_routed_experts
    self.n_local_experts = args.n_routed_experts // world_size
    self.n_activated_experts = args.n_activated_experts
    self.experts_start_idx = rank * self.n_local_experts
    self.experts_end_idx = self.experts_start_idx + self.n_local_experts
    self.gate = Gate(args)
    self.experts = nn.ModuleList([
        Expert(args.dim, args.moe_inter_dim)
        if self.experts_start_idx <= i < self.experts_end_idx
        else None
        for i in range(self.n_routed_experts)
    ])
    self.shared_experts = MLP(
        args.dim, args.n_shared_experts * args.moe_inter_dim
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the MoE module.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after expert routing and computation.
    """
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x)
    y = torch.zeros_like(x)
    counts = torch.bincount(
        indices.flatten(), minlength=self.n_routed_experts
    ).tolist()
    for i in range(self.experts_start_idx, self.experts_end_idx):
      if counts[i] == 0:
        continue
      expert = self.experts[i]
      idx, top = torch.where(indices == i)
      y[idx] += expert(x[idx]) * weights[idx, top, None]
    z = self.shared_experts(x)
    if world_size > 1:
      dist.all_reduce(y)
    return (y + z).view(shape)


class Block(nn.Module):
  """Transformer block combining attention and feed-forward layers.

  Attributes:
      attn (nn.Module): Attention layer (MLA).
      ffn (nn.Module): Feed-forward network (MLP or MoE).
      attn_norm (nn.Module): Layer normalization for attention.
      ffn_norm (nn.Module): Layer normalization for feed-forward network.
  """

  def __init__(self, layer_id: int, args: ModelArgs):
    """Initializes the Transformer block.

    Args:
        layer_id (int): Layer index in the transformer.
        args (ModelArgs): Model arguments containing block parameters.
    """
    super().__init__()
    self.attn = MLA(args)
    self.ffn = (
        MLP(args.dim, args.inter_dim)
        if layer_id < args.n_dense_layers
        else MoE(args)
    )
    self.attn_norm = RMSNorm(args.dim)
    self.ffn_norm = RMSNorm(args.dim)

  def forward(
      self,
      x: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
  ) -> torch.Tensor:
    """Forward pass for the Transformer block.

    Args:
        x (torch.Tensor): Input tensor.
        start_pos (int): Starting position in the sequence.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for
          rotary embeddings.
        mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions
          from attention.

    Returns:
        torch.Tensor: Output tensor after block computation.
    """
    x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  """Transformer model with positional embeddings, multiple layers, and output projection.

  Attributes:
      max_seq_len (int): Maximum sequence length for the transformer.
      embed (nn.Module): Embedding layer for input tokens.
      layers (torch.nn.ModuleList): List of transformer blocks.
      norm (nn.Module): Layer normalization applied after all blocks.
      head (nn.Module): Output projection layer mapping to vocabulary size.
      freqs_cis (torch.Tensor): Precomputed complex exponential values for
        rotary embeddings.
  """

  def __init__(self, args: ModelArgs):
    """Initializes the Transformer model.

    Args:
        args (ModelArgs): Model arguments containing transformer parameters.
    """
    global world_size, rank
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    Linear.dtype = (
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    )
    super().__init__()
    self.max_seq_len = args.max_seq_len
    self.embed = ParallelEmbedding(args.vocab_size, args.dim)
    self.layers = torch.nn.ModuleList()
    for layer_id in range(args.n_layers):
      self.layers.append(Block(layer_id, args))
    self.norm = RMSNorm(args.dim)
    self.head = ColumnParallelLinear(
        args.dim, args.vocab_size, dtype=torch.get_default_dtype()
    )
    self.register_buffer(
        "freqs_cis", precompute_freqs_cis(args), persistent=False
    )

  @torch.inference_mode()
  def forward(self, tokens: torch.Tensor, start_pos: int = 0):
    """Forward pass for the Transformer model.

    Args:
        tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size,
          seq_len).
        start_pos (int, optional): Starting position in the sequence for rotary
          embeddings. Defaults to 0.

    Returns:
        torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
    """
    seqlen = tokens.size(1)
    h = self.embed(tokens)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
      mask = torch.full(
          (seqlen, seqlen), float("-inf"), device=tokens.device
      ).triu_(1)
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)[:, -1]
    logits = self.head(h)
    if world_size > 1:
      all_logits = [torch.empty_like(logits) for _ in range(world_size)]
      dist.all_gather(all_logits, logits)
      logits = torch.cat(all_logits, dim=-1)
    return logits


def main(argv) -> None:
  torch.set_default_dtype(torch.bfloat16)
  torch.set_default_device("cuda")
  torch.manual_seed(0)
  args = ModelArgs()
  x = torch.randint(0, args.vocab_size, (2, 128))
  model = Transformer(args)
  print(model(x).size())


if __name__ == "__main__":
  app.run(main)

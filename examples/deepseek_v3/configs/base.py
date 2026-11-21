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
"""Base configuration for the model."""

import dataclasses
from typing import Literal


@dataclasses.dataclass
class ModelArgs:
  """Data class for defining model arguments and hyperparameters.

  Attributes:
      max_batch_size (int): Maximum batch size.
      max_seq_len (int): Maximum sequence length.
      dtype (Literal["bf16", "fp8"]): Data type for computations.
      vocab_size (int): Vocabulary size.
      dim (int): Model dimension.
      inter_dim (int): Intermediate dimension for MLP layers.
      moe_inter_dim (int): Intermediate dimension for MoE layers.
      n_layers (int): Number of transformer layers.
      n_dense_layers (int): Number of dense layers in the model.
      n_heads (int): Number of attention heads.
      n_routed_experts (int): Number of routed experts for MoE layers.
      n_shared_experts (int): Number of shared experts for MoE layers.
      n_activated_experts (int): Number of activated experts in MoE layers.
      n_expert_groups (int): Number of expert groups.
      n_limited_groups (int): Number of limited groups for MoE routing.
      score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE
        routing.
      route_scale (float): Scaling factor for routing scores.
      q_lora_rank (int): LoRA rank for query projections.
      kv_lora_rank (int): LoRA rank for key-value projections.
      qk_nope_head_dim (int): Dimension for query-key projections without
        positional embeddings.
      qk_rope_head_dim (int): Dimension for query-key projections with rotary
        embeddings.
      v_head_dim (int): Dimension for value projections.
      original_seq_len (int): Original sequence length.
      rope_theta (float): Base for rotary positional encoding.
      rope_factor (float): Scaling factor for extended sequence lengths.
      beta_fast (int): Fast beta correction factor.
      beta_slow (int): Slow beta correction factor.
      mscale (float): Scaling factor for extended attention.
      world_size (int): World size.
      rank (int): Rank.
      block_size (int): Block size.
      gemm_impl (Literal["bf16", "fp8"] | None): Implementation for GEMM
        operations.
      attn_impl (Literal["naive", "absorb"]): Implementation for attention
        operations.
  """

  max_batch_size: int = 8
  max_seq_len: int = 4096 * 4
  dtype: Literal["bf16", "fp8"] = "bf16"
  vocab_size: int = 102400
  dim: int = 2048
  inter_dim: int = 10944
  moe_inter_dim: int = 1408
  n_layers: int = 27
  n_dense_layers: int = 1
  n_heads: int = 16
  # moe
  n_routed_experts: int = 64
  n_shared_experts: int = 2
  n_activated_experts: int = 6
  n_expert_groups: int = 1
  n_limited_groups: int = 1
  score_func: Literal["softmax", "sigmoid"] = "softmax"
  route_scale: float = 1.0
  # mla
  q_lora_rank: int = 0
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  # yarn
  original_seq_len: int = 4096
  rope_theta: float = 10000.0
  rope_factor: float = 40
  beta_fast: int = 32
  beta_slow: int = 1
  mscale: float = 1.0
  # misc
  world_size: int = 1
  rank: int = 0
  block_size: int = 128
  gemm_impl: Literal["bf16", "fp8"] | None = "bf16"
  attn_impl: Literal["naive", "absorb"] = "absorb"

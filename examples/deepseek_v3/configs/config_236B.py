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
"""Configuration for the 236B model."""

from configs.base import ModelArgs


def get_config():
  """Returns the configuration for the model."""
  return ModelArgs(
      vocab_size=102400,
      dim=5120,
      inter_dim=12288,
      moe_inter_dim=1536,
      n_layers=60,
      n_dense_layers=1,
      n_heads=128,
      n_routed_experts=160,
      n_shared_experts=2,
      n_activated_experts=6,
      n_expert_groups=8,
      n_limited_groups=3,
      route_scale=16.0,
      q_lora_rank=1536,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
  )

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
"""Tests for the DeepSeek model."""


from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import model as model_lib
from deepseek import modeling_deepseek as model_lib_pt
import jax
import jax.numpy as jnp
import numpy as np
import torch
from deepseek_convert_utils import convert_config


def bf16_pt_to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.array(
      pt_tensor.detach().to(torch.float32).numpy().astype(jnp.bfloat16)
  )

def to_jax(pt_tensor: torch.Tensor, /) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class ModelTest(parameterized.TestCase):
  # def _copy_rs_norm_weights(
  #   self, model_jax: model_lib.RMSNorm, model_pt: model_lib_pt.DeepseekV3RMSNorm
  # ):
  #   model_jax.weight.value = to_jax(model_pt.weight)

  def test_rsnorm_parity(self):
    """Tests that the RSNorm layer is implemented correctly."""
    hidden_size = 16

    # pytorch
    model_pt = model_lib_pt.DeepseekV3RMSNorm(hidden_size)
    x_pt = torch.randn(1, 5, hidden_size)
    y_pt = model_pt(x_pt)

    # jax
    model_jax = model_lib.RMSNorm(hidden_size)
    self._copy_rs_norm_weights(model_jax, model_pt)
    x_jax = to_jax(x_pt)
    y_jax = model_jax(x_jax)

    np.testing.assert_allclose(to_jax(y_pt), y_jax, atol=1e-3)

  # def _copy_embedding_weights(
  #   self,
  #   model_jax: model_lib.YarnRotaryEmbedding,
  #   model_pt: model_lib_pt.DeepseekV3YarnRotaryEmbedding,
  # ):
  #   model_jax.inv_freq.value = to_jax(model_pt.inv_freq)
  #   model_jax.cos_cached.value = to_jax(model_pt.cos_cached)
  #   model_jax.sin_cached.value = to_jax(model_pt.sin_cached)

  def test_rotary_embedding(self):
    hidden_dim = 32
    seq_len = 8

    # pytorch
    model_pt = model_lib_pt.DeepseekV3YarnRotaryEmbedding(
      dim=hidden_dim, max_position_embeddings=128, base=10000
    )
    x_pt = torch.randn(2, seq_len, hidden_dim)
    cos_pt, sin_pt = model_pt(x_pt, seq_len=seq_len)

    # jax
    model_jax = model_lib.YarnRotaryEmbedding(hidden_dim, 128, 10000)
    self._copy_embedding_weights(model_jax, model_pt)
    x_jax = to_jax(x_pt)
    cos_jax, sin_jax = model_jax(x_jax, seq_len=seq_len)

    np.testing.assert_allclose(to_jax(cos_pt), cos_jax, atol=1e-3)
    np.testing.assert_allclose(to_jax(sin_pt), sin_jax, atol=1e-3)

  def _copy_mlp_weights(
    self,
    model_jax: model_lib.MLPLayer,
    model_pt: model_lib_pt.DeepseekV3MLP,
  ):
    model_jax.w_gate.value = to_jax(model_pt.gate_proj.weight).T
    model_jax.w_up.value = to_jax(model_pt.up_proj.weight).T
    model_jax.w_down.value = to_jax(model_pt.down_proj.weight).T

  def test_mlp_parity(self):
    """Tests that the MLP layer is implemented correctly."""
    hidden_size = 16
    mlp_dim = 32
    config_pt = model_lib_pt.DeepseekV3Config(
      hidden_size=hidden_size,
      intermediate_size=mlp_dim,
    )
    config_jax = convert_config(config_pt, quantized=False)

    # pytorch
    model_pt = model_lib_pt.DeepseekV3MLP(config_pt, hidden_size, mlp_dim)
    x_pt = torch.randn(1, 5, hidden_size)
    y_pt = model_pt(x_pt)

    # jax
    model_jax = model_lib.MLPLayer(config_jax, rngs=nnx.Rngs(0))

    self._copy_mlp_weights(model_jax, model_pt)
    x_jax = to_jax(x_pt)
    y_jax = model_jax(x_jax)

    np.testing.assert_allclose(to_jax(y_pt), y_jax, atol=1.5e-3)

  # def _copy_moe_gate_weights(
  #   self, model_jax: model_lib.MoEGate, model_pt: model_lib_pt.MoEGate
  # ):
  #   model_jax.weight.value = to_jax(model_pt.weight).T
  #   if model_jax.e_score_correction_bias is not None:
  #     model_jax.e_score_correction_bias.value = to_jax(
  #       model_pt.e_score_correction_bias
  #     )

  def test_moe_gate_parity(self):
    """Tests that the MoEGate layer is implemented correctly."""

    config = model_lib.Config(
      n_routed_experts=8,
      hidden_size=16,
      n_group=4,
      topk_group=2,
    )
    # pytorch
    model_pt = model_lib_pt.MoEGate(config)
    model_pt.eval()
    x_pt = torch.randn(1, 12, 16)
    topk_idx_pt, topk_weight_pt = model_pt(x_pt)
    topk_idx_pt, topk_weight_pt = to_jax(topk_idx_pt), to_jax(topk_weight_pt)

    # sort pytorch output
    idxs_pt = jnp.argsort(topk_idx_pt, axis=-1)
    topk_idx_pt = jnp.take_along_axis(topk_idx_pt, idxs_pt, axis=-1)
    topk_weight_pt = jnp.take_along_axis(topk_weight_pt, idxs_pt, axis=-1)

    # jax
    model_jax = model_lib.MoEGate(config, rngs=nnx.Rngs(0))
    self._copy_moe_gate_weights(model_jax, model_pt)
    x_jax = to_jax(x_pt)
    topk_idx_jax, topk_weight_jax = model_jax(x_jax)

    # sort jax output
    idxs_jax = jnp.argsort(topk_idx_jax, axis=-1)
    topk_idx_jax = jnp.take_along_axis(topk_idx_jax, idxs_jax, axis=-1)
    topk_weight_jax = jnp.take_along_axis(topk_weight_jax, idxs_jax, axis=-1)

    np.testing.assert_allclose(topk_idx_pt, topk_idx_jax, atol=1e-3)
    np.testing.assert_allclose(topk_weight_pt, topk_weight_jax, atol=1e-3)

  # def _copy_moe_weights(
  #   self, jax_model: model_lib.MoE, pt_model: model_lib_pt.DeepseekV3MoE
  # ):
  #   for expert_jax, expert_pt in zip(jax_model.experts, pt_model.experts):
  #     assert isinstance(expert_pt, model_lib_pt.DeepseekV3MLP)
  #     self._copy_mlp_weights(expert_jax, expert_pt)

  #   if jax_model.shared_experts is not None:
  #     self._copy_mlp_weights(jax_model.shared_experts, pt_model.shared_experts)

  #   self._copy_moe_gate_weights(jax_model.gate, pt_model.gate)

  def test_moe_parity(self):
    """Tests that the Moe layer is implemented correctly."""

    batch_size = 4
    seq_len = 10
    config = model_lib.Config(
      n_routed_experts=12,
      hidden_size=16,
      n_group=4,
      topk_group=2,
      num_experts_per_tok=3,
    )
    # pytorch
    model_pt = model_lib_pt.DeepseekV3MoE(config)
    model_pt.eval()
    x_pt = torch.randn(batch_size, seq_len, config.hidden_size)
    y_pt = model_pt(x_pt)

    # jax
    model_jax = model_lib.MoE(config, rngs=nnx.Rngs(0))

    self._copy_moe_weights(model_jax, model_pt)
    x_jax = to_jax(x_pt)
    y_jax = model_jax(x_jax)

    np.testing.assert_allclose(to_jax(y_pt), y_jax, atol=1e-3)


if __name__ == "__main__":
  absltest.main()

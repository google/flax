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
# ============================================================================å
"""Tests for the DeepSeek model."""

from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import model as model_lib
import model_pytorch as model_lib_pytorch
import jax
import jax.numpy as jnp
import numpy as np
import torch


def bf16_pt_to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.array(
      pt_tensor.detach().to(torch.float32).numpy().astype(jnp.bfloat16)
  )


class ModelTest(parameterized.TestCase):

  def test_parallel_embedding_parity(self):
    vocab_size = 100
    dim = 64
    batch_size = 2
    seq_len = 10
    config = model_lib.ModelArgs(vocab_size=vocab_size, dim=dim)

    # Create PyTorch embedding
    embedding_pytorch = model_lib_pytorch.ParallelEmbedding(vocab_size, dim)
    input_pytorch = torch.randint(0, vocab_size, (batch_size, seq_len))
    output_pytorch = embedding_pytorch(input_pytorch)

    # Create Flax embedding
    embedding_flax = model_lib.ParallelEmbedding(config, dim)
    input_flax = jnp.array(input_pytorch.detach().numpy())

    # Copy weights from PyTorch to Flax
    embedding_flax.weight.value = jnp.array(
        embedding_pytorch.weight.detach().numpy()
    )

    # Run Flax embedding
    output_flax = embedding_flax(input_flax)

    # Compare outputs
    self.assertTrue(
        jnp.allclose(output_flax, jnp.array(output_pytorch.detach().numpy()))
    )

  def test_linear_parity_bf16(self):
    gemm_impl = "bf16"
    in_features = 6
    out_features = 7
    config = model_lib.ModelArgs(gemm_impl=gemm_impl)

    x_pytorch = torch.rand(in_features).to(torch.bfloat16)
    linear_pytorch = model_lib_pytorch.ColumnParallelLinear(
        in_features=in_features, out_features=out_features, bias=True
    )
    linear_pytorch.weight.data = torch.rand(out_features, in_features).to(
        torch.bfloat16
    )
    linear_pytorch.bias.data = torch.rand(out_features).to(torch.bfloat16)
    y_pytorch = linear_pytorch(x_pytorch)

    # jax
    x_jax = bf16_pt_to_jax(x_pytorch)
    linear_jax = model_lib.ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        config=config,
    )
    linear_jax.weight.value = bf16_pt_to_jax(linear_pytorch.weight).T
    linear_jax.bias.value = bf16_pt_to_jax(linear_pytorch.bias)
    y_jax = linear_jax(x_jax)

    np.testing.assert_allclose(
        y_jax,
        bf16_pt_to_jax(y_pytorch),
        rtol=0.0072,
    )

  def test_linear_parity_bf16_fp8(self):
    gemm_impl = "fp8"
    in_features = 6
    out_features = 7
    config = model_lib.ModelArgs(gemm_impl=gemm_impl)

    # pytorch
    dtype_pytorch = torch.float8_e4m3fn
    x_pytorch = torch.rand(in_features).to(torch.bfloat16)
    linear_pytorch = model_lib_pytorch.ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        dtype=dtype_pytorch,
        bias=True,
    )
    linear_pytorch.weight.data = torch.rand(out_features, in_features).to(
        dtype_pytorch
    )
    linear_pytorch.bias.data = torch.rand(out_features).to(dtype_pytorch)
    y_pytorch = linear_pytorch(x_pytorch)


if __name__ == "__main__":
  absltest.main()

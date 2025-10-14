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
"""Base layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike  # pylint: disable=g-importing-member,g-multiple-import


Shape = Sequence[Union[int, Any]]


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: Shape,
      *,
      kernel_init: nnx.Initializer = nnx.initializers.normal(),
      rngs: nnx.Rngs,
      dtype: Any = jnp.float32,
  ):
    self.einsum_str = einsum_str
    self.w = nnx.Param(kernel_init(rngs.params(), shape, dtype))

  def __call__(self, x: ArrayLike) -> Array:
    return jnp.einsum(self.einsum_str, x, self.w[...])

  @property
  def shape(self) -> Shape:
    return self.w.shape


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      scale_init: nnx.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
      dtype: Any = jnp.float32,
  ):
    self.scale = nnx.Param(scale_init(rngs.params(), dim, dtype))

  def __call__(self, x: Array) -> Array:
    dtype = self.scale.dtype
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = jnp.asarray(x * jax.lax.rsqrt(var + 1e-06), dtype=dtype)
    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(self.scale, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs

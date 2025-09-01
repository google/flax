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
      weight_dtype: Any = jnp.float32,
  ):
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.einsum_str = einsum_str
    self.w = nnx.Param(kernel_init(rngs.params(), shape, weight_dtype))

  def __call__(self, x: ArrayLike) -> Array:
    x = jnp.asarray(x, self.dtype)
    w = jnp.asarray(self.w[...], self.dtype)
    return jnp.einsum(self.einsum_str, x, w)

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
      weight_dtype: Any = jnp.float32,
  ):
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scale = nnx.Param(scale_init(rngs.params(), dim, weight_dtype))

  def __call__(self, x: Array) -> Array:
    x = jnp.asarray(x, self.dtype)
    mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = jnp.asarray(x * jax.lax.rsqrt(mean2 + 1e-06), dtype=self.dtype)
    scale = jnp.asarray(self.scale, self.dtype)
    normed_inputs = normed_inputs * (1.0 + scale)
    return normed_inputs

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

from __future__ import annotations

import typing as tp

import jax.numpy as jnp
from flax.linen import pooling as linen_pooling
from flax.nnx.module import Module
from flax.typing import PaddingLike

class MaxPool(Module):
  """Max pooling layer.

  Args:
    window_shape: int or sequence of ints determining the window shape.
    strides: int or sequence of ints determining the stride.
    padding: 'SAME', 'VALID' or a sequence of n (low, high) integer pairs.
  """

  def __init__(
    self,
    window_shape: int | tp.Sequence[int],
    strides: int | tp.Sequence[int] | None = None,
    padding: PaddingLike = 'VALID',
  ):
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return linen_pooling.max_pool(
      inputs,
      window_shape=self.window_shape,
      strides=self.strides,
      padding=self.padding,
    )


class MinPool(Module):
  """Min pooling layer.

  Args:
    window_shape: int or sequence of ints determining the window shape.
    strides: int or sequence of ints determining the stride.
    padding: 'SAME', 'VALID' or a sequence of n (low, high) integer pairs.
  """

  def __init__(
    self,
    window_shape: int | tp.Sequence[int],
    strides: int | tp.Sequence[int] | None = None,
    padding: PaddingLike = 'VALID',
  ):
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return linen_pooling.min_pool(
      inputs,
      window_shape=self.window_shape,
      strides=self.strides,
      padding=self.padding,
    )


class AvgPool(Module):
  """Average pooling layer.

  Args:
    window_shape: int or sequence of ints determining the window shape.
    strides: int or sequence of ints determining the stride.
    padding: 'SAME', 'VALID' or a sequence of n (low, high) integer pairs.
    count_include_pad: a boolean whether to include padded tokens
      in the average calculation (default: True).
  """

  def __init__(
    self,
    window_shape: int | tp.Sequence[int],
    strides: int | tp.Sequence[int] | None = None,
    padding: PaddingLike = 'VALID',
    count_include_pad: bool = True,
  ):
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.count_include_pad = count_include_pad

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return linen_pooling.avg_pool(
      inputs,
      window_shape=self.window_shape,
      strides=self.strides,
      padding=self.padding,
      count_include_pad=self.count_include_pad,
    )


class GlobalAveragePool(Module):
  """Global Average Pooling layer.

  Pools the input by taking the average over the spatial dimensions.
  Assumes channel-last layout (e.g., NHWC or NDC).

  Args:
    keepdims: If True, keeps the reduced dimensions as singleton dimensions.
      Defaults to False.
  """

  def __init__(self, keepdims: bool = False):
    self.keepdims = keepdims

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Assume inputs are (Batch, Spatial..., Features)
    # We want to reduce over the Spatial dimensions (1 to ndim-2 inclusive)
    assert inputs.ndim >= 2, f"Input must have at least 2 dimensions (batch, features), got {inputs.ndim}"
    spatial_axes = tuple(range(1, inputs.ndim - 1))
    return jnp.mean(inputs, axis=spatial_axes, keepdims=self.keepdims)
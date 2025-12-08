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

import typing as tp
from types import MappingProxyType
from jax.nn import (
  celu,
  elu,
  gelu,
  glu,
  hard_sigmoid,
  hard_silu,
  hard_swish,
  hard_tanh,
  leaky_relu,
  log_sigmoid,
  log_softmax,
  logsumexp,
  one_hot,
  relu,
  identity,
  relu6,
  selu,
  sigmoid,
  silu,
  soft_sign,
  softmax,
  softplus,
  standardize,
  swish,
)
import jax.numpy as jnp
from jax.numpy import tanh

from flax import nnx
from flax.nnx.nn import dtypes
from flax.typing import Array, Dtype, PromoteDtypeFn


__all__ = [
  'celu',
  'elu',
  'gelu',
  'glu',
  'hard_sigmoid',
  'hard_silu',
  'hard_swish',
  'hard_tanh',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'logsumexp',
  'one_hot',
  'relu',
  'identity',
  'relu6',
  'selu',
  'sigmoid',
  'silu',
  'soft_sign',
  'softmax',
  'softplus',
  'standardize',
  'swish',
  'tanh',
  'PReLU',
]


class PReLU(nnx.Module):
  """Parametric Rectified Linear Unit (PReLU) activation function.

  Note that PReLU is a Flax layer and not a simple activation function, so
  it needs to be initialized before being called.

  Example::

    >>> import flax.nnx as nnx

    >>> class MLP(nnx.Module):
    ...   def __init__(self):
    ...     self.linear = nnx.Linear(3, 2)
    ...     self.act = nnx.PReLU(negative_slope_init=0.1)
    ...
    ...   def __call__(self, x):
    ...     x = self.linear(x)
    ...     x = self.act(x)
    ...     return x

  Args:
    negative_slope_init: the value to initialize the negative slope (default 0.01).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    promote_dtype: function to promote the dtype of all input array arguments
      (including Variables accessed through ``self``) to the desired dtype. The
      function should accept a tuple of ``(inputs, negative_slope)`` and a ``dtype``
      keyword argument, and return a tuple of arrays with the promoted dtype.
    negative_slope_metadata: Optional metadata dictionary to set when initializing
      the negative slope.
  """
  def __init__(
    self,
    negative_slope_init: float = 0.01,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    negative_slope_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  ):
    self.negative_slope = nnx.Param(
      jnp.asarray(negative_slope_init, dtype=param_dtype), **negative_slope_metadata
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.promote_dtype = promote_dtype

  def __call__(self, inputs: Array) -> Array:
    negative_slope = self.negative_slope[...]
    if self.dtype is not None:
      inputs, negative_slope = self.promote_dtype(
        (inputs, negative_slope), dtype=self.dtype
      )
    else:
      # Match Linen behavior: cast parameter to input dtype
      negative_slope = jnp.asarray(negative_slope, inputs.dtype)

    return jnp.where(
      inputs >= 0,
      inputs,
      negative_slope * inputs,
    )

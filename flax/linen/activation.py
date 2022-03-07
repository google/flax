# Copyright 2022 The Flax Authors.
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

"""Activation functions.
"""

import jax.numpy as jnp
# pylint: disable=unused-import
# re-export activation functions from jax.nn and jax.numpy
from jax.nn import (celu, elu, gelu, glu, hard_sigmoid, hard_swish, hard_tanh,
                    leaky_relu, log_sigmoid, log_softmax, normalize, relu,
                    relu6, selu, sigmoid, silu, soft_sign, softmax, softplus,
                    swish)
from jax.numpy import tanh
# pylint: enable=unused-import

from .dtypes import Array, FloatingDType, canonicalize_inexact_dtypes
from .module import Module, compact


class PReLU(Module):
  """Parametric Rectified Linear Unit (PReLU) activation function.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    negative_slope_init: the value to initialize the negative slope
      (default 0.01).
  """
  dtype: Optional[FloatingDType] = jnp.float32
  param_dtype: Optional[FloatingDType] = jnp.float32
  negative_slope_init: float = 0.01

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies an activation to the inputs.

    Args:
      inputs: the nd-array to apply the activation function to.

    Returns:
      The transformed input.
    """
    assert jnp.issubdtype(inputs.dtype, jnp.floating)
    inputs = jnp.asarray(inputs, dtype)
    param_dtype, dtype = canonicalize_inexact_dtypes(inputs.dtype, param_dtype,
                                                     self.dtype)
    negative_slope = self.param(
      'negative_slope',
      lambda k: jnp.asarray(self.negative_slope_init, param_dtype)
    )
    negative_slope = jnp.asarray(negative_slope, dtype)
    return jnp.where(inputs >= 0, inputs, negative_slope * inputs)

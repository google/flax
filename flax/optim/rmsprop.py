# Copyright 2020 The Flax Authors.
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

import jax.numpy as jnp
import numpy as onp
from .. import struct
from .base import OptimizerDef


@struct.dataclass
class _RMSPropHyperParams:
  """RMSProp hyper parameters"""

  learning_rate: float
  beta2: float
  eps: float


@struct.dataclass
class _RMSPropParamState:
  """RMSProp parameter state"""

  v: onp.ndarray


class RMSProp(OptimizerDef):
  """RMSProp optimizer"""
  def __init__(self, learning_rate: float = None, beta2=0.9, eps=1e-8):
    """Constructor for the RMSProp optimizer
    
    Args:
      learning_rate: the step size used to update the parameters.
      beta2: the coefficient used for the moving average of the
        gradient magnitude (default: 0.9).
      eps: the term added to the gradient magnitude estimate for
        numerical stability.
    """
    hyper_params = _RMSPropHyperParams(learning_rate, beta2, eps)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    """Initialize parameter state"""
    return _RMSPropParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply per-parameter gradients"""

    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    new_v = hyper_params.beta2 * state.v + (
        1.0 - hyper_params.beta2) * jnp.square(grad)
    new_param = param - hyper_params.learning_rate * grad / (jnp.sqrt(new_v) +
                                                             hyper_params.eps)
    new_state = _RMSPropParamState(new_v)

    return new_param, new_state

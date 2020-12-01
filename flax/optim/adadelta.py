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
class _AdadeltaHyperParams:
  """Adadelta hyper parameters"""

  learning_rate: float
  rho: float
  eps: float
  weight_decay: float


@struct.dataclass
class _AdadeltaParamState:
  """Adadelta parameter state"""

  sq_avg: onp.ndarray
  acc_delta: onp.ndarray


class Adadelta(OptimizerDef):
  """Adadelta optimizer"""
  def __init__(self, learning_rate: float = None, rho=0.9, eps=1e-8, weight_decay=0):
    """Constructor for the Adadelta optimizer.
        
    Args:
      learning_rate: the step size used to update the parameters.
      rho: coefficient used for computing a running average
      eps: term added to the denominator to improve numerical stability
    """
    hyper_params = _AdadeltaHyperParams(learning_rate, rho, eps, weight_decay)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    """Initialize parameter state"""
    return _AdadeltaParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply per-parameter gradients"""
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    assert hyper_params.rho is not None, 'no rho provided.'
    assert hyper_params.eps is not None, 'no eps provided.'
    assert hyper_params.weight_decay is not None, 'no weight decay provided.'


    rho = hyper_params.rho
    eps = hyper_params.eps
    weight_decay = hyper_params.weight_decay

    if hyper_params.weight_decay != 0:
        grad = param + weight_decay

    sq_avg = rho * (state.sq_avg + jnp.multiply(grad, grad) * (1- rho))
    std = jnp.sqrt(sq_avg + eps)
    delta = jnp.divide(jnp.sqrt(state.acc_delta+eps)/std)

    new_param = param - hyper_params.learning_rate * grad * delta
    acc_delta = rho * (state.acc_delta + jnp.multiply(delta, delta)*(1 - rho))
    new_state = _AdadeltaParamState(sq_avg, acc_delta)

    return new_param, new_state
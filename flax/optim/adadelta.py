# Copyright 2021 The Flax Authors.
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

"""Adadelta Optimizer."""

from .. import struct
from .base import OptimizerDef
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class _AdadeltaHyperParams:
  learning_rate: float
  rho: float
  eps: float
  weight_decay: float


@struct.dataclass
class _AdadeltaParamState:
  sq_avg: np.ndarray
  acc_delta: np.ndarray


class Adadelta(OptimizerDef):
  """Adadelta optimizer.

  Reference:
  [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)
  (Zeiler 2012)
  """

  def __init__(self,
               learning_rate: float = None,
               rho: float = 0.9,
               eps: float = 1e-6,
               weight_decay: float = 0.0):
    """Constructor for the Adadelta optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      rho: coefficient used for computing a running average
      eps: term added to the denominator to improve numerical stability
      weight_decay: the weight decay parameter for l2 regularization
    """
    hyper_params = _AdadeltaHyperParams(learning_rate, rho, eps, weight_decay)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    """Initialize parameter state."""
    return _AdadeltaParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply per-parameter gradients."""
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'

    rho = hyper_params.rho
    eps = hyper_params.eps
    weight_decay = hyper_params.weight_decay

    sq_avg = rho * state.sq_avg + jnp.square(grad) * (1 - rho)
    delta = (jnp.sqrt(state.acc_delta+eps) / jnp.sqrt(sq_avg + eps)) * grad

    new_param = param - hyper_params.learning_rate * delta
    new_param -= hyper_params.learning_rate * weight_decay * param
    acc_delta = rho * state.acc_delta + jnp.square(delta) * (1 - rho)
    new_state = _AdadeltaParamState(sq_avg, acc_delta)

    return new_param, new_state

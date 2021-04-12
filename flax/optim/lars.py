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

from .. import struct

import jax.numpy as jnp

import numpy as np

from .base import OptimizerDef


@struct.dataclass
class _LARSHyperParams:
  learning_rate: np.ndarray
  beta: np.ndarray
  weight_decay: np.ndarray
  trust_coefficient: np.ndarray
  eps: np.ndarray
  nesterov: bool


@struct.dataclass
class _LARSParamState:
  momentum: np.ndarray


class LARS(OptimizerDef):
  """Layerwise adaptive rate scaling (LARS) optimizer.

  See https://arxiv.org/abs/1708.03888
  """

  def __init__(self, learning_rate=None, beta=0.9, weight_decay=0,
               trust_coefficient=0.001, eps=0, nesterov=False):
    """Constructor for the LARS optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta: the coefficient used for the moving average of the
        gradient (default: 0.9).
      weight_decay: weight decay coefficient to apply
      trust_coefficient: coefficient for trust ratio computation
        (default: 0.001).
      eps: epsilon used for trust ratio computation (default: no epsilon).
      nesterov: whether to use Nesterov momentum (default: False).
    """

    hyper_params = _LARSHyperParams(
        learning_rate, beta, weight_decay, trust_coefficient, eps, nesterov)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _LARSParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'

    param_norm = jnp.linalg.norm(param)
    grad_norm = jnp.linalg.norm(grad)
    trust_ratio = hyper_params.trust_coefficient * param_norm / (
        grad_norm + hyper_params.weight_decay * param_norm + hyper_params.eps)
    clipped_trust_ratio = jnp.where(
        jnp.logical_or(grad_norm == 0., param_norm == 0.), 1., trust_ratio)
    scaled_lr = hyper_params.learning_rate * clipped_trust_ratio
    if hyper_params.weight_decay != 0:
      grad += hyper_params.weight_decay * param

    scaled_grad = scaled_lr * grad
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + scaled_grad
    if hyper_params.nesterov:
      d_p = scaled_grad + hyper_params.beta * new_momentum
    else:
      d_p = new_momentum
    new_param = param - d_p
    new_state = _LARSParamState(new_momentum)
    return new_param, new_state

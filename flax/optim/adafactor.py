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

"""Adafactor Optimizer.

This is a so-called "1+epsilon" optimizer that is extremely memory efficient
compared to Adam, and has had wide success when applied to large-scale training
of attention-based models.
"""
from typing import Optional, Any, Sequence, Tuple

from .. import struct
from .base import OptimizerDef

import jax
import jax.numpy as jnp

import numpy as np


Dtype = Any


@struct.dataclass
class _AdafactorHyperParams:
  learning_rate: Optional[float]
  factored: bool
  multiply_by_parameter_scale: bool
  beta1: Optional[float]
  decay_rate: float
  step_offset: int
  clipping_threshold: Optional[float]
  weight_decay_rate: Optional[float]
  min_dim_size_to_factor: int
  epsilon1: float
  epsilon2: float


@struct.dataclass
class _AdafactorParamState:
  v_row: np.ndarray  # used in normal factored version
  v_col: np.ndarray
  v: np.ndarray  # only used without factoring
  m: np.ndarray  # only used with momentum


class Adafactor(OptimizerDef):
  """Adafactor optimizer.
  
  Adafactor is described in https://arxiv.org/abs/1804.04235.
  """

  def __init__(self,
               learning_rate: Optional[float] = None,
               factored: bool = True,
               multiply_by_parameter_scale: bool = True,
               beta1: Optional[float] = None,
               decay_rate: float = 0.8,
               step_offset: int = 0,
               clipping_threshold: Optional[float] = 1.0,
               weight_decay_rate: Optional[float] = None,
               min_dim_size_to_factor: int = 128,
               epsilon1: float = 1e-30,
               epsilon2: float = 1e-3,
               dtype_momentum: Dtype = jnp.float32):
    """Constructor for the Adafactor optimizer.

    Args:
      learning_rate: float: learning rate.  NB: the natural scale for adafactor
        LR is markedly different from Adam, one doesn't use the 1/sqrt(hidden)
        correction for this optimizer with attention-based models.
      factored: boolean: whether to use factored second-moment estimator for 2d
        variables.
      multiply_by_parameter_scale: boolean: if True, then scale provided
        learning_rate by parameter norm. if False, provided learning_rate is
        absolute step size.
      beta1: an optional float value between 0 and 1, enables momentum and
        uses extra memory if non-None! None by default.
      decay_rate: float: controls second-moment exponential decay schedule.
      step_offset: for finetuning, one may set this to the starting step-number
        of the finetuning phase.
      clipping_threshold: an optional float >= 1, if None no update clipping.
      weight_decay_rate: optional rate at which to decay weights.
      min_dim_size_to_factor: only factor accumulator if two array dimensions
        are at least this size.
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
      dtype_momentum: dtype of momentum buffers.
    """
    hyper_params = _AdafactorHyperParams(
        learning_rate, factored, multiply_by_parameter_scale,
        beta1, decay_rate, step_offset, clipping_threshold,
        weight_decay_rate, min_dim_size_to_factor, epsilon1, epsilon2)
    self.dtype_momentum = jax.dtypes.canonicalize_dtype(dtype_momentum)
    super().__init__(hyper_params)

  @staticmethod
  def _decay_rate_pow(i: int, exponent: float = 0.8) -> float:
    """Default Adafactor second-moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return 1.0 - t**(-exponent)

  def _factored_dims(self, shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor. If we do factor the accumulator, then this function returns a
    tuple of the two largest axes to reduce over.

    Args:
      shape: a Shape

    Returns:
      None or a tuple of ints
    """
    if not self.hyper_params.factored or len(shape) < 2:
      return None
    sorted_dims = np.argsort(shape)
    if shape[sorted_dims[-2]] < self.hyper_params.min_dim_size_to_factor:
      return None
    return int(sorted_dims[-2]), int(sorted_dims[-1])

  def init_param_state(self, param):
    shape = param.shape
    state = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v', 'm']}
    factored_dims = self._factored_dims(shape)
    if factored_dims is not None:
      d1, d0 = factored_dims
      vr_shape = np.delete(shape, d0)
      vc_shape = np.delete(shape, d1)
      state['v_row'] = jnp.zeros(vr_shape, dtype=jnp.float32)
      state['v_col'] = jnp.zeros(vc_shape, dtype=jnp.float32)
    else:
      state['v'] = jnp.zeros(param.shape, dtype=jnp.float32)
    if self.hyper_params.beta1 is not None:
      state['m'] = jnp.zeros(param.shape, dtype=self.dtype_momentum)
    return _AdafactorParamState(**state)

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    learning_rate = hyper_params.learning_rate
    beta1 = hyper_params.beta1
    decay_rate = hyper_params.decay_rate
    step_offset = hyper_params.step_offset
    clipping_threshold = hyper_params.clipping_threshold
    weight_decay_rate = hyper_params.weight_decay_rate
    epsilon1 = hyper_params.epsilon1
    epsilon2 = hyper_params.epsilon2

    grad = grad.astype(jnp.float32)

    updates = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v', 'm']}
    decay_rate = self._decay_rate_pow(step - step_offset, exponent=decay_rate)
    update_scale = learning_rate
    if self.hyper_params.multiply_by_parameter_scale:
      update_scale *= jnp.maximum(
          jnp.sqrt(jnp.mean(param * param)), epsilon2)
    mixing_rate = 1.0 - decay_rate

    grad_sqr = grad * grad + epsilon1
    factored_dims = self._factored_dims(param.shape)
    if factored_dims is not None:
      d1, d0 = factored_dims
      new_v_row = (
          decay_rate * state.v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0))
      new_v_col = (
          decay_rate * state.v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1))
      updates['v_row'] = new_v_row
      updates['v_col'] = new_v_col
      reduced_d1 = d1-1 if d1 > d0 else d1
      row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
      row_factor = (new_v_row / row_col_mean) ** -0.5
      col_factor = (new_v_col) ** -0.5
      y = (grad *
           jnp.expand_dims(row_factor, axis=d0) *
           jnp.expand_dims(col_factor, axis=d1))
    else:
      new_v = decay_rate * state.v + mixing_rate * grad_sqr
      updates['v'] = new_v
      y = grad * (new_v)**-0.5

    if clipping_threshold is not None:
      clipping_denom = (
          jnp.maximum(1.0, jnp.sqrt(jnp.mean(y * y)) / clipping_threshold))
      y /= clipping_denom

    subtrahend = update_scale * y
    if beta1 is not None:
      new_m = beta1 * state.m + (1.0 - beta1) * subtrahend
      subtrahend = new_m
      updates['m'] = new_m.astype(self.dtype_momentum)

    if weight_decay_rate is not None:
      new_param = (1.0 - weight_decay_rate) * param - subtrahend
    else:
      new_param = param - subtrahend
    new_state = _AdafactorParamState(**updates)
    return new_param.astype(param.dtype), new_state

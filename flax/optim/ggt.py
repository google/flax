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

import jax
import flax
import jax.numpy as jnp
import numpy as onp
from .. import struct
from .base import OptimizerDef
from .base import OptimizerState
from .sgd import GradientDescent
from typing import Any
from jax import lax


def flatten(tensors):
  return (jnp.concatenate([tensor.flatten() for tensor in tensors]),
          [tensor.shape for tensor in tensors])


def unflatten(flattened_tensor, shapes):
  pieces = jnp.split(flattened_tensor,
                     onp.cumsum([onp.prod(shape) for shape in shapes]))[:-1]
  return [piece.reshape(shape) for piece, shape in zip(pieces, shapes)]


def transform_vector(grad, buffer, hyper_params, t, d):
  w = hyper_params.history_size
  grad = grad.reshape(d, 1)
  m = buffer[:, 0].reshape((d, 1))

  new_m = hyper_params.beta * m + (1 - hyper_params.beta) * grad
  new_buffer = jnp.concatenate([new_m, buffer[:, :(w - 1)]], axis=1)
  M = new_buffer / jnp.sqrt(jnp.minimum(t + 1, w))
  I = jnp.eye(buffer.shape[1])
  U, sigma, _ = jnp.linalg.svd(M.T.dot(M) + I * hyper_params.svd_epsilon)
  sqrt_sigma = jnp.sqrt(sigma)
  sigma_sqrt_inv = (sqrt_sigma + hyper_params.sigma_epsilon)**(-3)
  sigma_sqrt_min = jnp.min(sqrt_sigma)
  new_grad = M.dot(U).dot(jnp.diag(sigma_sqrt_inv)).dot(U.T).dot(
      (M.T.dot(grad)))
  new_grad -= lax.cond(
      sigma_sqrt_min > hyper_params.epsilon, 0, lambda x: (grad - M.dot(U).dot(
          jnp.diag(1 / sigma)).dot(U.T).dot(M.T.dot(grad))) / sigma_sqrt_min, 0,
      lambda x: jnp.full((d, 1), 0.0))
  return new_grad, new_buffer


@struct.dataclass
class _GGTOptimizerHyperParams:
  learning_rate: onp.ndarray
  history_size: int
  epsilon: float
  sigma_epsilon: float
  svd_epsilon: float
  beta: float


@struct.dataclass
class _GGTOptimizerState(OptimizerState):
  gradients_buffer: Any


class GGTOptimizer(GradientDescent):
  #class GGTOptimizer(OptimizerDef):

  def __init__(self, learning_rate, history_size, epsilon, sigma_epsilon,
               svd_epsilon, beta):
    hyper_params = _GGTOptimizerHyperParams(learning_rate, history_size,
                                            epsilon, sigma_epsilon, svd_epsilon,
                                            beta)
    OptimizerDef.__init__(self, hyper_params)

  def init_state(self, params):
    parent_state = super().init_state(params)
    #parent_state = OptimizerDef.init_state(self, params)
    self.num_params = jnp.sum(
        jax.tree_leaves(jax.tree_map(lambda x: jnp.prod(x.shape), params)))
    return _GGTOptimizerState(
        parent_state.step, parent_state.param_states,
        jnp.full((self.num_params, self.hyper_params.history_size), 0))

  def apply_gradient(self, hyper_params, params, state, grads):
    step = state.step
    params_flat, treedef = jax.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)

    flattened_grads_flat, grads_flat_shape = flatten(grads_flat)
    flattened_grads_flat, new_gradients_buffer = transform_vector(
        flattened_grads_flat, state.gradients_buffer, hyper_params, step,
        self.num_params)
    grads_flat = unflatten(flattened_grads_flat, grads_flat_shape)

    out = [
        self.apply_param_gradient(step, hyper_params, param, state, grad)
        for param, state, grad in zip(params_flat, states_flat, grads_flat)
    ]

    new_params_flat, new_states_flat = list(zip(*out))
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = _GGTOptimizerState(step + 1, new_param_states,
                                   new_gradients_buffer)
    return new_params, new_state

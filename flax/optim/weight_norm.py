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

from typing import Any

from .. import struct

import jax
import jax.numpy as jnp

import numpy as np

from .base import OptimizerDef


@struct.dataclass
class _WeightNormHyperParams:
  inner: Any
  wn_decay: np.ndarray
  wn_eps: np.ndarray


@struct.dataclass
class _WeightNormParamState:
  direction_state: Any
  scale_state: Any
  mult: np.ndarray


class WeightNorm(OptimizerDef):
  """Adds weight normalization to an optimizer def.

  See https://arxiv.org/abs/1602.07868
  """

  def __init__(self, wrapped_optimizer, wn_decay=0, wn_eps=1e-8):
    """Constructor for a WeightNorm optimizer.

    Weight vectors are decomposed as :math:`w = g * v/||v||_2`, for scalar
    scale parameter g, and raw weight vector v. The original optimizer is then
    applied to the (g,v) parameterization and the updated parameters are
    transformed back to w-space, i.e.
    w,state --> (g,v) --(original optimizer)--> (g',v') --> w',state'

    We assume the output axis of any kernel matrix is the last one,
    as per the Tensorflow convention.

    Args:
      wrapped_optimizer: another OptimizerDef
      wn_decay: apply l2 decay to the unnormalized weight vector
      wn_eps: additive constant for stability of
        the normalization (default: 1e-8).
    """
    hps = _WeightNormHyperParams(
        wrapped_optimizer.hyper_params, wn_decay, wn_eps)
    super().__init__(hps)
    self.wrapped_optimizer = wrapped_optimizer

  def update_hyper_params(self, **hyper_param_overrides):
    decay = hyper_param_overrides.pop('wn_decay', self.hyper_params.wn_decay)
    eps = hyper_param_overrides.pop('wn_eps', self.hyper_params.wn_eps)
    inner = self.wrapped_optimizer.update_hyper_params(
        **hyper_param_overrides)
    return self.hyper_params.replace(inner=inner, wn_decay=decay, wn_eps=eps)

  def init_state(self, params):
    leaves, treedef = jax.tree_flatten(params)
    directions, scales = zip(*(self._split_param(p) for p in leaves))
    directions = treedef.unflatten(directions)
    scales = treedef.unflatten(scales)
    wn_params = {'direction': directions, 'scale': scales}
    state = self.wrapped_optimizer.init_state(wn_params)
    direction_state = state.param_states['direction']
    scale_state = state.param_states['scale']
    param_states = jax.tree_multimap(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, scales)
    return state.replace(param_states=param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    p_leaves, treedef = jax.tree_flatten(params)
    s_leaves = treedef.flatten_up_to(state.param_states)
    g_leaves = treedef.flatten_up_to(grads)
    split_grads = zip(*(self._split_grad(p, s, g, hyper_params.wn_decay)
                        for p, s, g in zip(p_leaves, s_leaves, g_leaves)))
    d_p, d_s, d_g, s_p, s_s, s_g = [
        jax.tree_unflatten(treedef, x) for x in split_grads]
    wn_params = {'direction': d_p, 'scale': s_p}
    wn_state = {'direction': d_s, 'scale': s_s}
    wn_grads = {'direction': d_g, 'scale': s_g}
    new_wn_params, new_state = self.wrapped_optimizer.apply_gradient(
        hyper_params.inner, wn_params,
        state.replace(param_states=wn_state), wn_grads)

    directions = treedef.flatten_up_to(new_wn_params['direction'])
    scales = treedef.flatten_up_to(new_wn_params['scale'])
    new_params, mults = zip(*(self._merge_param(d, s, hyper_params.wn_eps)
                              for d, s in zip(directions, scales)))
    new_params = jax.tree_unflatten(treedef, new_params)
    mults = jax.tree_unflatten(treedef, mults)

    direction_state = new_state.param_states['direction']
    scale_state = new_state.param_states['scale']
    param_states = jax.tree_multimap(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, mults)
    return new_params, new_state.replace(param_states=param_states)

  def _split_param(self, param):
    if param.size > param.shape[-1]:
      scale = jnp.sqrt(jnp.square(param).sum(
          tuple(range(param.ndim-1)), keepdims=True))
      direction = param / scale
      return direction, scale
    else:
      return param, ()

  def _merge_param(self, direction, scale, eps):
    if direction.size > direction.shape[-1]:
      norm = jnp.sqrt(jnp.square(direction).sum(
          tuple(range(direction.ndim - 1)), keepdims=True))
      mult = scale / (eps + norm)
      param = direction * mult
      return param, mult
    else:
      return direction, ()

  def _split_grad(self, param, state, grad, decay):
    """Split the gradient for the direction and scale."""
    if param.size > param.shape[-1]:
      red_dims = tuple(range(param.ndim-1))
      direction = param / state.mult
      norm = jnp.sqrt(jnp.square(param).sum(red_dims, keepdims=True))
      scale = norm * jnp.sign(state.mult)
      scale_grad = jnp.sum(
          grad * direction, axis=red_dims, keepdims=True)
      direction_grad = state.mult * (grad - scale_grad * direction)
      if decay != 0:
        direction_grad = direction_grad + decay * direction
      direction_info = direction, state.direction_state, direction_grad
      scale_info = scale, state.scale_state, scale_grad
      return direction_info + scale_info
    else:
      return (param, state.direction_state, grad, (), (), ())

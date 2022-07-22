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

from typing import Any

from .. import struct

import jax
import jax.numpy as jnp
from jax import lax

import numpy as np

from .base import OptimizerDef

Array = Any


@struct.dataclass
class _WeightNormHyperParams:
  inner: Any
  wn_decay: Array
  wn_eps: Array


@struct.dataclass
class _WeightNormParamState:
  direction_state: Any
  scale_state: Any
  direction: Array
  scale: Array


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
    def split_param(param):
      if param.size > param.shape[-1]:
        norms = jnp.sqrt(jnp.square(param).sum(
            tuple(range(param.ndim-1)), keepdims=True) + eps)
        direction = param / norms
        return direction, norms
      else:
        return param, ()

    leaves, treedef = jax.tree_flatten(params)
    eps = self.hyper_params.wn_eps
    directions, scales = zip(*(split_param(p) for p in leaves))
    directions = treedef.unflatten(directions)
    scales = treedef.unflatten(scales)
    wn_params = {'direction': directions, 'scale': scales}
    state = self.wrapped_optimizer.init_state(wn_params)
    direction_state = state.param_states['direction']
    scale_state = state.param_states['scale']
    param_states = jax.tree_util.tree_map(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, directions, scales)
    return state.replace(param_states=param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    treedef = jax.tree_structure(params)
    s_leaves = treedef.flatten_up_to(state.param_states)
    direction = treedef.unflatten(x.direction for x in s_leaves)
    scale = treedef.unflatten(x.scale for x in s_leaves)
    dir_state = treedef.unflatten(x.direction_state for x in s_leaves)
    scale_state = treedef.unflatten(x.scale_state for x in s_leaves)
    eps = hyper_params.wn_eps
    decay = hyper_params.wn_decay

    def merge_param(direction, scale):
      if direction.size > direction.shape[-1]:
        norm = jnp.square(direction).sum(
          tuple(range(direction.ndim - 1)), keepdims=True) + eps
        mult = scale * lax.rsqrt(norm)
        return direction * mult
      else:
        return direction
    merge_params = lambda d, s: jax.tree_util.tree_map(merge_param, d, s)
    _, vjp_fn = jax.vjp(merge_params, direction, scale)
    dir_grad, scale_grad = vjp_fn(grads)
    def add_decay(direction, dir_grad):
      if direction.size > direction.shape[-1]:
        return dir_grad + decay * direction
      return dir_grad
    dir_grad = jax.tree_util.tree_map(add_decay, direction, dir_grad)

    wn_params = {'direction': direction, 'scale': scale}
    wn_state = {'direction': dir_state, 'scale': scale_state}
    wn_grads = {'direction': dir_grad, 'scale': scale_grad}
    new_wn_params, new_state = self.wrapped_optimizer.apply_gradient(
        hyper_params.inner, wn_params,
        state.replace(param_states=wn_state), wn_grads)
    direction = new_wn_params['direction']
    scale = new_wn_params['scale']
    new_params = merge_params(direction, scale)

    direction_state = new_state.param_states['direction']
    scale_state = new_state.param_states['scale']
    param_states = jax.tree_util.tree_map(
        lambda _, *args: _WeightNormParamState(*args),
        params, direction_state, scale_state, direction, scale)
    return new_params, new_state.replace(param_states=param_states)

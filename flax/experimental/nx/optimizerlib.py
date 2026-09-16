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
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from flax.experimental.nx import statelib, variablelib
from flax.experimental.nx import pytreelib


class OptState(variablelib.Variable): ...


class OptArray(OptState): ...


class OptVariable(OptState): ...


def to_opt_state(tree):
  def _to_opt_state(x):
    if isinstance(x, variablelib.Variable):
      opt_state = OptVariable(x[...], **x.metadata)  # type: ignore
    else:
      opt_state = OptArray(x)
    return opt_state

  tree = jax.tree.map(
    _to_opt_state,
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable),
  )
  return tree


class OptaxOptimizer(pytreelib.Pytree):
  __nodes__ = ('step', 'opt_state')

  def __init__(self, params, tx: optax.GradientTransformation):
    self.tx = tx
    self.step = OptArray(jnp.array(0, dtype=jnp.uint32))
    self.opt_state = to_opt_state(tx.init(params))

  def update(self, params, grads, **kwargs):
    param_arrays = statelib.pure(params)
    grad_arrays = statelib.pure(grads)
    opt_state_arrays = statelib.pure(self.opt_state)

    updates, new_opt_state = self.tx.update(
      grad_arrays, opt_state_arrays, param_arrays, **kwargs
    )
    new_params = optax.apply_updates(param_arrays, updates)

    def _update_variable(variable, new_value):
      variable[...] = new_value

    jax.tree.map(
      _update_variable,
      (params, self.opt_state),
      (new_params, new_opt_state),
      is_leaf=lambda x: isinstance(x, variablelib.Variable),
    )
    self.step[...] += 1
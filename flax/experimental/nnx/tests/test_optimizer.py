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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

from flax.experimental import nnx

from absl.testing import parameterized


class Model(nnx.Module):
  def __init__(self, in_features, out_features, rngs):
    self.linear1 = nnx.Linear(in_features, 3, rngs=rngs)
    self.linear2 = nnx.Linear(3, out_features, rngs=rngs)
  def __call__(self, x):
    return self.linear2(self.linear1(x))


class TestOptimizer(parameterized.TestCase):
  @parameterized.parameters(
    {'module_cls': nnx.Linear},
    {'module_cls': Model},
  )
  def test_split_merge(self, module_cls):
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optax.adam(1e-3)
    state = nnx.optimizer.Optimizer(model, tx)
    state, static = state.split()
    state = static.merge(state)

  @parameterized.parameters(
    {'module_cls': nnx.Linear},
    {'module_cls': Model},
  )
  def test_train(self, module_cls):
    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optax.adam(1e-3)
    state = nnx.optimizer.Optimizer(model, tx)
    loss_fn = lambda model: ((model(x)-y)**2).mean()
    initial_loss = loss_fn(state.model)
    grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    state.apply_gradients(grads=grads)
    new_loss = loss_fn(state.model)
    self.assertTrue(new_loss < initial_loss)
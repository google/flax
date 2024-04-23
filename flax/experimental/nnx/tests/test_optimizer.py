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
    x = jax.random.normal(jax.random.key(0), (1, 2))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optax.adam(1e-3)
    state = nnx.Optimizer(model, tx)
    out = state.model(x)
    graphdef, state = state.split()
    state = nnx.merge(graphdef, state)
    np.testing.assert_allclose(out, state.model(x))

  @parameterized.product(
    module_cls=[nnx.Linear, Model],
    jit_decorator=[lambda f: f, nnx.jit, jax.jit],
    optimizer=[optax.sgd, optax.adam],
  )
  def test_jit(self, module_cls, jit_decorator, optimizer):
    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optimizer(1e-3) # TODO: this doesn't work with adam optimizer for some reason
    state = nnx.Optimizer(model, tx)

    if jit_decorator == jax.jit:
      model_static, model_state = nnx.split(state.model)
      loss_fn = lambda graphdef, state, x, y: (
        (nnx.merge(graphdef, state)(x) - y) ** 2
      ).mean()
      initial_loss = loss_fn(model_static, model_state, x, y)

      def train_step(graphdef, state, x, y):
        state = nnx.merge(graphdef, state)
        model_static, model_state = nnx.split(state.model)
        grads = jax.grad(loss_fn, argnums=1)(model_static, model_state, x, y)
        state.update(grads)
        return state.split()

      graphdef, state = jit_decorator(train_step)(*state.split(), x, y)
      state = nnx.merge(graphdef, state)
      new_loss = loss_fn(*nnx.split(state.model), x, y)

    else:
      loss_fn = lambda model, x, y: ((model(x)-y)**2).mean()
      initial_loss = loss_fn(state.model, x, y)

      def train_step(optimizer: nnx.Optimizer, x, y):
        grads = nnx.grad(loss_fn, wrt=nnx.Param)(optimizer.model, x, y)
        optimizer.update(grads)

      jit_decorator(train_step)(state, x, y)
      new_loss = loss_fn(state.model, x, y)

    self.assertTrue(new_loss < initial_loss)

  @parameterized.product(
    module_cls=[nnx.Linear, Model],
    optimizer=[optax.sgd, optax.adam],
  )
  def test_metrics(self, module_cls, optimizer):
    class TrainState(nnx.Optimizer):
      def __init__(self, model, tx, metrics):
        self.metrics = metrics
        super().__init__(model, tx)
      def update(self, *, grads, **updates):
        self.metrics.update(**updates)
        super().update(grads)

    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optax.adam(1e-3)
    metrics = nnx.metrics.Average()
    state = TrainState(model, tx, metrics)

    loss_fn = lambda model: ((model(x)-y)**2).mean()
    grads = nnx.grad(loss_fn, wrt=nnx.Param)(state.model)
    state.update(grads=grads, values=loss_fn(state.model))
    initial_loss = state.metrics.compute()
    state.update(grads=grads, values=loss_fn(state.model))
    self.assertTrue(state.metrics.compute() < initial_loss)
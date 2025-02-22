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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def assert_equal(path, x, y):
  np.testing.assert_array_equal(x, y, err_msg=f'Mismatch at path: {path}')


def assert_not_equal(path, x, y):
  np.testing.assert_(
      np.any(np.not_equal(x, y)), msg=f'Unexpected match at path: {path}'
  )


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
    optimizer = nnx.Optimizer(model, tx)
    out = optimizer.model(x)
    graphdef, optimizer = nnx.split(optimizer)
    optimizer = nnx.merge(graphdef, optimizer)
    np.testing.assert_allclose(out, optimizer.model(x))

  def test_update(self):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(0.1))

    def loss_fn(model):
      params = nnx.state(model)
      loss = sum(jnp.sum(x**2) for x in jax.tree.leaves(params))
      return loss

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(grads)

  def test_sharding_propagation(self):
    model = nnx.Linear(
        2,
        3,
        rngs=nnx.Rngs(0),
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            sharding=('a', 'b'),
        ),
        use_bias=False,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(0.1))

    state = nnx.state(optimizer)
    partition_spec = nnx.get_partition_spec(state)

    self.assertEqual(state['opt_state'][0]['mu']['kernel'].sharding, ('a', 'b'))
    self.assertEqual(
      partition_spec['opt_state'][0]['mu']['kernel'].value,
      jax.sharding.PartitionSpec('a', 'b'),
    )

  @parameterized.product(
    module_cls=[nnx.Linear, Model],
    jit_decorator=[lambda f: f, nnx.jit, jax.jit],
    optimizer=[optax.sgd, optax.adam],
  )
  def test_jit(self, module_cls, jit_decorator, optimizer):
    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optimizer(
        1e-3
    )  # TODO: this doesn't work with adam optimizer for some reason
    state = nnx.Optimizer(model, tx)

    if jit_decorator == jax.jit:
      model_static, model_state = nnx.split(state.model)
      loss_fn = lambda graphdef, state, x, y: (
          (nnx.merge(graphdef, state)(x) - y) ** 2
      ).mean()
      initial_loss = loss_fn(model_static, model_state, x, y)

      def jax_jit_train_step(graphdef, state, x, y):
        state = nnx.merge(graphdef, state)
        model_static, model_state = nnx.split(state.model)
        grads = jax.grad(loss_fn, argnums=1)(model_static, model_state, x, y)
        state.update(grads)
        return nnx.split(state)

      graphdef, state = jit_decorator(jax_jit_train_step)(
          *nnx.split(state), x, y
      )
      state = nnx.merge(graphdef, state)
      new_loss = loss_fn(*nnx.split(state.model), x, y)

    else:
      loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
      initial_loss = loss_fn(state.model, x, y)

      def nnx_jit_train_step(optimizer: nnx.Optimizer, x, y):
        grads = nnx.grad(loss_fn)(optimizer.model, x, y)
        optimizer.update(grads)

      jit_decorator(nnx_jit_train_step)(state, x, y)
      new_loss = loss_fn(state.model, x, y)

    self.assertTrue(new_loss < initial_loss)

  @parameterized.product(
      module_cls=[nnx.Linear, Model],
      jit_decorator=[lambda f: f, nnx.jit, jax.jit],
      optimizer=[optax.lbfgs],
  )
  def test_jit_linesearch(self, module_cls, jit_decorator, optimizer):
    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optimizer(1e-3)
    state = nnx.Optimizer(model, tx)

    if jit_decorator == jax.jit:
      model_static, model_state = nnx.split(state.model)
      loss_fn = lambda graphdef, state, x, y: (
          (nnx.merge(graphdef, state)(x) - y) ** 2
      ).mean()
      initial_loss = loss_fn(model_static, model_state, x, y)

      def jax_jit_train_step(graphdef, state, x, y):
        state = nnx.merge(graphdef, state)
        model_static, model_state = nnx.split(state.model)
        grads = jax.grad(loss_fn, argnums=1)(model_static, model_state, x, y)
        state.update(
            grads,
            grad=grads,
            value=initial_loss,
            value_fn=lambda state: loss_fn(model_static, state, x, y),
        )
        return nnx.split(state)

      graphdef, state = jit_decorator(jax_jit_train_step)(
          *nnx.split(state), x, y
      )
      state = nnx.merge(graphdef, state)
      new_loss = loss_fn(*nnx.split(state.model), x, y)

    else:
      graphdef = nnx.graphdef(model)
      loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()

      loss_fn_split = lambda state: loss_fn(nnx.merge(graphdef, state), x, y)

      initial_loss = loss_fn(state.model, x, y)

      def nnx_jit_train_step(optimizer: nnx.Optimizer, x, y):
        grads = nnx.grad(loss_fn)(optimizer.model, x, y)
        optimizer.update(
            grads, grad=grads, value=initial_loss, value_fn=loss_fn_split
        )

      jit_decorator(nnx_jit_train_step)(state, x, y)
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

      def update(self, *, grads, **updates):  # type: ignore[signature-mismatch]
        self.metrics.update(**updates)
        super().update(grads)

    x = jax.random.normal(jax.random.key(0), (1, 2))
    y = jnp.ones((1, 4))
    model = module_cls(2, 4, rngs=nnx.Rngs(0))
    tx = optax.adam(1e-3)
    metrics = nnx.metrics.Average()
    state = TrainState(model, tx, metrics)

    loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    grads = nnx.grad(loss_fn)(state.model)
    state.update(grads=grads, values=loss_fn(state.model))
    initial_loss = state.metrics.compute()
    state.update(grads=grads, values=loss_fn(state.model))
    self.assertTrue(state.metrics.compute() < initial_loss)

  @parameterized.parameters(
      {'variable': nnx.Param},
      {'variable': nnx.LoRAParam},
      {'variable': (nnx.Param, nnx.LoRAParam)},
  )
  def test_wrt_update(self, variable):
    in_features = 4
    out_features = 10
    model = nnx.LoRA(
        in_features=in_features,
        lora_rank=2,
        out_features=out_features,
        base_module=Model(
            in_features=in_features, out_features=out_features, rngs=nnx.Rngs(0)
        ),
        rngs=nnx.Rngs(1),
    )
    state = nnx.Optimizer(model, optax.adam(1e-3), wrt=variable)
    prev_variables, prev_other_variables = nnx.state(model, variable, ...)

    x = jnp.ones((1, 4))
    y = jnp.ones((1, 10))
    loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
    grad_fn = nnx.grad(loss_fn, argnums=nnx.DiffState(0, variable))

    def step():
      grads = grad_fn(state.model, x, y)
      initial_loss = loss_fn(model, x, y)
      state.update(grads=grads)
      self.assertTrue(loss_fn(model, x, y) < initial_loss)

    # Since lora_b is initialized to zeros by default, the gradient flow to lora_a
    # will be zeroed out in first call. Thus, run the step twice to make sure
    # lora_a is updated.
    for _ in range(2):
      step()

    # make sure only the Variable's filtered in `wrt` are changed, and the others are unchanged
    variables, other_variables = nnx.state(model, variable, ...)

    jax.tree.map_with_path(assert_not_equal, prev_variables, variables)

    if other_variables:
      jax.tree.map_with_path(
          assert_equal, prev_other_variables, other_variables
      )

  @parameterized.parameters(
      {'variable': nnx.Param},
      # {'variable': nnx.LoRAParam},
      {'variable': (nnx.Param, nnx.LoRAParam)},
  )
  def test_wrt_update_linesearch(self, variable):
    in_features = 4
    out_features = 10
    model = nnx.LoRA(
        in_features=in_features,
        lora_rank=2,
        out_features=out_features,
        base_module=Model(
            in_features=in_features, out_features=out_features, rngs=nnx.Rngs(0)
        ),
        rngs=nnx.Rngs(1),
    )
    state = nnx.Optimizer(model, optax.lbfgs(), wrt=variable)
    prev_variables, prev_other_variables = nnx.state(model, variable, ...)

    x = jnp.ones((1, 4))
    y = jnp.ones((1, 10))
    loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()

    grad_fn = nnx.grad(loss_fn, argnums=nnx.DiffState(0, variable))
    graphdef = nnx.graphdef(model)
    loss_fn_split = lambda state: loss_fn(nnx.merge(graphdef, state), x, y)

    def step():
      grads = grad_fn(state.model, x, y)
      initial_loss = loss_fn(model, x, y)
      state.update(
          grads, grad=grads, value_fn=loss_fn_split, value=initial_loss
      )
      self.assertTrue(loss_fn(model, x, y) < initial_loss)

    # Since lora_b is initialized to zeros by default, the gradient flow to lora_a
    # will be zeroed out in first call. Thus, run the step twice to make sure
    # lora_a is updated.
    for _ in range(2):
      step()

    # make sure only the Variable's filtered in `wrt` are changed, and the others are unchanged
    variables, other_variables = nnx.state(model, variable, ...)

    jax.tree.map_with_path(assert_not_equal, prev_variables, variables)

    if other_variables:
      jax.tree.map_with_path(
          assert_equal, prev_other_variables, other_variables
      )


if __name__ == '__main__':
  absltest.main()

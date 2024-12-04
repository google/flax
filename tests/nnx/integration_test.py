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

import tempfile
import typing as tp

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from flax import nnx

A = tp.TypeVar('A')


class TestIntegration(absltest.TestCase):
  def test_shared_modules(self):
    class Block(nnx.Module):
      def __init__(self, linear: nnx.Linear, *, rngs):
        self.linear = linear
        self.bn = nnx.BatchNorm(2, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return nnx.relu(x)

    class Model(nnx.Module):
      def __init__(self, *, rngs):
        shared = nnx.Linear(2, 2, rngs=rngs)
        self.block1 = Block(shared, rngs=rngs)
        self.block2 = Block(shared, rngs=rngs)

      def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    @nnx.jit
    def train_step(model: Model, x, y):
      @nnx.grad
      def loss_fn(model: Model):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

      grads = loss_fn(model)
      nnx.update(
        model,
        jax.tree.map(
          lambda w, g: w - 0.1 * g, nnx.state(model, nnx.Param), grads
        ),
      )

    model = Model(rngs=nnx.Rngs(0))

    x = np.random.uniform(size=(4, 2))
    y = np.random.uniform(size=(4, 2))
    model.set_attributes(use_running_average=False)

    for _i in range(3):
      train_step(model, x, y)

    assert model.block1.linear is model.block2.linear
    assert model.block1.linear.bias is not None
    assert model.block1.bn is not model.block2.bn

  def test_shared_modules_pure(self):
    class Block(nnx.Module):
      def __init__(self, linear: nnx.Linear, *, rngs: nnx.Rngs):
        self.linear = linear
        self.bn = nnx.BatchNorm(2, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return nnx.relu(x)

    class Model(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        shared = nnx.Linear(2, 2, rngs=rngs)
        self.block1 = Block(shared, rngs=rngs)
        self.block2 = Block(shared, rngs=rngs)

      def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    @jax.jit
    def train_step(state: nnx.State, graphdef: nnx.GraphDef[Model], x, y):
      model = nnx.merge(graphdef, state)
      model.set_attributes(use_running_average=False)

      @nnx.grad
      def loss_fn(model: Model):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

      grads = loss_fn(model)
      nnx.update(
        model,
        jax.tree.map(
          lambda w, g: w - 0.1 * g, nnx.state(model, nnx.Param), grads
        ),
      )

      return nnx.split(model)

    graphdef: nnx.GraphDef[Model]
    graphdef, state = nnx.split(Model(rngs=nnx.Rngs(0)))

    x = np.random.uniform(size=(4, 2))
    y = np.random.uniform(size=(4, 2))

    for _i in range(3):
      graphdef, state = train_step(state, graphdef, x, y)

    model = nnx.merge(graphdef, state)

    assert model.block1.linear.bias is not None
    assert model.block2.linear.bias is not None
    assert model.block1.linear.kernel is model.block2.linear.kernel
    assert model.block1.linear.bias is model.block2.linear.bias
    assert model.block1.bn is not model.block2.bn

  def test_stateful_example(self):
    class State(nnx.Variable[A]):
      pass

    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = State(0)

      def __call__(self, x):
        self.count.value += 1
        return x @ self.w.value + self.b.value[None]

    model = Linear(din=12, dout=2, rngs=nnx.Rngs(0))
    # forward pass
    x = jnp.ones((8, 12))
    y = model(x)
    assert model.count.value == 1

    @nnx.jit
    def train_step(model, x, y):
      def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)

      # compute gradient
      grads: nnx.State = nnx.grad(loss_fn)(model)
      # SGD update
      nnx.update(
        model,
        jax.tree.map(
          lambda w, g: w - 0.1 * g, nnx.state(model, nnx.Param), grads
        ),
      )

    # execute the training step
    train_step(model, x, y)
    assert model.count.value == 2

  def test_functional_example(self):
    class Count(nnx.Variable[A]):
      pass

    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = Count(0)

      def __call__(self, x):
        self.count.value += 1
        return x @ self.w.value + self.b.value[None]

    model = Linear(din=12, dout=2, rngs=nnx.Rngs(0))
    # forward pass
    x = jnp.ones((8, 12))
    y = model(x)
    assert model.count.value == 1

    graphdef, params, counts = nnx.split(model, nnx.Param, Count)

    @jax.jit
    def train_step(params, counts, x, y):
      def loss_fn(params):
        y_pred, (_, updates) = graphdef.apply(params, counts)(x)
        loss = jax.numpy.mean((y_pred - y) ** 2)
        return loss, nnx.filter_state(updates, Count)

      # compute gradient
      grads, counts = jax.grad(loss_fn, has_aux=True)(params)
      # SGD update
      params = jax.tree.map(lambda w, g: w - 0.1 * g, params, grads)

      return params, counts

    # execute the training step
    params, counts = train_step(params, counts, x, y)
    model = nnx.merge(graphdef, params, counts)
    assert model.count.value == 2

  def test_intermediates_example(self):
    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

      def __call__(self, x):
        y = x @ self.w.value + self.b.value[None]
        self.y = nnx.Intermediate(y)
        return y

    model = Linear(12, 2, rngs=nnx.Rngs(0))

    y = model(jnp.ones((8, 12)))

    intermediates = nnx.pop(model, nnx.Intermediate)

    assert 'y' in intermediates

  def test_intermediates_example_functional(self):
    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

      def __call__(self, x):
        y = x @ self.w.value + self.b.value[None]
        self.y = nnx.Intermediate(y)
        return y

    model = Linear(12, 2, rngs=nnx.Rngs(0))

    graphdef, state = nnx.split(model)

    y, (_, state) = graphdef.apply(state)(jnp.ones((8, 12)))

    intermediates, state = nnx.split_state(state, nnx.Intermediate, ...)

    assert 'y' in intermediates

  def test_replace_by_pure_dict(self):
    class MLPs(nnx.Module):
      def __init__(self, dim, rngs: nnx.Rngs):
        self.layers = []
        for _ in range(4):
          self.layers.append(nnx.Linear(dim, dim, rngs=rngs, use_bias=False))

      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    model = MLPs(4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(42), (3, 4))
    assert model(x).shape == (3, 4)

    _, state = nnx.split(model)
    pure_dict_state = nnx.to_pure_dict(state)
    nnx.display(pure_dict_state)

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = ocp.test_utils.erase_and_create_empty(
        tmpdir + '/my-checkpoints/'
      )
      checkpointer = ocp.StandardCheckpointer()
      # checkpointer.save(ckpt_dir / 'state', state)
      checkpointer.save(ckpt_dir / 'pure_dict', pure_dict_state)

      # Restore as a pure dictionary.
      restored_pure_dict = checkpointer.restore(ckpt_dir / 'pure_dict')
      nnx.display(restored_pure_dict)

      abstract_model = nnx.eval_shape(lambda: MLPs(4, rngs=nnx.Rngs(0)))
      graphdef, abstract_state = nnx.split(abstract_model)
      nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
      model = nnx.merge(graphdef, abstract_state)
      assert model(x).shape == (3, 4)  # The model still works!


if __name__ == '__main__':
  absltest.main()

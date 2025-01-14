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

# %%
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from time import time

from flax import nnx

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  'mode', 'all', ['all', 'nnx', 'jax'], 'Mode to run the script in'
)
flags.DEFINE_integer('total_steps', 10_000, 'Total number of training steps')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('width', 32, 'Hidden layer size')
flags.DEFINE_integer('depth', 5, 'Depth of the model')


def dataset(X, Y, batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))

  def __call__(self, x):
    return x @ self.w + self.b

class Block(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.linear = Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)

  def __call__(self, x):
    return nnx.relu(self.bn(self.linear(x)))

class Count(nnx.Variable):
  pass


class MLP(nnx.Module):
  def __init__(self, din, dhidden, dout, depth, *, rngs: nnx.Rngs):
    self.count = Count(jnp.array(0))
    self.linear_in = Block(din, dhidden, rngs=rngs)
    self.intermediates = [
      Block(dhidden, dhidden, rngs=rngs) for _ in range(depth - 2)
    ]
    self.linear_out = Block(dhidden, dout, rngs=rngs)

  def __call__(self, x):
    self.count.value += 1
    x = nnx.relu(self.linear_in(x))
    for layer in self.intermediates:
      x = nnx.relu(layer(x))
    x = self.linear_out(x)
    return x


def main(argv):
  print(argv)
  mode: str = FLAGS.mode
  total_steps: int = FLAGS.total_steps
  batch_size: int = FLAGS.batch_size
  width: int = FLAGS.width
  depth: int = FLAGS.depth

  print(f'{mode=}, {total_steps=}, {batch_size=}, {width=}')

  X = np.linspace(0, 1, 100)[:, None]
  Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)

  if mode == 'nnx' or mode == 'all':
    model = MLP(din=1, dhidden=width, dout=1, depth=depth, rngs=nnx.Rngs(0))
    tx = optax.sgd(1e-3)
    optimizer = nnx.Optimizer(model, tx)
    t0 = time()

    @nnx.jit(donate_argnums=(0, 1))
    def train_step_nnx(model: MLP, optimizer: nnx.Optimizer, batch):
      x, y = batch

      def loss_fn(model: MLP):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

      grads: nnx.State = nnx.grad(loss_fn)(model)
      optimizer.update(grads)

    @nnx.jit(donate_argnums=0)
    def test_step_nnx(model: MLP, batch):
      x, y = batch
      y_pred = model(x)
      loss = jnp.mean((y - y_pred) ** 2)
      return {'loss': loss}

    cached_train_step_nnx = nnx.cached_partial(train_step_nnx, model, optimizer)
    cached_test_step_nnx = nnx.cached_partial(test_step_nnx, model)

    for step, batch in enumerate(dataset(X, Y, batch_size)):
      cached_train_step_nnx(batch)

      if step % 1000 == 0:
        logs = cached_test_step_nnx((X, Y))

      if step >= total_steps - 1:
        break

    print('### NNX ###')
    print(f"final loss: {logs['loss']}")
    total_time = time() - t0
    print('total time:', total_time)
    print(f'time per step: {total_time / total_steps * 1e6:.2f} µs')
    print('times called:', model.count.value)

  if mode == 'jax' or mode == 'all':
    model = MLP(din=1, dhidden=width, dout=1, depth=depth, rngs=nnx.Rngs(0))
    tx = optax.sgd(1e-3)
    optimizer = nnx.Optimizer(model, tx)
    t0 = time()

    @partial(jax.jit, donate_argnums=0)
    def train_step_jax(state, batch):
      model, optimizer = nnx.merge(graphdef, state)
      x, y = batch

      def loss_fn(model: MLP):
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

      grads = nnx.grad(loss_fn)(model)
      optimizer.update(grads)

      return nnx.state((model, optimizer))

    @partial(jax.jit, donate_argnums=0)
    def test_step_jax(state, batch):
      model, optimizer = nnx.merge(graphdef, state)
      x, y = batch
      y_pred = model(x)
      loss = jnp.mean((y - y_pred) ** 2)
      state = nnx.state((model, optimizer))
      return state, {'loss': loss}

    graphdef, state = nnx.split((model, optimizer))

    for step, batch in enumerate(dataset(X, Y, batch_size)):
      state = train_step_jax(state, batch)

      if step % 1000 == 0:
        state, logs = test_step_jax(state, (X, Y))

      if step >= total_steps - 1:
        break

    model, optimizer = nnx.merge(graphdef, state)

    print('### JAX ###')
    print(f"final loss: {logs['loss']}")
    total_time = time() - t0
    print('total time:', total_time)
    print(f'time per step: {total_time / total_steps * 1e6:.2f} µs')
    print('times called:', model.count.value)


if __name__ == '__main__':
  app.run(main)

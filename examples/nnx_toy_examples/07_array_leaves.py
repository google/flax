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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from flax import nnx, struct

X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = jax.random.normal(rngs.params(), (din, dout))
    self.b = jnp.zeros((dout,))

  def __call__(self, x):
    return x @ self.w + self.b


class MLP(nnx.Module):
  def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
    self.count = jnp.array(0)
    self.linear1 = Linear(din, dhidden, rngs=rngs)
    self.linear2 = Linear(dhidden, dout, rngs=rngs)

  def __call__(self, x):
    self.count += 1
    return self.linear2(nnx.relu(self.linear1(x)))

def is_param(path, value):
  key = path[-1]
  return key == 'w' or key == 'b'

model = MLP(din=1, dhidden=32, dout=1, rngs=nnx.Rngs(0))
tx = optax.sgd(1e-3)
optimizer = nnx.Optimizer(model, tx, wrt=is_param)


@nnx.jit
def train_step(model: MLP, optimizer: nnx.Optimizer, batch):
  x, y = batch

  def loss_fn(model: MLP):
    y_pred = model(x)
    return jnp.mean((y - y_pred) ** 2)

  diff_state = nnx.DiffState(0, is_param)
  grads: nnx.State = nnx.grad(loss_fn, argnums=diff_state)(model)
  optimizer.update(grads)


@nnx.jit
def test_step(model: MLP, batch):
  x, y = batch
  y_pred = model(x)
  loss = jnp.mean((y - y_pred) ** 2)
  return {'loss': loss}


total_steps = 10_000
for step, batch in enumerate(dataset(32)):
  train_step(model, optimizer, batch)

  if step % 1000 == 0:
    logs = test_step(model, (X, Y))
    print(f"step: {step}, loss: {logs['loss']}")

  if step >= total_steps - 1:
    break

print('times called:', model.count)

y_pred = model(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()

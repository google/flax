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

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental import mesh_utils
import matplotlib.pyplot as plt

# create a mesh + shardings
num_devices = jax.local_device_count()
mesh = jax.sharding.Mesh(
  mesh_utils.create_device_mesh((num_devices,)), ('data',)
)
model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))


# create model
class MLP(nnx.Module):
  def __init__(self, din, dmid, dout, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    return self.linear2(nnx.relu(self.linear1(x)))


model = MLP(1, 64, 1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))

# replicate state
state = nnx.state((model, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((model, optimizer), state)

# visualize model sharding
print('model sharding')
jax.debug.visualize_array_sharding(model.linear1.kernel.value)


@nnx.jit
def train_step(model: MLP, optimizer: nnx.Optimizer, x, y):
  def loss_fn(model: MLP):
    y_pred = model(x)
    return jnp.mean((y - y_pred) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  return loss


def dataset(steps, batch_size):
  for _ in range(steps):
    x = np.random.uniform(-2, 2, size=(batch_size, 1))
    y = 0.8 * x**2 + 0.1 + np.random.normal(0, 0.1, size=x.shape)
    yield x, y


for step, (x, y) in enumerate(dataset(1000, 16)):
  # shard data
  x, y = jax.device_put((x, y), data_sharding)
  # train
  loss = train_step(model, optimizer, x, y)

  if step == 0:
    print('data sharding')
    jax.debug.visualize_array_sharding(x)

  if step % 100 == 0:
    print(f'step={step}, loss={loss}')

# dereplicate state
state = nnx.state((model, optimizer))
state = jax.device_get(state)
nnx.update((model, optimizer), state)

X, Y = next(dataset(1, 1000))
x_range = np.linspace(X.min(), X.max(), 100)[:, None]
y_pred = model(x_range)

# plot
plt.scatter(X, Y, label='data')
plt.plot(x_range, y_pred, color='black', label='model')
plt.legend()
plt.show()
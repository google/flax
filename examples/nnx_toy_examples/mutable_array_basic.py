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
import os

os.environ['FLAX_MUTABLE_ARRAY'] = 'true'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from flax import nnx
from flax.nnx.variablelib import is_mutable_array


def mutable_like(path, x):
  return (
    isinstance(x, nnx.Variable) and x.mutable
  ) or nnx.variablelib.is_mutable_array(x)


def freeze(x, only: nnx.filterlib.Filter = mutable_like):
  freeze_filter = nnx.filterlib.to_predicate(only)
  mutable_arrays: set[int] = set()

  def check_mutable_array(path, x):
    m_array_id = id(x)
    if m_array_id in mutable_arrays:
      path_str = jax.tree_util.keystr(path)
      raise ValueError(
        f'Found duplicate MutableArray found at path {path_str}: {x}'
      )
    mutable_arrays.add(m_array_id)

  def _freeze_fn(jax_path, x):
    path = tuple(nnx.graph._key_path_to_key(part) for part in jax_path)
    if freeze_filter(path, x):
      if isinstance(x, nnx.Variable):
        check_mutable_array(jax_path, x.raw_value)
        return x.from_metadata(x[...], x.get_metadata().copy())
      elif nnx.variablelib.is_mutable_array(x):
        check_mutable_array(jax_path, x)
        return x[...]
    return x

  return jax.tree.map_with_path(
    _freeze_fn, x, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )

X = np.linspace(-jnp.pi, jnp.pi, 100)[:, None]
Y = 0.8 * jnp.sin(X) + 0.1 + np.random.normal(0, 0.1, size=X.shape)

def dataset(batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


class Linear(nnx.Module):
  __data__ = ('w', 'b')

  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))

  def __call__(self, x):
    return x @ self.w[...] + self.b[None]


class Count(nnx.Variable[nnx.A]):
  pass


class MLP(nnx.Module):
  __data__ = ('count', 'linear1', 'linear2')

  def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
    self.count = Count(jnp.array(0))
    self.linear1 = Linear(din, dhidden, rngs=rngs)
    self.linear2 = Linear(dhidden, dout, rngs=rngs)

  def __call__(self, x):
    self.count[...] += 1
    return self.linear2(jax.nn.gelu(self.linear1(x)) * 0.5)


model = MLP(din=1, dhidden=64, dout=1, rngs=nnx.Rngs(0))


@jax.jit
def train_step(model, x, y):
  graphdef, params, counts = nnx.split(model, nnx.Param, Count)

  def loss_fn(params):
    model = nnx.merge(graphdef, params, counts)
    return jnp.mean((y - model(x)) ** 2)

  grads = jax.grad(loss_fn)(freeze(params))

  def sgd(w, g):
    w[...] -= 0.1 * g[...]

  jax.tree.map(sgd, params, grads)


@jax.jit
def test_step(model: MLP, x, y):
  return {'loss': jnp.mean((y - model(x)) ** 2)}


total_steps = 10_000
for step, (x, y) in enumerate(dataset(32)):
  train_step(model, x, y)

  if step % 1000 == 0:
    logs = test_step(model, X, Y)
    print(f"step: {step}, loss: {logs['loss']}")

  if step >= total_steps - 1:
    break

print('times called:', model.count.value)

y_pred = model(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()

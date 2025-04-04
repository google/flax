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

from flax.experimental import nx

# -------------------------------------------------
# temporary fix for mutable_array
# -------------------------------------------------
from jax._src import core
from jax._src.interpreters import batching

batching.defvectorized(core.mutable_array_p)

# -------------------------------------------------
# Data
# -------------------------------------------------
X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


# -------------------------------------------------
# Model
# -------------------------------------------------
class Linear(nx.Pytree):
  __nodes__ = ('w', 'b')

  def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
    self.din, self.dout = din, dout
    initializer = jax.nn.initializers.lecun_normal()
    self.w = nx.Param(initializer(rngs.params(), (din, dout)))
    self.b = nx.Param(jnp.zeros((dout,)))

  def __call__(self, x: jax.Array):
    return x @ self.w[...] + self.b[None]


class Block(nx.Pytree):
  __nodes__ = ('w', 'b', 'mean', 'var', 'scale', 'bias', 'rng')

  def __init__(
    self,
    din: int,
    dout: int,
    *,
    dropout_rate: float = 0.1,
    moumentum: float = 0.99,
    use_stats: bool = False,
    deterministic: bool = False,
    rngs: nx.Rngs,
  ):
    # ----------- linear -------------------
    self.din, self.dout = din, dout
    initializer = jax.nn.initializers.lecun_normal()
    self.w = nx.Param(initializer(rngs.params(), (din, dout)))
    self.b = nx.Param(jnp.zeros((dout,)))
    # ----------- batch norm ---------------
    self.mu = moumentum  # momentum
    self.use_stats = use_stats
    self.mean = nx.BatchStat(jnp.zeros((dout,)))
    self.var = nx.BatchStat(jnp.ones((dout,)))
    self.scale = nx.Param(jnp.ones((dout,)))
    self.bias = nx.Param(jnp.zeros((dout,)))
    # ----------- dropout ------------------
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.rng = rngs.dropout.fork()

  def __call__(self, x: jax.Array) -> jax.Array:
    # ----------- linear --------------------
    x = x @ self.w[...] + self.b[None]
    # ----------- batch norm ----------------
    if self.use_stats:
      mean = self.mean[...]
      var = self.var[...]
    else:
      mean = jnp.mean(x, axis=0)
      var = jnp.var(x, axis=0)
      # ema updates. using stop gradient until mutable array in grad is fixed.
      sg = jax.lax.stop_gradient
      self.mean[...] = sg(self.mu * self.mean[...] + (1 - self.mu) * mean)
      self.var[...] = sg(self.mu * self.var[...] + (1 - self.mu) * var)
    # normalize and scale
    x = (x - mean[None]) / jnp.sqrt(var[None] + 1e-5)
    x = x * self.scale[...] + self.bias[...]
    # ----------- dropout -------------------
    if not self.deterministic and self.dropout_rate > 0.0:
      keep_prob = 1.0 - self.dropout_rate
      mask = jax.random.bernoulli(self.rng(), keep_prob, x.shape)
      x = jnp.where(mask, x / keep_prob, jnp.zeros_like(x))
    # ----------- activation ---------------
    x = jax.nn.relu(x)
    return x


class Count(nx.Variable): ...


class Model(nx.Pytree):
  __nodes__ = ('block_in', 'blocks', 'linear_out', 'count')

  def __init__(
    self,
    num_blocks: int,
    din: int,
    dhidden: int,
    dout: int,
    *,
    use_scan: bool = True,
    rngs: nx.Rngs,
  ):
    self.count = Count(jnp.array(0))
    self.block_in = Block(din, dhidden, rngs=rngs)
    self.linear_out = Linear(dhidden, dout, rngs=rngs)

    if use_scan:

      @jax.vmap
      def create_block(rngs, /):
        return Block(dhidden, dhidden, rngs=rngs)

      self.blocks = create_block(nx.split_rngs(rngs, num_blocks))
    else:
      self.blocks = [
        Block(dhidden, dhidden, rngs=rngs) for i in range(num_blocks)
      ]

  def __call__(self, x: jax.Array):
    self.count[...] += 1
    x = self.block_in(x)

    if isinstance(self.blocks, list):
      for block in self.blocks:
        x = block(x)
    else:
      # scan over layers
      def block_fw(x, block: Block):
        x = block(x)
        return x, None

      x, _ = jax.lax.scan(block_fw, x, self.blocks)
    x = self.linear_out(x)
    return x


# -------------------------------------------------
# Optimizer
# -------------------------------------------------


class OptState(nx.Variable): ...


class SGD(nx.Pytree):
  __nodes__ = ('momentum',)

  def __init__(self, params, lr: float, decay: float = 0.9):
    self.lr = lr
    self.decay = decay
    self.momentum = jax.tree.map(
      lambda x: OptState(jnp.zeros_like(x), **x.metadata),
      params,
      is_leaf=lambda x: isinstance(x, nx.Variable),
    )

  def update(self, params, grads):
    def update_fn(param: nx.Variable, momentum: OptState, grad: nx.Variable):
      momentum[...] = self.decay * momentum[...] + (1 - self.decay) * grad[...]
      param[...] -= self.lr * momentum[...]

    is_variable = lambda x: isinstance(x, nx.Variable)
    jax.tree.map(update_fn, params, self.momentum, grads, is_leaf=is_variable)


# -------------------------------------------------
# Training
# -------------------------------------------------
def eval_mode(path, x):
  if isinstance(x, Block):
    return x.replace(use_stats=True, deterministic=True)
  return x


model = Model(3, din=1, dhidden=64, dout=1, use_scan=False, rngs=nx.Rngs(0))
optimizer = SGD(params=nx.state(model, nx.Param), lr=1e-3, decay=0.99)
model, optimizer = nx.mutable((model, optimizer))
model_eval = nx.recursive_map(model, eval_mode)


@jax.jit
def train_step(model: Model, optimizer: nx.OptaxOptimizer, x, y):
  treedef, params, nondiff = nx.split(model, nx.Param, ...)

  def loss_fn(params):
    model = nx.merge(treedef, params, nondiff)
    loss = jnp.mean((model(x) - y) ** 2)
    return loss

  grads = jax.grad(loss_fn)(nx.freeze(params))
  optimizer.update(params, grads)


@jax.jit
def test_step(model: Model, x, y):
  loss = jnp.mean((model(x) - y) ** 2)
  return {'loss': loss}


total_steps = 10_000
for step, (x, y) in enumerate(dataset(32)):
  train_step(model, optimizer, x, y)

  if step % 1000 == 0:
    logs = test_step(model_eval, X, Y)
    print(f'step: {step}, loss: {logs["loss"]}')

  if step >= total_steps - 1:
    break

# -------------------------------------------------
# Sample
# -------------------------------------------------


print('times called:', model_eval.count[...])

y_pred = model_eval(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()

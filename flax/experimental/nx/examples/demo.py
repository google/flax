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
from jax._src.core import MutableArray

from flax.experimental import nx

# temporary fix for mutable_array
from jax._src import core
from jax._src.interpreters import batching

batching.defvectorized(core.mutable_array_p)

# ## Data
# We create a simple dataset of points sampled from a parabola with some noise.
X = np.linspace(-1, 1, 200)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


# ## Model
# Here we define a MLP made of a stack of blocks. Each block contains a linear layer,
# batch normalization, and a dropout layer.
#
# In this version we want the Modules to be pytrees so they can be used with JAX transforms
# so we use a new Pytree type as the base. Pytree implements the pytree protocol but tries
# to have a programming model that looks more like a regular python class e.g. uses __init__
# and __call__, but instance are frozen after __init__ is done. A big sintactic difference
# with current NNX is that users have to specify which attributes are nodes using the __nodes__
# class variable (similar to __slots__).
#
# Variable changes in a couple of ways:
# * its now implements the pytree protocol
# * it can only hold arrays
# * it has a mutable attribute, when True it will hold a MutableArray
# * its immutable
# * [...] is used to access & mutate underlying array
#
class Linear(nx.Pytree):
  __nodes__ = ('w', 'b')

  # we mark 'w' and 'b' as nodes, the rest of the attributes are
  # are treated as static.
  def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
    self.din, self.dout = din, dout
    initializer = jax.nn.initializers.lecun_normal()
    # Param, BatchState, and Cache are built-in Variable subtypes
    self.w = nx.Param(initializer(rngs.params(), (din, dout)))
    self.b = nx.Param(jnp.zeros((dout,)))

  # [...] is used to access the array
  def __call__(self, x: jax.Array):
    return x @ self.w[...] + self.b[None]


# Block implements linear, batch norm, and dropout. Its behavior
# is controlled by the 'use_stats' and 'deterministic' flags.
class Block(nx.Pytree):
  __nodes__ = ('w', 'b', 'mean', 'var', 'scale', 'bias', 'rng')

  def __init__(
    self,
    din: int,
    dout: int,
    *,
    dropout_rate: float = 0.05,
    moumentum: float = 0.95,
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
    # 'fork' is used to get a derived frozen stream, this is done
    # to avoid aliasing MutableArray as as its not supported by JAX
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
      # ema updates
      # stop gradient is used until a MutableArray supports updates from grad tracers
      sg = jax.lax.stop_gradient
      self.mean[...] = sg(self.mu * self.mean[...] + (1 - self.mu) * mean)
      self.var[...] = sg(self.mu * self.var[...] + (1 - self.mu) * var)
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


# Trivial Variables subclasses are used to easily
# query state groups. In this case Count will be used to hold
# non-differentiable state containing the number of times the model
# is called.
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

    # 'blocks' is either a list of blocks or single block
    # whose parameters contain an additional 'layer' dimension,
    # here created using jax.vmap
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

    # on the forward pass we either iterate over the block
    # list or use jax.lax.scan to apply the blocks, if we
    # had shared state we would use split and merge to
    # pass the shared state as a capture
    if isinstance(self.blocks, list):
      for block in self.blocks:
        x = block(x)
    else:

      def block_fw(x, block: Block):
        x = block(x)
        return x, None

      x, _ = jax.lax.scan(block_fw, x, self.blocks)
    x = self.linear_out(x)
    return x


# ## Optimizer


class OptState(nx.Variable): ...


# Optimizer are an interesting case as they are inherently stateful and
# pose a good use case for MutableArray. Here we implement SGD with
# momentum. The optimizer receives the params as constructor arguments but doesn't
# hold a reference to them, it only uses the params to initialize its state
# by creating new OptState Variables that reuse the param's metadata.
class SGD(nx.Pytree):
  __nodes__ = ('momentum',)

  def __init__(self, params, lr: float, decay: float = 0.9):
    self.lr = lr
    self.decay = decay

    def make_opt_state(x):
      if isinstance(x, nx.Variable):
        return OptState(jnp.zeros_like(x), **x.metadata)
      else:
        return OptState(jnp.zeros_like(x))

    self.momentum = jax.tree.map(
      make_opt_state, params, is_leaf=lambda x: isinstance(x, nx.Variable)
    )

  # during the update we simply map over (params, momentum, grads),
  # for each triplet we implement the SGD update rule which updates
  # both the optimizer's state (momentum) and the params in place.
  def update(self, params, grads):
    def update_fn(param: MutableArray, momentum: OptState, grad: jax.Array):
      momentum[...] = self.decay * momentum[...] + (1 - self.decay) * grad[...]
      param[...] -= self.lr * momentum[...]

    is_variable = lambda x: isinstance(x, nx.Variable)
    jax.tree.map(update_fn, params, self.momentum, grads, is_leaf=is_variable)


# ## Training
# To setup the training loop we first instantiate the model and optimizer.
# Variables are immutable (only contain Arrays) by default as it can make
# initialization easier, however this means we have to use 'mutable' to
# create the MutableArrays that will be updated during training.


model = Model(
  num_blocks=3, din=1, dhidden=256, dout=1, use_scan=False, rngs=nx.Rngs(0)
)
optimizer = SGD(params=nx.state(model, nx.Param), lr=3e-3, decay=0.99)
model, optimizer = nx.mutable((model, optimizer))


# We also need to create a version of the model that can be used for evaluation.
# This is done by leveraging 'recursive_map' to replace the Block's 'use_stats' and
# 'deterministic' flags with their evaluation values. This returns a new model
# but uses the same parameters as the original model so it will get automatically
# updated during training.
def eval_mode(path, x):
  if isinstance(x, Block):
    return x.replace(use_stats=True, deterministic=True)
  return x


model_eval = nx.recursive_map(model, eval_mode)


# The training step uses 'jax.jit' and receives the model and optimizer as arguments,
# this is supported as they are now pytrees. The first thing we do is group the model
# state into the params and the non-differentiable state using 'split'. We differentiate
# the loss function using 'jax.grad' with respect to the params-only. Inside the loss
# function we merge the params and non-diff state back into a single model and then
# compute the loss by calling the model with the inputs.
@jax.jit
def train_step(model: Model, optimizer: nx.OptaxOptimizer, x, y):
  treedef, params, nondiff = nx.split(model, nx.Param, ...)

  def loss_fn(params):
    model = nx.merge(treedef, params, nondiff)
    loss = jnp.mean((model(x) - y) ** 2)
    return loss

  # For the time being we have to use 'freeze' make the Variables immutable
  # as 'jax.grad' doesn't support MutableArrays yet.
  grads = jax.grad(loss_fn)(nx.freeze(params))
  # 'update' mutates the optimizer's state and the params in place
  # so we don't need to return anything 🚀
  optimizer.update(params, grads)


# simple test step that computes the loss
@jax.jit
def test_step(model: Model, x, y):
  return {'loss': jnp.mean((model(x) - y) ** 2)}


# minimalistic training loop
total_steps = 10_000
for step, (x, y) in enumerate(dataset(32)):
  train_step(model, optimizer, x, y)

  if step % 1000 == 0:
    logs = test_step(model_eval, X, Y)
    print(f'step: {step}, loss: {logs["loss"]}')

  if step >= total_steps - 1:
    break

# ## Sample
# Sampling is trivial, just use 'model_eval'

print('times called:', model_eval.count[...])

y_pred = model_eval(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()

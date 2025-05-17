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


# ## Data
# We create a simple dataset of points sampled from a parabola with some noise.
X = np.linspace(-jnp.pi, jnp.pi, 100)[:, None]
Y = 0.8 * jnp.sin(X) + 0.1 + np.random.normal(0, 0.1, size=X.shape)


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
# with current NNnX is that users have to specify which attributes are nodes using the __data__
# class variable (similar to __slots__).
#
# Variable changes in a couple of ways:
# * its now implements the pytree protocol
# * it can only hold arrays
# * it has a mutable attribute, when True it will hold a MutableArray
# * its immutable
# * [...] is used to access & mutate underlying array
#
class Linear(nnx.Module):
  __data__ = ('w', 'b')

  # we mark 'w' and 'b' as nodes, the rest of the attributes are
  # are treated as static.
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.din, self.dout = din, dout
    initializer = jax.nn.initializers.lecun_normal()
    # Param, BatchState, and Cache are built-in Variable subtypes
    self.w = nnx.Param(initializer(rngs.params(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))

  # [...] is used to access the array
  def __call__(self, x: jax.Array):
    return x @ self.w[...] + self.b[None]


# Block implements linear, batch norm, and dropout. Its behavior
# is controlled by the 'use_stats' and 'deterministic' flags.
class Block(nnx.Module):
  __data__ = ('w', 'b', 'mean', 'var', 'scale', 'bias', 'rng')

  def __init__(
    self,
    din: int,
    dout: int,
    *,
    dropout_rate: float = 0.05,
    moumentum: float = 0.95,
    use_stats: bool = False,
    deterministic: bool = False,
    rngs: nnx.Rngs,
  ):
    # ----------- linear -------------------
    self.din, self.dout = din, dout
    initializer = jax.nn.initializers.lecun_normal()
    self.w = nnx.Param(initializer(rngs.params(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    # ----------- batch norm ---------------
    self.mu = moumentum  # momentum
    self.use_stats = use_stats
    self.mean = nnx.BatchStat(jnp.zeros((dout,)))
    self.var = nnx.BatchStat(jnp.ones((dout,)))
    self.scale = nnx.Param(jnp.ones((dout,)))
    self.bias = nnx.Param(jnp.zeros((dout,)))
    # ----------- dropout ------------------
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic

  def __call__(
    self, x: jax.Array, *, rngs: nnx.Rngs | None = None
  ) -> jax.Array:
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
      assert rngs is not None
      keep_prob = 1.0 - self.dropout_rate
      mask = jax.random.bernoulli(rngs.dropout(), keep_prob, x.shape)
      x = jnp.where(mask, x / keep_prob, jnp.zeros_like(x))
    # ----------- activation ---------------
    x = jax.nn.gelu(x)
    return x


class Model(nnx.Module):
  __data__ = ('block_in', 'blocks', 'linear_out', 'count')

  def __init__(
    self,
    num_blocks: int,
    din: int,
    dhidden: int,
    dout: int,
    *,
    use_scan: bool = True,
    rngs: nnx.Rngs,
  ):
    self.count: nnx.MutableArray = nnx.mutable_array(jnp.array(0))
    self.block_in = Block(din, dhidden, rngs=rngs)
    self.linear_out = Linear(dhidden, dout, rngs=rngs)

    # 'blocks' is either a list of blocks or single block
    # whose parameters contain an additional 'layer' dimension,
    # here created using jax.vmap
    if use_scan:

      @jax.vmap
      def create_block(rngs, /):
        return nnx.freeze(Block(dhidden, dhidden, rngs=rngs))

      self.blocks = nnx.mutable(create_block(rngs.fork(split=num_blocks)))
    else:
      self.blocks = [
        Block(dhidden, dhidden, rngs=rngs) for i in range(num_blocks)
      ]

  def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None):
    self.count[...] += 1
    x = self.block_in(x, rngs=rngs)

    # on the forward pass we either iterate over the block
    # list or use jax.lax.scan to apply the blocks, if we
    # had shared state we would use split and merge to
    # pass the shared state as a capture
    if isinstance(self.blocks, list):
      for block in self.blocks:
        x = block(x, rngs=rngs)
    else:

      def block_fw(x, block: Block):
        x = block(x, rngs=rngs)
        return x, None

      x, _ = jax.lax.scan(block_fw, x, self.blocks)
    x = self.linear_out(x)
    return x


# ## Optimizer
class OptState(nnx.Variable): ...


# Optimizer are an interesting case as they are inherently stateful and
# pose a good use case for MutableArray. Here we implement SGD with
# momentum. The optimizer receives the params as constructor arguments but doesn't
# hold a reference to them, it only uses the params to initialize its state
# by creating new OptState Variables that reuse the param's metadata.
class SGD(nnx.Object):
  __data__ = ('momentum',)

  def __init__(self, params, lr: float, decay: float = 0.9):
    self.lr = lr
    self.decay = decay

    def make_opt_state(x):
      if isinstance(x, nnx.Variable | nnx.VariableState):
        return OptState(jnp.zeros_like(x.value), **x.get_metadata())
      else:
        return OptState(jnp.zeros_like(x))

    self.momentum = jax.tree.map(
      make_opt_state,
      params,
      is_leaf=lambda x: isinstance(x, nnx.Variable | nnx.VariableState),
    )

  # during the update we simply map over (params, momentum, grads),
  # for each triplet we implement the SGD update rule which updates
  # both the optimizer's state (momentum) and the params in place.
  def update(self, params, grads):
    params = nnx.pure(params)
    grads = nnx.pure(grads)
    momentum = nnx.pure(self.momentum)

    def update_fn(
      param: nnx.MutableArray, momentum: nnx.MutableArray, grad: jax.Array
    ):
      momentum[...] = self.decay * momentum[...] + (1 - self.decay) * grad[...]
      param[...] -= self.lr * momentum[...]

    jax.tree.map(update_fn, params, momentum, grads)


# ## Training
# To setup the training loop we first instantiate the model and optimizer.
# Variables are immutable (only contain Arrays) by default as it can make
# initialization easier, however this means we have to use 'mutable' to
# create the MutableArrays that will be updated during training.

rngs = nnx.Rngs(params=0, dropout=1)
model = Model(
  num_blocks=3, din=1, dhidden=256, dout=1, use_scan=False, rngs=rngs
)
optimizer = SGD(params=nnx.state(model, nnx.Param), lr=3e-3, decay=0.99)
# Create a copy of the model structure and set its attributes to eval model.
# This works because they share the underlying MutableArrays so both models
# will always be in sync.
eval_model = nnx.merge(*nnx.split(model))
eval_model.set_attributes(use_stats=True, deterministic=True)


# The training step uses 'jax.jit' and receives the model and optimizer as arguments,
# this is supported as they are now pytrees. The first thing we do is group the model
# state into the params and the non-differentiable state using 'split'. We differentiate
# the loss function using 'jax.grad' with respect to the params-only. Inside the loss
# function we merge the params and non-diff state back into a single model and then
# compute the loss by calling the model with the inputs.
@jax.jit
def train_step(model: Model, optimizer: SGD, rngs: nnx.Rngs, x, y):
  treedef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params):
    model = nnx.merge(treedef, params, nondiff)
    loss = jnp.mean((model(x, rngs=rngs) - y) ** 2)
    return loss

  # For the time being we have to use 'freeze' make the Variables immutable
  # as 'jax.grad' doesn't support MutableArrays yet.
  grads = jax.grad(loss_fn)(nnx.freeze(params))
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
  train_step(model, optimizer, rngs, x, y)

  if step % 1000 == 0:
    logs = test_step(eval_model, X, Y)
    print(f'step: {step}, loss: {logs["loss"]}')

  if step >= total_steps - 1:
    break

# ## Sample
# Sampling is trivial, just use 'model_eval'
print('times called:', eval_model.count[...])

y_pred = eval_model(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()

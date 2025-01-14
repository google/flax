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
from flax import nnx
import optax
import numpy as np
from einop import einop
from time import time
from tqdm import tqdm

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
flags.DEFINE_integer('depth', 4, 'Depth of the model')


class MlpBlock(nnx.Module):
  def __init__(self, din: int, mlp_dim: int, rngs: nnx.Rngs):
    self.din, self.mlp_dim = din, mlp_dim
    self.linear_in = nnx.Linear(din, mlp_dim, rngs=rngs)
    self.linear_out = nnx.Linear(mlp_dim, din, rngs=rngs)

  def __call__(self, x):
    return self.linear_out(nnx.gelu(self.linear_in(x)))


class MixerBlock(nnx.Module):
  def __init__(
    self,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    hidden_dim: int,
    rngs: nnx.Rngs,
  ):
    self.tokens_mlp_dim = tokens_mlp_dim
    self.channels_mlp_dim = channels_mlp_dim
    self.hidden_dim = hidden_dim
    self.token_mixing = MlpBlock(tokens_mlp_dim, hidden_dim, rngs=rngs)
    self.channel_mixing = MlpBlock(channels_mlp_dim, hidden_dim, rngs=rngs)
    self.ln1 = nnx.LayerNorm(channels_mlp_dim, rngs=rngs)
    self.ln2 = nnx.LayerNorm(channels_mlp_dim, rngs=rngs)

  def __call__(self, x):
    y = self.ln1(x)
    y = y.swapaxes(1, 2)
    y = self.token_mixing(y)
    y = y.swapaxes(1, 2)
    x = x + y
    y = self.ln2(x)
    return x + self.channel_mixing(y)


class MlpMixer(nnx.Module):
  def __init__(
    self,
    din: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    num_blocks: int,
    hidden_dim: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    rngs: nnx.Rngs,
  ):
    self.din = din
    self.kernel_size = kernel_size
    self.num_blocks = num_blocks
    self.hidden_dim = hidden_dim
    self.tokens_mlp_dim = tokens_mlp_dim
    self.channels_mlp_dim = channels_mlp_dim
    self.stem = nnx.Conv(
      din + 1,
      channels_mlp_dim,
      kernel_size=kernel_size,
      strides=strides,
      rngs=rngs,
    )
    self.blocks = [
      MixerBlock(tokens_mlp_dim, channels_mlp_dim, hidden_dim, rngs=rngs)
      for _ in range(num_blocks)
    ]
    self.pre_head_layer_norm = nnx.LayerNorm(channels_mlp_dim, rngs=rngs)
    self.conv_t = nnx.ConvTranspose(
      channels_mlp_dim, din, kernel_size=kernel_size, strides=strides, rngs=rngs
    )

  def __call__(self, *, x, t):
    # add time feature to input
    t = einop(t, 'n -> n h w c', h=x.shape[1], w=x.shape[2], c=1)
    x = jnp.concatenate([x, t], axis=-1)
    # create patches
    x = self.stem(x)
    h, w = x.shape[1], x.shape[2]
    x = einop(x, 'n h w c -> n (h w) c')
    # apply blocks
    for block in self.blocks:
      x = block(x)
    x = self.pre_head_layer_norm(x)
    # recreate image
    x = einop(x, 'n (h w) c -> n h w c', h=h, w=w)
    x = self.conv_t(x)
    return x


def main(argv):
  print(argv)
  mode: str = FLAGS.mode
  total_steps: int = FLAGS.total_steps
  batch_size: int = FLAGS.batch_size
  width: int = FLAGS.width
  depth: int = FLAGS.depth

  print(f'{mode=}, {total_steps=}, {batch_size=}, {width=}')

  X = np.random.uniform(size=(batch_size, 28, 28, 1))

  if mode == 'nnx' or mode == 'all':
    rngs = nnx.Rngs(0)
    flow = MlpMixer(
      din=1,
      kernel_size=(2, 2),
      strides=(2, 2),
      num_blocks=4,
      hidden_dim=512,
      tokens_mlp_dim=196,
      channels_mlp_dim=512,
      rngs=rngs,
    )
    optimizer = nnx.Optimizer(flow, tx=optax.adamw(1e-4))
    t0 = time()

    mse = lambda a, b: jnp.mean((a - b) ** 2)

    @nnx.jit(donate_argnums=(0, 1, 2))
    def train_step_nnx(flow, optimizer, rngs, x_1):
      print('JITTING NNX')
      x_0 = jax.random.normal(rngs(), x_1.shape)
      t = jax.random.uniform(rngs(), (len(x_1),))

      x_t = jax.vmap(lambda x_0, x_1, t: (1 - t) * x_0 + t * x_1)(x_0, x_1, t)
      dx_t = x_1 - x_0

      loss, grads = nnx.value_and_grad(
        lambda flow: mse(flow(x=x_t, t=t), dx_t)
      )(flow)
      optimizer.update(grads)
      return loss

    losses = []
    t0 = time()
    for step in tqdm(range(total_steps), desc='NNX'):
      loss = train_step_nnx(flow, optimizer, rngs, X)
      losses.append(loss)

    total_time = time() - t0
    print('### NNX ###')
    print(f'final loss: {losses[-1]}')
    print('total time:', total_time)
    print(f'time per step: {total_time / total_steps * 1e6:.2f} µs')

  if mode == 'jax' or mode == 'all':
    rngs = nnx.Rngs(0)
    flow = MlpMixer(
      din=1,
      kernel_size=(2, 2),
      strides=(2, 2),
      num_blocks=depth,
      hidden_dim=width,
      tokens_mlp_dim=196,
      channels_mlp_dim=width,
      rngs=rngs,
    )
    optimizer = nnx.Optimizer(flow, tx=optax.adamw(1e-4))
    graphdef, state = nnx.split((flow, optimizer, rngs))
    t0 = time()

    mse = lambda a, b: jnp.mean((a - b) ** 2)

    @partial(nnx.jit, donate_argnums=0)
    def train_step_jax(state, x_1):
      print('JITTING JAX')
      flow, optimizer, rngs = nnx.merge(graphdef, state)
      x_0 = jax.random.normal(rngs(), x_1.shape)
      t = jax.random.uniform(rngs(), (len(x_1),))

      x_t = jax.vmap(lambda x_0, x_1, t: (1 - t) * x_0 + t * x_1)(x_0, x_1, t)
      dx_t = x_1 - x_0

      loss, grads = nnx.value_and_grad(
        lambda flow: mse(flow(x=x_t, t=t), dx_t)
      )(flow)
      optimizer.update(grads)
      state = nnx.state((flow, optimizer, rngs))
      return loss, state

    losses = []
    t0 = time()
    for step in tqdm(range(total_steps), desc='JAX'):
      loss, state = train_step_jax(state, X)
      losses.append(loss)

    nnx.update((flow, optimizer, rngs), state)
    total_time = time() - t0
    print('### JAX ###')
    print(f'final loss: {losses[-1]}')
    print('total time:', total_time)
    print(f'time per step: {total_time / total_steps * 1e6:.2f} µs')


if __name__ == '__main__':
  app.run(main)

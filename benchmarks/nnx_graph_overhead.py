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
import numpy as np
import optax
from time import time

from flax import nnx

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'all', ['all', 'nnx', 'jax'], 'Mode to run the script in')
flags.DEFINE_integer('total_steps', 100, 'Total number of training steps')
flags.DEFINE_integer('width', 32, 'Hidden layer size')
flags.DEFINE_integer('depth', 5, 'Depth of the model')



class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.list = [
      nnx.Param(jax.random.uniform(rngs.params(), (din, dout))),
      nnx.Param(jnp.zeros((dout,))),
    ]
    self.dict = {
      'w': nnx.Param(jax.random.uniform(rngs.params(), (din, dout))),
      'b': nnx.Param(jnp.zeros((dout,))),
    }



class MLP(nnx.Module):
  def __init__(self, depth, *, rngs: nnx.Rngs):
    self.intermediates = [
      Linear(10, 10, rngs=rngs) for _ in range(depth)
    ]


def main(argv):
  print(argv)
  mode: str = FLAGS.mode
  total_steps: int = FLAGS.total_steps
  width: int = FLAGS.width
  depth: int = FLAGS.depth

  print(f'{mode=}, {total_steps=}, {width=}')

  X = np.linspace(0, 1, 100)[:, None]
  Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)

  model = MLP(depth=depth, rngs=nnx.Rngs(0))
  tx = optax.sgd(1e-3)
  optimizer = nnx.Optimizer(model, tx)

  #------------------------------------------------------------
  # NNX
  #------------------------------------------------------------
  if mode in ['all', 'nnx']:
    @nnx.jit
    def step_nnx(model: MLP, optimizer: nnx.Optimizer):
      pass

    t0 = time()
    for _ in range(total_steps):
      step_nnx(model, optimizer)

    total_time = time() - t0
    time_per_step = total_time / total_steps
    time_per_layer = time_per_step / depth
    print("### NNX ###")
    print('total time:', total_time)
    print(f'time per step: {time_per_step * 1e6:.2f} µs')
    print(f'time per layer: {time_per_layer * 1e6:.2f} µs')


  #------------------------------------------------------------
  # JAX
  #------------------------------------------------------------

  if mode in ['all', 'jax']:
    @jax.jit
    def step_jax(graphdef, state):
      return graphdef, state

    graphdef, state = nnx.split((model, optimizer))
    t0 = time()
    for _ in range(total_steps):
      graphdef, state = step_jax(graphdef, state)

    total_time = time() - t0
    time_per_step = total_time / total_steps
    time_per_layer = time_per_step / depth
    print("### JAX ###")
    print('total time:', total_time)
    print(f'time per step: {time_per_step * 1e6:.2f} µs')
    print(f'time per layer: {time_per_layer * 1e6:.2f} µs')
    print()



if __name__ == '__main__':
  app.run(main)

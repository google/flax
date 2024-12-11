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

# Example profile command:
#   python -m cProfile -o ~/tmp/overhead.prof benchmarks/nnx_graph_overhead.py --mode=nnx --depth=100 --total_steps=1000
# View profile (need to install snakeviz):
#   snakeviz ~/tmp/overhead.prof

import jax
from time import time

from flax import nnx

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_integer('total_steps', 1000, 'Total number of training steps')
flags.DEFINE_integer('width', 4, 'Width of each level')
flags.DEFINE_integer('depth', 4, 'Depth of the model')


class NestedClass(nnx.Module):
  def __init__(self, width, depth):
    self.x = nnx.Variable(jax.numpy.ones((depth+1, )))
    if depth > 0:
      for i in range(width):
        setattr(self, f'child{i}', NestedClass(width, depth-1))


def main(argv):
  print(argv)
  total_steps: int = FLAGS.total_steps
  width: int = FLAGS.width
  depth: int = FLAGS.depth


  model = NestedClass(width, depth)
  to_test = nnx.state(model)

  print(f'{total_steps=}, {width=}')

  #------------------------------------------------------------
  # tree_flatten_with_path
  #------------------------------------------------------------
  t0 = time()
  for _ in range(total_steps):
    jax.tree_util.tree_flatten_with_path(to_test)

  total_time = time() - t0
  time_per_step = total_time / total_steps
  time_per_layer = time_per_step / depth
  print("### tree_flatten_with_path ###")
  print('total time:', total_time)
  print(f'time per step: {time_per_step * 1e6:.2f} µs')
  print(f'time per layer: {time_per_layer * 1e6:.2f} µs')


  #------------------------------------------------------------
  # tree_map_with_path
  #------------------------------------------------------------

  t0 = time()
  for _ in range(total_steps):
    jax.tree_util.tree_map_with_path(lambda _, x: x, to_test)

  total_time = time() - t0
  time_per_step = total_time / total_steps
  time_per_layer = time_per_step / depth
  print("### tree_map_with_path ###")
  print('total time:', total_time)
  print(f'time per step: {time_per_step * 1e6:.2f} µs')
  print(f'time per layer: {time_per_layer * 1e6:.2f} µs')


  #------------------------------------------------------------
  # tree_flatten
  #------------------------------------------------------------

  t0 = time()
  for _ in range(total_steps):
    jax.tree_util.tree_flatten(to_test)

  total_time = time() - t0
  time_per_step = total_time / total_steps
  time_per_layer = time_per_step / depth
  print("### tree_flatten ###")
  print('total time:', total_time)
  print(f'time per step: {time_per_step * 1e6:.2f} µs')
  print(f'time per layer: {time_per_layer * 1e6:.2f} µs')



if __name__ == '__main__':
  app.run(main)

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


import jax
import jax.numpy as jnp

from flax import nnx


class Block(nnx.Module):
  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.linear = nnx.Linear(dim, dim, rngs=rngs)
    self.bn = nnx.BatchNorm(dim, rngs=rngs)
    self.dropout = nnx.Dropout(0.5, rngs=rngs)

  def __call__(self, x: jax.Array):
    return jax.nn.gelu(self.dropout(self.bn(self.linear(x))))


class ScanMLP(nnx.Module):
  """
  An MLP that uses `vmap` during `__init__` to create a Block instance
  with an additional `layer` axis, and `scan` during `__call__` to apply
  the sequence of layers iteratively over the input / output `x`.
  """

  def __init__(self, dim: int, *, n_layers: int, rngs: nnx.Rngs):
    self.n_layers = n_layers

    @nnx.split_rngs(splits=n_layers)
    @nnx.vmap(axis_size=n_layers)
    def create_block(rngs: nnx.Rngs):
      return Block(dim, rngs=rngs)

    self.layers = create_block(rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    @nnx.split_rngs(splits=self.n_layers)
    @nnx.scan
    def scan_fn(x: jax.Array, block: Block):
      x = block(x)
      return x, None

    x, _ = scan_fn(x, self.layers)

    return x


model = ScanMLP(10, n_layers=5, rngs=nnx.Rngs(0))

x = jnp.ones((3, 10))
y = model(x)

print(jax.tree.map(jnp.shape, nnx.state(model)))
print(y.shape)

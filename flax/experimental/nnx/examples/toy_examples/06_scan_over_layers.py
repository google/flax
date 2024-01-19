# Copyright 2023 The Flax Authors.
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

from typing import Tuple

import jax
import jax.numpy as jnp

from flax.experimental import nnx


class Block(nnx.Module):
  def __init__(self, dim: int, *, ctx: nnx.Ctx):
    self.linear = nnx.Linear(dim, dim, ctx=ctx)
    self.dropout = nnx.Dropout(0.5)

  def __call__(self, x: jax.Array, *, ctx: nnx.Ctx) -> jax.Array:
    x = self.linear(x)
    x = self.dropout(x, ctx=ctx)
    x = jax.nn.gelu(x)
    return x


class ScanMLP(nnx.Module):
  """
  An MLP that uses `vmap` during `__init__` to create a Block instance
  with an additional `layer` axis, and `scan` during `__call__` to apply
  the sequence of layers iteratively over the input / output `x`.
  """

  def __init__(self, dim: int, *, n_layers: int, ctx: nnx.Ctx):
    self.n_layers = n_layers
    # fork Rngs, split keys into `n_layers`
    keys = ctx.fork(n_layers)

    def create_block(keys):
      # create Block instance and return its split
      return Block(dim, ctx=nnx.Ctx(keys)).split()

    # call vmap over create_block, passing the split `params` key
    # and immediately merge to get a Block instance
    self.layers = nnx.merge(jax.vmap(create_block)(keys))

  def __call__(self, x: jax.Array, *, ctx: nnx.Ctx) -> jax.Array:
    # fork Rngs, split keys into `n_layers`
    keys, flags = ctx.fork(self.n_layers)
    # split Module to get params
    params, static = self.layers.split(nnx.Param)

    def scan_fn(
      x: jax.Array, inputs: Tuple[nnx.State, dict[str, nnx.RngStream]]
    ) -> Tuple[jax.Array, nnx.State]:
      params, keys = inputs
      # merge back Module and Rngs
      module = static.merge(params)
      # forward pass
      x = module(x, ctx=nnx.Ctx(keys, flags=flags))
      # split state and return
      params, _ = module.split(nnx.Param)
      return x, params

    # call scan passing x as the carry, and params + keys as the input
    x, params = jax.lax.scan(scan_fn, x, (params, keys))
    # update layers state and return
    self.layers.update(params)
    return x


model = ScanMLP(10, n_layers=5, ctx=nnx.Ctx(0))

x = jnp.ones((3, 10))
flags = dict(deterministic=False)
y = model(x, ctx=nnx.Ctx(dropout=1, flags=flags))

print(jax.tree_map(jnp.shape, model.get_state()))
print(y.shape)

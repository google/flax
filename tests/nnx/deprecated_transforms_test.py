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

from functools import partial

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

from flax.nnx.transforms.deprecated import vmap, Vmap, pmap, Pmap


class TestVmap(absltest.TestCase):
  def test_basic(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    vectorized_create_block = vmap(
      create_block, state_axes={nnx.Param: 0}, axis_size=5
    )

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = vectorized_create_block(rngs)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    def forward_block(module, x):
      return module(x)

    vectorized_forward_block = vmap(
      forward_block, state_axes={nnx.Param: 0}, axis_size=5
    )

    y = vectorized_forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    y2 = vectorized_forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_basic_demo(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    @partial(vmap, axis_size=5)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(vmap, axis_size=5)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    assert rngs.default.count.value == 1
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )

    x = jnp.ones((5, 1, 3))

    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

  def test_replicate(self):
    din = 3
    dout = 10

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(
      vmap,
      state_axes={},  # replicate all state
      split_rngs=True,  # different rngs for each replica
    )
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (din, dout)
    assert module.linear.bias.value.shape == (dout,)

    x = jnp.ones((5, 1, din))

    y = forward_block(module, x)

    assert y.shape == (5, 1, dout)
    assert rngs.default.count.value == 3

    assert not jnp.allclose(y[0], y[1])

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

    assert rngs.default.key.value == initial_key

  def test_combinator(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = Vmap.constructor(Block, state_axes={nnx.Param: 0}, axis_size=5)

    module = MLP(rngs=nnx.Rngs(0))

    assert not jnp.allclose(
      module.vmap_module.linear.kernel.value[0],
      module.vmap_module.linear.kernel.value[1],
    )
    assert module.vmap_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.vmap_module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))
    y = module(x)

    assert y.shape == (5, 1, 3)

  def test_combinator_init(self):
    class Block(nnx.Module):
      def __init__(self, *, graphdef: str, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.graphdef = graphdef

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = Vmap.constructor(Block, state_axes={nnx.Param: 0}, axis_size=5)

    module = MLP(graphdef='hello', rngs=nnx.Rngs(0))

    assert module.vmap_module.graphdef == 'hello'

  def test_state_axes(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.param = nnx.Param(jnp.arange(5))

    foo = Foo()

    @partial(vmap, state_axes={...: 0})
    def f(foo: Foo):
      assert foo.param.value.shape == ()

    f(foo)


class TestPmap(absltest.TestCase):
  def test_basic_single(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 10, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.elu(x)
        x = self.dropout(x)
        return x

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    vectorized_create_block = pmap(
      create_block, state_axes={nnx.Param: 0}, axis_size=1
    )

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = vectorized_create_block(rngs)

    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key
    assert module.linear.kernel.value.shape == (1, 3, 10)
    assert module.linear.bias.value.shape == (1, 10)

    x = jnp.ones((1, 1, 3))

    def forward_block(module, x):
      return module(x)

    vectorized_forward_block = vmap(
      forward_block, state_axes={nnx.Param: 0}, axis_size=1
    )

    y = vectorized_forward_block(module, x)

    assert y.shape == (1, 1, 10)
    assert rngs.default.count.value == 3
    assert rngs.default.key.value == initial_key

    y2 = vectorized_forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_basic_demo_single(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    @partial(pmap, axis_size=1)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(pmap, axis_size=1)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (1, 3, 3)
    assert module.linear.bias.value.shape == (1, 3)

    x = jnp.ones((1, 10, 3))

    y = forward_block(module, x)

    assert y.shape == (1, 10, 3)
    assert rngs.default.count.value == 3

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

  def test_replicate_single(self):
    din = 3
    dout = 10

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(
      pmap,
      state_axes={},  # replicate all state
      split_rngs=True,  # different rngs for each replica
    )
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (din, dout)
    assert module.linear.bias.value.shape == (dout,)

    x = jnp.ones((1, 5, din))

    y = forward_block(module, x)

    assert y.shape == (1, 5, dout)
    assert rngs.default.count.value == 3

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

    assert rngs.default.key.value == initial_key

  def test_combinator_single(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = Pmap.constructor(Block, state_axes={nnx.Param: 0}, axis_size=1)

    module = MLP(rngs=nnx.Rngs(0))

    assert module.pmap_module.linear.kernel.value.shape == (1, 3, 3)
    assert module.pmap_module.linear.bias.value.shape == (1, 3)

    x = jnp.ones((1, 5, 3))
    y = module(x)

    assert y.shape == (1, 5, 3)

if __name__ == '__main__':
  absltest.main()
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
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx


class TestExperimentalVmap(absltest.TestCase):
  def test_basic(self):
    @partial(nnx.experimental_vmap, in_axes=0, out_axes=0, axis_size=5)
    def create_block(rngs: nnx.Rngs):
      return nnx.Linear(2, 3, rngs=rngs)

    rngs = nnx.Rngs(0)
    backups = nnx.split_rngs(rngs, 5)

    block = create_block(rngs)
    nnx.restore_rngs(backups)

    self.assertEqual(block.kernel.value.shape, (5, 2, 3))
    self.assertEqual(rngs.default.count.value, 1)

    @partial(nnx.experimental_vmap, in_axes=(0, 1), out_axes=1)
    def forward(block: nnx.Linear, x):
      self.assertEqual(block.kernel.value.shape, (2, 3))
      self.assertEqual(block.bias.value.shape, (3,))
      self.assertEqual(x.shape, (2,))
      return block(x)

    x = jax.random.uniform(rngs(), (2, 5))
    y = forward(block, x)

    self.assertEqual(y.shape, (3, 5))

  def test_state_axes(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    @nnx.experimental_vmap(
      in_axes=0,
      out_axes=nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None}),
    )
    def create_block(rngs: nnx.Rngs):
      rngs = nnx.clone(rngs)
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value

    backups = nnx.split_rngs(rngs, 5)
    module = create_block(rngs)
    nnx.restore_rngs(backups)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.experimental_vmap(
      in_axes=(nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None}), 0),
    )
    def forward_block(module, x):
      return module(x)

    backups = nnx.split_rngs(rngs, 5)
    y = forward_block(module, x)
    nnx.restore_rngs(backups)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_state_axes_simple(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    state_axes = nnx.StateAxes({(nnx.BatchStat, 'dropout'): 0, ...: None})

    @nnx.experimental_vmap(in_axes=(state_axes,), out_axes=state_axes)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(params=0, dropout=1)
    nnx.split_rngs(rngs, 5, 'dropout')

    module = create_block(rngs)

    assert module.linear.kernel.value.shape == (2, 3)
    assert module.bn.scale.value.shape == (3,)
    assert module.bn.mean.value.shape == (5, 3)

    @nnx.experimental_vmap(in_axes=(state_axes, 0), out_axes=0)
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  def test_state_axes_super_simple(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    @nnx.experimental_vmap(in_axes=0, out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    nnx.split_rngs(rngs, 5)

    module = create_block(rngs)

    assert module.linear.kernel.value.shape == (5, 2, 3)
    assert module.bn.scale.value.shape == (5, 3)
    assert module.bn.mean.value.shape == (5, 3)

    @nnx.experimental_vmap(in_axes=(0, 0), out_axes=0)
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  def test_consistent_aliasing_inputs(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jnp.zeros((5, 5))

    m = Foo()

    @nnx.experimental_vmap(in_axes=(0, 1))
    def f(m1, m2):
      pass

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      f(m, m)

  def test_consistent_aliasing_input_output(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jnp.zeros((2, 3))

    m = Foo()

    @partial(nnx.experimental_vmap, in_axes=0, out_axes=1)
    def f(m):
      return m

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      m2 = f(m)

  def test_consistent_aliasing_shared(self):
    class Shared(nnx.Module):
      def __init__(self):
        self.a = jnp.zeros((3, 3))

    class Foo(nnx.Module):
      def __init__(self, shared: Shared):
        self.a = shared

    shared = Shared()
    m1 = Foo(shared)
    m2 = Foo(shared)

    @partial(nnx.experimental_vmap, in_axes=(0, 1))
    def f(m1, m2):
      pass

    with self.assertRaisesRegex(
      ValueError,
      r'Inconsistent aliasing detected([\s\S]*)Shared([\s\S]*)a: 0([\s\S]*)a: 1',
    ):
      f(m1, m2)

  def test_vmap_and_cond_passthrough(self):
    class Broadcast(nnx.Variable[nnx.A]): ...

    class Vectorized(nnx.Variable[nnx.A]): ...

    class Env(nnx.Module):
      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.experimental_vmap(
      in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),)
    )
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(is_even, increment, no_nothing, env)

    f(env)

    np.testing.assert_array_equal(env.step.value, [1, 0, 1, 0, 1, 0, 1, 0])

  def test_vmap_and_cond_passthrough_error(self):
    class Broadcast(nnx.Variable[nnx.A]): ...

    class Vectorized(nnx.Variable[nnx.A]): ...

    class Env(nnx.Module):
      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.experimental_vmap(
      in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),)
    )
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step += 1
        env.broadcast += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(is_even, increment, no_nothing, env)

    with self.assertRaisesRegex(
      ValueError,
      r"at vmap.*'broadcast'.*got axis spec None but output was batched on axis 0",
    ):
      f(env)


if __name__ == '__main__':
  absltest.main()
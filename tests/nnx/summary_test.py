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

import jax.numpy as jnp
from absl.testing import absltest

from flax import nnx

CONSOLE_TEST_KWARGS = dict(force_terminal=False, no_color=True, width=10_000)


class SummaryTest(absltest.TestCase):
  def test_tabulate(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def forward(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    class Foo(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.block1 = Block(32, 128, rngs=rngs)
        self.block2 = Block(128, 10, rngs=rngs)

      def __call__(self, x):
        return self.block2.forward(self.block1.forward(x))

    foo = Foo(nnx.Rngs(0))
    x = jnp.ones((1, 32))
    table_repr = nnx.tabulate(
      foo, x, console_kwargs=CONSOLE_TEST_KWARGS
    ).splitlines()

    self.assertIn('Foo Summary', table_repr[0])
    self.assertIn('path', table_repr[2])
    self.assertIn('type', table_repr[2])
    self.assertIn('BatchStat', table_repr[2])
    self.assertIn('Param', table_repr[2])
    self.assertIn('block1/forward', table_repr[6])
    self.assertIn('Block', table_repr[6])
    self.assertIn('block1/linear', table_repr[8])
    self.assertIn('Linear', table_repr[8])
    self.assertIn('block1/bn', table_repr[13])
    self.assertIn('BatchNorm', table_repr[13])
    self.assertIn('block1/dropout', table_repr[18])
    self.assertIn('Dropout', table_repr[18])
    self.assertIn('block2/forward', table_repr[20])
    self.assertIn('Block', table_repr[20])
    self.assertIn('block2/linear', table_repr[22])
    self.assertIn('Linear', table_repr[22])
    self.assertIn('block2/bn', table_repr[27])
    self.assertIn('BatchNorm', table_repr[27])
    self.assertIn('block2/dropout', table_repr[32])
    self.assertIn('Dropout', table_repr[32])

    self.assertIn('Total', table_repr[34])
    self.assertIn('276 (1.1 KB)', table_repr[34])
    self.assertIn('5,790 (23.2 KB)', table_repr[34])
    self.assertIn('4 (24 B)', table_repr[34])
    self.assertIn('Total Parameters: 6,070 (24.3 KB)', table_repr[37])

  def test_multiple_inputs_and_outputs(self):
    class CustomMLP(nnx.Module):
      def __init__(self):
        self.weight = nnx.Param(jnp.ones((4, 8)))
        self.bias = nnx.Param(jnp.ones(8))

      def __call__(self, x, x2):
        y = x @ self.weight
        y += self.bias[None]
        y += x2
        return x, y, 2 * y

    cmlp = CustomMLP()
    x = jnp.ones((1, 4))
    x2 = jnp.ones((1, 8))
    table_repr = nnx.tabulate(
      cmlp, x, x2, console_kwargs=CONSOLE_TEST_KWARGS
    ).splitlines()

    self.assertIn('CustomMLP Summary', table_repr[0])
    self.assertIn('float32[1,4]', table_repr[4])
    self.assertIn('float32[1,8]', table_repr[5])
    self.assertIn('float32[1,8]', table_repr[6])


  def test_no_dup_flops(self):
    class Model(nnx.Module):
      def g(self, x):
        return x**2
      def __call__(self, x):
        return self.g(x)
    m = Model()
    x = jnp.ones(4)
    table_rep = nnx.tabulate(m, x, compute_flops=True)
    table_lines = table_rep.splitlines()
    self.assertEqual(sum(" g " in l for l in table_lines), 1)


  def test_flops(self):
    class Model(nnx.Module):
      def __init__(self):
        self.weight = nnx.Param(jnp.ones(4))

      def __call__(self, x1):
        return jnp.sum((x1 * self.weight)**2)
    m = Model()
    x = jnp.ones(4)
    table_repr1 = nnx.tabulate(
      m, x, compute_flops=True
    ).splitlines()
    self.assertIn('flops', table_repr1[2])
    self.assertNotIn('vjp_flops', table_repr1[2])
    table_repr2 = nnx.tabulate(
      m, x, compute_flops=True, compute_vjp_flops=True
    ).splitlines()
    self.assertIn('vjp_flops', table_repr2[2])

  def test_nested(self):
    class Block(nnx.Module):
      def __init__(self, rngs):
        self.linear = nnx.Linear(2, 2, rngs=rngs)
        self.bn = nnx.BatchNorm(2, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return nnx.relu(x)

    class Model(nnx.Module):
      def __init__(self, rngs):
        self.block1 = Block(rngs)
        self.block2 = Block(rngs)

      def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    m = Model(nnx.Rngs(0))
    x = jnp.ones((4, 2))
    table = nnx.tabulate(m, x, compute_flops=True, compute_vjp_flops=True)
    # We should see 3 calls per block, plus one overall call
    self.assertEqual(sum([s.startswith("├─") for s in table.splitlines()]), 7)

  def test_shared(self):
    class Block(nnx.Module):
      def __init__(self, linear: nnx.Linear, *, rngs):
        self.linear = linear
        self.bn = nnx.BatchNorm(2, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return nnx.relu(x)

    class Model(nnx.Module):
      def __init__(self, rngs):
        shared = nnx.Linear(2, 2, rngs=rngs)
        self.block1 = Block(shared, rngs=rngs)
        self.block2 = Block(shared, rngs=rngs)

      def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    m = Model(nnx.Rngs(0))
    x = jnp.ones((4, 2))
    table = nnx.tabulate(m, x, compute_vjp_flops=True)
    # We should see 3 calls per block, plus one overall call, minus the shared call
    self.assertEqual(sum([s.startswith("├─") for s in table.splitlines()]), 6)

if __name__ == '__main__':
  absltest.main()

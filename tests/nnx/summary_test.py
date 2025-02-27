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
    self.assertIn('2 (12 B)', table_repr[34])
    self.assertIn('Total Parameters: 6,068 (24.3 KB)', table_repr[37])

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


if __name__ == '__main__':
  absltest.main()
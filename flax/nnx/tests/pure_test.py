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
from __future__ import annotations

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

class StatefulLinear(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))

  def increment(self):
    self.count.value += 1

  def __call__(self, x):
    self.count.value += 1
    return x @ self.w + self.b


class PureTest(absltest.TestCase):
  def test_jit_update(self):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = jnp.zeros(())

      def inc(self):
        self.count += 1
        return 1

    p_counter: nnx.Pure[Counter] = nnx.pure(Counter())

    @jax.jit
    def update(p_counter: nnx.Pure[Counter]):
      out, p_counter = p_counter.inc()
      self.assertEqual(out, 1)
      return p_counter

    p_counter = update(p_counter)
    p_counter = update(p_counter)

    counter = nnx.stateful(p_counter)

    self.assertEqual(counter.count, 2)

  def test_stateful_linear(self):
    linear = StatefulLinear(3, 2, nnx.Rngs(0))
    pure_linear: nnx.Pure[StatefulLinear] = nnx.pure(linear)

    @jax.jit
    def forward(x, pure_linear: nnx.Pure[StatefulLinear]):
      y, pure_linear = pure_linear(x)
      return y, pure_linear

    x = jnp.ones((1, 3))
    y, pure_linear = forward(x, pure_linear)
    y, pure_linear = forward(x, pure_linear)

    self.assertEqual(linear.count.value, 0)
    new_linear = nnx.stateful(pure_linear)
    self.assertEqual(new_linear.count.value, 2)

  def test_getitem(self):
    rngs = nnx.Rngs(0)
    nodes = dict(
      a=StatefulLinear(3, 2, rngs),
      b=StatefulLinear(2, 1, rngs),
    )
    pure = nnx.pure(nodes)
    _, pure = pure['b'].increment()
    nodes = nnx.stateful(pure)

    self.assertEqual(nodes['a'].count.value, 0)
    self.assertEqual(nodes['b'].count.value, 1)

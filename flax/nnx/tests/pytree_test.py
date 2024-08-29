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

import dataclasses
from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx


class StatefulLinear(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.din, self.dout = din, dout
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))

  def increment(self):
    self.count.value += 1

  def __call__(self, x):
    self.count.value += 1
    return x @ self.w + self.b[None]


class TestPytree(absltest.TestCase):
  def test_basic_example(self):
    module = nnx.Pytree(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
    module = jax.tree.map(lambda x: x, module)
    y = module(jnp.ones((1, 2)))
    self.assertEqual(y.shape, (1, 3))
    self.assertEqual(module.in_features, 2)
    self.assertEqual(module.out_features, 3)
    self.assertEqual(module.kernel.shape, (2, 3))

  def test_call_jit_update(self):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = jnp.zeros(())

      def inc(self):
        self.count += 1
        return 1

    pt_counter = nnx.Pytree(Counter())

    @jax.jit
    def update(pt_counter: nnx.Pytree[Counter]):
      out = pt_counter.inc()
      self.assertEqual(out, 1)
      return pt_counter

    pt_counter = update(pt_counter)
    pt_counter = update(pt_counter)

    counter = nnx.merge(*pt_counter)

    self.assertEqual(counter.count, 2)

  def test_stateful_linear(self):
    linear = StatefulLinear(3, 2, nnx.Rngs(0))
    pt_linear: nnx.Pytree[StatefulLinear] = nnx.Pytree(linear)

    @jax.jit
    def forward(x, pt_linear: nnx.Pytree[StatefulLinear]):
      y = pt_linear(x)
      return y, pt_linear

    x = jnp.ones((1, 3))
    y, pt_linear = forward(x, pt_linear)
    y, pt_linear = forward(x, pt_linear)

    self.assertEqual(linear.count.value, 0)
    new_linear = nnx.merge(pt_linear.graphdef, pt_linear.state)
    self.assertEqual(new_linear.count.value, 2)

  def test_getitem(self):
    rngs = nnx.Rngs(0)
    nodes = dict(
      a=StatefulLinear(3, 2, rngs),
      b=StatefulLinear(2, 1, rngs),
    )
    pt_nodes = nnx.Pytree(nodes)
    _ = pt_nodes['b'].increment()

    nodes = nnx.merge(*pt_nodes)

    self.assertEqual(nodes['a'].count.value, 0)
    self.assertEqual(nodes['b'].count.value, 1)

  def test_static_attribute_access(self):
    rngs = nnx.Rngs(0)
    nodes = dict(
      a=StatefulLinear(3, 2, rngs),
      b=StatefulLinear(2, 1, rngs),
    )
    pt_nodes = nnx.Pytree(nodes)

    self.assertEqual(pt_nodes['a'].w.shape, (3, 2))
    self.assertEqual(pt_nodes['b'].w.shape, (2, 1))
    self.assertEqual(pt_nodes['a'].din, 3)
    self.assertEqual(pt_nodes['a'].dout, 2)
    self.assertEqual(pt_nodes['b'].din, 2)
    self.assertEqual(pt_nodes['b'].dout, 1)

  def test_pytree_sentinel(self):
    # make sure issue #4142 is not present

    class SomeModule(nnx.Module):
      def __init__(self, epsilon):
        self.epsilon = epsilon

    def f(m, x) -> None:
      pass

    module = nnx.Pytree(SomeModule(jnp.zeros(1)))
    z = jnp.zeros(10)
    jax.vmap(f, in_axes=(None, 0))(module, z)

  def test_context_manager(self):
    @dataclasses.dataclass
    class Counter(nnx.Module):
      count: nnx.BatchStat[int]

    pt_counter = nnx.Pytree(Counter(nnx.BatchStat(0)))
    self.assertEqual(pt_counter.count.value, 0)

    with pt_counter as counter:
      counter.count += 1

    self.assertEqual(pt_counter.count.value, 1)

  def test_set_attr_error(self):
    @dataclasses.dataclass
    class Counter(nnx.Module):
      count: nnx.BatchStat[int]

    pt_counter = nnx.Pytree(Counter(nnx.BatchStat(0)))

    with self.assertRaisesRegex(
      AttributeError, "'Pytree' object has no attribute"
    ):
      pt_counter.count = 1
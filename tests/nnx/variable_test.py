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

import typing as tp

import jax
import jax.numpy as jnp

from absl.testing import absltest
from flax import nnx

A = tp.TypeVar('A')

class TestVariableState(absltest.TestCase):
  def test_pytree(self):
    r1 = nnx.VariableState(nnx.Param, 1)
    self.assertEqual(r1.value, 1)

    r2 = jax.tree.map(lambda x: x + 1, r1)

    self.assertEqual(r1.value, 1)
    self.assertEqual(r2.value, 2)
    self.assertIsNot(r1, r2)

  def test_overloads_module(self):
    class Linear(nnx.Module):
      def __init__(self, din, dout, rngs: nnx.Rngs):
        key = rngs()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)))
        self.b = nnx.Param(jax.numpy.zeros((dout,)))

      def __call__(self, x: jax.Array):
        return x @ self.w + self.b

    linear = Linear(3, 4, nnx.Rngs(0))
    x = jax.numpy.ones((3,))
    y = linear(x)
    self.assertEqual(y.shape, (4,))

  def test_jax_array(self):
    class Linear(nnx.Module):
      def __init__(self, din, dout, rngs: nnx.Rngs):
        key = rngs()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)))
        self.b = nnx.Param(jax.numpy.zeros((dout,)))

      def __call__(self, x: jax.Array):
        return jnp.dot(x, self.w) + self.b  # type: ignore[arg-type]

    linear = Linear(3, 4, nnx.Rngs(0))
    x = jax.numpy.ones((3,))
    y = linear(x)
    self.assertEqual(y.shape, (4,))

  def test_proxy_access(self):
    v = nnx.Param(jnp.ones((2, 3)))
    t = v.T

    self.assertEqual(t.shape, (3, 2))

  def test_proxy_call(self):
    class Callable(tp.NamedTuple):
      value: int

      def __call__(self, x):
        return self.value * x

    v = nnx.Param(Callable(2))
    result = v(3)

    self.assertEqual(result, 6)

  def test_binary_ops(self):
    v1 = nnx.Param(2)
    v2 = nnx.Param(3)

    result = v1 + v2

    self.assertEqual(result, 5)

    v1 += v2

    self.assertEqual(v1.value, 5)


if __name__ == '__main__':
  absltest.main()

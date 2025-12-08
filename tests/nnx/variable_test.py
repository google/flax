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
import numpy as np
import pytest

from absl.testing import absltest, parameterized
from flax import nnx

A = tp.TypeVar('A')

class TestVariable(parameterized.TestCase):
  def test_pytree(self):
    r1 = nnx.Param(1)
    self.assertEqual(r1.get_value(), 1)

    r2 = jax.tree.map(lambda x: x + 1, r1)

    self.assertEqual(r1.get_value(), 1)
    self.assertEqual(r2.get_value(), 2)
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
    v1 = nnx.Param(jnp.array(2))
    v2 = nnx.Param(jnp.array(3))

    result = v1 + v2

    self.assertEqual(result, 5)
    self.assertFalse(v1 == v2)

    v1[...] += v2

    self.assertEqual(v1[...], 5)

  @parameterized.product(
    v1=[np.array([1, 2]), np.array(2), 3],
    v2=[np.array([1, 2]), np.array(2), 3],
  )
  def test_eq_op(self, v1, v2):
    p1 = nnx.Param(jnp.asarray(v1) if isinstance(v1, np.ndarray) else v1)
    p2 = nnx.Param(jnp.asarray(v2) if isinstance(v2, np.ndarray) else v2)
    if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
      self.assertEqual((p1 == p2).all(), (v1 == v2).all())
    else:
      self.assertEqual(p1 == p2, v1 == v2)

  def test_mutable_array_context(self):
    initial_mode = nnx.using_hijax()
    with nnx.use_hijax(False):
      v = nnx.Variable(jnp.array(1.0))
      self.assertEqual(nnx.using_hijax(), False)
      self.assertNotIsInstance(v[...], jax.Ref)

      with nnx.use_hijax(True):
        v = nnx.Variable(jnp.array(1.0))
        self.assertEqual(nnx.using_hijax(), True)
        self.assertIsInstance(v[...], jax.Array)

      v = nnx.Variable(jnp.array(2.0))
      self.assertIsInstance(v[...], jax.Array)
      self.assertEqual(nnx.using_hijax(), False)

      nnx.use_hijax(True)

      v = nnx.Variable(jnp.array(0.0))
      self.assertEqual(nnx.using_hijax(), True)
      self.assertIsInstance(v[...], jax.Array)

    v = nnx.Variable(jnp.array(1.0))
    self.assertEqual(nnx.using_hijax(), initial_mode)
    self.assertIsInstance(v[...], jax.Array)

  def test_get_set_metadata(self):
    v = nnx.Variable(jnp.array(1.0))
    self.assertEqual(
        v.get_metadata(),
        {
            'is_hijax': False,
            'has_ref': False,
            'is_mutable': True,
            'eager_sharding': True,
        },
    )
    v.set_metadata(a=1, b=2)
    self.assertEqual(v.get_metadata('a'), 1)
    self.assertEqual(v.get_metadata('b'), 2)
    v.set_metadata({
        'b': 3,
        'c': 4,
        'is_hijax': False,
        'has_ref': False,
        'is_mutable': True,
        'eager_sharding': True,
    })
    self.assertEqual(
        v.get_metadata(),
        {
            'b': 3,
            'c': 4,
            'is_hijax': False,
            'has_ref': False,
            'is_mutable': True,
            'eager_sharding': True,
        },
    )
    self.assertEqual(v.get_metadata('b'), 3)
    self.assertEqual(v.get_metadata('c'), 4)
    c = v.get_metadata('c')
    self.assertEqual(c, 4)
    x = v.get_metadata('x', default=10)
    self.assertEqual(x, 10)

  def test_set_module_metadata(self):
    class Module(nnx.Module):
      def __init__(self):
        self.v = nnx.Variable(jnp.array(0.0))
        self.p = nnx.Param(jnp.array(1.0))

    m = Module()
    self.assertNotIn('foo', m.v.get_metadata())
    self.assertNotIn('foo', m.p.get_metadata())
    nnx.set_metadata(m, foo='bar')
    # Check that foo was added but the default metadata is still there
    v_metadata = m.v.get_metadata()
    p_metadata = m.p.get_metadata()
    self.assertEqual(v_metadata['foo'], 'bar')
    self.assertEqual(p_metadata['foo'], 'bar')
    # Check that default metadata is preserved
    self.assertIn('is_hijax', v_metadata)
    self.assertIn('has_ref', v_metadata)
    self.assertIn('is_mutable', v_metadata)

    self.assertNotIn('differentiable', m.v.get_metadata())
    self.assertNotIn('differentiable', m.p.get_metadata())
    nnx.set_metadata(m, differentiable=False, only=nnx.Param)
    # Check that v still has foo but not differentiable
    v_metadata = m.v.get_metadata()
    self.assertEqual(v_metadata['foo'], 'bar')
    self.assertNotIn('differentiable', v_metadata)
    # Check that p has both foo and differentiable
    p_metadata = m.p.get_metadata()
    self.assertEqual(p_metadata['foo'], 'bar')
    self.assertEqual(p_metadata['differentiable'], False)

  @pytest.mark.skip(reason="Ref doesn't support broadcasting yet")
  def test_broadcasting(self):
    v = nnx.Param(jnp.array([1.0, 2.0, 3.0]))
    x = v[None]
    self.assertEqual(x.shape, (1, 3))


if __name__ == '__main__':
  absltest.main()

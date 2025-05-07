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

import pytest

import flax.errors
from flax import config
from flax import nnx
import jax
import jax.numpy as jnp
from absl.testing import absltest

@pytest.mark.skipif(
  not config.flax_mutable_array, reason='MutableArray not enabled'
)
class TestMutableArray(absltest.TestCase):
  def test_tree_map(self):
    class Foo(nnx.Module):
      __data__ = ('node',)

      def __init__(self):
        self.node = jnp.array(1)
        self.meta = 1

    m = Foo()

    m = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2
    assert m.meta == 1

  def test_static(self):
    class C(nnx.Module):
      def __init__(self, meta):
        self.meta = meta

    n = 0

    @jax.jit
    def f(x):
      nonlocal n
      n += 1

    f(C(1))
    assert n == 1
    f(C(1))
    assert n == 1
    f(C(2))
    assert n == 2
    f(C(2))
    assert n == 2

  def test_variable_creation(self):
    v = nnx.Variable(1)
    self.assertEqual(v[...], 1)
    self.assertTrue(v.mutable)

  def test_variable_metadata(self):
    v = nnx.Variable(1, a=2, b=3)
    self.assertEqual(v.a, 2)
    self.assertEqual(v.b, 3)

  def test_object(self):
    class Params(nnx.Object):
      __data__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = nnx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nnx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nnx.Variable(0)

    params: Params
    params = Params(3, 4)

    paths_leaves, treedef = jax.tree.flatten_with_path(params)
    paths, leaves = zip(*paths_leaves)

    self.assertLen(paths_leaves, 3)
    self.assertEqual(leaves[0].shape, (4,))  # b
    self.assertEqual(leaves[1].shape, ())  # count
    self.assertEqual(leaves[2].shape, (3, 4))  # w
    self.assertEqual(
      paths[0],
      (jax.tree_util.GetAttrKey('b'), jax.tree_util.GetAttrKey('value')),
    )
    self.assertEqual(
      paths[1],
      (jax.tree_util.GetAttrKey('count'), jax.tree_util.GetAttrKey('value')),
    )
    self.assertEqual(
      paths[2],
      (jax.tree_util.GetAttrKey('w'), jax.tree_util.GetAttrKey('value')),
    )

    params = jax.tree.unflatten(treedef, leaves)

    self.assertEqual(params.w.shape, (3, 4))
    self.assertEqual(params.b.shape, (4,))
    self.assertEqual(params.count.shape, ())
    self.assertIsInstance(params.w, nnx.Variable)
    self.assertIsInstance(params.w[...], jax.Array)
    self.assertIsInstance(params.b, nnx.Variable)
    self.assertIsInstance(params.b[...], jax.Array)
    self.assertIsInstance(params.count, nnx.Variable)
    self.assertIsInstance(params.count[...], jax.Array)

    @jax.jit
    def linear(params: Params, x: jax.Array):
      params.count[...] += 1
      return x @ params.w[...] + params.b[None]

    x = jnp.ones((1, 3))
    y = linear(params, x)
    self.assertEqual(y.shape, (1, 4))
    self.assertEqual(params.count[...], 1)
    y = linear(params, x)
    self.assertEqual(params.count[...], 2)

  def test_object_state(self):
    class Params(nnx.Object):
      __data__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = jnp.zeros((din, dout), jnp.float32)
        self.b = jnp.zeros((dout,), jnp.float32)
        self.count = 0

    params = Params(3, 4)

    with self.assertRaises(flax.errors.TraceContextError):

      @jax.jit
      def f():
        params.count = 1

      f()

    @jax.jit
    def f(params: Params):
      params.count = 1
      return params

    params = f(params)
    self.assertEqual(params.count, 1)

  @pytest.mark.skip(reason='Rngs not yet supported')
  def test_rngs_create(self):
    rngs = nnx.Rngs(0)

    paths_leaves = jax.tree.leaves_with_path(rngs)
    paths, leaves = zip(*paths_leaves)
    self.assertLen(paths_leaves, 2)
    self.assertEqual(leaves[0].shape, ())  # key
    self.assertEqual(leaves[1].shape, ())  # count
    self.assertEqual(
      paths[0],
      (
        jax.tree_util.GetAttrKey('default'),
        jax.tree_util.GetAttrKey('count'),
        jax.tree_util.GetAttrKey('value'),
      ),
    )
    self.assertEqual(
      paths[1],
      (
        jax.tree_util.GetAttrKey('default'),
        jax.tree_util.GetAttrKey('key'),
        jax.tree_util.GetAttrKey('value'),
      ),
    )

  @pytest.mark.skip(reason='Keys in MutableArray failing')
  def test_rngs_call(self):
    rngs = nnx.Rngs(0)
    key = rngs()
    self.assertIsInstance(key, jax.Array)

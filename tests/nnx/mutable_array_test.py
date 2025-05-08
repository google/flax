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
from flax import config
from flax import nnx
import flax.errors
import jax
import jax.numpy as jnp
import pytest


@pytest.mark.skipif(
  not config.flax_mutable_array, reason='MutableArray not enabled'
)
class TestObject(absltest.TestCase):
  def test_pytree(self):
    class Foo(nnx.Module):
      __data__ = ('node',)

      def __init__(self):
        self.node = jnp.array(1)
        self.meta = 1

    m = Foo()

    m = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2
    assert m.meta == 1

  def test_pytree_dataclass(self):
    @nnx.dataclass
    class Foo(nnx.Module):
      node: jax.Array
      meta: nnx.Static[int]
      meta2: int = nnx.static(default=3)
      meta3: int = nnx.field(default=4, static=True)
      meta4: int = dataclasses.field(default=5, metadata={'static': True})
      node2: jax.Array = nnx.field(default=6)

    m = Foo(node=jnp.array(1), meta=1)

    m: Foo = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2

    assert m.meta == 1
    assert m.meta2 == 3
    assert m.meta3 == 4
    assert m.meta4 == 5
    assert m.node2 == 7


@pytest.mark.skipif(
  not config.flax_mutable_array, reason='MutableArray not enabled'
)
class TestMutableArrayGraph(absltest.TestCase):
  def test_split_mutable_array(self):
    m = nnx.mutable_array(1)
    graphdef, state = nnx.split(m)

    self.assertIs(m, state)

    m2 = nnx.merge(graphdef, state)

    self.assertIs(m2, m)

  def test_freeze_and_mutable(self):
    class Foo(nnx.Module):
      __data__ = ('a',)

      def __init__(self):
        self.a = nnx.Param(1)

    m = Foo()
    self.assertTrue(m.a.mutable)

    m2 = nnx.freeze(m)
    self.assertFalse(m2.a.mutable)
    self.assertIsNot(m, m2)

    m3 = nnx.mutable(m2)
    self.assertTrue(m3.a.mutable)
    self.assertIsNot(m2, m3)
    self.assertIsNot(m2.a, m3.a)

  def test_freeze_and_mutable_with_filter(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.BatchStat(2)

    m = Foo()
    self.assertTrue(m.a.mutable)
    self.assertTrue(m.b.mutable)

    m2 = nnx.freeze(m, only=nnx.BatchStat)
    self.assertTrue(m2.a.mutable)
    self.assertFalse(m2.b.mutable)
    self.assertIsNot(m, m2)

    m3 = nnx.mutable(m2, nnx.BatchStat)
    self.assertTrue(m3.a.mutable)
    self.assertTrue(m3.b.mutable)
    self.assertIsNot(m2, m3)
    self.assertIs(m.a, m3.a)

  def test_freeze_duplicate_error(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        self.a = nnx.mutable_array(1)
        self.b = self.a

    m = Foo()

    with self.assertRaisesRegex(
      ValueError, 'Found duplicate MutableArray found at path'
    ):
      nnx.freeze(m)

  def test_mutable_array_split(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        self.a = nnx.mutable_array(1)
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 1)
    self.assertLen(ref_map, 2)  # 1 Foo + 1 MutableArray

    m1 = nnx.merge(graphdef, state)
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, nnx.MutableArray)

  def test_mutable_array_split_merge_in_variable(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        self.a = nnx.Param(nnx.mutable_array(1))
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 1)
    self.assertLen(ref_map, 3)  # 1 Foo + 1 Param + 1 MutableArray

    m1 = nnx.merge(graphdef, state)
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, nnx.Param)

  def test_mutable_array_split_merge_in_variable_shared_array(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        m_array = nnx.mutable_array(1)
        self.a = nnx.Param(m_array)
        self.b = nnx.Param(m_array)

    m = Foo()
    self.assertIs(m.a.raw_value, m.b.raw_value)

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 1)
    self.assertLen(ref_map, 4)  # 1 Foo + 2 Param + 1 MutableArray

    m1 = nnx.merge(graphdef, state)
    self.assertIs(m1.a.raw_value, m1.b.raw_value)
    self.assertIsInstance(m1.a, nnx.Param)

  def test_mutable_array_split_freeze(self):
    class Foo(nnx.Module):
      __data__ = ('a', 'b')

      def __init__(self):
        self.a = nnx.mutable_array(1)
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    state = nnx.freeze(state)
    self.assertLen(state, 1)

    m1 = nnx.merge(graphdef, nnx.mutable(state))
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, nnx.MutableArray)

  def test_update_context(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    with nnx.update_context('example'):
      with nnx.split_context('example') as ctx:
        graphdef, state = ctx.split(m1)

      with nnx.merge_context('example', True) as ctx:
        m2 = ctx.merge(graphdef, state)

      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

      with nnx.split_context('example') as ctx:
        graphdef_out, state_out = ctx.split((m2, m_out1, m2))

      self.assertIsInstance(state_out[0]['kernel'].value, nnx.graph.NoUpdate)
      self.assertIsInstance(state_out[0]['bias'].value, nnx.graph.NoUpdate)
      self.assertIsInstance(
        state_out[1]['kernel'].value, nnx.graph.MutableArrayOutput
      )
      self.assertIsInstance(
        state_out[1]['bias'].value, nnx.graph.MutableArrayOutput
      )
      # 2 MutableArrayOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(state_out), 2)

      with nnx.merge_context('example', False) as ctx:
        m3, m_out2, _ = ctx.merge(graphdef_out, state_out)

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_flatten(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    with nnx.update_context('example'):
      with nnx.split_context('example') as ctx:
        graphdef, state = ctx.flatten(m1)

      with nnx.merge_context('example', True) as ctx:
        m2 = ctx.merge(graphdef, state)

      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

      with nnx.split_context('example') as ctx:
        graphdef_out, state_out = ctx.flatten((m2, m_out1, m2))

      state_out_dict = dict(state_out)

      self.assertIsInstance(
        state_out_dict[(0, 'kernel')].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        state_out_dict[(0, 'bias')].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        state_out_dict[(1, 'kernel')].value, nnx.graph.MutableArrayOutput
      )
      self.assertIsInstance(
        state_out_dict[(1, 'bias')].value, nnx.graph.MutableArrayOutput
      )
      # 2 MutableArrayOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(state_out), 2)

      with nnx.merge_context('example', False) as ctx:
        m3, m_out2, _ = ctx.merge(graphdef_out, state_out)

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree1(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    with nnx.update_context('example'):
      m1_tree = nnx.to_tree((m1,), ctxtag='example')

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True)

      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example')

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'].value, nnx.graph.MutableArrayOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'].value, nnx.graph.MutableArrayOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 MutableArrayOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree2(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    with nnx.update_context('example') as ctx:
      m1_tree = nnx.to_tree((m1,), ctxtag='example')

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True)

      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example')

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'].value, nnx.graph.MutableArrayOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'].value, nnx.graph.MutableArrayOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 MutableArrayOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree_trivial_prefix(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    with nnx.update_context('example'):
      m1_tree = nnx.to_tree((m1,), ctxtag='example', prefix=0)

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True, prefix=0)

      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example', prefix=0)

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'].value, nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'].value, nnx.graph.MutableArrayOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'].value, nnx.graph.MutableArrayOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 MutableArrayOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False, prefix=0
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)


@pytest.mark.skipif(
  not config.flax_mutable_array, reason='MutableArray not enabled'
)
class TestMutableArrayNNXTransforms(absltest.TestCase):
  def test_simple_jit(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    m_out1 = None

    @nnx.jit
    def f(m2):
      nonlocal m_out1
      m_out1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
      return m_out1

    m_out2 = f(m1)

    self.assertIsNot(m_out1, m_out2)
    self.assertIsInstance(m_out2.kernel, nnx.Param)
    self.assertTrue(nnx.is_mutable_array(m_out2.kernel.raw_value))

  def test_jit_mutable(self):
    @nnx.dataclass
    class Foo(nnx.Object):
      a: nnx.MutableArray

    m1 = Foo(a=nnx.mutable_array(1))

    @nnx.jit
    def f(m2: Foo):
      m2.a[...] += 1
      return m2

    m_out1 = f(m1)
    self.assertEqual(m_out1.a[...], 2)
    self.assertIs(m_out1, m1)
    self.assertIsInstance(m_out1.a, nnx.MutableArray)


@pytest.mark.skipif(
    not config.flax_mutable_array, reason='MutableArray not enabled'
)
class TestMutableArray(absltest.TestCase):

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

  def test_rngs_call(self):
    rngs = nnx.Rngs(0)
    key = rngs()
    self.assertIsInstance(key, jax.Array)


if __name__ == '__main__':
  absltest.main()

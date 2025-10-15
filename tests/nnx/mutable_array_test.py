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
import optax
import pytest
from flax import nnx
import flax.errors
import jax
import jax.numpy as jnp


class TestObject(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.using_hijax = nnx.current_variable_mode()
    nnx.variable_mode('hijax')

  @classmethod
  def tearDownClass(cls):
    nnx.variable_mode(cls.using_hijax)

  def test_pytree(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.node = jnp.array(1)
        self.meta = 1

    m = Foo()

    m = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2
    assert m.meta == 1

  def test_pytree_data_typehint(self):
    class Foo(nnx.Module):
      node: nnx.Data[jax.Array]

      def __init__(self):
        self.node = jnp.array(1)
        self.meta = 1

    m = Foo()

    m = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2
    assert m.meta == 1

  def test_pytree_data_instance(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.node = nnx.data(jnp.array(1))
        self.meta = 1

    m = Foo()

    m = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2
    assert m.meta == 1

  def test_pytree_dataclass(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      node: nnx.Data[jax.Array]
      meta: int
      meta2: int = 3
      meta3: int = 4
      meta4: int = 5
      node2: nnx.Data[int] = 6

    m = Foo(node=jnp.array(1), meta=1)

    m: Foo = jax.tree.map(lambda x: x + 1, m)

    assert m.node == 2

    assert m.meta == 1
    assert m.meta2 == 3
    assert m.meta3 == 4
    assert m.meta4 == 5
    assert m.node2 == 7

  def test_data_example(self):
    class Foo(nnx.Pytree):
      def __init__(self):
        self.data_attr = nnx.data(42)  # pytree data
        self.static_attr = 'hello'  # static attribute

    foo = Foo()

    self.assertEqual(jax.tree.leaves(foo), [42])

  def test_register_data_type(self):
    @dataclasses.dataclass(frozen=True)
    class MyType:
      value: int

    nnx.register_data_type(MyType)

    class Foo(nnx.Pytree):
      def __init__(self, a):
        self.a = MyType(a)  # Automatically registered as data
        self.b = 'hello'  # str not registered as data

    foo = Foo(42)

    self.assertTrue(nnx.is_data(foo.a))
    self.assertEqual(jax.tree.leaves(foo), [MyType(value=42)])



class TestMutableArrayGraph(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.using_hijax = nnx.current_variable_mode()
    nnx.variable_mode('hijax')

  @classmethod
  def tearDownClass(cls):
    nnx.variable_mode(cls.using_hijax)

  def test_split_mutable_array(self):
    m = jax.new_ref(1)
    graphdef, state = nnx.split(m)

    self.assertIs(m, state)

    m2 = nnx.merge(graphdef, state)

    self.assertIs(m2, m)

  def test_freeze_and_mutable(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)

    m = Foo()
    self.assertEqual(m.a.mode, 'hijax')

    m2 = nnx.to_lojax(m)
    self.assertEqual(m2.a.mode, 'lojax')
    self.assertIsNot(m, m2)

    m3 = nnx.to_hijax(m2)
    self.assertEqual(m3.a.mode, 'hijax')
    self.assertIsNot(m2, m3)
    self.assertIsNot(m2.a, m3.a)

  def test_to_arrays_example(self):

    node = [jnp.array(1.0), jax.new_ref(jnp.array(2.0))]
    mutable_node = nnx.to_refs(node)
    assert isinstance(mutable_node[0], jax.Ref)
    assert isinstance(mutable_node[1], jax.Ref)

    shared_array = jnp.array(1.0)
    node = [shared_array, shared_array]
    with self.assertRaisesRegex(
      ValueError,
      'Found duplicate at path'
    ):
      nnx.to_refs(node)

    node = [jnp.array(1.0), jnp.array(2.0)]
    mutable_node = nnx.to_refs(node, only=lambda path, x: path[0] == 0)
    assert isinstance(mutable_node[0], jax.Ref)
    assert isinstance(mutable_node[1], jax.Array)

  def test_freeze_and_mutable_with_filter(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.BatchStat(2)

    m = Foo()
    self.assertEqual(m.a.mode, 'hijax')
    self.assertEqual(m.b.mode, 'hijax')

    m2 = nnx.to_lojax(m, only=nnx.BatchStat)
    self.assertEqual(m2.a.mode, 'hijax')
    self.assertEqual(m2.b.mode, 'lojax')
    self.assertIsNot(m, m2)

    m3 = nnx.to_hijax(m2, only=nnx.BatchStat)
    self.assertEqual(m3.a.mode, 'hijax')
    self.assertEqual(m3.b.mode, 'hijax')
    self.assertIsNot(m2, m3)
    self.assertIs(m.a, m3.a)

  def test_freeze_duplicate_error(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jax.new_ref(1)
        self.b = self.a

    m = Foo()

    with self.assertRaisesRegex(ValueError, 'Found duplicate at path'):
      nnx.to_arrays(m)

  def test_mutable_array_split(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jax.new_ref(1)
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 1)
    self.assertLen(ref_map, 2)  # 1 Foo + 1 ArrayRef

    m1 = nnx.merge(graphdef, state)
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, jax.Ref)

  def test_mutable_array_split_merge_in_variable(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1, mode='hijax')
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 1)
    self.assertLen(ref_map, 2)  # 1 Foo + 1 Param

    m1 = nnx.merge(graphdef, state)
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, nnx.Param)

  def test_mutable_array_split_merge_in_variable_shared_array(self):
    class Foo(nnx.Module):
      def __init__(self):
        m_array = 1
        self.a = nnx.Param(m_array, mode='hijax')
        self.b = nnx.Param(m_array, mode='hijax')

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    self.assertLen(state, 2)
    self.assertLen(ref_map, 3)  # 1 Foo + 2 Param

    m1 = nnx.merge(graphdef, state)
    # Each variable will own its own array and ref.
    self.assertIsInstance(m1.a, nnx.Param)

  def test_mutable_example(self):
    tree = [jnp.array(1.0), jax.new_ref(jnp.array(2.0))]
    mutable_tree = nnx.to_refs(tree)
    assert isinstance(mutable_tree[0], jax.Ref)
    assert isinstance(mutable_tree[1], jax.Ref)

  def test_mutable_array_split_freeze(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jax.new_ref(1)
        self.b = self.a

    m = Foo()

    ref_map = nnx.graph.RefMap()
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_map)
    state = nnx.to_lojax(state)
    self.assertLen(state, 1)

    m1 = nnx.merge(graphdef, nnx.to_hijax(state))
    self.assertIs(m1.a, m1.b)
    self.assertIsInstance(m1.a, jax.Ref)

  def test_update_context(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.kernel = jax.new_ref(1)
        self.bias = jax.new_ref(2)

    m1 = Foo()
    with nnx.update_context('example'):
      with nnx.split_context('example') as ctx:
        graphdef, state = ctx.split(m1)

      with nnx.merge_context('example', True) as ctx:
        m2 = ctx.merge(graphdef, state)

      m_out1 = Foo()

      with nnx.split_context('example') as ctx:
        graphdef_out, state_out = ctx.split((m2, m_out1, m2))

      self.assertIsInstance(state_out[0]['kernel'], nnx.graph.NoUpdate)
      self.assertIsInstance(state_out[0]['bias'], nnx.graph.NoUpdate)
      self.assertIsInstance(state_out[1]['kernel'], nnx.graph.ArrayRefOutput)
      self.assertIsInstance(state_out[1]['bias'], nnx.graph.ArrayRefOutput)
      # 2 ArrayRefOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(state_out), 2)

      with nnx.merge_context('example', False) as ctx:
        m3, m_out2, _ = ctx.merge(graphdef_out, state_out)

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_flatten(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.kernel = jax.new_ref(1)
        self.bias = jax.new_ref(2)

    m1 = Foo()
    with nnx.update_context('example'):
      with nnx.split_context('example') as ctx:
        graphdef, state = ctx.flatten(m1)

      with nnx.merge_context('example', True) as ctx:
        m2 = ctx.merge(graphdef, state)

      m_out1 = Foo()

      with nnx.split_context('example') as ctx:
        graphdef_out, state_out = ctx.flatten((m2, m_out1, m2))

      state_out_dict = dict(state_out)

      self.assertIsInstance(state_out_dict[(0, 'kernel')], nnx.graph.NoUpdate)
      self.assertIsInstance(state_out_dict[(0, 'bias')], nnx.graph.NoUpdate)
      self.assertIsInstance(
        state_out_dict[(1, 'kernel')], nnx.graph.ArrayRefOutput
      )
      self.assertIsInstance(
        state_out_dict[(1, 'bias')], nnx.graph.ArrayRefOutput
      )
      # 2 ArrayRefOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(state_out), 2)

      with nnx.merge_context('example', False) as ctx:
        m3, m_out2, _ = ctx.merge(graphdef_out, state_out)

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree1(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.kernel = jax.new_ref(1)
        self.bias = jax.new_ref(2)

    m1 = Foo()
    with nnx.update_context('example'):
      m1_tree = nnx.to_tree((m1,), ctxtag='example')

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True)

      m_out1 = Foo()

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example')

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'], nnx.graph.ArrayRefOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'], nnx.graph.ArrayRefOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 ArrayRefOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree2(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.kernel = jax.new_ref(1)
        self.bias = jax.new_ref(2)

    m1 = Foo()
    with nnx.update_context('example') as ctx:
      m1_tree = nnx.to_tree((m1,), ctxtag='example')

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True)

      m_out1 = Foo()

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example')

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'], nnx.graph.ArrayRefOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'], nnx.graph.ArrayRefOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 ArrayRefOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)

  def test_update_context_to_tree_trivial_prefix(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.kernel = jax.new_ref(1)
        self.bias = jax.new_ref(2)

    m1 = Foo()
    with nnx.update_context('example'):
      m1_tree = nnx.to_tree((m1,), ctxtag='example', prefix=0)

      (m2,) = nnx.from_tree(m1_tree, ctxtag='example', is_inner=True, prefix=0)

      m_out1 = Foo()

      # with nnx.split_context('example') as ctx:
      #   graphdef_out, state_out = ctx.split((m2, m_out1))
      out_tree = nnx.to_tree(((m2,), m_out1, m2), ctxtag='example', prefix=0)

      self.assertIsInstance(
        out_tree[0][0].states[0]['kernel'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[0][0].states[0]['bias'], nnx.graph.NoUpdate
      )
      self.assertIsInstance(
        out_tree[1].states[0]['kernel'], nnx.graph.ArrayRefOutput
      )
      self.assertIsInstance(
        out_tree[1].states[0]['bias'], nnx.graph.ArrayRefOutput
      )
      self.assertEmpty(out_tree[2].states[0])  # Repeated m2 State

      # 2 ArrayRefOutput + 2 NoUpdate, however, NoUpdate are empty nodes
      self.assertLen(jax.tree.leaves(out_tree), 2)

      # with nnx.merge_context('example', False) as ctx:
      #   m3, m_out2 = ctx.merge(graphdef_out, state_out)
      (m3,), m_out2, _ = nnx.from_tree(
        out_tree, ctxtag='example', is_inner=False, prefix=0
      )

      self.assertIs(m3, m1)
      self.assertIsNot(m_out2, m_out1)



class TestMutableArrayNNXTransforms(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.using_hijax = nnx.current_variable_mode()
    nnx.variable_mode('hijax')

  @classmethod
  def tearDownClass(cls):
    nnx.variable_mode(cls.using_hijax)

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
    self.assertIsInstance(m_out2.kernel[...], jax.Array)

  def test_jit_mutable(self):
    @dataclasses.dataclass
    class Foo(nnx.Pytree):
      a: nnx.Data[jax.Ref]

    m1 = Foo(a=jax.new_ref(1))

    @nnx.jit
    def f(m2: Foo):
      m2.a[...] += 1
      return m2

    m_out1 = f(m1)
    self.assertEqual(m_out1.a[...], 2)
    self.assertIs(m_out1, m1)
    self.assertIsInstance(m_out1.a, jax.Ref)



class TestMutableArray(absltest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.using_hijax = nnx.current_variable_mode()
    nnx.variable_mode('hijax')

  @classmethod
  def tearDownClass(cls):
    nnx.variable_mode(cls.using_hijax)

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
    v = nnx.Variable(jnp.array(1))
    self.assertEqual(v[...], 1)
    self.assertEqual(v.mode, 'hijax')

  def test_variable_metadata(self):
    v = nnx.Variable(jnp.array(1), a=2, b=3)
    self.assertEqual(v.a, 2)
    self.assertEqual(v.b, 3)

  def test_object(self):
    class Params(nnx.Pytree):
      def __init__(self, din: int, dout: int):
        self.w = nnx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nnx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nnx.Variable(jnp.array(0))

    params: Params
    params = Params(3, 4)

    paths_leaves, treedef = jax.tree.flatten_with_path(params)
    paths, leaves = zip(*paths_leaves)

    self.assertLen(paths_leaves, 3)
    self.assertEqual(leaves[0].shape, (4,))  # b
    self.assertEqual(leaves[1].shape, ())  # count
    self.assertEqual(leaves[2].shape, (3, 4))  # w
    self.assertEqual(paths[0], (jax.tree_util.GetAttrKey('b'),))
    self.assertEqual(paths[1], (jax.tree_util.GetAttrKey('count'),))
    self.assertEqual(paths[2], (jax.tree_util.GetAttrKey('w'),))

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
    class Params(nnx.Pytree):
      def __init__(self, din: int, dout: int):
        self.w = jnp.zeros((din, dout), jnp.float32)
        self.b = jnp.zeros((dout,), jnp.float32)
        self.count = nnx.data(0)

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
      ),
    )
    self.assertEqual(
      paths[1],
      (
        jax.tree_util.GetAttrKey('default'),
        jax.tree_util.GetAttrKey('key'),
      ),
    )

  def test_rngs_call(self):
    rngs = nnx.Rngs(0)
    key = rngs()
    self.assertIsInstance(key, jax.Array)


class TestOptimizer(absltest.TestCase):
  def test_optimize_arrays(self):
    class Model(nnx.Module):
      def __init__(self, rngs):
        self.w = jax.random.uniform(rngs(), (2, 4))
        self.count = jnp.array(0)

      def __call__(self, x):
        self.count += 1
        return x @ self.w

    x = jax.random.normal(jax.random.key(0), (5, 2))
    y = jnp.ones((5, 4))

    with nnx.variable_mode('lojax'):
      wrt = lambda path, x: path[-1] == 'w'
      model = Model(nnx.Rngs(1))
      optimizer = nnx.Optimizer(
        model, tx=optax.adam(1e-3), wrt=wrt
      )

    @jax.jit
    def train_step(model, optimizer, x, y):
      graphdef, params, nondiff = nnx.split(model, wrt, ...)

      def loss_fn(params):
        model = nnx.merge(graphdef, params, nondiff)
        return jnp.mean((model(x) - y) ** 2), nnx.state(model, nnx.Not(wrt))

      (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
      nnx.update(model, updates)
      optimizer.update(model, grads)

      return loss, model, optimizer

    loss, model, optimizer = train_step(model, optimizer, x, y)

    self.assertNotEqual(loss, 0.0)
    self.assertEqual(model.count[...], 1)
    self.assertEqual(optimizer.step[...], 1)

  def test_optimize_mutable_arrays(self):
    class Model(nnx.Module):
      def __init__(self, rngs):
        self.w = nnx.Variable(jax.random.uniform(rngs(), (2, 4)))
        self.count = nnx.Variable(jnp.array(0))

      def __call__(self, x):
        self.count[...] += 1
        return x @ self.w

    x = jax.random.normal(jax.random.key(0), (5, 2))
    y = jnp.ones((5, 4))

    with nnx.variable_mode('hijax'):
      wrt = lambda path, x: path[-1] == 'w'
      model = Model(nnx.Rngs(1))
      optimizer = nnx.Optimizer(model, tx=optax.adam(1e-3), wrt=wrt)

    @jax.jit
    def train_step(model, optimizer, x, y):
      graphdef, params, nondiff = nnx.split(model, wrt, ...)

      def loss_fn(params):
        model = nnx.merge(graphdef, params, nondiff)
        return jnp.mean((model(x) - y) ** 2)

      loss, grads = jax.value_and_grad(loss_fn)(nnx.to_lojax(params))
      optimizer.update(params, grads)
      return loss

    loss = train_step(model, optimizer, x, y)

    self.assertNotEqual(loss, 0.0)

class TestHijaxVariables(absltest.TestCase):
  def test_variable_to_hijax(self):
    v_low = nnx.Param(jnp.array(1), a='hi')
    v_hi = nnx.to_hijax(v_low)

    self.assertEqual(v_hi.mode, 'hijax')
    self.assertEqual(v_hi[...], 1)
    self.assertIsInstance(v_hi, nnx.Param)

    v_hi[...] = 2
    self.assertEqual(v_hi[...], 2)

    @jax.jit
    def set(v_hi: nnx.Param, a):
      self.assertIsInstance(v_hi, nnx.Param)
      v_hi[...] = a
      self.assertEqual(v_hi.a, 'hi')
      self.assertEqual(v_hi.mode, 'hijax')
      v_hi[...] += 5
      return v_hi + 2

    y = set(v_hi, 10)
    self.assertEqual(v_hi[...], 15)
    self.assertEqual(y, 17)

    v_low = nnx.to_lojax(v_hi)
    self.assertEqual(v_low.mode, 'lojax')
    self.assertIsInstance(v_low, nnx.Param)

  def test_from_metadata(self):
    value = 1
    metadata = {'a': 'hi', 'mode': 'lojax'}
    v_low = nnx.Param.from_metadata(value, metadata)
    self.assertIsInstance(v_low, nnx.Param)
    self.assertEqual(v_low.mode, 'lojax')

    metadata['mode'] = 'hijax'
    v_hi = nnx.Param.from_metadata(value, metadata)
    self.assertIsInstance(v_hi, nnx.Param)
    self.assertEqual(v_hi.mode, 'hijax')

  def test_variable_to_hijax_clean(self):
    v_low = nnx.Param(jnp.array([1]), tag='hello')
    print()
    print(v_low)
    assert v_low.mode == 'lojax'
    v_hi = nnx.to_hijax(v_low)
    v_hi[...] = jnp.array([2])
    assert v_hi.mode == 'hijax'
    print(v_hi)
    assert v_hi[...] == 2

    @jax.jit
    def set(v_hi, a):
      v_hi[...] = a
      print(v_hi)
      assert v_hi.tag == 'hello'

    set(v_hi, 10)

    assert v_hi[...] == 10

    v_low = nnx.to_lojax(v_hi)

    assert v_low.mode == 'lojax'
    assert v_low[...] == 10

  def test_hijax_and_pytree(self):
    class Foo(nnx.Pytree):
      def __init__(self, din, dout, rngs: nnx.Rngs):
        self.w = nnx.Param(rngs.uniform((din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = nnx.Variable(0)

    foo = Foo(2, 4, nnx.Rngs(1))
    assert foo.w.mode == 'lojax'
    assert foo.b.mode == 'lojax'

    foo = nnx.to_hijax(foo)

    assert foo.w.mode == 'hijax'
    assert foo.b.mode == 'hijax'

    @jax.jit
    def forward(foo, x):
      foo.count[...] += 1
      return x @ foo.w + foo.b[None]

    x = jnp.ones((1, 2))
    y = forward(foo, x)
    assert y.shape == (1, 4)
    assert foo.count[...] == 1

  def test_use_hijax(self):
    v_low = nnx.Param(1, a='hi')
    self.assertEqual(v_low.mode, 'lojax')

    v_hi = nnx.Param(1, a='hi', mode='hijax')
    self.assertEqual(v_hi.mode, 'hijax')

    with nnx.variable_mode('hijax'):
      v2 = nnx.Param(1, a='hi')
      self.assertEqual(v2.mode, 'hijax')

  @nnx.variable_mode('hijax')
  def test_hijax_rngs(self):
    rngs = nnx.Rngs(0)

    @jax.jit
    def f(rngs: nnx.Rngs):
      return rngs()

    k1 = f(rngs)
    k2 = f(rngs)

    assert k1 != k2

  @pytest.mark.skip(reason='not yet supported')
  def test_return_hijax_from_transform(self):
    @jax.jit
    def create_var():
      return nnx.Param(1, mode='hijax')

    v = create_var()
    self.assertEqual(v.mode, 'hijax')

  @pytest.mark.skip(reason='not yet supported')
  @nnx.variable_mode('hijax')
  def test_lower(self):
    v = nnx.Param(jnp.ones((2, 3)))

    @jax.jit
    def f(v):
      v[...] += 1
      return v[...]

    e = f.lower(v)
    y = e.out_info[2]
    self.assertEqual(y.shape, ())

  @nnx.variable_mode('hijax')
  def test_eval_shape(self):
    v = nnx.Param(jnp.array(0))

    def f(v):
      v[...] += 1
      return v[...]

    y = jax.eval_shape(f, v)

    self.assertEqual(y.shape, ())


if __name__ == '__main__':
  absltest.main()

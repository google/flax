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
import numpy as np
import optax
from flax import struct
from flax.experimental import nx
import jax
import jax.numpy as jnp


class VariableTest(absltest.TestCase):
  def test_variable_creation(self):
    v = nx.Variable(1)
    self.assertEqual(v[...], 1)

  def test_variable_metadata(self):
    v = nx.Variable(1, a=2, b=3)
    self.assertEqual(v.a, 2)
    self.assertEqual(v.b, 3)

  @absltest.skip('proxy not implemented')
  def test_variable_is_proxy(self):
    x1 = nx.Variable(1)
    x2 = nx.Variable(2)
    x3 = x1 + x2
    self.assertEqual(x3, 3)

  def test_len(self):
    x1 = nx.Variable([1, 2, 3])
    self.assertLen(x1, 3)

  def test_replace(self):
    x = nx.Variable(1, a=2, mutable=True)

    self.assertTrue(x.mutable)
    self.assertEqual(x.a, 2)

    x = x.copy(value=4, mutable=False, a=5)
    self.assertFalse(x.mutable)
    self.assertEqual(x[...], 4)
    self.assertEqual(x.a, 5)


class StatelibTest(absltest.TestCase):
  def test_split_merge(self):
    @struct.dataclass
    class Params:
      w: nx.Param
      b: nx.Param

      @classmethod
      def create(cls, din: int, dout: int):
        return cls(
          w=nx.Param(jnp.zeros((din, dout), jnp.float32)),
          b=nx.Param(jnp.zeros((dout,), jnp.float32)),
        )

    params = Params.create(3, 4)

    is_kernel = nx.PathContains('w')
    is_bias = nx.PathContains('b')
    treedef, kernels, biases = nx.split(params, is_kernel, is_bias)

    self.assertIsInstance(kernels, dict)
    self.assertIsInstance(biases, dict)
    self.assertIn('w', kernels)
    self.assertIn('b', biases)

    params = nx.merge(treedef, kernels, biases)

    self.assertIsInstance(params, Params)
    self.assertEqual(params.w.shape, (3, 4))
    self.assertEqual(params.b.shape, (4,))

  def test_recursive_map(self):
    class Linear(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    class MLP(nx.Pytree):
      __nodes__ = ('linear1', 'linear2')

      def __init__(self, din: int, dmid: int, dout: int, *, rngs: nx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    model: MLP = nx.mutable(MLP(1, 64, 1, rngs=nx.Rngs(0)))

    def map_fn(path, x):
      if path == ('linear1',):
        return None
      elif path == ('linear2', 'b'):
        return None
      else:
        return x

    model = nx.recursive_map(model, map_fn)

    self.assertIsNone(model.linear1)
    self.assertIsNone(model.linear2.b)
    self.assertIsInstance(model.linear2.w, nx.Param)
    self.assertIsInstance(model.linear2.count, nx.Variable)

  def test_update_mutable(self):
    class Linear(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    class MLP(nx.Pytree):
      __nodes__ = ('linear1', 'linear2')

      def __init__(self, din: int, dmid: int, dout: int, *, rngs: nx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    model = nx.mutable(MLP(1, 3, 1, rngs=nx.Rngs(0)))
    updates = {
      'linear1': {
        'w': jnp.ones((1, 3)),
      },
    }

    updated_model = nx.update(model, updates)

    np.testing.assert_allclose(updated_model.linear1.w[...], jnp.ones((1, 3)))
    self.assertTrue(updated_model.linear1.w.mutable)

  def test_update_immutable(self):
    class Linear(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    class MLP(nx.Pytree):
      __nodes__ = ('linear1', 'linear2')

      def __init__(self, din: int, dmid: int, dout: int, *, rngs: nx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    model = MLP(1, 3, 1, rngs=nx.Rngs(0))
    updates = {
      'linear1': {
        'w': jnp.ones((1, 3)),
      },
    }

    updated_model = nx.update(model, updates)

    np.testing.assert_allclose(updated_model.linear1.w[...], jnp.ones((1, 3)))
    self.assertFalse(updated_model.linear1.w.mutable)
    self.assertIsNot(updated_model.linear1.w, model.linear1.w)


class ObjectTest(absltest.TestCase):
  def test_object_dataclass(self):
    @dataclasses.dataclass
    class Params(nx.Pytree):
      din: int
      dout: int

      __nodes__ = ('w', 'b', 'count')

      def __post_init__(self):
        self.w = nx.Param(jnp.zeros((self.din, self.dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((self.dout,), jnp.float32))
        self.count = nx.Variable(0)

    params = Params(3, 4)
    leaves = jax.tree.leaves(params)

    self.assertLen(leaves, 3)

    with self.assertRaises(AttributeError, msg='Cannot set attribute'):
      params.w = None

  def test_object(self):
    class Params(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = nx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

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
    self.assertIsInstance(params.w, nx.Variable)
    self.assertIsInstance(params.w[...], jax.Array)
    self.assertIsInstance(params.b, nx.Variable)
    self.assertIsInstance(params.b[...], jax.Array)
    self.assertIsInstance(params.count, nx.Variable)
    self.assertIsInstance(params.count[...], jax.Array)

    params = nx.mutable(params)

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

  def test_object_frozen(self):
    class Params(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = nx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    params = Params(3, 4)

    with self.assertRaises(AttributeError, msg='Cannot set attribute'):
      params.w = None

  @absltest.skip('setattr not implemented')
  def test_attrs(self):
    class Params(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = nx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    params = Params(3, 4)

    with self.assertRaises(AttributeError, msg='Cannot set attribute'):
      params.w = None

    nx.setattr(params, 'w', None)
    self.assertIsNone(params.w)

    @jax.jit
    def f(x: jax.Array):
      nx.setattr(params, 'w', x)

    x = jnp.ones((3, 4))
    f(x)

    self.assertEqual(params.w[...], x)

  def test_object_inheritance(self):
    class Base(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.a = 1
        self.w = nx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    self.assertEqual(Base._pytree__nodes, {'w', 'b', 'count'})

    class Child(Base):
      __nodes__ = ('new_field',)

      def __init__(self, din: int, dout: int):
        super().__init__(din, dout)
        self.new_field = nx.Variable(0)
        self.t = 2

    self.assertEqual(Child._pytree__nodes, {'w', 'b', 'count', 'new_field'})

    child = Child(3, 4)
    leaves = jax.tree.leaves(child)

    self.assertLen(leaves, 4)
    self.assertEqual(leaves[0].shape, (4,))  # b
    self.assertEqual(leaves[1].shape, ())  # count
    self.assertEqual(leaves[2].shape, ())  # new_field
    self.assertEqual(leaves[3].shape, (3, 4))  # w

class TestSow(absltest.TestCase):
  @absltest.skip('sow currently does work')
  def test_sow(self):
    @jax.jit
    def f(x):
      z = x * 2
      nx.sow('intermediates', 'z', z)
      return z**2

    with nx.sow_context('intermediates') as sowed:
      y = f(3)

    self.assertEqual(sowed.intermediates.z, 6)
    self.assertEqual(y, 36)

class RngsTest(absltest.TestCase):
  def test_rngs_basic(self):
    class Params(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    params = Params(3, 4, rngs=nx.Rngs(0))
    params = nx.mutable(params)

    @jax.jit
    def linear(x: jax.Array):
      params.count[...] += 1
      return x @ params.w[...] + params.b[None]

    x = jnp.ones((1, 3))
    y = linear(x)

    assert y.shape == (1, 4)
    assert params.count[...] == 1

    y = linear(x)

    assert params.count[...] == 2

  def test_split_rngs(self):
    rngs = nx.Rngs(params=0, dropout=1)

    self.assertEqual(rngs.params.tag, 'params')
    self.assertEqual(rngs.dropout.tag, 'dropout')
    self.assertEqual(rngs.params.key.shape, ())
    self.assertEqual(rngs.dropout.key.shape, ())
    self.assertEqual(rngs.params.count.shape, ())
    self.assertEqual(rngs.dropout.count.shape, ())

    split_rngs = nx.split_rngs(rngs, 5, only='params')

    self.assertEqual(split_rngs.params.key.shape, (5,))
    self.assertEqual(split_rngs.params.count.shape, (5,))
    self.assertEqual(split_rngs.dropout.key.shape, ())
    self.assertEqual(split_rngs.dropout.count.shape, ())
    self.assertEqual(rngs.params.count[...], 1)
    self.assertEqual(rngs.dropout.count[...], 0)


class TransformsTest(absltest.TestCase):
  def test_jit(self):
    @struct.dataclass
    class Params:
      w: nx.Param
      b: nx.Param
      count: nx.Variable

      @classmethod
      def create(cls, din: int, dout: int):
        return cls(
          w=nx.Param(jnp.zeros((din, dout), jnp.float32)),
          b=nx.Param(jnp.zeros((dout,), jnp.float32)),
          count=nx.Variable(0),
        )

    params = Params.create(3, 4)

    self.assertIsInstance(params.w[...], jax.Array)
    self.assertIsInstance(params.b[...], jax.Array)

    params = nx.mutable(params)

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

  def test_split_merge_nested(self):
    @struct.dataclass
    class Block:
      w: nx.Param
      b: nx.Param

      @staticmethod
      def create(din: int, dout: int):
        return Block(
          w=nx.Param(jnp.zeros((din, dout), jnp.float32)),
          b=nx.Param(jnp.zeros((dout,), jnp.float32)),
        )

    @struct.dataclass
    class Model:
      blocks: tuple[Block, ...]
      count: nx.Variable

      @staticmethod
      def create(num_blocks: int, dim: int):
        return Model(
          blocks=tuple(Block.create(dim, dim) for _ in range(num_blocks)),
          count=nx.Variable(0),
        )

    @jax.jit
    def forward(model: Model, x: jax.Array):
      for block in model.blocks:
        x = x @ block.w[...] + block.b[None]
      model.count[...] += 1
      return x

    model = nx.mutable(Model.create(3, 5))

    y = forward(model, jnp.ones((1, 5)))
    self.assertEqual(y.shape, (1, 5))
    self.assertEqual(model.count[...], 1)

    is_kernel = nx.PathContains('w')
    is_bias = nx.PathContains('b')
    treedef, kernels, biases, counts = nx.split(model, is_kernel, is_bias, ...)

    self.assertLen(jax.tree.leaves(kernels), 3)
    self.assertLen(jax.tree.leaves(biases), 3)
    self.assertLen(jax.tree.leaves(counts), 1)

  @absltest.skip('grad not yet supported')
  def test_grad(self):
    x = nx.Variable(1.0)

    @jax.grad
    def f(x):
      return x**2

    g = f(x)
    self.assertEqual(g, 2.0)


class DemosTest(absltest.TestCase):
  def test_demo_scan(self):
    class Count(nx.Variable):
      pass

    @struct.dataclass
    class Weights:
      w: nx.Param
      b: nx.Param
      count: Count

      @staticmethod
      def create(num_blocks: int, dim: int):
        return Weights(
          w=nx.Param(jnp.zeros((num_blocks, dim, dim))),
          b=nx.Param(jnp.zeros((num_blocks, dim))),
          count=Count(0),
        )

    weights = nx.mutable(Weights.create(3, 5))
    treedef, params, counts = nx.split(weights, nx.Param, Count)

    def forward(x: jax.Array, params):
      weights = nx.merge(treedef, params, counts)
      x = x @ weights.w[...] + weights.b[None]
      weights.count[...] += 1
      return x, None

    x = jnp.ones((1, 5))
    y, _ = jax.lax.scan(forward, x, params)

    assert y.shape == (1, 5)
    assert weights.count[...] == 3

  def test_clean_demo(self):
    @struct.dataclass
    class Params:
      w: nx.Variable
      b: nx.Variable
      count: nx.Variable

      @classmethod
      def create(cls, din, dout: int):
        return cls(
          w=nx.Variable(jnp.zeros((din, dout))),
          b=nx.Variable(jnp.zeros((dout,))),
          count=nx.Variable(0),
        )

    params = Params.create(3, 4)
    params = nx.mutable(params)

    @jax.jit  # <=== 🎉
    def linear(params, x):
      params.count[...] += 1
      return x @ params.w[...] + params.b[None]

    x = jnp.ones((1, 3))
    y = linear(params, x)

    assert y.shape == (1, 4)
    assert params.count[...] == 1

  def test_object_demo(self):
    class Params(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int):
        self.w = nx.Param(jnp.zeros((din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

    params = Params(3, 4)
    params = nx.mutable(params)

    @jax.jit
    def linear(params: Params, x: jax.Array):
      params.count[...] += 1
      return x @ params.w[...] + params.b[None]

    x = jnp.ones((1, 3))
    y = linear(params, x)

    assert y.shape == (1, 4)
    assert params.count[...] == 1

    y = linear(params, x)

    assert params.count[...] == 2

  def test_training_loop(self):
    class Linear(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

      def __call__(self, x: jax.Array):
        self.count[...] += 1
        return x @ self.w[...] + self.b[None]

    class MLP(nx.Pytree):
      __nodes__ = ('linear1', 'linear2')

      def __init__(self, din: int, dmid: int, dout: int, *, rngs: nx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

      def __call__(self, x: jax.Array):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    model = MLP(1, 64, 1, rngs=nx.Rngs(0))
    model = nx.mutable(model)

    @jax.jit
    def train_step(model, x, y):
      treedef, params, rest = nx.split(model, nx.Param, ...)

      def loss_fn(params):
        model = nx.merge(treedef, params, rest)
        return jnp.mean((model(x) - y) ** 2)

      loss, grads = jax.value_and_grad(loss_fn)(nx.freeze(params))

      def sgd(p: nx.Variable, g: jax.Array):
        p[...] -= 0.1 * g

      jax.tree.map(sgd, params, grads)
      return loss

    loss = 100
    for _ in range(10):
      x = np.linspace(0, 1, 32)[:, None]
      y = 0.8 * x**2 + 0.1 + np.random.normal(0, 0.1, size=x.shape)

      loss = train_step(model, x, y)

    assert loss < 1.0

  def test_optax_training_loop(self):
    class Linear(nx.Pytree):
      __nodes__ = ('w', 'b', 'count')

      def __init__(self, din: int, dout: int, *, rngs: nx.Rngs):
        self.w = nx.Param(jax.random.uniform(rngs(), (din, dout), jnp.float32))
        self.b = nx.Param(jnp.zeros((dout,), jnp.float32))
        self.count = nx.Variable(0)

      def __call__(self, x: jax.Array):
        self.count[...] += 1
        return x @ self.w[...] + self.b[None]

    class MLP(nx.Pytree):
      __nodes__ = ('linear1', 'linear2')

      def __init__(self, din: int, dmid: int, dout: int, *, rngs: nx.Rngs):
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

      def __call__(self, x: jax.Array):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    model = MLP(1, 64, 1, rngs=nx.Rngs(0))
    optimizer = nx.OptaxOptimizer(nx.state(model, nx.Param), optax.adamw(0.1))
    model, optimizer = nx.mutable((model, optimizer))

    @jax.jit
    def train_step(model: MLP, optimizer: nx.OptaxOptimizer, x, y):
      treedef, params, rest = nx.split(model, nx.Param, ...)

      def loss_fn(params):
        model = nx.merge(treedef, params, rest)
        return jnp.mean((model(x) - y) ** 2)

      loss, grads = jax.value_and_grad(loss_fn)(nx.freeze(params))
      optimizer.update(params, grads)

      return loss

    loss = 100
    for i in range(10):
      x = np.linspace(0, 1, 32)[:, None]
      y = 0.8 * x**2 + 0.1 + np.random.normal(0, 0.1, size=x.shape)

      loss = train_step(model, optimizer, x, y)

    assert loss < 1.0


if __name__ == '__main__':
  absltest.main()

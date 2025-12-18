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

from copy import deepcopy
import dataclasses
import pickle
import tempfile
from typing import TypeVar

from absl.testing import absltest
import cloudpickle
from flax import errors, nnx
import jax
import jax.numpy as jnp
import numpy as np

A = TypeVar('A')

class PytreeTest(absltest.TestCase):
  def test_pytree(self):
    class Foo(nnx.Pytree):
      def __init__(self, a, b):
        self.a = nnx.data(a)
        self.b = nnx.static(b)

    foo = Foo(a=1, b=2)

    self.assertEqual(jax.tree.leaves(foo), [1])

  def test_consistent_attrs(self):
    class Foo(nnx.Pytree):
      def __init__(self, a, b, c):
        self.a = nnx.data(a)
        self.b = nnx.static(b)
        self.c = c

    foo = Foo(a=1, b=2, c=jnp.array(3))
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.a = 3
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.a = nnx.static(3)
    self.assertLen(jax.tree.leaves(foo), 1)

    foo.b = 4  # ok
    self.assertLen(jax.tree.leaves(foo), 1)

    foo.b = nnx.data(4)
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.c = jnp.array(5)  # ok
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.c = nnx.static(5)
    self.assertLen(jax.tree.leaves(foo), 1)

    with self.assertRaisesRegex(
        ValueError,
        'Found Arrays on value of type',
    ):
      foo.a = ['hi', jnp.array(6)]

    with self.assertRaisesRegex(
        ValueError,
        'Found Arrays in value of type',
    ):
      foo.b = nnx.static(jnp.array(4))

  def test_consistent_attrs_frozen_dataclass(self):
    @nnx.dataclass
    class Foo(nnx.Pytree):
      a: int = nnx.data()
      b: int = nnx.static()
      c: jax.Array

    foo = Foo(a=1, b=2, c=jnp.array(3))
    self.assertLen(jax.tree.leaves(foo), 2)

  def test_consistent_attrs_dataclass_annotations(self):
    @dataclasses.dataclass
    class Foo(nnx.Pytree):
      a: nnx.Data[int]
      b: nnx.Static[int]
      c: jax.Array

    foo = Foo(a=1, b=2, c=jnp.array(3))
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.a = 3
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.a = nnx.static(3)
    self.assertLen(jax.tree.leaves(foo), 1)

    foo.b = 4  # ok
    self.assertLen(jax.tree.leaves(foo), 1)

    foo.b = nnx.data(4)
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.c = jnp.array(5)  # ok
    self.assertLen(jax.tree.leaves(foo), 2)

    foo.c = nnx.static(5)
    self.assertLen(jax.tree.leaves(foo), 1)

    with self.assertRaisesRegex(
        ValueError,
        'Found Arrays on value of type',
    ):
      foo.a = ['hi', jnp.array(6)]

    with self.assertRaisesRegex(
        ValueError,
        'Found Arrays in value of type',
    ):
      foo.b = nnx.static(jnp.array(4))

  def test_explicit_dont_change(self):
    class Foo(nnx.Pytree):
      def __init__(self):
        self.b = nnx.data(2)

    foo = Foo()
    self.assertEqual(jax.tree.leaves(foo), [2])
    foo.b = "hello"
    self.assertEqual(jax.tree.leaves(foo), ["hello"])

  def test_no_data_in_static(self):
    class Foo(nnx.Pytree):
      def __init__(self):
        self.a = nnx.static(jnp.array(1))

    with self.assertRaisesRegex(
        ValueError,
        'Found Arrays in value of type',
    ):
      foo = Foo()

class TestModule(absltest.TestCase):
  def test_has_module_state(self):
    class Foo(nnx.Module): ...

    foo = Foo()

    assert hasattr(foo, '_pytree__state')

  def test_trace_level(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def f():
      with self.assertRaisesRegex(
        errors.TraceContextError,
        "Cannot mutate 'Dict' from different trace level",
      ):
        m.a = 2

    f()

  def test_tree_map(self):
    m = nnx.Dict(a=nnx.Param(1))

    graphdef, state = nnx.split(m)

    state = jax.tree.map(lambda x: x + 1, state)

  def test_split_2(self):
    m = nnx.Dict(a=nnx.Param(1))

    graphdef, empty, some = nnx.split(m, None, ...)

    some = jax.tree.map(lambda x: x + 1, some)

  def test_split_merge(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(graphdef: nnx.GraphDef[nnx.Dict], state: nnx.State):
      m = nnx.merge(graphdef, state)
      m.a = 2
      return nnx.split(m)

    graphdef, state = g(*nnx.split(m))
    m2 = nnx.merge(graphdef, state)

    assert m2.a == 2

  def test_call(self):
    class Foo(nnx.Module):
      def __init__(self, c: float, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, ()))
        self.c = c

      def __call__(self, x, *, rngs: nnx.Rngs):
        return self.w * x + rngs.e.normal(()) + self.c

    foo = Foo(c=1.0, rngs=nnx.Rngs(0))

    y = foo(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_shared_module(self):
    m1 = nnx.Dict(a=nnx.Param(1), b=nnx.Param(2))
    m2 = nnx.Dict(x=m1, y=m1, z=nnx.Param(3))

    m3 = nnx.merge(*nnx.split(m2))

    assert m3['x'] is m3['y']
    assert m3['x']['a'] is m3['y']['a']
    assert m3['x']['b'] is m3['y']['b']

  def test_module_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.sub = self

    m = Foo()

    graphdef, state = nnx.split(m)
    assert len(state) == 1

    m2 = nnx.merge(graphdef, state)
    assert m2 is m2.sub

  def test_deref_through_jit(self):
    r1 = nnx.Variable(1)
    r2 = nnx.Variable(2)

    m = m0 = nnx.Dict({'a': nnx.List([r1, r2]), 'b': r1})

    @jax.jit
    def f(graphdef: nnx.GraphDef[nnx.Dict], state: nnx.State):
      m = nnx.merge(graphdef, state)

      assert m['a'][0] is m['b']
      assert m['a'][1] is not m['b']

      return nnx.split(m)

    graphdef, state = f(*nnx.split(m))
    m = nnx.merge(graphdef, state)

    assert m['a'][0] is m['b']
    assert m['a'][1] is not m['b']

    # compare with original
    assert m['a'][0] is not m0['a'][0]
    assert m['a'][1] is not m0['a'][1]
    assert m['b'] is not m0['b']

  def test_cross_barrier(self):
    m = nnx.Dict(a=nnx.Param(jnp.array(1)))

    @jax.jit
    def g(graphdef: nnx.GraphDef[nnx.Dict], state: nnx.State):
      m = nnx.merge(graphdef, state)
      m.a[...] += 1
      return nnx.split(m)

    graphdef, state = g(*nnx.split(m))
    m2 = nnx.merge(graphdef, state)
    assert m2 is not m
    assert m.a[...] == 1
    assert m2.a[...] == 2

  def test_no_rejit(self):
    n = 0
    m = nnx.Dict(a=nnx.Param(jnp.array(1)))

    @jax.jit
    def g(state_and_def):
      nonlocal n
      n += 1
      m = nnx.merge(*state_and_def)
      m.a[...] += 1
      return nnx.split(m)

    m2 = nnx.merge(*g(nnx.split(m)))

    assert n == 1
    assert m2 is not m
    assert m.a[...] == 1
    assert m2.a[...] == 2

    g(nnx.split(m))
    assert n == 1

    g(nnx.split(m2))
    assert n == 1

    m2.b = nnx.Param(10)
    g(nnx.split(m2))

    assert n == 2

  def test_deref_number_of_fields(self):
    r1 = nnx.Variable(1)
    r2 = nnx.Variable(2)
    v1 = 3
    m = nnx.Dict(
      {
        'a': nnx.List([r1, r2, v1]),
        'b': nnx.Dict({'c': r1, 'd': r2}),
      }
    )

    graphdef, p = nnx.split(m)
    assert len(nnx.to_flat_state(p)) == 2
    assert len(jax.tree_util.tree_leaves(p)) == 2

  def test_clone(self):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(1), nnx.Param(2), 3]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.Param(2)),
    )

    m2 = nnx.clone(m)

    assert m is not m2
    assert m2.a[0].get_value() == m2.b.c.get_value()
    assert m2.a[1].get_value() == m2.b.d.get_value()

    assert m.a[0].get_value() == m2.a[0].get_value()
    assert m.a[1].get_value() == m2.a[1].get_value()
    assert m.b.c.get_value() == m2.b.c.get_value()
    assert m.b.d.get_value() == m2.b.d.get_value()

  def test_sow_basic(self):
    class Foo(nnx.Module):
      def __call__(self, x):
        y = x + 1
        self.sow(nnx.Intermediate, 'y', y)
        return y

    m = Foo()
    y1 = m(2)
    y2 = m(10)

    assert y1 == 3
    assert y2 == 11
    assert m.y.get_value() == (3, 11)

    intermediates = nnx.pop(m, nnx.Intermediate)

    assert isinstance(intermediates['y'], nnx.Intermediate)
    assert intermediates['y'].get_value() == (3, 11)

    assert not hasattr(m, 'y')

  def test_sow_existing_non_variable_field(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.y = 10

      def __call__(self, x):
        y = x + 1
        self.sow(nnx.Intermediate, 'y', y)
        return y

    m = Foo()

    with self.assertRaisesRegex(ValueError, 'to be a Variable, got'):
      m(2)

  def test_sow_wrong_collection(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.y = nnx.Param(10)

      def __call__(self, x):
        y = x + 1
        self.sow(nnx.Intermediate, 'y', y)
        return y

    m = Foo()

    with self.assertRaisesRegex(ValueError, 'to be of type'):
      m(2)

  def test_perturb_basic(self):
    class Foo(nnx.Module):
      def __init__(self, rngs):
        self.linear = nnx.Linear(10, 10, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.perturb('before_multiply', x)
        x = 4 * x
        x = self.perturb('after_multiply', x)
        return x

    model = Foo(rngs=nnx.Rngs(0))
    # Perturbations are not created in init time. It needs some sample input.
    self.assertFalse(hasattr(model, 'before_multiply'))
    self.assertFalse(hasattr(model, 'after_multiply'))

    x = jax.random.uniform(jax.random.key(1), shape=(10,))
    y = jax.random.uniform(jax.random.key(2), shape=(10,))
    model(x)
    np.testing.assert_array_equal(model.before_multiply, jnp.zeros_like(x))
    np.testing.assert_array_equal(model.after_multiply, jnp.zeros_like(x))

    take_gradient_filter = nnx.Any(nnx.Param, nnx.Perturbation)
    @nnx.grad(argnums=nnx.DiffState(argnum=0, filter=take_gradient_filter))
    def grad_loss(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean()
    intm_grads = grad_loss(model, x, y)

    # Gradient should not be zero
    self.assertFalse(
      jnp.array_equal(intm_grads.before_multiply[...], jnp.zeros_like(x))
    )
    # activation * 4 so reverse gradient also * 4
    np.testing.assert_allclose(
      intm_grads.after_multiply[...] * 4, intm_grads.before_multiply[...]
    )

  def test_update_static_state_submodules(self):
    class Bar(nnx.Module):
      def __init__(self) -> None:
        self.x = 1

      def add_field(self):
        self.y = 2

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = Bar()
        self.b = self.a

    m1 = Foo()
    with nnx.update_context('test'):
      with nnx.split_context('test') as ctx:
        graphdef, state = ctx.split(m1)
      with nnx.merge_context('test', inner=True) as ctx:
        m2 = ctx.merge(graphdef, state)
      m2.a.add_field()
      with nnx.split_context('test') as ctx:
        new_graphdef, state = ctx.split(m2)

      with nnx.merge_context('test', inner=False) as ctx:
        m3 = ctx.merge(new_graphdef, state)

    assert m3 is m1
    assert m1.a.x == 1
    assert m1.a.y == 2
    assert m1.b.x == 1
    assert m1.b.y == 2

  def test_update_new_submodule(self):
    class Bar(nnx.Module):
      def __init__(self) -> None:
        self.x = 1

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = Bar()

      def add_module(self):
        self.b = Bar()

    m1 = Foo()
    with nnx.update_context('test'):
      with nnx.split_context('test') as ctx:
        graphdef, state = ctx.split(m1)
      with nnx.merge_context('test', inner=True) as ctx:
        m2 = ctx.merge(graphdef, state)
      m2.add_module()
      with nnx.split_context('test') as ctx:
        new_graphdef, state = ctx.split(m2)

      with nnx.merge_context('test', inner=False) as ctx:
        m3 = ctx.merge(new_graphdef, state)

    assert m3 is m1
    assert m1.a.x == 1
    assert m1.b.x == 1

  def test_update_update_submodule(self):
    class Bar(nnx.Module):
      def __init__(self) -> None:
        self.x = 1

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = Bar()
        self.b = self.a

    m1 = Foo()
    with nnx.update_context('test'):
      with nnx.split_context('test') as ctx:
        graphdef, state = ctx.split(m1)
      with nnx.merge_context('test', inner=True) as ctx:
        m2 = ctx.merge(graphdef, state)
      m2.a.x = 2
      with nnx.split_context('test') as ctx:
        new_graphdef, state = ctx.split(m2)
      with nnx.merge_context('test', inner=False) as ctx:
        m3 = ctx.merge(new_graphdef, state)

    assert m3 is m1
    assert m1.a.x == 2
    assert m1.b.x == 2

  def test_update_add_shared(self):
    class Bar(nnx.Module):
      def __init__(self) -> None:
        self.x = 1

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = Bar()
        self.b = self.a

      def add_submodule(self):
        self.c = self.a

    m1 = Foo()
    with nnx.update_context('test'):
      with nnx.split_context('test') as ctx:
        graphdef, state = ctx.split(m1)
      with nnx.merge_context('test', inner=True) as ctx:
        m2 = ctx.merge(graphdef, state)
      m2.add_submodule()
      with nnx.split_context('test') as ctx:
        new_graphdef, state = ctx.split(m2)
      with nnx.merge_context('test', inner=False) as ctx:
        m3 = ctx.merge(new_graphdef, state)

    assert m3 is m1
    assert hasattr(m1, 'c')

  def test_create_abstract(self):
    linear = nnx.eval_shape(lambda: nnx.Linear(2, 3, rngs=nnx.Rngs(0)))

    assert linear.kernel.get_value() == jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert linear.bias.get_value() == jax.ShapeDtypeStruct((3,), jnp.float32)

  def test_create_abstract_stateful(self):
    linear = nnx.eval_shape(lambda: nnx.Dropout(0.5, rngs=nnx.Rngs(0)))

    assert linear.rngs.key.get_value() == jax.ShapeDtypeStruct(
      (), jax.random.key(0).dtype
    )

  def test_partial_init(self):
    linear = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    state = nnx.state(linear)

    del state['bias']

    @nnx.jit
    def partial_init(state: nnx.State):
      m = nnx.Linear(
        2, 3, bias_init=nnx.initializers.ones_init(), rngs=nnx.Rngs(1)
      )
      nnx.update(m, state)
      return m

    linear2 = partial_init(state)

    np.testing.assert_allclose(linear.kernel[...], linear2.kernel[...])
    np.testing.assert_allclose(linear.bias[...], 0)
    np.testing.assert_allclose(linear2.bias[...], 1)

  def test_deepcopy(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = nnx.Param(jnp.array(1))
        self.b = [1, 2, 3]
        self.c = nnx.Param(jnp.array([1.0]))
        self.self = self

    m1 = Foo()
    m2 = deepcopy(m1)

    assert m1.a[...] == m2.a[...]
    assert vars(m1)['a'] is not vars(m2)['a']
    assert m1.b is not m2.b
    assert m1.c is not m2.c
    assert m1.self is m1

  def test_set_attributes(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False)
        self.batch_norm = nnx.BatchNorm(
          10, use_running_average=False, rngs=rngs
        )

    block = Block(2, 5, rngs=nnx.Rngs(0))
    assert block.dropout.deterministic == False
    assert block.batch_norm.use_running_average == False

    block.set_attributes(deterministic=True, use_running_average=True)
    assert block.dropout.deterministic == True
    assert block.batch_norm.use_running_average == True

    block = Block(2, 5, rngs=nnx.Rngs(0))
    block.set_attributes(nnx.Dropout, deterministic=True)
    # Only the dropout will be modified
    assert block.dropout.deterministic == True
    assert block.batch_norm.use_running_average == False

  def test_set_attribute_error(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False)
        self.batch_norm = nnx.BatchNorm(
          10, use_running_average=False, rngs=rngs
        )

    block = Block(2, 5, rngs=nnx.Rngs(0))

    with self.assertRaisesRegex(
        ValueError,
        (
            'Could not find at least one instance of the following attributes:'
            " {'unknown'}"
        ),
    ):
      block.set_attributes(
        deterministic=True, use_running_average=True, unknown=True
      )

    block.set_attributes(
      deterministic=True,
      use_running_average=True,
      unknown=True,
      raise_if_not_found=False,
    )

  def test_set_mode(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(
          10, use_running_average=False, rngs=rngs
        )

    block = Block(2, 5, rngs=nnx.Rngs(0))
    assert block.dropout.deterministic == False
    assert block.batch_norm.use_running_average == False

    new_block = nnx.set_mode(block, deterministic=True, use_running_average=True)
    assert new_block.dropout.deterministic == True
    assert new_block.batch_norm.use_running_average == True
    assert new_block.linear.kernel is block.linear.kernel

    block = Block(2, 5, rngs=nnx.Rngs(0))
    new_block = nnx.set_mode(block, only=nnx.Dropout, deterministic=True)
    # Only the dropout will be modified
    assert new_block.dropout.deterministic == True
    assert new_block.batch_norm.use_running_average == False

  def test_set_mode_error(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(
          10, use_running_average=False, rngs=rngs
        )
    block = Block(2, 5, rngs=nnx.Rngs(0))

    with self.assertRaisesRegex(
        ValueError,
        (
            "Unused keys found in set_mode: \\['unknown'\\]"
        ),
    ):
      nnx.set_mode(block, deterministic=True, use_running_average=True, unknown=True)

  def test_cloud_pickle(self):
    import platform
    if platform.python_version().startswith('3.11'):
      self.skipTest("Cloudpickle cannot pickle PRNGKeyArray on python 3.11")
    class Model(nnx.Module):
      def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

      def __call__(self, x):
        x = nnx.relu(self.dropout(self.bn(self.linear(x))))
        return self.linear_out(x)

    model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
    model.eval()

    y1 = model(jnp.ones((5, 2)))
    with tempfile.TemporaryDirectory() as tmpdir:
      path = f'{tmpdir}/model.pkl'
      with open(path, 'wb') as f:
        cloudpickle.dump(model, f)
        del model
      with open(path, 'rb') as f:
        model = pickle.load(f)

    self.assertIsInstance(model, Model)
    y2 = model(jnp.ones((5, 2)))
    np.testing.assert_allclose(y1, y2)

  def test_repr(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    class Foo(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.block1 = Block(32, 128, rngs=rngs)
        self.block2 = Block(128, 10, rngs=rngs)

      def __call__(self, x):
        return self.block2(self.block1(x))

    obj = Foo(nnx.Rngs(0))

    leaves = nnx.to_flat_state(nnx.state(obj)).leaves

    expected_total = sum(int(np.prod(x.shape)) for x in leaves)
    expected_total_params = sum(
      int(np.prod(x.shape)) for x in leaves if isinstance(x, nnx.Param)
    )
    expected_total_batch_stats = sum(
      int(np.prod(x.shape)) for x in leaves if isinstance(x, nnx.BatchStat)
    )
    expected_total_rng_states = sum(
      int(np.prod(x.shape)) for x in leaves if isinstance(x, nnx.RngState)
    )

    foo_repr = repr(obj).replace(',', '').splitlines()

    self.assertIn(str(expected_total), foo_repr[0])
    self.assertIn(str(expected_total_params), foo_repr[0])
    self.assertIn(str(expected_total_batch_stats), foo_repr[0])
    self.assertIn(str(expected_total_rng_states), foo_repr[0])

  def test_set_mode_info(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    class Foo(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.block1 = Block(32, 128, rngs=rngs)
        self.block2 = Block(128, 10, rngs=rngs)

      def __call__(self, x):
        return self.block2(self.block1(x))

    obj = Foo(rngs=nnx.Rngs(0))
    info_str = nnx.set_mode_info(obj)
    self.assertEqual(info_str.count("BatchNorm:"), 1)
    self.assertEqual(info_str.count("Dropout:"), 1)

  def test_set_mode_info_with_filter(self):
    class Block(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    obj = Block(4, 8, rngs=nnx.Rngs(0))
    info_str = nnx.set_mode_info(obj, only=nnx.Dropout)
    self.assertIn("Dropout:", info_str)
    self.assertNotIn("BatchNorm:", info_str)

    info_str = nnx.set_mode_info(obj, only=nnx.MultiHeadAttention)
    self.assertEmpty(info_str)

  def test_set_mode_info_with_custom_set_mode(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        pass

      def __call__(self, x):
        return x

      def set_mode(self, arg1: bool | None = None, arg2: int | None = None, **kwargs) -> dict:
        """Example set_mode docstring. This follows Google style docstrings.

        Args:
          arg1: The first argument.
          arg2: The second argument.
            This has two lines.
        """
        return kwargs

    obj = Block(rngs=nnx.Rngs(0))
    info_str = nnx.set_mode_info(obj)
    self.assertEqual(f"{obj.__class__.__qualname__}:\n  arg1: bool | None = None\n    The first argument.\n  arg2: int | None = None\n    The second argument.\n    This has two lines.", info_str)


class TestModuleDataclass(absltest.TestCase):
  def test_basic(self):

    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: int
      b: nnx.Variable[int]
      c: nnx.Param[int]
      d: nnx.Variable[int]
      e: nnx.Variable[int]
      f: int

    m = Foo(
      a=1,  # graphdef
      b=nnx.Variable(2),  # node
      c=nnx.Param(3),  # param
      d=nnx.Variable(4),  # var
      e=nnx.BatchStat(5),  # var
      f=6,  # graphdef int
    )

    graphdef, state = nnx.split(m)

    assert len(state) == 4
    assert state['b'].get_value() == 2
    assert isinstance(state['b'], nnx.Variable)
    assert state['c'].get_value() == 3
    assert isinstance(state['c'], nnx.Param)
    assert state['d'].get_value() == 4
    assert isinstance(state['d'], nnx.Variable)
    assert state['e'].get_value() == 5
    assert isinstance(state['e'], nnx.BatchStat)

  def test_field_specifiers(self):
    @nnx.dataclass
    class Foo(nnx.Pytree):
      a: int = nnx.static()
      b: jax.Array = nnx.data()

    m = Foo(a=1, b=jnp.array(2))

    leaves = jax.tree.leaves(m)
    assert len(leaves) == 1
    assert leaves[0] == jnp.array(2)

  def test_field_specifiers_forced(self):
    @nnx.dataclass
    class Bar(nnx.Pytree):
      a: int = nnx.data()

    m = Bar(a=1)

    leaves = jax.tree.leaves(m)
    assert len(leaves) == 1
    assert leaves[0] == 1

  def test_field_specifiers_with_defaults(self):
    @nnx.dataclass
    class Bar(nnx.Pytree):
      a: int = nnx.data(default=3)

    m = Bar()

    leaves = jax.tree.leaves(m)
    assert len(leaves) == 1
    assert leaves[0] == 3

  def test_field_specifiers_array_in_static(self):
    @nnx.dataclass
    class Bar(nnx.Pytree):
      a: jax.Array = nnx.static()

    with self.assertRaisesRegex(
      ValueError,
      'Found unexpected Arrays on value of type',
    ):
      m = Bar(a=jnp.array(3))

  def test_post_init(self):

    @dataclasses.dataclass
    class DFoo(nnx.Module):
      din: int
      dout: int
      rngs: nnx.Rngs

      def __post_init__(self):
        self.bar = nnx.Linear(self.din, self.dout, rngs=self.rngs)

      def __call__(self, x):
        return self.bar(x)

    m = DFoo(1, 1, rngs=nnx.Rngs(0))

    assert hasattr(m, 'bar')


class TestModuleDef:
  def test_apply(self):
    class Foo(nnx.Module):
      def __init__(self, c: float, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.uniform(rngs.params(), ()))
        self.c = c

      def __call__(self, x, *, rngs: nnx.Rngs):
        return self.w * x + rngs.e.normal(()) + self.c

    rngs = nnx.Rngs(0)
    foo = Foo(c=1.0, rngs=rngs)

    graphdef, states = nnx.split(foo)

    assert isinstance(states, nnx.State)
    assert isinstance(states['w'], nnx.Param)

    y, _updates = graphdef.apply(states)(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_derefed_mod_apply(self):
    class Foo(nnx.Module):
      def __init__(self, c: float, *, rngs: nnx.Rngs):
        self.w = nnx.Param(
          jax.random.uniform(rngs.params(), ()),
        )
        self.c = nnx.Variable(c)

      def __call__(self, x, *, rngs: nnx.Rngs):
        return self.w * x + rngs.e.normal(()) + self.c

    foo = Foo(c=1.0, rngs=nnx.Rngs(0))

    graphdef, state = nnx.split(foo)

    assert isinstance(graphdef.nodes[0], nnx.graph.NodeDef | nnx.graph.NodeRef)
    assert isinstance(state, nnx.State)
    assert isinstance(state['w'], nnx.Param)
    assert isinstance(state['c'], nnx.Variable)

    y, (graphdef, state) = graphdef.apply(state)(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_modules_iterator(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.submodules = nnx.data([
          {'a': nnx.Linear(1, 1, rngs=rngs)},
          {'b': nnx.Conv(1, 1, 1, rngs=rngs)},
        ])
        self.linear = nnx.Linear(1, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    module = Foo(rngs=nnx.Rngs(0))

    modules = list(nnx.iter_modules(module))

    assert len(modules) == 5
    assert modules[0][0] == ('dropout',)
    assert isinstance(modules[0][1], nnx.Dropout)
    assert modules[1][0] == ('linear',)
    assert isinstance(modules[1][1], nnx.Linear)
    assert modules[2][0] == ('submodules', 0, 'a')
    assert isinstance(modules[2][1], nnx.Linear)
    assert modules[3][0] == ('submodules', 1, 'b')
    assert isinstance(modules[3][1], nnx.Conv)
    assert modules[4][0] == ()
    assert isinstance(modules[4][1], Foo)

  def test_children_modules_iterator(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.submodules = nnx.data([
          {'a': nnx.Linear(1, 1, rngs=rngs)},
          {'b': nnx.Conv(1, 1, 1, rngs=rngs)},
        ])
        self.linear = nnx.Linear(1, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    module = Foo(rngs=nnx.Rngs(0))

    modules = list(nnx.iter_children(module))

    assert len(modules) == 2
    assert modules[0][0] == 'dropout'
    assert isinstance(modules[0][1], nnx.Dropout)
    assert modules[1][0] == 'linear'
    assert isinstance(modules[1][1], nnx.Linear)

  def test_state_in_module(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.data(nnx.State({'b': nnx.Param(jnp.array(1.0))}))

    foo = Foo()

    graphdef, state = nnx.split(foo)

    assert isinstance(state, nnx.State)
    assert isinstance(state['a'], nnx.State)

    foo2 = nnx.merge(graphdef, state)

    assert isinstance(foo2.a, nnx.State)

if __name__ == '__main__':
  absltest.main()

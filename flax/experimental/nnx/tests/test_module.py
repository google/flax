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
from copy import deepcopy
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax.experimental import nnx

A = TypeVar('A')


class TestModule:
  def test_has_module_state(self):
    class Foo(nnx.Module): ...

    foo = Foo()

    assert hasattr(foo, '_module__state')

  def test_trace_level(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def f():
      with pytest.raises(
        nnx.TraceContextError,
        match='Cannot mutate Module from different trace level',
      ):
        m.a = 2

    f()

  def test_tree_map(self):
    m = nnx.Dict(a=nnx.Param(1))

    state, static = m.split()

    state = jax.tree_map(lambda x: x + 1, state)

  def test_split_2(self):
    m = nnx.Dict(a=nnx.Param(1))

    empty, some, static = m.split(None, ...)

    some = jax.tree_map(lambda x: x + 1, some)

  def test_split_merge(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(state: nnx.State, graphdef: nnx.GraphDef[nnx.Dict[int]]):
      m = graphdef.merge(state)
      m.a = 2
      return m.split()

    state, graphdef = g(*m.split())
    m2 = graphdef.merge(state)

    assert m2.a == 2

  def test_no_trace_level_error_on_grad(self):
    # No trace level error occurs because jax doesn't update
    # its top trace for grad.
    m = nnx.Dict(a=nnx.Param(1.0))

    @jax.grad
    def f(_):
      m.a = 2.0
      return 1.0

    f(1.0)

  def test_call(self):
    class Foo(nnx.Module):
      def __init__(self, c: float, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, ()))
        self.c = c

      def __call__(self, x, *, rngs: nnx.Rngs):
        key = rngs.e()
        return self.w.value * x + jax.random.normal(key, ()) + self.c

    foo = Foo(c=1.0, rngs=nnx.Rngs(0))

    y = foo(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_shared_module(self):
    m1 = nnx.Dict(a=nnx.Param(1), b=nnx.Param(2))
    m2 = nnx.Dict(x=m1, y=m1, z=nnx.Param(3))

    m3 = nnx.merge(m2.split())

    assert m3['x'] is m3['y']
    assert m3['x']['a'] is m3['y']['a']
    assert m3['x']['b'] is m3['y']['b']

  def test_module_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.sub = self

    m = Foo()

    state, graphdef = m.split()
    assert len(state) == 1

    m2 = graphdef.merge(state)
    assert m2 is m2.sub

  def test_deref_through_jit(self):
    r1 = nnx.Variable(1)
    r2 = nnx.Variable(2)

    m = m0 = nnx.Dict({'a': nnx.Sequence([r1, r2]), 'b': r1})

    @jax.jit
    def f(state: nnx.State, graphdef: nnx.GraphDef[nnx.Dict[Any]]):
      m = graphdef.merge(state)

      assert m['a'][0] is m['b']
      assert m['a'][1] is not m['b']

      return m.split()

    state, graphdef = f(*m.split())
    m = graphdef.merge(state)

    assert m['a'][0] is m['b']
    assert m['a'][1] is not m['b']

    # compare with original
    assert m['a'][0] is not m0['a'][0]
    assert m['a'][1] is not m0['a'][1]
    assert m['b'] is not m0['b']

  def test_cross_barrier(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(state: nnx.State, graphdef: nnx.GraphDef[nnx.Dict[nnx.Param[int]]]):
      m = graphdef.merge(state)
      m.a.value += 1
      return m.split()

    state, graphdef = g(*m.split())
    m2 = graphdef.merge(state)
    assert m2 is not m
    assert m.a.value == 1
    assert m2.a.value == 2

  def test_no_rejit(self):
    n = 0
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(state_and_def):
      nonlocal n
      n += 1
      m = nnx.merge(state_and_def)
      m.a.value += 1
      return m.split()

    m2 = nnx.merge(g(m.split()))

    assert n == 1
    assert m2 is not m
    assert m.a.value == 1
    assert m2.a.value == 2

    g(m.split())
    assert n == 1

    g(m2.split())
    assert n == 1

    m2.b = nnx.Param(10)
    g(m2.split())

    assert n == 2

  def test_deref_number_of_fields(self):
    r1 = nnx.Variable(1)
    r2 = nnx.Variable(2)
    v1 = 3
    m = nnx.Dict(
      {
        'a': nnx.Sequence([r1, r2, v1]),
        'b': nnx.Dict({'c': r1, 'd': r2}),
      }
    )

    p, graphdef = m.split()
    assert len(p.flat_state()) == 2
    assert len(jax.tree_util.tree_leaves(p)) == 2

  def test_deref_array_attributes_not_allowed(self):
    # test arrays are nodes
    r1 = nnx.Variable(1)
    r2 = nnx.Variable(2)
    v1 = jax.numpy.array(3)

    with pytest.raises(
      ValueError,
      match=f"Trying to assign a '{type(v1).__name__}' to the Module",
    ):
      m = nnx.Dict(
        {
          'a': nnx.Sequence([r1, r2, v1]),
          'b': nnx.Dict({'c': r1, 'd': r2}),
        }
      )

  def test_clone(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(1), nnx.Param(2), 3]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.Param(2)),
    )

    m2 = m.clone()

    assert m is not m2
    assert m2.a[0] == m2.b.c
    assert m2.a[1] == m2.b.d

    assert m.a[0] == m2.a[0]
    assert m.a[1] == m2.a[1]
    assert m.b.c == m2.b.c
    assert m.b.d == m2.b.d

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
    assert m.y.value == (3, 11)

    intermediates = m.pop(nnx.Intermediate)

    assert isinstance(intermediates.y, nnx.Intermediate)
    assert intermediates['y'].raw_value == (3, 11)

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

    with pytest.raises(ValueError, match='to be a Variable, got'):
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

    with pytest.raises(ValueError, match='to be of type'):
      m(2)

  def test_update_static_state(self):
    class Foo(nnx.Module):
      def add_field(self):
        self.a = 1

    m1 = Foo()
    m2 = Foo()
    m2.add_field()

    m1.update(m2)

    assert m1.a == 1

  def test_update_moduledef(self):
    class Foo(nnx.Module):
      def add_field(self):
        self.a = 1

    m1 = Foo()
    m2 = Foo()
    m2.add_field()

    m1.update(m2.get_graphdef())

    assert m1.a == 1

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
    m2 = Foo()
    m2.a.add_field()

    m1.update(m2)

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
    m2 = Foo()
    m2.add_module()

    m1.update(m2)

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
    m2 = Foo()
    m2.a.x = 2

    m1.update(m2)

    assert m1.a.x == 2
    assert m1.b.x == 2

  def test_update_add_shared_error(self):
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
    m2 = Foo()
    m2.add_submodule()

    assert hasattr(m2, 'c')

    with pytest.raises(ValueError, match='Trying to add a new node at path'):
      m1.update(m2)

  def test_update_add_shared_error_new_first(self):
    class Bar(nnx.Module):
      def __init__(self) -> None:
        self.x = 1

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.b = Bar()
        self.c = self.b

      def add_submodule(self):
        self.a = self.b

    m1 = Foo()
    m2 = Foo()
    m2.add_submodule()

    assert hasattr(m2, 'a')

    m2 = m2.clone()  # clone to sort the fields

    with pytest.raises(ValueError, match='Trying to add a new node at path'):
      m1.update(m2)

  def test_create_abstract(self):
    linear = nnx.Linear.create_abstract(2, 3, rngs=nnx.Rngs(0))

    assert linear.kernel.value == jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert linear.bias.value == jax.ShapeDtypeStruct((3,), jnp.float32)

  def test_partial_init(self):
    linear = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    state = linear.get_state()

    del state['bias']

    linear2 = nnx.Linear.partial_init(state)(
      2, 3, bias_init=nnx.initializers.ones_init(), rngs=nnx.Rngs(1)
    )

    np.testing.assert_allclose(linear.kernel.value, linear2.kernel.value)
    np.testing.assert_allclose(linear.bias.value, 0)
    np.testing.assert_allclose(linear2.bias.value, 1)

  def test_deepcopy(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.a = nnx.Param(1)
        self.b = [1, 2, 3]
        self.c = nnx.Param(jnp.array([1.0]))
        self.self = self

    m1 = Foo()
    m2 = deepcopy(m1)

    assert m1.a == m2.a
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

    with pytest.raises(
      ValueError,
      match="Could not find at least one instance of the following attributes: {'unknown'}",
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


class TestModulePytree:
  def test_tree_map(self):
    class Foo(nnx.Module, experimental_pytree=True):
      def __init__(self):
        self.node = nnx.Param(1)
        self.static = 1

    m = Foo()

    m = jax.tree_map(lambda x: x + 1, m)

    assert m.node.value == 2
    assert m.static == 1


class TestModuleDataclass:
  def test_basic(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: int
      b: nnx.TreeNode[int]
      c: nnx.Param[int]
      d: nnx.Variable[int]
      e: nnx.Variable[int]
      f: int

    m = Foo(
      a=1,  # static
      b=nnx.TreeNode(2),  # node
      c=nnx.Param(3),  # param
      d=nnx.Variable(4),  # var
      e=nnx.BatchStat(5),  # var
      f=6,  # static int
    )

    state, graphdef = m.split()

    assert len(state) == 4
    assert state.b == nnx.TreeNode(2)
    assert state.c == nnx.Param(3)
    assert state.d == nnx.Variable(4)
    assert state.e == nnx.BatchStat(5)

  def test_context_none_after_init(self):
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
    assert m.rngs is None

  def test_setup_is_called(self):
    @dataclasses.dataclass
    class DFoo(nnx.Module):
      din: int
      dout: int
      rngs: nnx.Rngs

      def setup(self):
        self.bar = nnx.Linear(self.din, self.dout, rngs=self.rngs)

      def __call__(self, x):
        return self.bar(x)

    m = DFoo(1, 1, rngs=nnx.Rngs(0))

    assert hasattr(m, 'bar')
    assert m.rngs is None


class TestModuleDef:
  def test_apply(self):
    class Foo(nnx.Module):
      def __init__(self, c: float, *, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.uniform(rngs.params(), ()))
        self.c = c

      def __call__(self, x, *, rngs: nnx.Rngs):
        key = rngs.e()
        return self.w.value * x + jax.random.normal(key, ()) + self.c

    rngs = nnx.Rngs(0)
    foo = Foo(c=1.0, rngs=rngs)

    states, graphdef = foo.split()

    assert isinstance(states, nnx.State)
    assert isinstance(states.w, nnx.Param)
    # assert isinstance(states["c"], jax.Array)

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
        key = rngs.e()
        return self.w.value * x + jax.random.normal(key, ()) + self.c.value

    foo = Foo(c=1.0, rngs=nnx.Rngs(0))

    state, graphdef = foo.split()

    assert isinstance(graphdef, nnx.GraphDef)
    assert isinstance(state, nnx.State)
    assert isinstance(state.w, nnx.Param)
    assert isinstance(state.c, nnx.Variable)

    y, (state, graphdef) = graphdef.apply(state)(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_modules_iterator(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.submodules = [
          {'a': nnx.Linear(1, 1, rngs=rngs)},
          {'b': nnx.Conv(1, 1, 1, rngs=rngs)},
        ]

    module = Foo(rngs=nnx.Rngs(0))

    modules = list(module.modules())

    assert len(modules) == 3
    assert modules[0][0] == ''
    assert isinstance(modules[0][1], Foo)
    assert modules[1][0] == 'submodules/0/a'
    assert isinstance(modules[1][1], nnx.Linear)
    assert modules[2][0] == 'submodules/1/b'
    assert isinstance(modules[2][1], nnx.Conv)

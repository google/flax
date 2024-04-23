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
    class Foo(nnx.Module):
      ...

    foo = Foo()

    assert hasattr(foo, '_graph_node__state')

  def test_trace_level(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def f():
      with pytest.raises(
        nnx.errors.TraceContextError,
        match="Cannot mutate 'Dict' from different trace level",
      ):
        m.a = 2

    f()

  def test_tree_map(self):
    m = nnx.Dict(a=nnx.Param(1))

    graphdef, state = nnx.split(m)

    state = jax.tree_util.tree_map(lambda x: x + 1, state)

  def test_split_2(self):
    m = nnx.Dict(a=nnx.Param(1))

    graphdef, empty, some = nnx.split(m, None, ...)

    some = jax.tree_util.tree_map(lambda x: x + 1, some)

  def test_split_merge(self):
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(graphdef: nnx.GraphDef[nnx.Dict[int]], state: nnx.State):
      m = nnx.merge(graphdef, state)
      m.a = 2
      return nnx.split(m)

    graphdef, state = g(*nnx.split(m))
    m2 = nnx.merge(graphdef, state)

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
    def f(graphdef: nnx.GraphDef[nnx.Dict[Any]], state: nnx.State):
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
    m = nnx.Dict(a=nnx.Param(1))

    @jax.jit
    def g(graphdef: nnx.GraphDef[nnx.Dict[nnx.Param[int]]], state: nnx.State):
      m = nnx.merge(graphdef, state)
      m.a.value += 1
      return nnx.split(m)

    graphdef, state = g(*nnx.split(m))
    m2 = nnx.merge(graphdef, state)
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
      m = nnx.merge(*state_and_def)
      m.a.value += 1
      return nnx.split(m)

    m2 = nnx.merge(*g(nnx.split(m)))

    assert n == 1
    assert m2 is not m
    assert m.a.value == 1
    assert m2.a.value == 2

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
    assert len(p.flat_state()) == 2
    assert len(jax.tree_util.tree_leaves(p)) == 2

  def test_clone(self):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(1), nnx.Param(2), 3]),
      b=nnx.Dict(c=nnx.Param(1), d=nnx.Param(2)),
    )

    m2 = nnx.clone(m)

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

    intermediates = nnx.pop(m, nnx.Intermediate)

    assert issubclass(intermediates.y.type, nnx.Intermediate)
    assert intermediates['y'].value == (3, 11)

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
    with nnx.UpdateContext() as ctx:
      graphdef, state = ctx.split(m1)
      m2 = ctx.merge(graphdef, state)
      m2.a.add_field()
      new_graphdef, state = ctx.split(m2)

      ctx.update(new_graphdef, state)

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
    ctx = nnx.UpdateContext()
    graphdef, state = ctx.split(m1)
    m2 = ctx.merge(graphdef, state)
    m2.add_module()
    new_graphdef, state = ctx.split(m2)

    ctx.update(new_graphdef, state)

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
    ctx = nnx.UpdateContext()
    graphdef, state = ctx.split(m1)
    m2 = ctx.merge(graphdef, state)
    m2.a.x = 2
    new_graphdef, state = ctx.split(m2)
    ctx.update(new_graphdef, state)

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
    ctx = nnx.UpdateContext()
    graphdef, state = ctx.split(m1)
    m2 = ctx.merge(graphdef, state)
    m2.add_submodule()
    new_graphdef, state = ctx.split(m2)
    ctx.update(new_graphdef, state)

    assert hasattr(m1, 'c')

  def test_create_abstract(self):
    linear = nnx.eval_shape(lambda: nnx.Linear(2, 3, rngs=nnx.Rngs(0)))

    assert linear.kernel.value == jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert linear.bias.value == jax.ShapeDtypeStruct((3,), jnp.float32)

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

  def test_init(self):
    class Linear(nnx.Module):
      def __init__(self, dout, rngs: nnx.Rngs):
        self.dout = dout
        self.rngs = rngs

      def __call__(self, x):
        if self.is_initializing():
          din = x.shape[-1]
          if not hasattr(self, 'w'):
            key = self.rngs.params()
            self.w = nnx.Param(jax.random.uniform(key, (din, self.dout)))
          if not hasattr(self, 'b'):
            self.b = nnx.Param(jnp.zeros((self.dout,)))
        return x @ self.w + self.b[None]

    linear = Linear(3, nnx.Rngs(0))
    x = jnp.ones((5, 2))
    y = linear.init(x)
    assert linear.w.value.shape == (2, 3)
    assert linear.b.value.shape == (3,)
    assert y.shape == (5, 3)
    assert not linear.is_initializing()


class TestModulePytree:
  def test_tree_map(self):
    class Foo(nnx.Module, experimental_pytree=True):
      def __init__(self):
        self.node = nnx.Param(1)
        self.graphdef = 1

    m = Foo()

    m = jax.tree_util.tree_map(lambda x: x + 1, m)

    assert m.node.value == 2
    assert m.graphdef == 1


class TestModuleDataclass:
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
    assert state.b.value == 2
    assert state.b.type == nnx.Variable
    assert state.c.value == 3
    assert state.c.type == nnx.Param
    assert state.d.value == 4
    assert state.d.type == nnx.Variable
    assert state.e.value == 5
    assert state.e.type == nnx.BatchStat

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

    graphdef, states = nnx.split(foo)

    assert isinstance(states, nnx.State)
    assert issubclass(states.w.type, nnx.Param)

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

    graphdef, state = nnx.split(foo)

    assert isinstance(graphdef, nnx.GraphDef)
    assert isinstance(state, nnx.State)
    assert issubclass(state.w.type, nnx.Param)
    assert issubclass(state.c.type, nnx.Variable)

    y, (graphdef, state) = graphdef.apply(state)(x=2.0, rngs=nnx.Rngs(e=1))

    assert isinstance(y, jax.Array)

  def test_modules_iterator(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.submodules = [
          {'a': nnx.Linear(1, 1, rngs=rngs)},
          {'b': nnx.Conv(1, 1, 1, rngs=rngs)},
        ]

    module = Foo(rngs=nnx.Rngs(0))

    modules = list(module.iter_modules())

    assert len(modules) == 3
    assert modules[0][0] == ()
    assert isinstance(modules[0][1], Foo)
    assert modules[1][0] == ('submodules', 0, 'a')
    assert isinstance(modules[1][1], nnx.Linear)
    assert modules[2][0] == ('submodules', 1, 'b')
    assert isinstance(modules[2][1], nnx.Conv)

  def test_array_in_module(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jnp.array(1.0)

    foo = Foo()

    graphdef, state = nnx.split(foo)

    assert isinstance(state, nnx.State)
    assert isinstance(state.a, jax.Array)

    foo2 = nnx.merge(graphdef, state)

    assert isinstance(foo2.a, jax.Array)

  def test_state_in_module(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.State({'b': jnp.array(1.0)})

    foo = Foo()

    graphdef, state = nnx.split(foo)

    assert isinstance(state, nnx.State)
    assert isinstance(state.a, nnx.State)

    foo2 = nnx.merge(graphdef, state)

    assert isinstance(foo2.a, nnx.State)

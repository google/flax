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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import mesh_utils

from flax.experimental import nnx


class TestJIT:
  def test_jit(self):
    m = nnx.Dict(a=nnx.Param(1))

    @nnx.jit
    def g(m: nnx.Dict):
      m.a = 2
      return 1.0

    out = g(m)

    assert m.a == 2
    assert out == 1.0

  def test_jit_on_init(self):
    n = 0

    class Foo(nnx.Module):
      @partial(nnx.jit, static_argnums=(1, 2))
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        nonlocal n
        n += 1

        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1
    assert m.w.value.shape == (2, 3)
    assert m.din == 2
    assert m.dout == 3
    assert isinstance(m.din, int)
    assert isinstance(m.dout, int)
    assert isinstance(m.w.value, jax.Array)

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1

  def test_jit_on_call(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

      @nnx.jit
      def __call__(self, x: jax.Array) -> jax.Array:
        nonlocal n
        n += 1
        return jnp.dot(x, self.w.value)

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert m.w.value.shape == (2, 3)
    assert m.din == 2
    assert m.dout == 3
    assert isinstance(m.din, int)
    assert isinstance(m.dout, int)
    assert isinstance(m.w.value, jax.Array)

    y = m(jnp.ones((1, 2)))
    assert y.shape == (1, 3)
    assert n == 1
    y = m(jnp.ones((1, 2)))
    assert n == 1

  def test_jit_combinator(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

      @nnx.jit
      def __call__(self, x: jax.Array) -> jax.Array:
        nonlocal n
        n += 1
        return jnp.dot(x, self.w.value)

    m = nnx.Jit(Foo)(2, 3, rngs=nnx.Rngs(0))

    y = m(jnp.ones((1, 2)))
    assert y.shape == (1, 3)
    assert n == 1
    y = m(jnp.ones((1, 2)))
    assert n == 1

  def test_cached_unflatten(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.BatchNorm(2, rngs=rngs)

    @nnx.jit
    def f(m: Foo):
      nonlocal n
      n += 1
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b
    a_kernel = a.kernel.value
    a_bias = a.bias.value
    b_scale = b.scale.value
    b_bias = b.bias.value
    b_mean = b.mean.value
    b_var = b.var.value

    f(m)

    assert n == 1
    assert m.a is b
    assert m.b is a
    np.testing.assert_allclose(a_kernel, a.kernel.value)
    np.testing.assert_allclose(a_bias, a.bias.value)
    np.testing.assert_allclose(b_scale, b.scale.value)
    np.testing.assert_allclose(b_bias, b.bias.value)
    np.testing.assert_allclose(b_mean, b.mean.value)
    np.testing.assert_allclose(b_var, b.var.value)

    f(m)

    assert n == 2
    assert m.a is a
    assert m.b is b

    f(m)

    assert n == 2
    assert m.a is b
    assert m.b is a

    f(m)

    assert n == 2
    assert m.a is a
    assert m.b is b

  def test_cached_unflatten_same_type(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.Linear(2, 2, rngs=rngs)

    @nnx.jit
    def f(m: Foo):
      nonlocal n
      n += 1
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    f(m)

    assert n == 1
    assert m.a is b
    assert m.b is a

    f(m)

    assert n == 1
    assert m.a is a
    assert m.b is b

  def test_objects_in_pytree(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.Linear(2, 2, rngs=rngs)

    class FooDict(tp.TypedDict):
      foo: Foo

    @nnx.jit
    def f(tree: tuple[FooDict]):
      nonlocal n
      n += 1
      m = tree[0]['foo']
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    f(({'foo': m},))

    assert n == 1
    assert m.a is b
    assert m.b is a

    f(({'foo': m},))

    assert n == 1
    assert m.a is a
    assert m.b is b

  def test_cached_unflatten_swap_variables(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.Param(2)

    @nnx.jit
    def f(m: Foo):
      m.a, m.b = m.b, m.a

    m = Foo()
    a = m.a
    b = m.b

    f(m)

    assert m.a is b
    assert m.b is a

  def test_cached_unflatten_add_self_reference(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.ref: tp.Optional[Foo] = None

    @nnx.jit
    def f(m: Foo):
      nonlocal n
      n += 1
      m.ref = m

    m = Foo()

    f(m)

    assert n == 1
    assert m.ref is m

    f(m)

    assert n == 2
    assert m.ref is m

    f(m)

    assert n == 2
    assert m.ref is m

  def test_cached_unflatten_ref_in_output(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.ref: tp.Optional[Foo] = None

    @nnx.jit
    def f(m: Foo):
      nonlocal n
      n += 1
      m.ref = m
      return m

    m = Foo()

    m2 = f(m)

    assert n == 1
    assert m.ref is m
    assert m2 is m

    m2 = f(m)

    assert n == 2
    assert m.ref is m
    assert m2 is m

    m2 = f(m)

    assert n == 2
    assert m.ref is m
    assert m2 is m

  def test_apply_shardings(self):
    n_devices = max(jax.local_device_count() // 2, 1)
    devices = mesh_utils.create_device_mesh((n_devices, n_devices))
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    rngs = nnx.Rngs(0)
    m = nnx.Linear(
      16,
      32,
      rngs=rngs,
      kernel_init=nnx.with_partitioning(
        nnx.initializers.lecun_normal(), ('a', 'b')
      ),
    )

    @partial(nnx.jit, constrain_state=True)
    def constrain_object(m):
      pass

    with mesh:
      constrain_object(m)

    m.kernel.value.sharding



class TestGrad:
  def test_grad(self):
    p1 = nnx.Param(10.0)
    p2 = nnx.Param(20.0)

    m = nnx.Dict(
      a=nnx.List([p1, p2]),
      b=p1,
      c=7,
      d=5.0,
    )

    @nnx.grad
    def f(m: nnx.Dict):
      # sum all params
      return m['a'][0].value + m['a'][1].value + m['b'].value

    grads = f(m)

    assert m.a[0] is m.b
    assert isinstance(grads, nnx.State)
    assert grads['a'][0].value == 2.0
    assert issubclass(grads.a[0].type, nnx.Variable)
    assert grads['a'][1].value == 1.0
    assert issubclass(grads.a[1].type, nnx.Variable)
    assert len(grads.flat_state()) == 2

    nnx.update(m, grads)

    assert m.a[0] is m.b
    assert m['a'][0].value == 2.0
    assert m['a'][1].value == 1.0
    assert m['b'].value == 2.0
    assert m['c'] == 7
    assert m['d'] == 5.0

  def test_grad_with_multiple_ref_types(self):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(10.0), nnx.BatchStat(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    @nnx.grad
    def f(m: nnx.Dict):
      # sum all params
      return m.a[0].value + m.a[1].value + m.b.value

    grads = f(m)

    assert isinstance(grads, nnx.State)
    assert grads['a'][0].value == 1.0
    assert issubclass(grads.a[0].type, nnx.Param)
    assert len(grads) == 2

    nnx.update(m, grads)

    assert m.a[0].value == 1.0
    assert m.a[1].value == 20.0
    assert m.b.value == 1.0
    assert m.c == 7
    assert m.d == 5.0

  def test_grad_with_type_predicate(self):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(10.0), nnx.BatchStat(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    @partial(nnx.grad, wrt=nnx.BatchStat)
    def f(m: nnx.Dict):
      # sum all params
      return m.a[0].value + m.a[1].value + m.b.value

    grads = f(m)

    assert isinstance(grads, nnx.State)
    assert grads['a'][1].value == 1.0
    assert issubclass(grads.a[1].type, nnx.BatchStat)
    assert len(grads) == 1

    nnx.update(m, grads)

    assert m.a[0].value == 10.0
    assert m.a[1].value == 1.0
    assert m.b.value == 10.0
    assert m.c == 7
    assert m.d == 5.0

  def test_multiple_inputs(self):
    rngs = nnx.Rngs(0)
    m = nnx.Linear(2, 3, rngs=rngs)
    loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, wrt=nnx.Param)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads = grad_fn(m, x, y)

    assert 'kernel' in grads
    assert grads.kernel.value.shape == (2, 3)
    assert 'bias' in grads
    assert grads.bias.value.shape == (3,)

  def test_multiple_graph_nodes(self):
    rngs = nnx.Rngs(0)
    m1 = nnx.Linear(2, 3, rngs=rngs)
    m2 = nnx.Linear(3, 3, rngs=rngs)
    loss_fn = lambda m1, m2, x, y: jnp.mean((m2(m1(x)) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, argnums=(0, 1), wrt=nnx.Param)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads_m1, grads_m2 = grad_fn(m1, m2, x, y)

    assert 'kernel' in grads_m1
    assert grads_m1.kernel.value.shape == (2, 3)
    assert 'bias' in grads_m1
    assert grads_m1.bias.value.shape == (3,)
    assert 'kernel' in grads_m2
    assert grads_m2.kernel.value.shape == (3, 3)
    assert 'bias' in grads_m2
    assert grads_m2.bias.value.shape == (3,)

  def test_multiple_graph_nodes_mix_positions(self):
    rngs = nnx.Rngs(0)
    m1 = nnx.Linear(2, 3, rngs=rngs)
    m2 = nnx.Linear(3, 3, rngs=rngs)
    loss_fn = lambda x, m1, y, m2: jnp.mean((m2(m1(x)) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, argnums=(1, 3), wrt=nnx.Param)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads_m1, grads_m2 = grad_fn(x, m1, y, m2)

    assert 'kernel' in grads_m1
    assert grads_m1.kernel.value.shape == (2, 3)
    assert 'bias' in grads_m1
    assert grads_m1.bias.value.shape == (3,)
    assert 'kernel' in grads_m2
    assert grads_m2.kernel.value.shape == (3, 3)
    assert 'bias' in grads_m2
    assert grads_m2.bias.value.shape == (3,)


class TestScan:
  def test_basic(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        # self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    @partial(nnx.scan, state_axes={nnx.Param: 0}, length=5)
    def create_block(_, rngs: nnx.Rngs):
      return None, Block(rngs=rngs)

    _, module = create_block(None, nnx.Rngs(0))

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    # assert module.node.value.shape == (2,)

    @partial(nnx.scan, in_axes=None, state_axes={nnx.Param: 0}, length=5)
    def forward_block(_, block: Block, x: jax.Array):
      return None, block(x)

    x = jnp.ones((1, 3))
    out, y = forward_block(None, module, x)

    assert y.shape == (5, 1, 3)
    assert out is None

  def test_basic_combinator(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
    )

    module = MLP(rngs=nnx.Rngs(0))

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y, out = module(x)

    assert y.shape == (1, 3)
    assert out is None

  def test_no_scan_output(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      scan_output=False,
    )

    module = MLP(rngs=nnx.Rngs(0))

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y = module(x)

    assert y.shape == (1, 3)

  def test_out_axes(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, (x, x)

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      out_axes=(1, 2),
    )

    module = MLP(rngs=nnx.Rngs(0))

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    c, (y1, y2) = module(x)

    assert c.shape == (1, 3)
    assert y1.shape == (1, 5, 3)
    assert y2.shape == (1, 3, 5)

  def test_in_axes(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(
        self, x: jax.Array, a: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape
        x = x + a
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
    )

    module = MLP(rngs=nnx.Rngs(0))

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    y, out = module(x, a)

    assert y.shape == (1, 3)
    assert out is None

  def test_in_axes_broadcast(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(
        self, x: jax.Array, a: jax.Array, b: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape
        assert x.shape == b.shape
        x = x + a + b
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      in_axes=(None, None, 0, None),
    )

    module = MLP(rngs=nnx.Rngs(0))

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    b = jnp.ones((1, 3))
    y, out = module(x, a, b)

    assert y.shape == (1, 3)
    assert out is None

  def test_complex(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x, rngs=rngs)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Scan(
      Block, state_axes={nnx.Param: 0}, length=5, scan_output=False
    )

    module = MLP(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y = module(x, rngs=nnx.Rngs(1))

    assert y.shape == (1, 3)

  def test_complex_broadcast_dropout(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x, rngs=rngs)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      # params is split, dropout is broadcast
      split_rngs=['dropout'],
      scan_output=False,
    )

    module = MLP(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y = module(x, rngs=nnx.Rngs(1))

    assert y.shape == (1, 3)

  def test_complex_decorator(self):
    class Block(nnx.Module):
      @partial(
        nnx.vmap,
        state_axes={nnx.Param: 0},
        axis_size=5,
      )
      def __init__(self, *, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.node = nnx.Variable(jnp.ones((2,)))

      @partial(
        nnx.scan,
        state_axes={nnx.Param: 0},
        length=5,
        carry_argnum=1,
      )
      def __call__(
        self, x: jax.Array, _, *, rngs: nnx.Rngs
      ) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x, rngs=rngs)
        x = nnx.gelu(x)
        return x, None

    module = Block(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.d == 3
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y, out = module(x, None, rngs=nnx.Rngs(dropout=1))

    assert y.shape == (1, 3)
    assert out is None

  def test_scan_with_sharding(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
          3,
          3,
          kernel_init=nnx.with_metadata(
            nnx.initializers.lecun_normal(),
            sharding=('din', 'dout'),
          ),
          bias_init=nnx.with_metadata(
            nnx.initializers.zeros_init(),
            sharding=('dout',),
          ),
          rngs=rngs,
        )

      def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)

        # test sharding layer axes is not present inside scan
        state = nnx.state(self.linear)
        assert state.kernel.value.shape == (3, 3)
        assert state.kernel.sharding == ('din', 'dout')
        assert state.bias.value.shape == (3,)
        assert state.bias.sharding == ('dout',)

        return x, None

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      transform_metadata={nnx.PARTITION_NAME: 'layers'},
    )

    m = MLP(rngs=nnx.Rngs(0))

    # test sharding layers axes is set
    state = nnx.state(m)
    assert state.scan_module.linear.kernel.value.shape == (
      5,
      3,
      3,
    )
    assert state.scan_module.linear.kernel.sharding == (
      'layers',
      'din',
      'dout',
    )
    assert state.scan_module.linear.bias.value.shape == (5, 3)
    assert state.scan_module.linear.bias.sharding == (
      'layers',
      'dout',
    )

    x = jnp.ones((1, 3))
    y, out = m(x, None)

    # test sharding axes is preserved
    state = nnx.state(m)
    assert state.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert state.scan_module.linear.kernel.sharding == (
      'layers',
      'din',
      'dout',
    )
    assert state.scan_module.linear.bias.value.shape == (5, 3)
    assert state.scan_module.linear.bias.sharding == (
      'layers',
      'dout',
    )

  def test_type_error_less_than_one_args(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self):
        return None, None

    MLP = nnx.Scan(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
    )

    mlp = MLP(rngs=nnx.Rngs(0))

    with pytest.raises(
      TypeError, match='Expected at least 2 positional argument'
    ):
      mlp()


class TestRemat:
  def test_basic_remat(self):
    RematLinear = nnx.Remat(nnx.Linear)

    module = RematLinear(2, 3, rngs=nnx.Rngs(0))

    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)

  def test_remat_decorator(self):
    class RematLinear(nnx.Module):
      @partial(nnx.remat, static_argnums=(1, 2))
      def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)

      @nnx.remat
      def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)

    module = RematLinear(2, 3, nnx.Rngs(0))

    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)

  def test_remat_with_scan(self):
    class LinearBlock(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        return x, None

    RematLinear = nnx.Remat(LinearBlock)

    ScanRematLinear = nnx.Scan(
      RematLinear,
      state_axes={nnx.Param: 0},
      length=5,
    )

    m = ScanRematLinear(rngs=nnx.Rngs(0))

    assert m.scan_module.remat_module.linear.kernel.value.shape == (5, 3, 3)
    assert m.scan_module.remat_module.linear.bias.value.shape == (5, 3)

    y, _ = m(jnp.ones((1, 3)), None)
    assert y.shape == (1, 3)

    y, _ = m(jnp.ones((1, 3)), None)
    assert y.shape == (1, 3)

  def test_remat_with_scan_decorator(self):
    class ScanLinear(nnx.Module):
      @partial(
        nnx.vmap,
        state_axes={nnx.Param: 0},
        axis_size=5,
      )
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      @partial(
        nnx.scan,
        in_axes=None,
        state_axes={nnx.Param: 0},
        length=5,
        carry_argnum=1,
      )
      @nnx.remat
      def __call__(self, x: jax.Array, _) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        return x, None

    m = ScanLinear(rngs=nnx.Rngs(0))

    assert m.linear.kernel.value.shape == (5, 3, 3)
    assert m.linear.bias.value.shape == (5, 3)

    y, _ = m(jnp.ones((1, 3)), None)
    assert y.shape == (1, 3)


class TestVmap:
  def test_basic(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    vectorized_create_block = nnx.vmap(
      create_block, state_axes={nnx.Param: 0}, axis_size=5
    )

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = vectorized_create_block(rngs)

    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    def forward_block(module, x):
      return module(x)

    vectorized_forward_block = nnx.vmap(
      forward_block, state_axes={nnx.Param: 0}, axis_size=5
    )

    y = vectorized_forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 3
    assert rngs.default.key.value == initial_key

    y2 = vectorized_forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_basic_demo(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    @partial(nnx.vmap, axis_size=5)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(nnx.vmap, axis_size=5)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )

    x = jnp.ones((5, 1, 3))

    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 3

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

  def test_replicate(self):
    din = 3
    dout = 10

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @partial(
      nnx.vmap,
      state_axes={},  # replicate all state
      split_rngs=True,  # different rngs for each replica
    )
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (din, dout)
    assert module.linear.bias.value.shape == (dout,)

    x = jnp.ones((5, 1, din))

    y = forward_block(module, x)

    assert y.shape == (5, 1, dout)
    assert rngs.default.count.value == 3

    assert not jnp.allclose(y[0], y[1])

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

    assert rngs.default.key.value == initial_key

  def test_combinator(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Vmap(Block, state_axes={nnx.Param: 0}, axis_size=5)

    module = MLP(rngs=nnx.Rngs(0))

    assert not jnp.allclose(
      module.vmap_module.linear.kernel.value[0],
      module.vmap_module.linear.kernel.value[1],
    )
    assert module.vmap_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.vmap_module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))
    y = module(x)

    assert y.shape == (5, 1, 3)

  def test_combinator_init(self):
    class Block(nnx.Module):
      def __init__(self, *, graphdef: str, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.graphdef = graphdef

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Vmap(Block, state_axes={nnx.Param: 0}, axis_size=5)

    module = MLP(graphdef='hello', rngs=nnx.Rngs(0))

    assert module.vmap_module.graphdef == 'hello'

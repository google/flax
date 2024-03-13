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
import pytest

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

    m = nnx.JIT(Foo)(2, 3, rngs=nnx.Rngs(0))

    y = m(jnp.ones((1, 2)))
    assert y.shape == (1, 3)
    assert n == 1
    y = m(jnp.ones((1, 2)))
    assert n == 1


class TestGrad:
  def test_grad(self):
    p1 = nnx.Param(10.0)
    p2 = nnx.Param(20.0)

    m = nnx.Dict(
      a=nnx.Sequence([p1, p2]),
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
    assert grads['a']['0'].raw_value == 2.0
    assert isinstance(grads.a['0'], nnx.Variable)
    assert grads['a']['1'].raw_value == 1.0
    assert isinstance(grads.a['1'], nnx.Variable)
    assert len(grads.flat_state()) == 2

    m.update(grads)

    assert m.a[0] is m.b
    assert m['a'][0].value == 2.0
    assert m['a'][1].value == 1.0
    assert m['b'].value == 2.0
    assert m['c'] == 7
    assert m['d'] == 5.0

  def test_grad_with_multiple_ref_types(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(10.0), nnx.BatchStat(20.0)]),
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
    assert grads['a']['0'].raw_value == 1.0
    assert isinstance(grads.a['0'], nnx.Param)
    assert len(grads) == 2

    m.update(grads)

    assert m.a[0].value == 1.0
    assert m.a[1].value == 20.0
    assert m.b.value == 1.0
    assert m.c == 7
    assert m.d == 5.0

  def test_grad_with_type_predicate(self):
    m = nnx.Dict(
      a=nnx.Sequence([nnx.Param(10.0), nnx.BatchStat(20.0)]),
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
    assert grads['a']['1'].raw_value == 1.0
    assert isinstance(grads.a['1'], nnx.BatchStat)
    assert len(grads) == 1

    m.update(grads)

    assert m.a[0].value == 10.0
    assert m.a[1].value == 1.0
    assert m.b.value == 10.0
    assert m.c == 7
    assert m.d == 5.0


class TestScan:
  def test_basic(self):
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
      variable_axes={nnx.Param: 0},
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
      variable_axes={nnx.Param: 0},
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
      variable_axes={nnx.Param: 0},
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
      variable_axes={nnx.Param: 0},
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
      variable_axes={nnx.Param: 0},
      length=5,
      in_args_axes=(0, None),
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
      Block, variable_axes={nnx.Param: 0}, length=5, scan_output=False
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
      variable_axes={nnx.Param: 0},
      length=5,
      # params is split, dropout is broadcast
      broadcast_rngs=['dropout'],
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
    scan_over_layers = partial(
      nnx.scan,
      variable_axes={nnx.Param: 0},
      length=5,
    )

    class Block(nnx.Module):
      @scan_over_layers
      def __init__(self, *, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)
        self.node = nnx.Variable(jnp.ones((2,)))

      @scan_over_layers
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
        state = self.linear.get_state()
        assert state.kernel.raw_value.shape == (3, 3)
        assert state.kernel.sharding == ('din', 'dout')
        assert state.bias.raw_value.shape == (3,)
        assert state.bias.sharding == ('dout',)

        return x, None

    MLP = nnx.Scan(
      Block,
      variable_axes={nnx.Param: 0},
      length=5,
      scan_metadata={nnx.PARTITION_NAME: 'layers'},
    )

    m = MLP(rngs=nnx.Rngs(0))

    # test sharding layers axes is set
    state = m.get_state()
    assert state.scan_module.linear.kernel.raw_value.shape == (
      5,
      3,
      3,
    )
    assert state.scan_module.linear.kernel.sharding == (
      'layers',
      'din',
      'dout',
    )
    assert state.scan_module.linear.bias.raw_value.shape == (5, 3)
    assert state.scan_module.linear.bias.sharding == (
      'layers',
      'dout',
    )

    x = jnp.ones((1, 3))
    y, out = m(x, None)

    # test sharding axes is preserved
    state = m.get_state()
    assert state.scan_module.linear.kernel.raw_value.shape == (5, 3, 3)
    assert state.scan_module.linear.kernel.sharding == (
      'layers',
      'din',
      'dout',
    )
    assert state.scan_module.linear.bias.raw_value.shape == (5, 3)
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
      variable_axes={nnx.Param: 0},
      length=5,
    )

    mlp = MLP(rngs=nnx.Rngs(0))

    with pytest.raises(
      TypeError, match='Expected at least 1 positional argument'
    ):
      mlp()

  def test_value_error_positional_argument_type_context(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        return x, None

    MLP = nnx.Scan(
      Block,
      variable_axes={nnx.Param: 0},
      length=5,
    )

    with pytest.raises(
      ValueError, match='Rngs must be passed as a keyword argument named'
    ):
      MLP(nnx.Rngs(0))


class TestRemat:
  def test_basic_remat(self):
    RematLinear = nnx.Remat(nnx.Linear)

    module = RematLinear(2, 3, rngs=nnx.Rngs(0))

    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)

  def test_remat_decorator(self):
    class RematLinear(nnx.Module):
      @nnx.remat
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)

      @nnx.remat
      def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)

    module = RematLinear(2, 3, rngs=nnx.Rngs(0))

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
      variable_axes={nnx.Param: 0},
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
    scan = partial(
      nnx.scan,
      variable_axes={nnx.Param: 0},
      length=5,
    )

    class ScanLinear(nnx.Module):
      @scan
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      @scan
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
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Vmap(Block, variable_axes={nnx.Param: 0}, axis_size=5)

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

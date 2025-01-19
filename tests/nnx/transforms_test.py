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
from functools import partial
import typing as tp

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.nnx.transforms import general
import jax
from jax.experimental import mesh_utils, checkify
import jax.numpy as jnp
import numpy as np



class List(nnx.Module):
  def __init__(self, items):
    vars(self).update({str(i): item for i, item in enumerate(items)})

  def __getitem__(self, idx):
    return getattr(self, str(idx))

  def __setitem__(self, idx, value):
    setattr(self, str(idx), value)


class Dict(nnx.Module):
  def __init__(self, *args, **kwargs):
    vars(self).update(dict(*args, **kwargs))

  def __getitem__(self, key):
    return vars(self)[key]

  def __setitem__(self, key, value):
    vars(self)[key] = value

  if tp.TYPE_CHECKING:

    def __getattr__(self, key): ...


class TestJIT(absltest.TestCase):
  def test_jit(self):
    m = Dict(a=nnx.Param(1))

    @nnx.jit
    def g(m: Dict):
      m.a = 2
      return 1.0

    out = g(m)

    assert m.a == 2
    assert out == 1.0

  def test_simple_double_call(self):
    n = 0
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit
    def f(m: nnx.Linear, x: jnp.ndarray) -> jnp.ndarray:
      nonlocal n
      n += 1
      return m(x)

    x = jnp.ones((1, 2))
    y = f(m, x)

    self.assertEqual(n, 1)
    self.assertEqual(y.shape, (1, 3))

    y = f(m, x)

    self.assertEqual(n, 1)
    self.assertEqual(y.shape, (1, 3))

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

    m = nnx.Jit.constructor(Foo)(2, 3, rngs=nnx.Rngs(0))

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
      m.a, m.b = m.b, m.a  # type: ignore

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
        self.ref: tp.Optional[Foo] = None  # type: ignore[name-error]

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
        self.ref: tp.Optional[Foo] = None  # type: ignore[name-error]

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
    devices = mesh_utils.create_device_mesh((n_devices, jax.local_device_count() // n_devices))
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    def sharding(*args):
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

    state_sharding = nnx.StateSharding({
        nnx.PathContains('kernel'): sharding('a', 'b'),
        nnx.PathContains('bias'): sharding('b'),
    })

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    self.assertNotIsInstance(
        m.kernel.value.sharding, jax.sharding.NamedSharding
    )

    @nnx.jit(in_shardings=(state_sharding,))
    def constrain_object(m):
      pass

    constrain_object(m)

    self.assertIsInstance(m.kernel.value.sharding, jax.sharding.NamedSharding)

  def test_cache_args(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit
    def f(cached_m: nnx.Linear, m: nnx.Linear):
      self.assertIsNot(cached_m, m)
      self.assertIs(cached_m.kernel, m.kernel)
      self.assertIs(cached_m.bias, m.bias)
      return cached_m

    cached_f = nnx.cache_args(f, m)
    cached_m = cached_f(m)

    self.assertIsNot(m, cached_m)
    self.assertIs(m.kernel, cached_m.kernel)
    self.assertIs(m.bias, cached_m.bias)

    # test that cached m is reused
    cached_m2 = cached_f(m)
    self.assertIs(cached_m, cached_m2)


class TestGrad(parameterized.TestCase):
  def test_grad(self):
    p1 = nnx.Param(10.0)
    p2 = nnx.Param(20.0)

    m = Dict(
      a=List([p1, p2]),
      b=p1,
      c=7,
      d=5.0,
    )

    @nnx.grad
    def f(m: Dict):
      # sum all params
      return m['a'][0].value + m['a'][1].value + m['b'].value

    grads = f(m)

    assert m.a[0] is m.b
    assert isinstance(grads, nnx.State)
    assert grads['a']['0'].value == 2.0
    assert issubclass(grads.a['0'].type, nnx.Variable)
    assert grads['a']['1'].value == 1.0
    assert issubclass(grads.a['1'].type, nnx.Variable)
    assert len(grads.flat_state()) == 2

    nnx.update(m, grads)

    assert m.a[0] is m.b
    assert m['a'][0].value == 2.0
    assert m['a'][1].value == 1.0
    assert m['b'].value == 2.0
    assert m['c'] == 7
    assert m['d'] == 5.0

  def test_grad_with_multiple_ref_types(self):
    m = Dict(
      a=List([nnx.Param(10.0), nnx.BatchStat(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    @nnx.grad
    def f(m: Dict):
      # sum all params
      return m.a[0].value + m.a[1].value + m.b.value

    grads = f(m)

    assert isinstance(grads, nnx.State)
    assert grads['a']['0'].value == 1.0
    assert issubclass(grads.a['0'].type, nnx.Param)
    assert len(grads) == 2

    nnx.update(m, grads)

    assert m.a['0'].value == 1.0
    assert m.a['1'].value == 20.0
    assert m.b.value == 1.0
    assert m.c == 7
    assert m.d == 5.0

  def test_grad_with_type_predicate(self):
    m = Dict(
      a=List([nnx.Param(10.0), nnx.BatchStat(20.0)]),
      b=nnx.Param(10.0),
      c=7,
      d=5.0,
    )

    @nnx.grad(argnums=nnx.DiffState(0, nnx.BatchStat))
    def f(m: Dict):
      # sum all params
      return m.a[0].value + m.a[1].value + m.b.value

    grads = f(m)

    assert isinstance(grads, nnx.State)
    assert grads['a']['1'].value == 1.0
    assert issubclass(grads.a['1'].type, nnx.BatchStat)
    assert len(grads) == 1

    nnx.update(m, grads)

    assert m.a['0'].value == 10.0
    assert m.a['1'].value == 1.0
    assert m.b.value == 10.0
    assert m.c == 7
    assert m.d == 5.0

  def test_multiple_inputs(self):
    rngs = nnx.Rngs(0)
    m = nnx.Linear(2, 3, rngs=rngs)
    loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    grad_fn = nnx.grad(loss_fn)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads = grad_fn(m, x, y)

    assert 'kernel' in grads
    assert grads.kernel.value.shape == (2, 3)
    assert 'bias' in grads
    assert grads.bias.value.shape == (3,)

  @parameterized.parameters(
    {
      'loss_fn': lambda m1, m2, x, y: jnp.mean((m2(m1(x)) - y) ** 2),
      'argnums': (0, 1),
    },
    {
      'loss_fn': lambda x, m1, y, m2: jnp.mean((m2(m1(x)) - y) ** 2),
      'argnums': (1, 3),
    },
  )
  def test_multiple_graph_nodes(self, loss_fn, argnums):
    rngs = nnx.Rngs(0)
    m1 = nnx.Linear(2, 3, rngs=rngs)
    m2 = nnx.Linear(3, 3, rngs=rngs)
    grad_fn = nnx.grad(loss_fn, argnums=argnums)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    inputs = [x, y]
    inputs.insert(argnums[0], m1)
    inputs.insert(argnums[1], m2)
    grads_m1, grads_m2 = grad_fn(*inputs)

    assert 'kernel' in grads_m1
    assert grads_m1.kernel.value.shape == (2, 3)
    assert 'bias' in grads_m1
    assert grads_m1.bias.value.shape == (3,)
    assert 'kernel' in grads_m2
    assert grads_m2.kernel.value.shape == (3, 3)
    assert 'bias' in grads_m2
    assert grads_m2.bias.value.shape == (3,)

  def test_multiple_args(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    m1_diffstate = nnx.DiffState(0, nnx.PathContains('kernel'))
    m2_diffstate = nnx.DiffState(1, nnx.PathContains('bias'))

    @nnx.grad(argnums=(m1_diffstate, m2_diffstate))
    def loss_fn(m1: nnx.Linear, m2: nnx.Linear):
      return jnp.mean(m1.kernel * m2.kernel) + jnp.mean(m1.bias * m2.bias)

    grads_m1, grads_m2 = loss_fn(m1, m2)

    self.assertIn('kernel', grads_m1)
    self.assertNotIn('bias', grads_m1)
    self.assertNotIn('kernel', grads_m2)
    self.assertIn('bias', grads_m2)

  def test_multiple_args_in_pytrees(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    m1_diffstate = nnx.DiffState(0, nnx.PathContains('kernel'))
    m2_diffstate = nnx.DiffState(1, nnx.PathContains('bias'))

    @nnx.grad(argnums=(m1_diffstate, m2_diffstate))
    def loss_fn(l1: list[nnx.Linear], l2: list[nnx.Linear]):
      return jnp.mean(l1[0].kernel * l2[0].kernel) + jnp.mean(
          l1[0].bias * l2[0].bias
      )

    grads_m1, grads_m2 = loss_fn([m1], [m2])

    self.assertIn('kernel', grads_m1[0])
    self.assertNotIn('bias', grads_m1[0])
    self.assertNotIn('kernel', grads_m2[0])
    self.assertIn('bias', grads_m2[0])

  def test_value_and_grad_multiple_args_in_pytrees(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    m1_diffstate = nnx.DiffState(0, nnx.PathContains('kernel'))
    m2_diffstate = nnx.DiffState(1, nnx.PathContains('bias'))

    @nnx.value_and_grad(argnums=(m1_diffstate, m2_diffstate))
    def loss_fn(l1: list[nnx.Linear], l2: list[nnx.Linear]):
      return jnp.mean(l1[0].kernel * l2[0].kernel) + jnp.mean(
          l1[0].bias * l2[0].bias
      )

    loss, (grads_m1, grads_m2) = loss_fn([m1], [m2])

    self.assertEqual(loss.shape, ())
    self.assertIn('kernel', grads_m1[0])
    self.assertNotIn('bias', grads_m1[0])
    self.assertNotIn('kernel', grads_m2[0])
    self.assertIn('bias', grads_m2[0])

  def test_value_and_grad_with_aux(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    m1_diffstate = nnx.DiffState(0, nnx.PathContains('kernel'))
    m2_diffstate = nnx.DiffState(1, nnx.PathContains('bias'))

    @nnx.value_and_grad(argnums=(m1_diffstate, m2_diffstate), has_aux=True)
    def loss_fn(l1: list[nnx.Linear], l2: list[nnx.Linear]):
      loss = jnp.mean(l1[0].kernel * l2[0].kernel) + jnp.mean(
          l1[0].bias * l2[0].bias
      )
      l1[0].kernel.value = jnp.array(-1.0)
      m3 = nnx.Linear(2, 3, rngs=nnx.Rngs(2))
      return loss, m3

    (loss, m3), (grads_m1, grads_m2) = loss_fn([m1], [m2])

    self.assertEqual(m1.kernel.value, -1.0)
    self.assertEqual(loss.shape, ())
    self.assertIsInstance(m3, nnx.Linear)
    self.assertIn('kernel', grads_m1[0])
    self.assertNotIn('bias', grads_m1[0])
    self.assertNotIn('kernel', grads_m2[0])
    self.assertIn('bias', grads_m2[0])


class TestCustomVJP(parameterized.TestCase):

  def test_basic_call(self):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(1, 1, rngs=nnx.Rngs(1))

    @nnx.custom_vjp
    def f(m1: nnx.Linear, m2: nnx.Linear):
      y = m1.kernel * m2.kernel
      m1.kernel.value = jnp.array(-1.0)
      return y

    def f_fwd(m1, m2):
      y = f(m1, m2)
      return y, (m1, m2)

    def f_bwd(res, g):
      inputs_g, out_g = g
      m1, m2 = res
      return inputs_g

    f.defvjp(f_fwd, f_bwd)

    y = f(m1, m2)

    self.assertEqual(m1.kernel.value, -1.0)
    self.assertEqual(y.shape, (1, 1))

  def test_jax_example(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

      def __hash__(self):
        return id(self)

    @nnx.custom_vjp
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(m_g, res, out_g):
      cos_x, sin_x, m = res

      self.assertIsInstance(m_g, nnx.State)
      self.assertEqual(out_g.shape, ())
      self.assertIsInstance(m, Foo)

      # m_g = nnx.State({'x': cos_x * out_g * m.y, 'y': sin_x * out_g})
      m_g.x.value = cos_x * out_g * m.y
      m_g.y.value = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    grad: nnx.State = nnx.grad(f, argnums=nnx.DiffState(0, ...))(m)

    np.testing.assert_allclose(grad['x'].value, jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grad['y'].value, jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  def test_diff_state(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

      def __hash__(self):
        return id(self)

    x_in_path = nnx.PathContains('x')
    diff_state = nnx.DiffState(0, x_in_path)

    @nnx.custom_vjp(nondiff_argnums=(diff_state,))
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), m)  # type: ignore
      return y, res

    def f_bwd(m_g, res, out_g):
      cos_x, m = res

      self.assertIsInstance(m_g, nnx.State)
      self.assertEqual(out_g.shape, ())
      self.assertIsInstance(m, Foo)

      m_g.x.value = cos_x * out_g * m.y
      del m_g['y']
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    grad: nnx.State = nnx.grad(f, argnums=nnx.DiffState(0, x_in_path))(m)

    np.testing.assert_allclose(grad['x'].value, jnp.cos(1.0) * 2.0)  # type: ignore
    self.assertEqual(m.z, 1)

  def test_jax_example_with_remat(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

      def __hash__(self):
        return id(self)

    @nnx.custom_vjp
    @nnx.remat
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x.value) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x.value), jnp.sin(m.x.value), m)  # type: ignore
      return y, res

    def f_bwd(m_g, res, out_g):
      cos_x, sin_x, m = res

      self.assertIsInstance(m_g, nnx.State)
      self.assertEqual(out_g.shape, ())
      self.assertIsInstance(m, Foo)

      # m_g = nnx.State({'x': cos_x * out_g * m.y, 'y': sin_x * out_g})
      m_g.x.value = cos_x * out_g * m.y
      m_g.y.value = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    @nnx.jit
    def loss_fn(m):
      return f(m)

    grad: nnx.State = nnx.grad(loss_fn, argnums=nnx.DiffState(0, ...))(m)

    np.testing.assert_allclose(grad['x'].value, jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grad['y'].value, jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  def test_two_args(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

      def __hash__(self):
        return id(self)

    @nnx.custom_vjp
    def f(m1: Foo, m2: Foo):
      m1.z += 1
      y = jnp.sin(m1.x) * m1.y  # type: ignore
      return y, m2

    def f_fwd(m1: Foo, m2: Foo):
      y, m2 = f(m1, m2)
      res = (jnp.cos(m1.x), jnp.sin(m1.x), m1)  # type: ignore
      return (y, m2), res

    def f_bwd(m1_g, m2_g, res, g):
      y_g, _ = g
      cos_x, sin_x, m = res

      self.assertIsInstance(m1_g, nnx.State)
      self.assertIsInstance(m2_g, nnx.State)
      self.assertEqual(y_g.shape, ())
      self.assertIsInstance(m, Foo)

      m1_g = nnx.State(dict(x=cos_x * y_g * m.y, y=sin_x * y_g))
      m2_g = nnx.State(dict(x=m2_g['x'], y=m2_g['y']))

      return m1_g, m2_g

    f.defvjp(f_fwd, f_bwd)

    m1 = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)
    m2 = Foo(nnx.Param(jnp.array(3.0)), nnx.Param(jnp.array(4.0)), 0)

    def loss_fn(m1, m2):
      y, m2 = f(m1, m2)
      return y + m2.x * m2.y

    m1_grad: nnx.State
    m2_grad: nnx.State
    m1_grad, m2_grad = nnx.grad(
        loss_fn, argnums=(nnx.DiffState(0, ...), nnx.DiffState(1, ...))
    )(m1, m2)

    np.testing.assert_allclose(m1_grad['x'].value, jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(m1_grad['y'].value, jnp.sin(1.0))  # type: ignore
    self.assertEqual(m1.z, 1)
    np.testing.assert_allclose(m2_grad['x'].value, 4.0)  # type: ignore
    np.testing.assert_allclose(m2_grad['y'].value, 3.0)  # type: ignore

  def test_non_diff_args(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

      def __hash__(self):
        return id(self)

    @nnx.custom_vjp(nondiff_argnums=(0, 2))
    def f(a, m: Foo, b):
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(a, m: Foo, b):
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      y = f(a, m, b)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(a, m_g, b, res, out_g):
      cos_x, sin_x, m = res

      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      self.assertIsInstance(m_g, nnx.State)
      self.assertEqual(out_g.shape, ())
      self.assertIsInstance(m, Foo)

      # m_g = nnx.State({'x': cos_x * out_g * m.y, 'y': sin_x * out_g})
      m_g.x.value = cos_x * out_g * m.y
      m_g.y.value = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    def loss_fn(m):
      a = 1
      b = 2
      return f(a, m, b)

    grad: nnx.State = nnx.grad(loss_fn, argnums=nnx.DiffState(0, ...))(m)

    np.testing.assert_allclose(grad['x'].value, jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grad['y'].value, jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  def test_docs_example(self):
    import jax.numpy as jnp
    from flax import nnx

    class Foo(nnx.Module):

      def __init__(self, x, y):
        self.x = nnx.Param(x)
        self.y = nnx.Param(y)

    @nnx.custom_vjp
    def f(m: Foo):
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore

    def f_bwd(m_g, res, out_g):
      cos_x, sin_x, m = res
      m_t = nnx.State(dict(x=cos_x * out_g * m.y, y=sin_x * out_g))
      return (m_t,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
    grads = nnx.grad(f)(m)

  @parameterized.parameters(
    {'use_custom_vjp': False},
    {'use_custom_vjp': True},
  )
  def test_issue(self, use_custom_vjp: bool):
    class MyLinear(nnx.Module):
      def __init__(
        self, in_features: int, out_features: int, *, rngs: nnx.Rngs
      ):
        kernel_init = nnx.initializers.normal(in_features**-0.5)
        self.kernel = nnx.Param(
          kernel_init(rngs.params(), (in_features, out_features), jnp.float32)
        )
        self.bias = nnx.Param(jnp.zeros((out_features,), jnp.float32))
        self.n = nnx.BatchStat(jnp.array(0, jnp.uint32))

    def linear(m: MyLinear, x: jax.Array) -> jax.Array:
      m.n.value += 1
      y = x @ m.kernel + m.bias
      return y

    def linear_fwd(m: MyLinear, x: jax.Array):
      return linear(m, x), (m, x)

    def linear_bwd(m_g, _, res, outputs_g):
      m, x = res
      kernel_grad = outputs_g[None, :] * x[:, None]
      bias_grad = outputs_g
      x_grad = m.kernel @ outputs_g
      assert x_grad.shape == x.shape, 'Shape mismatch for x'
      assert (
        m.kernel.value.shape == kernel_grad.shape
      ), 'Shape mismatch for kernel'
      assert m.bias.value.shape == bias_grad.shape, 'Shape mismatch for bias'
      return (m_g, x_grad)

    if use_custom_vjp:
      linear = nnx.custom_vjp(linear)
      linear.defvjp(linear_fwd, linear_bwd)

    @nnx.jit
    def loss_fn(x, mod):
      y = linear(mod, x)
      return y.mean()

    mod = MyLinear(10, 5, rngs=nnx.Rngs(0))
    self.assertEqual(mod.n.value, 0)
    x = jax.random.normal(jax.random.key(0), (10,))
    loss, grad = nnx.value_and_grad(loss_fn)(x, mod)
    self.assertEqual(loss.shape, ())
    self.assertEqual(grad.shape, (10,))
    self.assertEqual(mod.n.value, 1)


class TestScan(absltest.TestCase):
  def test_basic(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    @nnx.split_rngs(splits=5)
    @nnx.scan(in_axes=(nnx.Carry, 0), length=5)
    def create_block(_, rngs: nnx.Rngs):
      return None, Block(rngs=rngs)

    _, module = create_block(None, nnx.Rngs(0))

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    @nnx.scan(in_axes=(nnx.Carry, 0, None), length=5)
    def forward_block(_, block: Block, x: jax.Array):
      return None, block(x)

    x = jnp.ones((1, 3))
    out, y = forward_block(None, module, x)

    assert y.shape == (5, 1, 3)
    assert out is None

  def test_basic_no_carry(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    @nnx.split_rngs(splits=5)
    @nnx.scan(in_axes=(0,), out_axes=0, length=5)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs=rngs)

    module = create_block(nnx.Rngs(0))

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    # assert module.node.value.shape == (2,)

    @nnx.scan(in_axes=(0, None), out_axes=0, length=5)
    def forward_block(block: Block, x: jax.Array):
      return block(x)

    x = jnp.ones((1, 3))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  def test_all_carry(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      n: nnx.BatchStat[int]

      def __hash__(self):
        return id(self)

    foo = Foo(n=nnx.BatchStat(0))

    @nnx.scan(in_axes=nnx.Carry, out_axes=nnx.Carry, length=3)
    def loop(foo: Foo):
      foo.n += 1
      return foo

    foo2 = loop(foo)

    self.assertIs(foo2, foo)
    self.assertEqual(foo.n.value, 3)

  def test_all_carry_one_argument_error(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      n: nnx.BatchStat[int]

    foo = Foo(n=nnx.BatchStat(0))

    @nnx.scan(in_axes=nnx.Carry, out_axes=nnx.Carry, length=3)
    def loop(foo: Foo, x):
      ...

    with self.assertRaisesRegex(
      ValueError,
      'When in_axes=Carry, the function must take exactly one argument',
    ):
      loop(foo, 0)

  def test_all_carry_new_reference_error(self):
    class Foo(nnx.Module):
      def __init__(self, n: nnx.BatchStat[int]):
        self.n = n

    xs = jnp.arange(3)
    foo = Foo(n=nnx.BatchStat(0))

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def loop(foo: Foo, x):
      x = x + 1
      foo = Foo(nnx.BatchStat(foo.n.value + 1))  # new reference
      return foo, x

    with self.assertRaisesRegex(
      ValueError,
      'Carry references must be the same between iterations',
    ):
      loop(foo, xs)

  def test_all_scan(self):
    class Foo(nnx.Module):
      def __init__(self, n: nnx.BatchStat[jax.Array]):
        self.n = n

    xs = jnp.arange(3)
    foo = Foo(n=nnx.BatchStat(jnp.arange(3)))

    @nnx.scan(in_axes=0, out_axes=0)
    def loop(foo: Foo, x):
      x = x + 1
      foo.n += 1
      return x

    ys = loop(foo, xs)

    np.testing.assert_allclose(ys, jnp.arange(1, 4))
    np.testing.assert_allclose(foo.n.value, jnp.arange(1, 4))

  def test_all_broadcast(self):
    class Foo(nnx.Module):
      def __init__(self, n: nnx.BatchStat[int]):
        self.n = n

    xs = jnp.array(1)
    foo = Foo(n=nnx.BatchStat(2))

    @nnx.scan(in_axes=None, out_axes=0, length=4)
    def loop(foo: Foo, x):
      return x + foo.n

    ys = loop(foo, xs)

    np.testing.assert_allclose(ys, 3)
    self.assertEqual(ys.shape, (4,))

  def test_input_output_carry_mismatch_error(self):
    with self.assertRaisesRegex(
      ValueError,
      'If one of in_axes or out_axes has Carry, the other must also have Carry',
    ):

      @nnx.scan(in_axes=0, out_axes=(nnx.Carry, 0))
      def loop(a, b):
        ...

    with self.assertRaisesRegex(
      ValueError,
      'If one of in_axes or out_axes has Carry, the other must also have Carry',
    ):

      @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=0)
      def loop(a, b):
        ...

  def test_double_carry_error(self):
    with self.assertRaisesRegex(
      ValueError,
      'Found multiple Carry definitions',
    ):

      @nnx.scan(in_axes=(nnx.Carry, nnx.Carry))
      def loop(a, b):
        ...

  def test_broadcast_in_output_error(self):
    with self.assertRaisesRegex(
      ValueError,
      'Cannot broadcast output state',
    ):

      @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, None))
      def loop(a, b):
        ...

    with self.assertRaisesRegex(
      ValueError,
      'Cannot broadcast output state. Got StateAxes',
    ):

      @nnx.scan(
        in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, nnx.StateAxes({...: None}))
      )
      def loop(a, b):
        ...

  def test_basic_combinator(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    MLP = nnx.Scan.constructor(
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

  def test_only_carry(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.c = nnx.BatchStat(jnp.array(0))

    @nnx.scan(in_axes=(nnx.Carry,), length=5)
    def loop(foo: Foo) -> tuple[Foo, jax.Array]:
      foo.c.value += 1
      return foo, foo.c.value

    foo = Foo()
    foo2, cs = loop(foo)
    self.assertIs(foo2, foo)
    np.testing.assert_allclose(cs, jnp.arange(1, 6))

  def test_no_scan_output(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Scan.constructor(
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
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), out_axes=(nnx.Carry, 1, 2))
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, x, x

    module = MLP(rngs=nnx.Rngs(0))

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    c, y1, y2 = module(x)

    assert c.shape == (1, 3)
    assert y1.shape == (1, 5, 3)
    assert y2.shape == (1, 3, 5)

  def test_in_axes_simple(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.vmap(in_axes=(state_axes, 0))
      def __init__(self, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), out_axes=nnx.Carry)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    key = jax.random.split(jax.random.key(0), 5)
    module = MLP(key=key)

    x = jnp.ones((1, 3))
    y = module(x)

    assert y.shape == (1, 3)

  def test_in_axes(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes))
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry, 0))
      def __call__(
          self, x: jax.Array, a: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape
        x = x + a
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    y, out = module(x, a)

    assert y.shape == (1, 3)
    assert out is None

  def test_in_axes_combinator(self):
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

    MLP = nnx.Scan.constructor(
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
    test = self
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes))
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry, 0, None))
      def __call__(
          self, x: jax.Array, a: jax.Array, b: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        test.assertEqual(x.shape, a.shape)
        test.assertEqual(x.shape, b.shape)
        x = x + a + b
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))

    self.assertEqual(module.linear.kernel.value.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.value.shape, (5, 3))
    self.assertEqual(module.node.value.shape, (2,))

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    b = jnp.ones((1, 3))
    y, out = module(x, a, b)

    self.assertEqual(y.shape, (1, 3))
    self.assertIsNone(out)

  def test_in_axes_broadcast_combinator(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(
        self, x: jax.Array, a: jax.Array, b: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape and x.shape == b.shape
        x = x + a + b
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    MLP = nnx.Scan.constructor(
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
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes))
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.split_rngs(splits=5)
      @nnx.scan(in_axes=(state_axes, nnx.Carry))
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = module(x)

    assert y.shape == (1, 3)

  def test_complex_combinator(self):
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

    MLP = nnx.Scan.constructor(
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
    state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5, only='params')
      @nnx.vmap(in_axes=(state_axes, state_axes))
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.split_rngs(splits=5, only='params')
      @nnx.scan(in_axes=(state_axes, nnx.Carry))
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(params=0, dropout=1))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = module(x)

    assert y.shape == (1, 3)

  def test_complex_broadcast_dropout_combinator(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Scan.constructor(
        Block,
        state_axes={nnx.Param: 0},
        length=5,
        # params is split, dropout is broadcast
        split_rngs=['params'],
        scan_output=False,
    )

    module = MLP(nnx.Rngs(params=0, dropout=1))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert module.scan_module.linear.bias.value.shape == (5, 3)
    assert module.scan_module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y = module(x)

    assert y.shape == (1, 3)

  def test_complex_decorator(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class Block(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5)
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.split_rngs(splits=5)
      @nnx.scan(in_axes=(state_axes, nnx.Carry))
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = Block(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.d == 3
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)
    assert module.node.value.shape == (2,)

    x = jnp.ones((1, 3))
    y, out = module(x)

    assert y.shape == (1, 3)
    assert out is None

  def test_scan_with_sharding(self):
    test = self
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})
    transform_metadata = {nnx.PARTITION_NAME: 'layers'}

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(
          in_axes=(state_axes, state_axes),
          transform_metadata=transform_metadata,
      )
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            3,
            3,
            kernel_init=nnx.with_metadata(
                nnx.initializers.lecun_normal(), sharding=('din', 'dout')
            ),
            bias_init=nnx.with_metadata(
                nnx.initializers.zeros_init(), sharding=('dout',)
            ),
            rngs=rngs,
        )

      @nnx.scan(
          in_axes=(state_axes, nnx.Carry), transform_metadata=transform_metadata
      )
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        # test sharding layer axes is not present inside scan
        test.assertEqual(self.linear.kernel.shape, (3, 3))
        test.assertEqual(self.linear.kernel.sharding, ('din', 'dout'))
        test.assertEqual(self.linear.bias.shape, (3,))
        test.assertEqual(self.linear.bias.sharding, ('dout',))
        return x, None

    m = MLP(rngs=nnx.Rngs(0))

    # test sharding layers axes is set
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.sharding, ('layers', 'dout'))

    x = jnp.ones((1, 3))
    y, out = m(x)

    # test sharding axes is preserved
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.sharding, ('layers', 'dout'))

  def test_scan_with_sharding_decorator(self):
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
        assert state.kernel.value.shape == (3, 3)  # type: ignore
        assert state.kernel.sharding == ('din', 'dout')  # type: ignore
        assert state.bias.value.shape == (3,)  # type: ignore
        assert state.bias.sharding == ('dout',)  # type: ignore

        return x, None

    MLP = nnx.Scan.constructor(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
      transform_metadata={nnx.PARTITION_NAME: 'layers'},
    )

    m = MLP(rngs=nnx.Rngs(0))

    # test sharding layers axes is set
    state = nnx.state(m)
    assert state.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert state.scan_module.linear.kernel.sharding == ('layers', 'din', 'dout')
    assert state.scan_module.linear.bias.value.shape == (5, 3)
    assert state.scan_module.linear.bias.sharding == ('layers', 'dout')

    x = jnp.ones((1, 3))
    y, out = m(x, None)

    # test sharding axes is preserved
    state = nnx.state(m)
    assert state.scan_module.linear.kernel.value.shape == (5, 3, 3)
    assert state.scan_module.linear.kernel.sharding == ('layers', 'din', 'dout')
    assert state.scan_module.linear.bias.value.shape == (5, 3)
    assert state.scan_module.linear.bias.sharding == ('layers', 'dout')

  def test_type_error_less_than_one_args(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self):
        return None, None

    MLP = nnx.Scan.constructor(
      Block,
      state_axes={nnx.Param: 0},
      length=5,
    )

    mlp = MLP(rngs=nnx.Rngs(0))

    with self.assertRaisesRegex(
        TypeError, 'Expected at least 2 positional argument'
    ):
      mlp()

  def test_cache_tracing_simple(self):
    n = 0
    x = jnp.arange(5)
    count = jnp.array(0)

    @nnx.scan
    def f(count, x):
      nonlocal n
      n += 1
      return count + 1, x**2

    count, y = f(count, x)
    assert n == 1
    assert count == 5
    np.testing.assert_allclose(y, x**2)

    count, y = f(count, x)
    assert n == 1
    assert count == 10

  def test_cache_tracing_object(self):
    n = 0
    x = jnp.arange(5)
    count = jnp.array(0)

    class Foo(nnx.Object):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(axis_size=5)
      def __init__(self, rngs: nnx.Rngs):
        self.x = nnx.Param(jax.random.normal(rngs(), shape=(3,)))

    foo = Foo(rngs=nnx.Rngs(0))
    assert foo.x.value.shape == (5, 3)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0))
    def f(count, x, foo):
      nonlocal n
      n += 1
      assert foo.x.value.shape == (3,)
      return count + 1, x**2

    count, y = f(count, x, foo)
    assert n == 1
    assert count == 5
    np.testing.assert_allclose(y, x**2)

    count, y = f(count, x, foo)
    assert n == 1
    assert count == 10

  def test_scan_broadcast_keys(self):
    params_key = jax.random.split(jax.random.key(0), 3)
    rngs = nnx.Rngs(params=params_key, dropout=1)
    state_axes = nnx.StateAxes({'params': 0, ...: None})

    @nnx.scan(in_axes=(nnx.Carry, state_axes), length=3)
    def f(_, rngs: nnx.Rngs):
      param_key = rngs.params()
      dropout_key = rngs.dropout()
      return (), (param_key, dropout_key)

    _, (param_keys, dropout_keys) = f((), rngs)

    assert jnp.not_equal(param_keys[0], param_keys[1])
    assert jnp.not_equal(param_keys[1], param_keys[2])
    assert jnp.equal(dropout_keys[0], dropout_keys[1])
    assert jnp.equal(dropout_keys[1], dropout_keys[2])

  def test_rnn_example(self):
    class RNNCell(nnx.Module):
      def __init__(self, input_size, hidden_size, rngs):
        self.linear = nnx.Linear(
          hidden_size + input_size, hidden_size, rngs=rngs
        )
        self.drop = nnx.Dropout(0.1, rngs=rngs)
        self.hidden_size = hidden_size

      def __call__(self, carry, x) -> tuple[jax.Array, jax.Array]:
        carry = self.drop(carry)  # recurrent dropout
        x = nnx.relu(self.linear(jnp.concatenate([carry, x], axis=-1)))
        return x, x

      def initial_state(self, batch_size: int):
        return jnp.zeros((batch_size, self.hidden_size))

    cell = RNNCell(20, 20, nnx.Rngs(params=0, dropout=1))

    state_axes = nnx.StateAxes({'dropout': None, ...: nnx.Carry})

    def rnn_forward(cell: RNNCell, x: jax.Array):
      carry = cell.initial_state(x.shape[0])

      @nnx.scan(in_axes=(state_axes, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
      def unroll(cell: RNNCell, carry, x) -> tuple[jax.Array, jax.Array]:
        return cell(carry, x)

      _, y = unroll(cell, carry, x)
      return y

    x = jnp.ones((16, 10, 20))
    y = rnn_forward(cell, x)


class TestRemat(absltest.TestCase):
  def test_basic_remat(self):
    RematLinear = nnx.Remat.constructor(nnx.Linear)

    module = RematLinear(2, 3, rngs=nnx.Rngs(0))

    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)

  def test_remat_decorator(self):
    class RematLinear(nnx.Module):

      @nnx.remat(static_argnums=(1, 2))
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

    RematLinear = nnx.Remat.constructor(LinearBlock)

    ScanRematLinear = nnx.Scan.constructor(
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
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class ScanLinear(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      @nnx.scan(in_axes=(state_axes, nnx.Carry))
      @nnx.remat
      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        return x, None

    m = ScanLinear(nnx.Rngs(0))

    assert m.linear.kernel.value.shape == (5, 3, 3)
    assert m.linear.bias.value.shape == (5, 3)

    y, _ = m(jnp.ones((1, 3)))
    assert y.shape == (1, 3)


class TestVmap(absltest.TestCase):
  def test_basic(self):

    @partial(nnx.vmap, in_axes=0, out_axes=0, axis_size=5)
    def create_block(rngs: nnx.Rngs):
      return nnx.Linear(2, 3, rngs=rngs)

    rngs = nnx.Rngs(0)
    backups = nnx.split_rngs(rngs, splits=5)

    block = create_block(rngs)
    nnx.restore_rngs(backups)

    self.assertEqual(block.kernel.value.shape, (5, 2, 3))
    self.assertEqual(rngs.default.count.value, 1)

    @partial(nnx.vmap, in_axes=(0, 1), out_axes=1)
    def forward(block: nnx.Linear, x):
      self.assertEqual(block.kernel.value.shape, (2, 3))
      self.assertEqual(block.bias.value.shape, (3,))
      self.assertEqual(x.shape, (2,))
      return block(x)

    x = jax.random.uniform(rngs(), (2, 5))
    y = forward(block, x)

    self.assertEqual(y.shape, (3, 5))

  def test_state_axes(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    @nnx.vmap(
        in_axes=0,
        out_axes=nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None}),
    )
    def create_block(rngs: nnx.Rngs):
      rngs = nnx.clone(rngs)
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value

    backups = nnx.split_rngs(rngs, splits=5)
    module = create_block(rngs)
    nnx.restore_rngs(backups)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
        module.linear.kernel.value[0],
        module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(
        in_axes=(nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None}), 0),
    )
    def forward_block(module, x):
      return module(x)

    backups = nnx.split_rngs(rngs, splits=5)
    y = forward_block(module, x)
    nnx.restore_rngs(backups)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_split_rngs_context_manager(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value

    with nnx.split_rngs(rngs, splits=5):
      module = create_block(rngs)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
        module.linear.kernel.value[0],
        module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(in_axes=(state_axes, 0))
    def forward_block(module, x):
      return module(x)

    with nnx.split_rngs(module, splits=5):
      y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    with nnx.split_rngs(module, splits=5):
      y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_split_rngs_decorator(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    @nnx.split_rngs(splits=5)
    @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value

    module = create_block(rngs)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert not jnp.allclose(
      module.linear.kernel.value[0],
      module.linear.kernel.value[1],
    )
    assert module.linear.kernel.value.shape == (5, 3, 3)
    assert module.linear.bias.value.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.split_rngs(splits=5)
    @nnx.vmap(in_axes=(state_axes, 0))
    def forward_block(module, x):
      self.assertEqual(x.shape, (1, 3))
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_state_axes_simple(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    state_axes = nnx.StateAxes({(nnx.BatchStat, 'dropout'): 0, ...: None})

    @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(params=0, dropout=1)
    nnx.split_rngs(rngs, splits=5, only='dropout')

    module = create_block(rngs)

    assert module.linear.kernel.value.shape == (2, 3)
    assert module.bn.scale.value.shape == (3,)
    assert module.bn.mean.value.shape == (5, 3)

    @nnx.vmap(in_axes=(state_axes, 0), out_axes=0)
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  def test_split_rngs_decorator_simple(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    state_axes = nnx.StateAxes({(nnx.BatchStat, 'dropout'): 0, ...: None})

    @nnx.split_rngs(splits=5, only='dropout')
    @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(params=0, dropout=1)

    module = create_block(rngs)

    assert module.linear.kernel.value.shape == (2, 3)
    assert module.bn.scale.value.shape == (3,)
    assert module.bn.mean.value.shape == (5, 3)
    assert module.dropout.rngs is not None
    self.assertEqual(module.dropout.rngs.params.key.shape, ())
    self.assertEqual(module.dropout.rngs.dropout.key.shape, ())

    @nnx.split_rngs(splits=5, only='dropout')
    @nnx.vmap(in_axes=(state_axes, 0), out_axes=0)
    def forward_block(module: Block, x):
      assert module.dropout.rngs is not None
      self.assertEqual(module.dropout.rngs.params.key.shape, ())
      self.assertEqual(module.dropout.rngs.dropout.key.shape, ())
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert module.dropout.rngs is not None
    self.assertEqual(module.dropout.rngs.params.key.shape, ())
    self.assertEqual(module.dropout.rngs.dropout.key.shape, ())
    assert y.shape == (5, 1, 3)

  def test_state_axes_super_simple(self):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    @nnx.vmap(in_axes=0, out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    nnx.split_rngs(rngs, splits=5)

    module = create_block(rngs)

    assert module.linear.kernel.value.shape == (5, 2, 3)
    assert module.bn.scale.value.shape == (5, 3)
    assert module.bn.mean.value.shape == (5, 3)

    @nnx.vmap(in_axes=(0, 0), out_axes=0)
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

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

    state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

    @nnx.split_rngs(splits=5)
    @partial(nnx.vmap, in_axes=(state_axes, 0), out_axes=0)
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

  def test_consistent_aliasing_inputs(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.a = nnx.Param(jnp.zeros((5, 5)))

    m = Foo()

    @nnx.vmap(in_axes=(0, 1))
    def f(m1, m2):
      pass

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      f(m, m)

  def test_consistent_aliasing_input_output(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.a = nnx.Param(jnp.zeros((2, 3)))

    m = Foo()

    @partial(nnx.vmap, in_axes=0, out_axes=1)
    def f(m):
      return m

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      m2 = f(m)

  def test_consistent_aliasing_shared(self):
    class Shared(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.zeros((3, 3)))

    class Foo(nnx.Module):

      def __init__(self, shared: Shared):
        self.a = shared

    shared = Shared()
    m1 = Foo(shared)
    m2 = Foo(shared)

    @nnx.vmap(in_axes=(0, 1))
    def f(m1, m2):
      pass

    with self.assertRaisesRegex(
      ValueError,
      r'Inconsistent aliasing detected([\s\S]*)Param([\s\S]*)a:'
      r' 0([\s\S]*)a: 1',
    ):
      f(m1, m2)

  def test_equivalent_state_axes_mapping(self):
    m = nnx.Linear(3, 3, rngs=nnx.Rngs(0))

    sa1 = nnx.StateAxes({...: 0})
    sa2 = nnx.StateAxes({nnx.Param: 0})

    @nnx.vmap(in_axes=(0, sa1, sa2))
    def f(m1, m2, m3):
      pass

    f(m, m, m)

  def test_equivalent_state_sharding_mapping(self):
    m = nnx.Linear(3, 3, rngs=nnx.Rngs(0))

    mesh = jax.sharding.Mesh(jax.devices(), ('mp',))
    sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('mp')
    )

    sa1 = nnx.StateSharding({...: sharding})
    sa2 = nnx.StateSharding({nnx.Param: sharding})

    @nnx.jit(in_shardings=(sharding, sa1, sa2))
    def f(m1, m2, m3):
      pass

    f(m, m, m)

  @absltest.skip('Enable once jax#19586 resolved')
  def test_captured_module_in_return_error(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.a = jnp.zeros((5, 5))

    m = Foo()

    @nnx.vmap(in_axes=0, out_axes=0)
    def f(x):
      return x, m

    with self.assertRaisesRegex(
        ValueError,
        r'Trying to extract graph node from different trace level.*Foo',
    ):
      x = jnp.zeros((5,))
      f(x)

  def test_vmap_and_cond_passthrough(self):
    class Broadcast(nnx.Variable[nnx.A]):
      ...

    class Vectorized(nnx.Variable[nnx.A]):
      ...

    class Env(nnx.Module):

      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.vmap(in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),))
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(is_even, increment, no_nothing, env)

    f(env)

    np.testing.assert_array_equal(env.step.value, [1, 0, 1, 0, 1, 0, 1, 0])

  def test_vmap_and_cond_passthrough_error(self):
    class Broadcast(nnx.Variable[nnx.A]):
      ...

    class Vectorized(nnx.Variable[nnx.A]):
      ...

    class Env(nnx.Module):

      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.vmap(in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),))
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step += 1
        env.broadcast += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(is_even, increment, no_nothing, env)

    with self.assertRaisesRegex(
        ValueError,
        r"at vmap.*'broadcast'.*got axis spec None but output was batched on"
        r' axis 0',
    ):
      f(env)

  def test_example(self):
    class Model(nnx.Module):

      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    @nnx.vmap(in_axes=0, out_axes=0)
    def initialize_ensemble(key):
      rngs = nnx.Rngs(key)
      return Model(2, 3, rngs=rngs)

    keys = jax.random.split(jax.random.key(0), 5)
    ensemble = initialize_ensemble(keys)

    self.assertEqual(ensemble.linear.kernel.shape, (5, 2, 3))

    @nnx.vmap(in_axes=(0, None), out_axes=0)
    def forward(model, x):
      return model(x)

    x = jnp.ones((4, 2))
    y = forward(ensemble, x)
    self.assertEqual(y.shape, (5, 4, 3))

  def test_example_with_vectorization(self):
    class LinearEnsemble(nnx.Module):

      def __init__(self, num, rngs):
        self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))

    model = LinearEnsemble(5, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(0, None), out_axes=0)
    def forward(model, x):
      self.assertEqual(model.w.shape, (2, 3))
      return jnp.dot(x, model.w.value)

    x = jnp.ones((4, 2))
    y = forward(model, x)

    self.assertEqual(y.shape, (5, 4, 3))

  def test_metadata(self):
    @nnx.vmap(
      in_axes=(None,),
      out_axes=0,
      axis_size=5,
      transform_metadata={nnx.spmd.PARTITION_NAME: 'c'},
    )
    def create_block(rngs: nnx.Rngs):
      return nnx.Linear(
        16,
        32,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
          nnx.initializers.lecun_normal(), ('a', 'b')
        ),
      )

    m = create_block(nnx.Rngs(0))
    self.assertEqual(m.kernel.value.shape, (5, 16, 32))
    self.assertEqual(m.kernel.sharding, ('c', 'a', 'b'))

  def test_state_axes_from_state(self):
    class Model(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)

    model = Model(2, 3, rngs=nnx.Rngs(0))
    state = nnx.state(model)

    state['linear']['kernel'] = 0
    state['linear']['bias'] = 1
    state['bn']['scale'] = 0
    state['bn']['mean'] = 1
    state['bn']['var'] = 0
    state['bn']['bias'] = None

    state_axes = nnx.StateAxes(state)

    self.assertEqual(state_axes.map_prefix(('linear', 'kernel'), None), 0)
    self.assertEqual(state_axes.map_prefix(('linear', 'bias'), None), 1)
    self.assertEqual(state_axes.map_prefix(('bn', 'scale'), None), 0)
    self.assertEqual(state_axes.map_prefix(('bn', 'mean'), None), 1)
    self.assertEqual(state_axes.map_prefix(('bn', 'var'), None), 0)
    self.assertEqual(state_axes.map_prefix(('bn', 'bias'), None), None)

    @nnx.vmap(out_axes=state_axes, axis_size=5)
    def create_block():
      return Model(2, 3, rngs=nnx.Rngs(0))

    model = create_block()

    self.assertEqual(model.linear.kernel.shape, (5, 2, 3))
    self.assertEqual(model.linear.bias.shape, (3, 5))
    self.assertEqual(model.bn.scale.shape, (5, 3))
    self.assertEqual(model.bn.mean.shape, (3, 5))
    self.assertEqual(model.bn.var.shape, (5, 3))
    self.assertEqual(model.bn.bias.shape, (3,))


class TestPmap(absltest.TestCase):

  def test_basic_single(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 10, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.elu(x)
        x = self.dropout(x)
        return x

    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    @nnx.split_rngs(splits=1)
    @nnx.pmap(in_axes=(state_axes,), out_axes=state_axes, axis_size=1)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = create_block(rngs)

    assert rngs.default.count.value == 1
    assert rngs.default.key.value == initial_key
    assert module.linear.kernel.value.shape == (1, 3, 10)
    assert module.linear.bias.value.shape == (1, 10)

    x = jnp.ones((1, 1, 3))

    @nnx.split_rngs(splits=1)
    @nnx.pmap(in_axes=(state_axes, 0), axis_size=1)
    def forward_block(module, x):
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (1, 1, 10)
    assert rngs.default.count.value == 2
    assert rngs.default.key.value == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  def test_basic_demo_single(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    @nnx.split_rngs(splits=1)
    @nnx.pmap(axis_size=1)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @nnx.split_rngs(splits=1)
    @nnx.pmap(axis_size=1)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    assert rngs.default.count.value == 1
    assert module.linear.kernel.value.shape == (1, 3, 3)
    assert module.linear.bias.value.shape == (1, 3)

    x = jnp.ones((1, 10, 3))

    y = forward_block(module, x)

    assert y.shape == (1, 10, 3)
    assert rngs.default.count.value == 2

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

  def test_replicate_single(self):
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

    state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

    @nnx.split_rngs(splits=1)
    @partial(nnx.pmap, in_axes=(state_axes, 0), out_axes=0, axis_size=1)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key.value
    module = create_block(rngs)

    assert rngs.default.count.value == 2
    assert module.linear.kernel.value.shape == (din, dout)
    assert module.linear.bias.value.shape == (dout,)

    x = jnp.ones((1, 5, din))

    y = forward_block(module, x)

    assert y.shape == (1, 5, dout)
    assert rngs.default.count.value == 3

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

    assert rngs.default.key.value == initial_key

  def test_combinator_single(self):
    class Block(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    MLP = nnx.Pmap.constructor(Block, state_axes={nnx.Param: 0}, axis_size=1)

    module = MLP(rngs=nnx.Rngs(0))

    assert module.pmap_module.linear.kernel.value.shape == (1, 3, 3)
    assert module.pmap_module.linear.bias.value.shape == (1, 3)

    x = jnp.ones((1, 5, 3))
    y = module(x)

    assert y.shape == (1, 5, 3)


class TestCond(absltest.TestCase):
  def test_basic(self):
    class TimeStep(tp.NamedTuple):
      step: nnx.Variable[jax.Array]
      reward: nnx.Variable[jax.Array]

      @staticmethod
      def zero():
        return TimeStep(
            step=nnx.Variable(jnp.array(0)), reward=nnx.Variable(jnp.array(0.0))
        )

    @dataclasses.dataclass
    class Foo(nnx.Object):
      timestep: TimeStep

      def __hash__(self):
        return id(self)

      def update(self):
        def reward_2(self: Foo):
          self.timestep = TimeStep(
              step=nnx.Variable(self.timestep.step + 1),
              reward=nnx.Variable(jnp.array(2.0)),
          )

        def reward_0(self: Foo):
          self.timestep = TimeStep(
              step=nnx.Variable(self.timestep.step + 1),
              reward=nnx.Variable(jnp.array(0.0)),
          )

        nnx.cond(self.timestep.step % 2 == 0, reward_2, reward_0, self)

    foo = Foo(timestep=TimeStep.zero())
    foo.update()
    self.assertEqual(foo.timestep.step.value, 1)
    self.assertEqual(foo.timestep.reward.value, 2.0)
    foo.update()
    self.assertEqual(foo.timestep.step.value, 2)
    self.assertEqual(foo.timestep.reward.value, 0.0)
    foo.update()
    self.assertEqual(foo.timestep.step.value, 3)
    self.assertEqual(foo.timestep.reward.value, 2.0)
    foo.update()
    self.assertEqual(foo.timestep.step.value, 4)
    self.assertEqual(foo.timestep.reward.value, 0.0)

  def test_cond_and_vmap(self):

    class Env(nnx.Object):

      def __init__(self):
        self.index = nnx.Variable(jnp.arange(8))
        self.step = nnx.Variable(jnp.zeros((8,), jnp.uint32))

    env = Env()
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(0, None), out_axes=None)
    def f(env: Env, model: nnx.Linear):
      self.assertEqual(env.index.shape, ())

      def increment(env: Env):
        env.step += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(is_even, increment, no_nothing, env)

    f(env, model)

    np.testing.assert_array_equal(
        env.step.value, np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint32)
    )


class TestSwitch(absltest.TestCase):
  def test_basic(self):
    class RoundTable(nnx.Module):
      def __init__(self):
        self.next_index = 0
        self.linear = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
        self.linear.kernel.value = jnp.identity(10)
        self.rounds_count = nnx.Variable(jnp.array(0))

      def __call__(self, x):
        def fn0(m, x):
          m.rounds_count += 1
          return m.linear(x)
        def fn1(m, x):
          return m.linear(x) * 2
        def fn2(m, x):
          m.linear.kernel.value = jnp.zeros((10, 10))
          return m.linear(x)

        # y = nnx.cond(self.next_index.value == 0, fn0, fn1, self, x)
        y = nnx.switch(self.next_index, (fn0, fn1, fn2), self, x)
        self.next_index = (self.next_index + 1) % 3
        return y

    model = RoundTable()
    x = jnp.ones((10,))
    np.testing.assert_array_equal(model(x), x)
    assert model.rounds_count.value == 1
    assert model.next_index == 1
    np.testing.assert_array_equal(model(x), x * 2)
    assert model.rounds_count.value == 1
    assert model.next_index == 2
    np.testing.assert_array_equal(model(x), jnp.zeros((10,)))
    assert model.rounds_count.value == 1
    assert model.next_index == 0
    np.testing.assert_array_equal(model(x), jnp.zeros((10,)))
    assert model.rounds_count.value == 2
    assert model.next_index == 1


class TestWhileLoop(absltest.TestCase):
  def test_basic(self):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel.value = jnp.identity(10) * 2
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0))
    np.testing.assert_array_equal(y, x * 8)

  def test_multiple_objects(self):
    def fwd_fn(input):
      m1, (w2,), x, c = input
      y = m1(x) @ w2
      return m1, (w2,), y, c - 1.0

    m1 = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    m1.kernel.value = jnp.identity(10) * 2
    w2 = nnx.Variable(jnp.identity(10) * 0.5)
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (m1, (w2,), x, 3.0))
    np.testing.assert_allclose(y, x)

  def test_nested_module(self):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel.value = jnp.identity(10) * 2
    module = nnx.Sequential(module)
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0))
    np.testing.assert_array_equal(y, x * 8)


  def test_shared_module(self):
    m1 = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(10, 10, use_bias=False, rngs=nnx.Rngs(0))
    m2.kernel = m1.kernel
    module = nnx.Sequential(m1, m2)
    self.assertLen(jax.tree.leaves(nnx.state(module)), 2)  # only m1 params

    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      m.layers[0].kernel.value = jnp.zeros_like(m.layers[0].kernel.value)
      return m, y, c - 1.0

    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))
    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 2.0))
    self.assertLen(jax.tree.leaves(nnx.state(module)), 2)  # only m1 params
    np.testing.assert_array_equal(m1.kernel.value, jnp.zeros((10, 10,)))
    np.testing.assert_array_equal(m2.kernel.value, jnp.zeros((10, 10,)))
    np.testing.assert_array_equal(y, jnp.zeros((10,)))


  def test_value_changed(self):
    def fwd_fn(input):
      m, x, c = input
      m.kernel.value = jnp.zeros_like(m.kernel.value)
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0))
    np.testing.assert_array_equal(module.kernel.value, jnp.zeros((10, 10,)))
    np.testing.assert_array_equal(y, jnp.zeros((10,)))


  def test_ref_changed(self):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      m.kernel = nnx.Param(jnp.zeros_like(m.kernel.value))
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    with self.assertRaises(ValueError):
      _, y, _ = nnx.while_loop(
        lambda input: input[-1] > 0, fwd_fn, (module, x, 2.0))


  def test_structure_changed(self):
    def fwd_fn(input):
      m, x, c = input
      m = nnx.Linear(10, 10, use_bias=False, rngs=nnx.Rngs(1))
      m.kernel.value = jnp.identity(10) * 2
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, use_bias=True, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    with self.assertRaises(ValueError):
      _, y, _ = nnx.while_loop(
        lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0))

  def test_repeated_object(self):
    m = nnx.Linear(10, 10, rngs=nnx.Rngs(0))

    def body_fn(val):
      count, m, _ = val
      return count + 1, m, m

    count, m, _ = nnx.while_loop(
      lambda val: val[0] < 2,
      body_fn,
      (0, m, m),
    )

  def test_fori_loop_basic(self):
    def fwd_fn(i, input):
      m, x = input
      m.kernel.value = jnp.identity(10) * i
      return m, m(x)

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (10,))

    _, y = nnx.fori_loop(2, 4, fwd_fn, (module, x))
    np.testing.assert_array_equal(y, x * 2 * 3)

  def test_fori_loop_with_sharing(self):
    class A(nnx.Object):
      def __init__(self):
        self.params = nnx.Param(jnp.zeros((10,), dtype=int))

    class B(nnx.Object):
      def __init__(self, a: A):
        self.a = a

    class C(nnx.Object):
      def __init__(self, a: A):
        self.a = a

    class D(nnx.Object):
      def __init__(self):
        self.a = A()
        self.b = B(self.a)
        self.c = C(self.a)

    def increment(_, d: D) -> D:
      d.a.params += 1
      return d

    @nnx.jit
    def rollout(d: D):
      nnx.fori_loop(0, 10, increment, d)

    d = D()
    rollout(d)

    np.testing.assert_array_equal(
      d.a.params.value, np.full((10,), 10, dtype=int)
    )

  def test_loops_multiple_modules(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.param = nnx.Param(jnp.zeros((1,)))
      def __call__(self, x):
        return self.param

    def loop_fn(inputs):
      return inputs
    while_loop_fn = lambda inputs: (*loop_fn(inputs[:-1]), inputs[-1]-1)
    fori_loop_fn = lambda i, inputs: loop_fn(inputs)
    a = Foo()
    b = Foo()
    nnx.while_loop(lambda input: input[-1] > 0, while_loop_fn, (a, b, 2))
    nnx.fori_loop(0, 2, fori_loop_fn, (a, b))


class TestSplitMergeInputs(absltest.TestCase):
  def test_split_inputs(self):
    class StatefulLinear(nnx.Linear):
      def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        super().__init__(din, dout, rngs=rngs)
        self.counter = nnx.BatchStat(jnp.array(0, jnp.uint32))

      def __call__(self, x):
        self.counter += 1
        return super().__call__(x)

    model = StatefulLinear(3, 4, rngs=nnx.Rngs(0))

    @general.split_inputs
    @jax.jit
    @general.merge_inputs
    def forward(model, x):
      return model(x)

    x = jnp.ones((2, 3))
    y = forward(model, x)

    self.assertEqual(model.counter.value, 1)

  def test_split_inputs_cond(self):
    class Counter(nnx.Linear):
      def __init__(self):
        self.count = nnx.BatchStat(jnp.array(0, jnp.uint32))

      def increment(self):
        self.count += 1

    counter = Counter()

    @general.merge_inputs
    def increment(counter: Counter):
      counter.increment()

    @general.merge_inputs
    def no_nothing(counter: Counter):
      pass

    general.split_inputs(jax.lax.cond)(True, increment, no_nothing, counter)

    self.assertEqual(counter.count.value, 1)

    general.split_inputs(jax.lax.cond)(False, increment, no_nothing, counter)

    self.assertEqual(counter.count.value, 1)

  def test_split_inputs_vmap(self):
    class EnvState(nnx.Variable[nnx.A]):
      pass

    class Env(nnx.Object):
      def __init__(self):
        self.index = EnvState(jnp.arange(8))
        self.step = EnvState(jnp.zeros((8,), jnp.uint32))

    env = Env()
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    # internally merge_inputs returns (args, out)
    in_axes = (0, None)
    out_axes = (in_axes, None)

    @general.split_inputs
    @partial(jax.vmap, in_axes=in_axes, out_axes=out_axes)
    @general.merge_inputs
    def f(env: Env, model: nnx.Linear):
      self.assertEqual(env.index.value.shape, ())

      @general.merge_inputs
      def increment(env: Env):
        env.step.value += 1

      @general.merge_inputs
      def no_nothing(env: Env):
        pass

      is_even = env.index.value % 2 == 0
      general.split_inputs(jax.lax.cond)(is_even, increment, no_nothing, env)

    f(env, model)

    np.testing.assert_array_equal(
      env.step.value, np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint32)
    )

class TestCheckify(absltest.TestCase):
  def test_basic(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: nnx.Param

      def __hash__(self):
        return id(self)

    @nnx.jit
    def f(m):
      y = jnp.sin(m.a.value)  # error
      return m.a + y

    m = Foo(a=nnx.Param(jnp.inf))
    err, out = nnx.checkify(f, errors=checkify.float_checks)(m)

    with self.assertRaisesRegex(ValueError, 'nan generated by primitive: sin'):
      err.throw()

if __name__ == '__main__':
  absltest.main()

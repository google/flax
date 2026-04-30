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

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import dataclasses
from functools import partial
import typing as tp

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.nnx.transforms import general
import jax
from jax.experimental import checkify, mesh_utils
import jax.numpy as jnp
import numpy as np
import optax


class TestJIT(parameterized.TestCase):

  def test_jit_graph_updates(self):
    m = nnx.Dict(a=nnx.Param(1))

    @nnx.jit(graph=True, graph_updates=True)
    def g(m: nnx.Dict):
      m.a = 2
      return 1.0

    out = g(m)

    self.assertEqual(m.a, 2)
    self.assertEqual(out, 1.0)

  def test_jit_graph_updates_functional(self):
    m = nnx.Dict(a=nnx.Param(1))

    @nnx.jit(graph=True, graph_updates=False)
    def g(m: nnx.Dict):
      m.a.set_value(2)
      return 1.0

    out = g(m)

    self.assertEqual(m.a, 2)
    self.assertEqual(out, 1.0)

  def test_prefix_graph_node_error(self):
    m = nnx.Dict(a=nnx.Param(1))

    with self.assertRaisesRegex(
      ValueError, 'Graph nodes are not allowed as prefixes'
    ):
      nnx.jit(lambda x: x, in_shardings=m, graph=True, graph_updates=True)

  def test_prefix_mapping_tree_mode_error(self):
    sharding = nnx.StateSharding({nnx.PathContains('a'): 1})

    with self.assertRaisesRegex(
      ValueError, 'cannot contain `StateSharding` objects'
    ):
      nnx.jit(
        lambda x: x, in_shardings=sharding, graph=False, graph_updates=False
      )

  def test_mutable_array_input_output(self):
    m = jax.new_ref(jnp.array(1.0))

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: jax.Ref):
      m[...] += 1.0
      m2 = jax.new_ref(jnp.array(10.0))
      return m2, m

    m2, m_out = f(m)

    self.assertEqual(m[...], 2.0)
    self.assertIs(m, m_out)
    self.assertIsInstance(m2, jax.Ref)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_simple_double_call(self, graph_mode, graph_updates):
    n = 0
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit(graph=graph_mode, graph_updates=graph_updates)
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
      @nnx.jit(static_argnums=(1, 2), graph=True, graph_updates=True)
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        nonlocal n
        n += 1

        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1
    assert m.w.shape == (2, 3)
    assert m.din == 2
    assert m.dout == 3
    assert isinstance(m.din, int)
    assert isinstance(m.dout, int)
    assert isinstance(m.w[...], jax.Array)

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1

  def test_jit_on_init_functional(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

    @nnx.jit(static_argnums=(0, 1), graph=True, graph_updates=False)
    def create_foo(din: int, dout: int, *, rngs: nnx.Rngs):
      nonlocal n
      n += 1
      return Foo(din, dout, rngs=rngs)

    m = create_foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1
    assert m.w.shape == (2, 3)
    assert m.din == 2
    assert m.dout == 3
    assert isinstance(m.din, int)
    assert isinstance(m.dout, int)
    assert isinstance(m.w[...], jax.Array)

    m = create_foo(2, 3, rngs=nnx.Rngs(0))
    assert n == 1

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_jit_on_call(self, graph_mode, graph_updates):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, shape=(din, dout)))
        self.din = din
        self.dout = dout

      @nnx.jit(graph=graph_mode, graph_updates=graph_updates)
      def __call__(self, x: jax.Array) -> jax.Array:
        nonlocal n
        n += 1
        return jnp.dot(x, self.w)

    m = Foo(2, 3, rngs=nnx.Rngs(0))
    assert m.w.shape == (2, 3)
    assert m.din == 2
    assert m.dout == 3
    assert isinstance(m.din, int)
    assert isinstance(m.dout, int)
    assert isinstance(m.w[...], jax.Array)

    y = m(jnp.ones((1, 2)))
    assert y.shape == (1, 3)
    assert n == 1
    y = m(jnp.ones((1, 2)))
    assert n == 1

  def test_graph_updates_unflatten(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.BatchNorm(2, rngs=rngs)

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: Foo):
      nonlocal n
      n += 1
      m.a, m.b = m.b, m.a  # type: ignore

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b
    a_kernel = a.kernel[...]
    a_bias = a.bias[...]
    b_scale = b.scale[...]
    b_bias = b.bias[...]
    b_mean = b.mean[...]
    b_var = b.var[...]

    f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.a, b)
    self.assertIs(m.b, a)
    np.testing.assert_allclose(a_kernel, a.kernel[...])
    np.testing.assert_allclose(a_bias, a.bias[...])
    np.testing.assert_allclose(b_scale, b.scale[...])
    np.testing.assert_allclose(b_bias, b.bias[...])
    np.testing.assert_allclose(b_mean, b.mean[...])
    np.testing.assert_allclose(b_var, b.var[...])

    f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.a, a)
    self.assertIs(m.b, b)

    f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.a, b)
    self.assertIs(m.b, a)

    f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.a, a)
    self.assertIs(m.b, b)

  def test_graph_updates_unflatten_functional(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.array(1))
        self.b = nnx.Param(jnp.array(2))

    @nnx.jit(graph=True, graph_updates=False)
    def f(m: Foo):
      nonlocal n
      n += 1
      a_val = m.a.get_value()
      b_val = m.b.get_value()
      m.a.set_value(b_val)
      m.b.set_value(a_val)

    m = Foo()
    a = m.a
    b = m.b

    f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.a, a)
    self.assertIs(m.b, b)
    self.assertEqual(m.a, 2)
    self.assertEqual(m.b, 1)

    f(m)

    self.assertEqual(n, 1)  # Should NOT retrace
    self.assertEqual(m.a, 1)
    self.assertEqual(m.b, 2)

    f(m)

    self.assertEqual(n, 1)
    self.assertEqual(m.a, 2)
    self.assertEqual(m.b, 1)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_jit_custom_vjp(self, graph_mode, graph_updates):
    @nnx.custom_vjp(graph=graph_mode, graph_updates=graph_updates)
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

    nnx_out = nnx.jit(f, graph=graph_mode, graph_updates=graph_updates)(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    jax_out = jax.jit(f)(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert (nnx_out == jax_out).all()

  def test_graph_updates_same_type(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.Linear(2, 2, rngs=rngs)

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: Foo):
      nonlocal n
      n += 1
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.a, b)
    self.assertIs(m.b, a)

    f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.a, a)
    self.assertIs(m.b, b)

  @parameterized.parameters(True, False)
  def test_graph_updates_same_type_functional(self, graph):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Variable(1.0)
        self.b = nnx.Variable(2.0)

    @nnx.jit(graph=graph, graph_updates=False)
    def f(m: Foo):
      nonlocal n
      n += 1
      # Functional swap using set_value
      a_val = m.a.get_value()
      b_val = m.b.get_value()
      m.a.set_value(b_val)
      m.b.set_value(a_val)

    m = Foo()

    f(m)

    self.assertEqual(n, 1)
    self.assertEqual(m.a, 2.0)
    self.assertEqual(m.b, 1.0)

    f(m)

    self.assertEqual(n, 1)
    self.assertEqual(m.a, 1.0)
    self.assertEqual(m.b, 2.0)

  @parameterized.parameters(True, False)
  def test_objects_in_pytree(self, graph_updates):
    n = 0

    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.Linear(2, 2, rngs=rngs)

    class FooDict(tp.TypedDict):
      foo: Foo

    @nnx.jit(graph=True, graph_updates=graph_updates)
    def f(tree: tuple[FooDict]):
      nonlocal n
      n += 1
      m = tree[0]['foo']
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    f(({'foo': m},))

    self.assertEqual(n, 1)
    if graph_updates:
      self.assertIs(m.a, b)
      self.assertIs(m.b, a)
    else:
      self.assertIs(m.a, a)
      self.assertIs(m.b, b)

    f(({'foo': m},))

    self.assertEqual(n, 1)
    self.assertIs(m.a, a)
    self.assertIs(m.b, b)

  def test_graph_updates_swap_variables(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.Param(2)

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: Foo):
      m.a, m.b = m.b, m.a

    m = Foo()
    a = m.a
    b = m.b

    f(m)

    self.assertIs(m.a, b)
    self.assertIs(m.b, a)

  def test_graph_updates_swap_variables_functional(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.Param(2)

    @nnx.jit(graph=True, graph_updates=False)
    def f(m: Foo):
      a_val = m.a.get_value()
      b_val = m.b.get_value()
      m.a.set_value(b_val)
      m.b.set_value(a_val)

    m = Foo()
    a = m.a
    b = m.b

    f(m)

    self.assertIs(m.a, a)  # References should NOT change in functional mode
    self.assertIs(m.b, b)
    self.assertEqual(m.a, 2)  # Values should be swapped
    self.assertEqual(m.b, 1)

  def test_graph_updates_add_self_reference(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.ref: tp.Optional[Foo] = nnx.data(None)  # type: ignore[name-error]

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: Foo):
      nonlocal n
      n += 1
      m.ref = m

    m = Foo()

    f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.ref, m)

    f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.ref, m)

    f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.ref, m)

  def test_graph_updates_ref_in_output(self):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.ref: tp.Optional[Foo] = nnx.data(None)  # type: ignore[name-error]

    @nnx.jit(graph=True, graph_updates=True)
    def f(m: Foo):
      nonlocal n
      n += 1
      m.ref = m
      return m

    m = Foo()

    m2 = f(m)

    self.assertEqual(n, 1)
    self.assertIs(m.ref, m)
    self.assertIs(m2, m)

    m2 = f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.ref, m)
    self.assertIs(m2, m)

    m2 = f(m)

    self.assertEqual(n, 2)
    self.assertIs(m.ref, m)
    self.assertIs(m2, m)

  def test_apply_shardings(self):
    n_devices = max(jax.local_device_count() // 2, 1)
    devices = mesh_utils.create_device_mesh(
      (n_devices, jax.local_device_count() // n_devices)
    )
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    def sharding(*args):
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

    state_sharding = nnx.StateSharding(
      {
        nnx.PathContains('kernel'): sharding('a', 'b'),
        nnx.PathContains('bias'): sharding('b'),
      }
    )

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    self.assertNotIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

    @nnx.jit(in_shardings=(state_sharding,), graph=True, graph_updates=True)
    def constrain_object(m):
      pass

    constrain_object(m)

    self.assertIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

  @parameterized.parameters(True, False)
  def test_apply_shardings_functional(self, graph):
    n_devices = max(jax.local_device_count() // 2, 1)
    devices = mesh_utils.create_device_mesh(
      (n_devices, jax.local_device_count() // n_devices)
    )
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    def sharding(*args):
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

    m_dummy = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    state_sharding = nnx.prefix(
      m_dummy,
      {
        nnx.PathContains('kernel'): sharding('a', 'b'),
        nnx.PathContains('bias'): sharding('b'),
      },
      graph=graph,
    )

    @nnx.jit(out_shardings=state_sharding, graph=graph, graph_updates=False)
    def constrain_object():
      m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
      return m

    m = constrain_object()

    self.assertIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

  def test_cache_args(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit(graph=True, graph_updates=True)
    def f(cached_m: nnx.Linear, m: nnx.Linear):
      self.assertIsNot(cached_m, m)
      self.assertIs(cached_m.kernel, m.kernel)
      self.assertIs(cached_m.bias, m.bias)
      return cached_m

    cached_f = nnx.compat.cached_partial(f, m)
    cached_m = cached_f(m)

    self.assertIsNot(m, cached_m)
    self.assertIs(m.kernel, cached_m.kernel)
    self.assertIs(m.bias, cached_m.bias)

    # test that cached m is reused
    cached_m2 = cached_f(m)
    self.assertIs(cached_m, cached_m2)

  @parameterized.parameters(True, False)
  def test_cache_args_functional(self, graph_mode):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      return m(x)

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    x = jnp.ones((1, 2))
    y = f_jit(x)
    self.assertEqual(y.shape, (1, 3))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_jit_wrapped(self, graph_mode, graph_updates):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.count = nnx.Variable(jnp.array(0))

      @nnx.jit(graph=graph_mode, graph_updates=graph_updates)
      def __call__(self, x: jax.Array) -> jax.Array:
        self.count[...] += 1
        return x * 2

    m = Foo(rngs=nnx.Rngs(0))
    x = jnp.array(3.0)

    @nnx.jit(graph=graph_mode, graph_updates=graph_updates)
    def f(m: nnx.Linear, x):
      return m(x)

    lowered = f.lower(m, x)
    compiled = lowered.compile()
    text = compiled.as_text()
    cost_analysis = compiled.cost_analysis()
    self.assertIsNotNone(cost_analysis)
    self.assertIsNotNone(text)

    y = compiled(m, x)
    np.testing.assert_allclose(y, 6.0)
    self.assertEqual(m.count[...], 1)
    y = compiled(m, x)
    self.assertEqual(m.count[...], 2)

  @parameterized.parameters(
    {'graph_mode': True, 'graph_updates': True, 'static_argnums': (2,), 'static_argnames': None},
    {'graph_mode': True, 'graph_updates': True, 'static_argnums': None, 'static_argnames': ('use_relu',)},
    {'graph_mode': True, 'graph_updates': False, 'static_argnums': (2,), 'static_argnames': None},
    {'graph_mode': True, 'graph_updates': False, 'static_argnums': None, 'static_argnames': ('use_relu',)},
    {'graph_mode': False, 'graph_updates': False, 'static_argnums': (2,), 'static_argnames': None},
    {'graph_mode': False, 'graph_updates': False, 'static_argnums': None, 'static_argnames': ('use_relu',)},
  )
  def test_jit_static_args_with_shardings(self, graph_mode, graph_updates, static_argnums, static_argnames):
    """Test static arguments work correctly with in_shardings."""
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('data',))

    def fn(x, scale, use_relu):
      y = x * scale
      if use_relu:
        y = jnp.maximum(y, 0)
      return y.sum()

    x = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
    x_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))

    f = nnx.jit(fn, in_shardings=(x_sharding, None),
                static_argnums=static_argnums, static_argnames=static_argnames,
                graph=graph_mode, graph_updates=graph_updates)
    y_relu = f(x, 0.5, True)
    y_no_relu = f(x, 0.5, False)
    self.assertNotEqual(y_relu, y_no_relu)

  @parameterized.parameters(
    {
      'static_args': {'static_argnums': (2, 3)},
    },
    {
      'static_args': {'static_argnames': ('static_arg1', 'static_arg2')},
    },
  )
  def test_with_sharding_and_static_args(self, static_args):
    n_devices = max(jax.local_device_count() // 2, 1)
    devices = mesh_utils.create_device_mesh(
      (n_devices, jax.local_device_count() // n_devices)
    )
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    def sharding(*args):
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

    state_sharding = nnx.StateSharding(
      {
        nnx.PathContains('kernel'): sharding('a', 'b'),
        nnx.PathContains('bias'): sharding('b'),
      }
    )

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
    self.assertNotIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

    @nnx.jit(
        in_shardings=(state_sharding, None),
        graph=True,
        graph_updates=True,
        **static_args,
    )
    def constrain_object(m, scale: float, static_arg1: bool, static_arg2: bool):
      new_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('b', 'a'))
      m.kernel = jax.lax.with_sharding_constraint(m.kernel, new_sharding)
      return None

    constrain_object(m, 0.5, True, True)
    self.assertEqual(m.kernel.sharding.spec, jax.sharding.PartitionSpec("a", "b"))

  @parameterized.parameters(
    {
      'graph': True,
      'static_args': {'static_argnums': (1, 2)},
    },
    {
      'graph': False,
      'static_args': {'static_argnums': (1, 2)},
    },
    {
      'graph': True,
      'static_args': {'static_argnames': ('static_arg1', 'static_arg2')},
    },
    {
      'graph': False,
      'static_args': {'static_argnames': ('static_arg1', 'static_arg2')},
    },
  )
  def test_with_sharding_and_static_args_functional(self, graph, static_args):
    n_devices = max(jax.local_device_count() // 2, 1)
    devices = mesh_utils.create_device_mesh(
      (n_devices, jax.local_device_count() // n_devices)
    )
    mesh = jax.sharding.Mesh(devices, ('a', 'b'))

    def sharding(*args):
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

    m_dummy = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    state_sharding = nnx.prefix(
      m_dummy,
      {
        nnx.PathContains('kernel'): sharding('a', 'b'),
        nnx.PathContains('bias'): sharding('b'),
      },
      graph=graph,
    )

    @nnx.jit(out_shardings=state_sharding, graph=graph, graph_updates=False, **static_args)
    def constrain_object(scale: float, static_arg1: bool, static_arg2: bool):
      m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
      new_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('b', 'a'))
      m.kernel.set_value(jax.lax.with_sharding_constraint(m.kernel[...], new_sharding))
      return m

    m = constrain_object(0.5, True, True)
    self.assertEqual(m.kernel.sharding.spec, jax.sharding.PartitionSpec("a", "b"))


class TestTreeJIT(parameterized.TestCase):
  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_jit_basic(self, graph, graph_updates):
    m = nnx.Dict(a=nnx.Param(jnp.array(1)))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def g(m: nnx.Dict):
      m.a[...] = 2
      return 1.0

    out = g(m)

    assert m.a[...] == 2
    assert out == 1.0

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_jit_module(self, graph, graph_updates):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(m, x):
      return m(x)

    x = jnp.ones((1, 2))
    y = f(m, x)
    self.assertEqual(y.shape, (1, 3))

  def test_tree_jit_variable_update(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

      @nnx.jit(graph=False)
      def __call__(self, x):
        self.count[...] += 1
        return x * 2

    m = Foo()
    y = m(jnp.array(3.0))
    np.testing.assert_allclose(y, 6.0)
    self.assertEqual(m.count[...], 1)
    y = m(jnp.array(3.0))
    self.assertEqual(m.count[...], 2)

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_jit_no_retrace(self, graph, graph_updates):
    n = 0
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(m, x):
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

  def test_tree_jit_static_argnums(self):
    @nnx.jit(graph=False, static_argnums=(1,))
    def f(x, use_relu):
      if use_relu:
        return jnp.maximum(x, 0)
      return x

    x = jnp.array([-1.0, 2.0])
    y_relu = f(x, True)
    np.testing.assert_allclose(y_relu, jnp.array([0.0, 2.0]))
    y_no_relu = f(x, False)
    np.testing.assert_allclose(y_no_relu, x)

  def test_tree_jit_no_input_output_aliasing(self):
    v = nnx.Param(jnp.array(1.0))

    @nnx.jit(graph=False)
    def f(v):
      return v

    with self.assertRaisesRegex(ValueError, 'does not support Variable aliasing'):
      f(v)

  def test_tree_jit_no_shared_variable_refs(self):
    v = nnx.Param(jnp.array(1.0))

    @nnx.jit(graph=False)
    def f(v1, v2):
      pass

    with self.assertRaisesRegex(
        ValueError, 'found at paths'
    ):
      f(v, v)

  def test_tree_jit_new_variable_output_ok(self):
    @nnx.jit(graph=False)
    def f(x):
      return nnx.Param(x + 1)

    v = f(jnp.array(1.0))
    self.assertIsInstance(v, nnx.Param)
    np.testing.assert_allclose(v[...], 2.0)

  def test_tree_jit_donate_argnums_unchanged_var(self):
    v = nnx.Param(jnp.array(1.0))

    @nnx.jit(graph=False, donate_argnums=(0,))
    def f(v):
      return v[...] + 1.0

    out = f(v)
    np.testing.assert_allclose(out, 2.0)
    np.testing.assert_allclose(v[...], 1.0)

    out = f(v)
    np.testing.assert_allclose(out, 2.0)
    np.testing.assert_allclose(v[...], 1.0)

  def test_tree_jit_donate_argnums_module(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    original_kernel = jnp.copy(m.kernel[...])

    @nnx.jit(graph=False, donate_argnums=(0,))
    def f(m, x):
      return m(x)

    x = jnp.ones((1, 2))
    y = f(m, x)
    self.assertEqual(y.shape, (1, 3))
    np.testing.assert_allclose(m.kernel[...], original_kernel)

    y = f(m, x)
    self.assertEqual(y.shape, (1, 3))
    np.testing.assert_allclose(m.kernel[...], original_kernel)

  def test_tree_jit_donate_argnums_with_mutation(self):
    v = nnx.Param(jnp.array(0.0))

    @nnx.jit(graph=False, donate_argnums=(0,))
    def f(v):
      v[...] += 1.0
      return None

    f(v)
    np.testing.assert_allclose(v[...], 1.0)
    f(v)
    np.testing.assert_allclose(v[...], 2.0)

  def test_tree_jit_donate_argnames(self):
    v = nnx.Param(jnp.array(1.0))

    @nnx.jit(graph=False, donate_argnames=('v',))
    def f(v):
      return v[...] + 1.0

    out = f(v=v)
    np.testing.assert_allclose(out, 2.0)
    np.testing.assert_allclose(v[...], 1.0)

    out = f(v=v)
    np.testing.assert_allclose(out, 2.0)
    np.testing.assert_allclose(v[...], 1.0)

  def test_tree_jit_donate_selective(self):
    donated = nnx.Param(jnp.array(1.0))
    not_donated = nnx.Param(jnp.array(2.0))

    @nnx.jit(graph=False, donate_argnums=(0,))
    def f(donated, not_donated):
      return donated[...] + not_donated[...]

    out = f(donated, not_donated)
    np.testing.assert_allclose(out, 3.0)
    np.testing.assert_allclose(donated[...], 1.0)
    np.testing.assert_allclose(not_donated[...], 2.0)

    out = f(donated, not_donated)
    np.testing.assert_allclose(out, 3.0)

  @parameterized.parameters(True, False)
  def test_jit_partial_basic(self, graph_mode):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      return m(x)

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    x = jnp.ones((1, 2))
    y = f_jit(x)
    self.assertEqual(y.shape, (1, 3))

  @parameterized.parameters(True, False)
  def test_jit_partial_lower_compile(self, graph_mode):
    class Foo(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))
        self.w = nnx.Param(jnp.ones((2, 3)))

    m = Foo()

    def f(m, x):
      m.count[...] += 1
      return x @ m.w[...]

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    compiled = f_jit.lower(jnp.ones((1, 2))).compile()

    x = jnp.ones((1, 2))
    y = compiled(x)
    self.assertEqual(y.shape, (1, 3))
    np.testing.assert_allclose(y, jnp.ones((1, 3)) * 2)
    self.assertEqual(m.count[...], 1)

    y = compiled(x)
    self.assertEqual(m.count[...], 2)

  @parameterized.parameters(True, False)
  def test_jit_partial_variable_update(self, graph_mode):
    class Foo(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

    m = Foo()

    def f(m, x):
      m.count[...] += 1
      return x * 2

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    y = f_jit(jnp.array(3.0))
    np.testing.assert_allclose(y, 6.0)
    self.assertEqual(m.count[...], 1)
    y = f_jit(jnp.array(3.0))
    self.assertEqual(m.count[...], 2)

  @parameterized.parameters(True, False)
  def test_jit_partial_multiple_args(self, graph_mode):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(3, 4, rngs=nnx.Rngs(1))

    def f(m1, m2, x):
      return m2(m1(x))

    f_jit = nnx.jit_partial(f, m1, m2, graph=graph_mode, graph_updates=False)
    x = jnp.ones((1, 2))
    y = f_jit(x)
    self.assertEqual(y.shape, (1, 4))

  @parameterized.parameters(True, False)
  def test_jit_partial_no_retrace(self, graph_mode):
    n = 0
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      nonlocal n
      n += 1
      return m(x)

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    x = jnp.ones((1, 2))
    f_jit(x)
    self.assertEqual(n, 1)
    f_jit(x)
    self.assertEqual(n, 1)

  @parameterized.parameters(True, False)
  def test_jit_partial_no_retrace_after_mutation(self, graph_mode):
    n = 0

    class Foo(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 3)))

    m = Foo()

    def f(m, x):
      nonlocal n
      n += 1
      return x @ m.w[...]

    f_jit = nnx.jit_partial(f, m, graph=graph_mode, graph_updates=False)
    x = jnp.ones((1, 2))
    f_jit(x)
    self.assertEqual(n, 1)
    m.w[...] = jnp.zeros((2, 3))
    y = f_jit(x)
    self.assertEqual(n, 1)
    np.testing.assert_allclose(y, jnp.zeros((1, 3)))

  @parameterized.parameters(True, False)
  def test_jit_partial_no_partial_args(self, graph_mode):
    f_partial = nnx.jit_partial(lambda x: x * 2, graph=graph_mode, graph_updates=False)
    y = f_partial(jnp.array(3.0))
    np.testing.assert_allclose(y, 6.0)

  @parameterized.parameters((False,))
  def test_jit_partial_in_shardings_none_broadcast(self, graph_mode):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('data',))

    m = nnx.Linear(4, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      return m(x)

    f_jit = nnx.jit_partial(f, m, in_shardings=(None, None), graph=graph_mode, graph_updates=False)
    x = jnp.ones((n_devices, 4))
    y = f_jit(x)
    self.assertEqual(y.shape, (n_devices, 3))

  @parameterized.parameters((False,))
  def test_jit_partial_in_shardings_named(self, graph_mode):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('data',))
    PS = jax.sharding.PartitionSpec

    v = nnx.Param(jnp.ones((n_devices, 4)))

    def f(v, x):
      return v[...] + x

    x_sharding = jax.sharding.NamedSharding(mesh, PS('data'))
    v_sharding = jax.sharding.NamedSharding(mesh, PS('data'))
    f_jit = nnx.jit_partial(
        f, v, in_shardings=(v_sharding, x_sharding), graph=graph_mode,
        graph_updates=False)
    x = jnp.ones((n_devices, 4))
    y = f_jit(x)
    self.assertEqual(y.shape, (n_devices, 4))

  @parameterized.parameters((False,))
  def test_jit_partial_mixed_shardings(self, graph_mode):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('data',))
    PS = jax.sharding.PartitionSpec

    m1 = nnx.Linear(4, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(3, 2, rngs=nnx.Rngs(1))

    def f(m1, m2, x):
      return m2(m1(x))

    x_sharding = jax.sharding.NamedSharding(mesh, PS('data'))
    f_jit = nnx.jit_partial(
        f, m1, m2, in_shardings=(None, None, x_sharding), graph=graph_mode,
        graph_updates=False)
    x = jnp.ones((n_devices, 4))
    y = f_jit(x)
    self.assertEqual(y.shape, (n_devices, 2))

  @parameterized.parameters(True, False)
  def test_jit_partial_in_shardings_non_tuple(self, graph_mode):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('data',))
    PS = jax.sharding.PartitionSpec

    m = nnx.Linear(4, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      return m(x)

    sharding = jax.sharding.NamedSharding(mesh, PS())
    f_jit = nnx.jit_partial(f, m, in_shardings=sharding, graph=graph_mode, graph_updates=False)
    x = jnp.ones((n_devices, 4))
    y = f_jit(x)
    self.assertEqual(y.shape, (n_devices, 3))

  @parameterized.parameters(True, False)
  def test_jit_partial_train_step(self, graph_mode):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    def train_step(model, optimizer, x, y):
      def loss_fn(model):
        return jnp.mean((model(x) - y) ** 2)
      loss, grads = nnx.value_and_grad(loss_fn)(model)
      optimizer.update(model, grads)
      return loss

    train_step_fn = nnx.jit_partial(train_step, model, optimizer, graph=graph_mode, graph_updates=False)
    for _ in range(2):
      x, y = jnp.ones((10, 2)), jnp.ones((10, 3))
      loss = train_step_fn(x, y)
      self.assertIsInstance(loss, jax.Array)

  @parameterized.parameters(True, False)
  def test_jit_partial_shared_variable(self, graph):
    v = nnx.Param(jnp.array(1.0))

    class Container(nnx.Module):
      def __init__(self, v):
        self.v = v

    c1 = Container(v)
    c2 = Container(v)

    def f(c1, c2, x):
      c1.v[...] += x
      return c1.v[...] + c2.v[...]

    f_jit = nnx.jit_partial(f, c1, c2, graph=graph, graph_updates=False)
    if not graph:
      with self.assertRaisesRegex(ValueError, 'Duplicate Param'):
        f_jit(jnp.array(1.0))
      return

    y = f_jit(jnp.array(1.0))
    np.testing.assert_allclose(y, 4.0)
    np.testing.assert_allclose(v[...], 2.0)

  @parameterized.parameters(True, False)
  def test_jit_inconsistent_aliasing(self, graph_updates):
    v = nnx.Param(jnp.array(1.0))
    P = jax.sharding.PartitionSpec

    @nnx.jit(
      in_shardings=(P(), P('x')),
      graph=True,
      graph_updates=graph_updates,
    )
    def f(a, b):
      return a[...] + b[...]

    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    with mesh:
      with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
        f(v, v)


class TestEvalShape(parameterized.TestCase):
  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_eval_shape(self, graph, graph_updates):
    abs_model = nnx.eval_shape(
      lambda: nnx.Linear(1, 2, rngs=nnx.Rngs(0)),
      graph=graph, graph_updates=graph_updates,
    )
    self.assertIsInstance(abs_model, nnx.Linear)
    self.assertIsInstance(abs_model.kernel.get_value(), jax.ShapeDtypeStruct)

  def test_eval_shape_mutable_array(self):
    with nnx.var_defaults(hijax=True):
      abs_model = nnx.eval_shape(lambda: nnx.Linear(1, 2, rngs=nnx.Rngs(0)), graph=True, graph_updates=True)
    self.assertIsInstance(abs_model, nnx.Linear)
    self.assertIsInstance(abs_model.kernel.get_value(), jax.ShapeDtypeStruct)
    self.assertEqual(abs_model.kernel.shape, (1, 2))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_eval_shape_with_module_input(self, graph, graph_updates):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    def f(m, x):
      return m(x)

    x = jnp.ones((4, 2))
    out = nnx.eval_shape(f, model, x, graph=graph, graph_updates=graph_updates)
    self.assertIsInstance(out, jax.ShapeDtypeStruct)
    self.assertEqual(out.shape, (4, 3))

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_eval_shape_no_state_update(self, graph, graph_updates):
    count = nnx.Variable(jnp.array(0))

    def f(c):
      c[...] += 1
      return jnp.ones((3, 4)) * c[...]

    out = nnx.eval_shape(f, count, graph=graph, graph_updates=graph_updates)
    self.assertIsInstance(out, jax.ShapeDtypeStruct)
    self.assertEqual(out.shape, (3, 4))
    self.assertEqual(count[...], 0)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_eval_shape_no_input_output_aliasing(self, graph, graph_updates):
    v = nnx.Param(jnp.array(1.0))

    def f(v):
      return v

    with self.assertRaises(ValueError):
      nnx.eval_shape(f, v, graph=graph, graph_updates=graph_updates)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_eval_shape_no_shared_variable_refs(self, graph, graph_updates):
    v = nnx.Param(jnp.array(1.0))

    def f(v1, v2):
      v1[...] += 1
      return None

    with self.assertRaises(ValueError):
      nnx.eval_shape(f, v, v, graph=graph, graph_updates=graph_updates)

class TestShardMap(parameterized.TestCase):
  def test_basic_shardmap(self):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('a',))
    PS = jax.sharding.PartitionSpec

    state_sharding = nnx.StateSharding(
      {
        nnx.PathContains('kernel'): PS(None, 'a'),
        nnx.PathContains('bias'): PS(),
      }
    )

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    self.assertNotIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(state_sharding,),
        out_specs=None,
        graph=True,
        graph_updates=True,
    )
    def f(m: nnx.Linear):
      self.assertEqual(
        m.kernel.shape, (m.in_features, m.out_features // n_devices)
      )
      self.assertEqual(m.bias.shape, (m.out_features,))

    f(m)

    self.assertIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

  @parameterized.parameters(True, False)
  def test_basic_shardmap_functional(self, graph):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('a',))
    PS = jax.sharding.PartitionSpec

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
    state_sharding = nnx.prefix(
        m,
        {
            nnx.PathContains('kernel'): PS(None, 'a'),
            nnx.PathContains('bias'): PS(),
        },
        graph=graph,
    )

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(state_sharding,),
        out_specs=None,
        graph=graph,
        graph_updates=False,
    )
    def f(m: nnx.Linear):
      self.assertEqual(
          m.kernel.shape, (m.in_features, m.out_features // n_devices)
      )
      self.assertEqual(m.bias.shape, (m.out_features,))

    f(m)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic_shardmap_variables(self, graph, graph_updates):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('a',))
    P = jax.sharding.PartitionSpec

    rngs = nnx.Rngs(0)
    w = nnx.Param(jax.random.normal(rngs.params(), (16, 32)))
    b = nnx.Param(jax.random.normal(rngs.params(), (32,)))
    count = nnx.BatchStat(jnp.array(0))

    self.assertNotIsInstance(w.sharding, jax.sharding.NamedSharding)

    @nnx.shard_map(
      mesh=mesh, in_specs=(P(None, 'a'), P(), P()), out_specs=None,
      graph=graph, graph_updates=graph_updates,
    )
    def f(w, b, count):
      count[...] += 1
      self.assertEqual(w.shape, (16, 32 // n_devices))
      self.assertEqual(b.shape, (32,))

    f(w, b, count)

    if graph and graph_updates:
      self.assertIsInstance(w.sharding, jax.sharding.NamedSharding)
      self.assertIsInstance(b.sharding, jax.sharding.NamedSharding)
    self.assertEqual(count[...], 1)

  def test_from_state(self):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('a',))
    PS = jax.sharding.PartitionSpec

    state_spec = nnx.State(
      {
        'kernel': PS(None, 'a'),
        'bias': PS(),
      }
    )
    state_sharding = nnx.StateSharding(state_spec)

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

    self.assertNotIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(state_sharding,),
        out_specs=None,
        graph=True,
        graph_updates=True,
    )

    def f(m: nnx.Linear):
      self.assertEqual(
        m.kernel.shape, (m.in_features, m.out_features // n_devices)
      )
      self.assertEqual(m.bias.shape, (m.out_features,))

    f(m)

    self.assertIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)
    self.assertIsInstance(m.bias.sharding, jax.sharding.NamedSharding)

  @parameterized.parameters(True, False)
  def test_from_state_functional(self, graph):
    n_devices = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = jax.sharding.Mesh(devices, ('a',))
    PS = jax.sharding.PartitionSpec

    state_spec = nnx.State({
        'kernel': PS(None, 'a'),
        'bias': PS(),
    })

    m = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
    graphdef, s = nnx.split(m)

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(state_spec,),
        out_specs=None,
        graph=graph,
        graph_updates=False,
    )
    def f(s):
      m = nnx.merge(graphdef, s)
      self.assertEqual(
          m.kernel.shape, (m.in_features, m.out_features // n_devices)
      )
      self.assertEqual(m.bias.shape, (m.out_features,))

    f(s)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_simple_data_parallel(self, graph, graph_updates):
    P = jax.sharding.PartitionSpec
    n_devices = jax.local_device_count()

    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    with jax.set_mesh(mesh):
      m = nnx.Linear(
          in_features=2,
          out_features=3,
          kernel_metadata={'out_sharding': jax.P(None)},
          bias_metadata={'out_sharding': jax.P(None)},
          rngs=nnx.Rngs(0),
    )
    x = jnp.ones((32, 2))

    @nnx.shard_map(
      mesh=mesh, in_specs=(P(None), P('data')), out_specs=P('data'),
      graph=graph, graph_updates=graph_updates,
    )
    def f(m, x):
      self.assertEqual(x.shape, (32 // n_devices, 2))
      return m(x)

    y = f(m, x)

    self.assertEqual(y.shape, (32, 3))
    self.assertIsInstance(y.sharding, jax.sharding.NamedSharding)
    if graph and graph_updates:
      self.assertIsInstance(m.kernel.sharding, jax.sharding.NamedSharding)
      self.assertIsInstance(m.bias.sharding, jax.sharding.NamedSharding)

  def test_simple_tensor_parallel(self):
    n_devices = jax.local_device_count()
    P = jax.sharding.PartitionSpec

    mesh = jax.sharding.Mesh(jax.local_devices(), ('model',))

    class MLP(nnx.Module):
      def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

      def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    m = MLP(2, 64, 3, rngs=nnx.Rngs(0))
    x = jnp.ones((32, 2))

    def path_ends_with(path_suffix):
      return lambda path, value: path[-len(path_suffix) :] == path_suffix

    model_sharding = nnx.StateSharding(
      {
        path_ends_with(('linear1', 'kernel')): P(None, 'model'),
        path_ends_with(('linear2', 'kernel')): P('model', None),
      }
    )

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(model_sharding, P(None)),
        out_specs=P(None),
        graph=True,
        graph_updates=True,
    )
    def f(m, x):
      self.assertEqual(m.linear1.kernel.shape, (2, 64 // n_devices))
      self.assertEqual(m.linear2.kernel.shape, (64 // n_devices, 3))
      y = m(x)
      return jax.lax.psum(y, 'model')

    y = f(m, x)
    self.assertEqual(y.shape, (32, 3))
    self.assertEqual(y.sharding.spec, P(None))

  @parameterized.parameters(True, False)
  def test_simple_tensor_parallel_functional(self, graph):
    n_devices = jax.local_device_count()
    P = jax.sharding.PartitionSpec

    mesh = jax.sharding.Mesh(jax.local_devices(), ('model',))

    class MLP(nnx.Module):

      def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

      def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))

    m = MLP(2, 64, 3, rngs=nnx.Rngs(0))
    x = jnp.ones((32, 2))

    def path_ends_with(path_suffix):
      return lambda path, value: path[-len(path_suffix) :] == path_suffix

    model_sharding = nnx.prefix(
        m,
        {
            path_ends_with(('linear1', 'kernel')): P(None, 'model'),
            path_ends_with(('linear2', 'kernel')): P('model', None),
        },
        graph=graph,
    )

    @nnx.shard_map(
        mesh=mesh,
        in_specs=(model_sharding, P(None)),
        out_specs=P(None),
        graph=graph,
        graph_updates=False,
    )
    def f(m, x):
      self.assertEqual(m.linear1.kernel.shape, (2, 64 // n_devices))
      self.assertEqual(m.linear2.kernel.shape, (64 // n_devices, 3))
      y = m(x)
      return jax.lax.psum(y, 'model')

    y = f(m, x)
    self.assertEqual(y.shape, (32, 3))
    self.assertEqual(y.sharding.spec, P(None))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_shardmap_with_sharding_names(self, graph, graph_updates):
    n_devices = jax.local_device_count()
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    with jax.set_mesh(mesh):
      w = nnx.Param(jnp.ones((8, 4)), out_sharding=('data', None))
      b = nnx.Param(jnp.ones((4,)), out_sharding=(None,))

    self.assertIsInstance(w.get_raw_value().sharding, jax.sharding.NamedSharding)
    self.assertEqual(w.out_sharding, ('data', None))
    self.assertEqual(b.out_sharding, (None,))

    @nnx.shard_map(
      mesh=mesh, in_specs=(P('data', None), P(None)), out_specs=P('data', None),
      graph=graph, graph_updates=graph_updates,
    )
    def f(w, b):
      self.assertEqual(w.shape, (8 // n_devices, 4))
      self.assertEqual(b.shape, (4,))
      return w + b[None]

    y = f(w, b)
    self.assertEqual(y.shape, (8, 4))
    self.assertIsInstance(y.sharding, jax.sharding.NamedSharding)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_shardmap_sharding_names_mutation(self, graph, graph_updates):
    n_devices = jax.local_device_count()
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    with jax.set_mesh(mesh):
      w = nnx.Param(jnp.zeros((8, 4)), out_sharding=('data', None))
      count = nnx.BatchStat(jnp.array(0))

    @nnx.shard_map(
      mesh=mesh, in_specs=(P('data', None), P()), out_specs=P('data', None),
      graph=graph, graph_updates=graph_updates,
    )
    def f(w, count):
      count[...] += 1
      self.assertEqual(w.shape, (8 // n_devices, 4))
      return w + 1.0

    y = f(w, count)
    self.assertEqual(count[...], 1)
    self.assertEqual(y.shape, (8, 4))
    np.testing.assert_allclose(w[...], jnp.zeros((8, 4)))

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_shardmap_shared_variable(self, graph, graph_updates):
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    v = nnx.Param(jnp.array(1.0))

    class Container(nnx.Module):
      def __init__(self, v):
        self.v = v

    c1 = Container(v)
    c2 = Container(v)

    @nnx.shard_map(
      mesh=mesh, in_specs=(P(), P(), P()), out_specs=P(),
      graph=graph, graph_updates=graph_updates,
    )
    def f(c1, c2, x):
      c1.v[...] += x
      return c1.v[...] + c2.v[...]

    if not graph and not graph_updates:
      with self.assertRaises(ValueError):
        f(c1, c2, jnp.array(1.0))
    else:
      y = f(c1, c2, jnp.array(1.0))
      np.testing.assert_allclose(y, 4.0)
      np.testing.assert_allclose(v[...], 2.0)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_shardmap_module_variable_update(self, graph, graph_updates):
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))

    class Foo(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

    m = Foo()

    @nnx.shard_map(
      mesh=mesh, in_specs=(P(), P()), out_specs=P(),
      graph=graph, graph_updates=graph_updates,
    )
    def f(m, x):
      m.count[...] += 1
      return x * 2

    y = f(m, jnp.array(3.0))
    np.testing.assert_allclose(y, 6.0)
    self.assertEqual(m.count[...], 1)
    y = f(m, jnp.array(3.0))
    self.assertEqual(m.count[...], 2)

  @parameterized.parameters(
    (True, True),
    (True, False),
  )
  def test_shard_map_inconsistent_aliasing(self, graph, graph_updates):
    v = nnx.Param(jnp.array(1.0))
    P = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))

    @nnx.shard_map(
      mesh=mesh,
      in_specs=(P(), P('x')),
      out_specs=P(),
      graph=graph, graph_updates=graph_updates,
    )
    def f(a, b):
      return a[...] + b[...]

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
      f(v, v)


class TestGrad(parameterized.TestCase):

  def test_prefix_graph_node_error(self):
    m = nnx.Dict(a=nnx.Param(1))

    with self.assertRaisesRegex(
      ValueError, 'Graph nodes are not allowed as prefixes'
    ):
      nnx.grad(lambda x: x, argnums=m, graph=True, graph_updates=True)

  def test_prefix_mapping_tree_mode_error(self):
    diff_state = nnx.DiffState(0, nnx.PathContains('a'))

    with self.assertRaisesRegex(
      ValueError, 'cannot contain `DiffState` objects'
    ):
      nnx.grad(
        lambda x: x, argnums=diff_state, graph=False, graph_updates=False
      )

  @parameterized.parameters(True, False)
  def test_grad(self, graph_updates: bool):
    p1 = nnx.Param(10.0)
    p2 = nnx.Param(20.0)

    m = nnx.Dict(
      a=nnx.List([p1, p2]),
      b=p1,
      c=7,
      d=5.0,
    )

    def f(m: nnx.Dict):
      # sum all params
      return m['a'][0][...] + m['a'][1][...] + m['b'][...]

    f_grad = nnx.grad(f, graph=True, graph_updates=graph_updates)
    grads = f_grad(m)
    if not graph_updates:
      grads = nnx.compat.state(grads)

    assert m.a[0] is m.b
    assert isinstance(grads, nnx.State)
    assert grads['a'][0][...] == 2.0
    assert issubclass(type(grads['a'][0]), nnx.Variable)
    assert grads['a'][1][...] == 1.0
    assert issubclass(type(grads['a'][1]), nnx.Variable)
    assert len(nnx.to_flat_state(grads)) == 2

    nnx.update(m, grads)

    assert m['a'][0] is m.b
    assert m['a'][0][...] == 2.0
    assert m['a'][1][...] == 1.0
    assert m['b'][...] == 2.0
    assert m['c'] == 7
    assert m['d'] == 5.0

  @parameterized.parameters(True, False)
  def test_grad_with_multiple_ref_types(self, graph_updates: bool):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(jnp.array(10.0)), nnx.BatchStat(jnp.array(20.0))]),
      b=nnx.Param(jnp.array(10.0)),
      c=7,
      d=5.0,
    )

    def f(m: nnx.Dict):
      # sum all params
      return m.a[0] + m.a[1] + m.b

    f_grad = nnx.grad(f, graph=True, graph_updates=graph_updates)
    grads = f_grad(m)

    self.assertEqual(grads['a'][0][...], 1.0)
    self.assertTrue(issubclass(type(grads['a'][0]), nnx.Param))

    if graph_updates:
      self.assertIsInstance(grads, nnx.State)
      self.assertEqual(len(grads), 2)
    else:
      # In tree mode (graph_updates=False), nnx.grad treats the Module as a regular
      # pytree and differentiates all array leaves, including non-Param variables.
      self.assertIsInstance(grads, nnx.Dict)
      self.assertEqual(len(grads), 4)
      self.assertEqual(grads['a'][1][...], 1.0)
      self.assertTrue(issubclass(type(grads['a'][1]), nnx.BatchStat))

    if not graph_updates:
      grads = nnx.state(grads)

    nnx.update(m, grads)

    self.assertEqual(m.a[0][...], 1.0)
    self.assertEqual(m.b[...], 1.0)
    self.assertEqual(m.c, 7)
    self.assertEqual(m.d, 5.0)

    if graph_updates:
      self.assertEqual(m.a[1][...], 20.0)
    else:
      self.assertEqual(m.a[1][...], 1.0)

  def test_grad_with_type_predicate(self):
    m = nnx.Dict(
      a=nnx.List([nnx.Param(jnp.array(10.0)), nnx.BatchStat(jnp.array(20.0))]),
      b=nnx.Param(jnp.array(10.0)),
      c=7,
      d=5.0,
    )

    @nnx.compat.grad(argnums=nnx.DiffState(0, nnx.BatchStat))
    def f(m: nnx.Dict):
      # sum all params
      return m.a[0] + m.a[1] + m.b

    grads = f(m)

    self.assertIsInstance(grads, nnx.State)
    self.assertEqual(grads['a'][1][...], 1.0)
    self.assertTrue(issubclass(type(grads['a'][1]), nnx.BatchStat))
    self.assertEqual(len(grads), 1)

    nnx.update(m, grads)

    self.assertEqual(m.a[0][...], 10.0)
    self.assertEqual(m.a[1][...], 1.0)
    self.assertEqual(m.b[...], 10.0)
    self.assertEqual(m.c, 7)
    self.assertEqual(m.d, 5.0)

  def test_grad_functional(self):
    m = nnx.Dict(
        a=nnx.List(
            [nnx.Param(jnp.array(10.0)), nnx.BatchStat(jnp.array(20.0))]
        ),
        b=nnx.Param(jnp.array(10.0)),
        c=7,
        d=5.0,
    )

    graphdef, batch_stats, rest = nnx.split(m, nnx.BatchStat, ...)

    @nnx.grad
    def f(batch_stats: nnx.State, rest: nnx.State):
      m_inner = nnx.merge(graphdef, batch_stats, rest)
      # sum all params
      return m_inner.a[0] + m_inner.a[1] + m_inner.b

    grads = f(batch_stats, rest)

    self.assertIsInstance(grads, nnx.State)
    self.assertEqual(grads['a'][1][...], 1.0)
    self.assertTrue(issubclass(type(grads['a'][1]), nnx.BatchStat))
    self.assertEqual(len(grads), 1)

    nnx.update(m, grads)

    self.assertEqual(m.a[0][...], 10.0)
    self.assertEqual(m.a[1][...], 1.0)
    self.assertEqual(m.b[...], 10.0)
    self.assertEqual(m.c, 7)
    self.assertEqual(m.d, 5.0)

  @parameterized.parameters(True, False)
  def test_multiple_inputs(self, graph_updates: bool):
    rngs = nnx.Rngs(0)
    m = nnx.Linear(2, 3, rngs=rngs)
    loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, graph=True, graph_updates=graph_updates)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads = grad_fn(m, x, y)
    if not graph_updates:
      grads = nnx.state(grads)

    assert 'kernel' in grads
    assert grads['kernel'].shape == (2, 3)
    assert 'bias' in grads
    assert grads['bias'].shape == (3,)

  @parameterized.named_parameters(
      {
          'testcase_name': '0_1_updates_True',
          'loss_fn': lambda m1, m2, x, y: jnp.mean((m2(m1(x)) - y) ** 2),
          'argnums': (0, 1),
          'graph_updates': True,
      },
      {
          'testcase_name': '0_1_updates_False',
          'loss_fn': lambda m1, m2, x, y: jnp.mean((m2(m1(x)) - y) ** 2),
          'argnums': (0, 1),
          'graph_updates': False,
      },
      {
          'testcase_name': '1_3_updates_True',
          'loss_fn': lambda x, m1, y, m2: jnp.mean((m2(m1(x)) - y) ** 2),
          'argnums': (1, 3),
          'graph_updates': True,
      },
      {
          'testcase_name': '1_3_updates_False',
          'loss_fn': lambda x, m1, y, m2: jnp.mean((m2(m1(x)) - y) ** 2),
          'argnums': (1, 3),
          'graph_updates': False,
      },
  )
  def test_multiple_graph_nodes(self, loss_fn, argnums, graph_updates: bool):
    rngs = nnx.Rngs(0)
    m1 = nnx.Linear(2, 3, rngs=rngs)
    m2 = nnx.Linear(3, 3, rngs=rngs)
    grad_fn = nnx.compat.grad(loss_fn, argnums=argnums) if graph_updates else nnx.grad(loss_fn, argnums=argnums)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    inputs = [x, y]
    inputs.insert(argnums[0], m1)
    inputs.insert(argnums[1], m2)
    grads_m1, grads_m2 = grad_fn(*inputs)
    if not graph_updates:
      grads_m1 = nnx.state(grads_m1)
      grads_m2 = nnx.state(grads_m2)

    assert 'kernel' in grads_m1
    assert grads_m1['kernel'].shape == (2, 3)
    assert 'bias' in grads_m1
    assert grads_m1['bias'].shape == (3,)
    assert 'kernel' in grads_m2
    assert grads_m2['kernel'].shape == (3, 3)
    assert 'bias' in grads_m2
    assert grads_m2['bias'].shape == (3,)

  def test_multiple_args(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    m1_diffstate = nnx.DiffState(0, nnx.PathContains('kernel'))
    m2_diffstate = nnx.DiffState(1, nnx.PathContains('bias'))

    @nnx.compat.grad(argnums=(m1_diffstate, m2_diffstate))
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

    @nnx.compat.grad(argnums=(m1_diffstate, m2_diffstate))
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

    @nnx.compat.value_and_grad(argnums=(m1_diffstate, m2_diffstate))
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

    @nnx.value_and_grad(
        argnums=(m1_diffstate, m2_diffstate), has_aux=True,
        graph=True, graph_updates=True,
    )
    def loss_fn(l1: list[nnx.Linear], l2: list[nnx.Linear]):
      loss = jnp.mean(l1[0].kernel * l2[0].kernel) + jnp.mean(
        l1[0].bias * l2[0].bias
      )
      l1[0].kernel.set_value(jnp.array(-1.0))
      m3 = nnx.Linear(2, 3, rngs=nnx.Rngs(2))
      return loss, m3

    (loss, m3), (grads_m1, grads_m2) = loss_fn([m1], [m2])

    self.assertEqual(loss.shape, ())
    self.assertIsInstance(m3, nnx.Linear)
    self.assertIn('kernel', grads_m1[0])
    self.assertNotIn('bias', grads_m1[0])
    self.assertNotIn('kernel', grads_m2[0])
    self.assertIn('bias', grads_m2[0])

  def test_value_and_grad_with_aux_functional(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

    def diff_filter(path, x):
      if path:
        first = path[0]
        idx = getattr(first, 'idx', first)
        if idx == 0:
          return nnx.PathContains('kernel')(path, x)
        if idx == 1:
          return nnx.PathContains('bias')(path, x)
      return False

    graphdef, diff, nondiff = nnx.split((m1, m2), diff_filter, ...)

    @nnx.value_and_grad(has_aux=True, graph=False)
    def loss_fn(diff: nnx.State, nondiff: nnx.State):
      m1, m2 = nnx.merge(graphdef, diff, nondiff)

      loss = jnp.mean(m1.kernel * m2.kernel) + jnp.mean(m1.bias * m2.bias)
      m1.kernel.set_value(jnp.array(-1.0))
      m3 = nnx.Linear(2, 3, rngs=nnx.Rngs(2))
      return loss, m3

    (loss, m3), grads = loss_fn(diff, nondiff)

    self.assertEqual(m1.kernel[...], -1.0)
    self.assertEqual(loss.shape, ())
    self.assertIsInstance(m3, nnx.Linear)
    self.assertIn(0, grads)
    self.assertIn('kernel', grads[0])
    self.assertNotIn('bias', grads[0])
    self.assertIn(1, grads)
    self.assertNotIn('kernel', grads[1])
    self.assertIn('bias', grads[1])

  @parameterized.parameters(True, False)
  def test_variables_in_grad(self, graph_updates: bool):
    p1 = nnx.Param(10.0)
    p2 = nnx.Param(20.0)

    m = dict(a=[p1, p2], b=p1)

    @nnx.grad(graph=True, graph_updates=graph_updates)
    def f(m: dict):
      return m['a'][0] + m['a'][1] + m['b']

    grads = f(m)

    self.assertIs(m['a'][0], m['b'])
    self.assertIsInstance(grads, dict)
    self.assertIsInstance(grads['a'][0], nnx.Variable)
    self.assertEqual(grads['a'][1][...], 1.0)
    self.assertIsInstance(grads['a'][1], nnx.Variable)

    if graph_updates:
      self.assertEqual(len(jax.tree.leaves(grads)), 2)
      self.assertIsInstance(grads['b'], nnx.State)
    else:
      self.assertEqual(len(jax.tree.leaves(grads)), 3)
      self.assertIsInstance(grads['b'], nnx.Variable)

    nnx.update(m, nnx.state(grads, graph=True))

    self.assertIs(m['a'][0], m['b'])
    self.assertEqual(m['a'][0][...], 2.0)
    self.assertEqual(m['a'][1][...], 1.0)
    self.assertEqual(m['b'][...], 2.0)

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_mode_grad(self, graph, graph_updates):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.grad(graph=graph, graph_updates=graph_updates)
    def loss_fn(m: nnx.Linear):
      return jnp.mean(m.kernel) + jnp.mean(m.bias)

    grads = loss_fn(m)

    grad_type = nnx.State if graph and graph_updates else nnx.Linear
    self.assertIsInstance(grads, grad_type)
    self.assertEqual(grads.kernel.shape, (2, 3))
    self.assertEqual(grads.bias.shape, (3,))

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_mode_grad_multiple_inputs(self, graph, graph_updates):
    rngs = nnx.Rngs(0)
    m = nnx.Linear(2, 3, rngs=rngs)
    loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, graph=graph, graph_updates=graph_updates)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads = grad_fn(m, x, y)

    grad_type = nnx.State if graph and graph_updates else nnx.Linear
    self.assertIsInstance(grads, grad_type)
    self.assertEqual(grads.kernel.shape, (2, 3))
    self.assertEqual(grads.bias.shape, (3,))

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_mode_grad_multiple_graph_nodes(self, graph, graph_updates):
    rngs = nnx.Rngs(0)
    m1 = nnx.Linear(2, 3, rngs=rngs)
    m2 = nnx.Linear(3, 3, rngs=rngs)
    loss_fn = lambda m1, m2, x, y: jnp.mean((m2(m1(x)) - y) ** 2)
    grad_fn = nnx.grad(loss_fn, argnums=(0, 1), graph=graph,
                       graph_updates=graph_updates)
    x = jax.random.uniform(rngs(), (1, 2))
    y = jnp.ones((1, 3))
    grads_m1, grads_m2 = grad_fn(m1, m2, x, y)

    grad_type = nnx.State if graph and graph_updates else nnx.Linear
    self.assertIsInstance(grads_m1, grad_type)
    self.assertEqual(grads_m1.kernel.shape, (2, 3))
    self.assertEqual(grads_m1.bias.shape, (3,))
    self.assertIsInstance(grads_m2, grad_type)
    self.assertEqual(grads_m2.kernel.shape, (3, 3))
    self.assertEqual(grads_m2.bias.shape, (3,))

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_tree_mode_value_and_grad_with_aux(self, graph, graph_updates):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.value_and_grad(has_aux=True, graph=graph, graph_updates=graph_updates)
    def loss_fn(m: nnx.Linear):
      loss = jnp.mean(m.kernel)
      m.kernel[...] = jnp.ones_like(m.kernel[...])
      return loss, {'aux': 1}

    (loss, aux), grads = loss_fn(m)

    self.assertEqual(loss.shape, ())
    self.assertEqual(aux, {'aux': 1})
    grad_type = nnx.State if graph and graph_updates else nnx.Linear
    self.assertIsInstance(grads, grad_type)
    self.assertEqual(grads.kernel.shape, (2, 3))
    np.testing.assert_allclose(m.kernel[...], jnp.ones((2, 3)))


class TestCustomVJP(parameterized.TestCase):
  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_basic_call(self, graph, graph_updates):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(1, 1, rngs=nnx.Rngs(1))

    @nnx.custom_vjp(graph=graph, graph_updates=graph_updates)
    def f(m1: nnx.Linear, m2: nnx.Linear):
      y = m1.kernel * m2.kernel
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

    self.assertEqual(y.shape, (1, 1))

  @parameterized.parameters(
    (True,),
    (False,),
  )
  def test_basic_call_with_state(self, graph):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(1, 1, rngs=nnx.Rngs(1))
    state = nnx.BatchStat(jnp.array(0.0))

    @nnx.custom_vjp(
      nondiff_argnums=(2,), graph=graph, graph_updates=False,
    )
    def f(m1: nnx.Linear, m2: nnx.Linear, state):
      y = m1.kernel * m2.kernel
      state[...] = jnp.array(-1.0)
      return y

    def f_fwd(m1, m2, state):
      y = f(m1, m2, state)
      return y, (m1, m2)

    def f_bwd(state, res, g):
      inputs_g, out_g = g
      m1, m2 = res
      return inputs_g

    f.defvjp(f_fwd, f_bwd)

    y = f(m1, m2, state)

    self.assertEqual(state[...], -1.0)
    self.assertEqual(y.shape, (1, 1))

  def test_jax_example_graph_updates(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(graph=True, graph_updates=True)
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      (m_g,), out_g = g
      self.assertIsInstance(m_g, nnx.State)
      m_g['x'][...] = cos_x * out_g * m.y
      m_g['y'][...] = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    grads = nnx.grad(
        f, argnums=nnx.DiffState(0, ...), graph=True, graph_updates=True
    )(m)
    self.assertIsInstance(grads, nnx.State)

    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  @parameterized.parameters(True, False)
  @nnx.set_graph_updates(False)
  def test_jax_example_functional(self, graph):
    @nnx.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: nnx.Variable[jax.Array]

    m = Foo(
        nnx.Param(jnp.array(1.0)),
        nnx.Param(jnp.array(2.0)),
        nnx.Variable(jnp.array(0)),
    )

    @nnx.custom_vjp(graph=graph)
    def f(m: Foo):
      m.z[...] += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(res, g, updates_g):
      (m_up_g,) = updates_g
      self.assertIsInstance(updates_g, tuple)
      self.assertLen(jax.tree.leaves(m_up_g), 1)
      self.assertIsNone(m_up_g.x)
      self.assertIsNone(m_up_g.y)
      self.assertIsInstance(m_up_g.z, nnx.Variable)

      cos_x, sin_x, m = res
      m_g = nnx.clone(m)
      m_g.x[...] = cos_x * g * m.y
      m_g.y[...] = sin_x * g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    params, nondiff = nnx.unpack(m, nnx.Param, ...)
    
    @nnx.jit(graph=graph)
    @nnx.grad(graph=graph)
    def grad_fn(params, nondiff):
      m = nnx.merge(params, nondiff)
      return f(m)

    grads = grad_fn(params, nondiff)
    self.assertIsInstance(grads, nnx.GraphState)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  def test_diff_state(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    x_in_path = nnx.PathContains('x')
    diff_state = nnx.DiffState(0, x_in_path)

    @nnx.custom_vjp(nondiff_argnums=(diff_state,), graph=True, graph_updates=True)
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), m)  # type: ignore
      return y, res

    def f_bwd(res, g):
      (m_g,), out_g = g
      cos_x, m = res

      self.assertIsInstance(m_g, nnx.State)
      self.assertEqual(out_g.shape, ())
      self.assertIsInstance(m, Foo)

      m_g['x'][...] = cos_x * out_g * m.y
      del m_g['y']
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    grad: nnx.State = nnx.grad(f, argnums=nnx.DiffState(0, x_in_path), graph=True, graph_updates=True)(m)

    np.testing.assert_allclose(grad['x'][...], jnp.cos(1.0) * 2.0)  # type: ignore
    self.assertEqual(m.z, 1)

  @parameterized.parameters(True, False)
  def test_diff_state_functional(self, graph):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    x_in_path = nnx.PathContains('x')
    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    graphdef, state_x, state_rest = nnx.split(m, x_in_path, ..., graph=graph)

    @nnx.custom_vjp(nondiff_argnums=(1,), graph=graph, graph_updates=False)
    def f(state_x, state_rest):
      m = nnx.merge(graphdef, state_x, state_rest)
      m.z += 1
      y = jnp.sin(m.x) * m.y
      return y

    def f_fwd(state_x, state_rest):
      y = f(state_x, state_rest)
      m = nnx.merge(graphdef, state_x, state_rest)
      res = (jnp.cos(m.x), state_x)
      return y, res

    def f_bwd(state_rest, res, g):
      cos_x, state_x = res
      m = nnx.merge(graphdef, state_x, state_rest)
      state_x_g = nnx.clone(state_x)
      state_x_g.x[...] = cos_x * g * m.y
      return (state_x_g,)

    f.defvjp(f_fwd, f_bwd)

    def loss_fn(state_x, state_rest):
      y = f(state_x, state_rest)
      return y

    grad_fn = nnx.grad(loss_fn, argnums=0, graph=graph, graph_updates=False)
    grad_x = grad_fn(state_x, state_rest)

    self.assertIsInstance(grad_x, nnx.State)
    np.testing.assert_allclose(grad_x['x'][...], jnp.cos(1.0) * 2.0)
    self.assertEqual(m.z, 0)

  def test_jax_example_with_remat_graph_updates(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(graph=True, graph_updates=True)
    @nnx.remat(graph=True, graph_updates=True)
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      (m_g,), out_g = g
      self.assertIsInstance(m_g, nnx.State)
      m_g['x'][...] = cos_x * out_g * m.y
      m_g['y'][...] = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    @nnx.jit(graph=True, graph_updates=True)
    def loss_fn(m):
      return f(m)

    grads = nnx.grad(
        loss_fn, argnums=nnx.DiffState(0, ...), graph=True, graph_updates=True
    )(m)
    self.assertIsInstance(grads, nnx.State)

    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 1)

  @parameterized.parameters(
      (True,),
      (False,),
  )
  def test_jax_example_with_remat_functional(self, graph):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(graph=graph, graph_updates=False)
    @nnx.remat(graph=graph, graph_updates=False)
    def f(m: Foo):
      m.z += 1
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore
      return y, res

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      out_g = g
      m_g = jax.tree.map(lambda x: x, m)
      m_g.x[...] = cos_x * out_g * m.y
      m_g.y[...] = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    @nnx.jit(graph=graph, graph_updates=False)
    def loss_fn(m):
      return f(m)

    grads = nnx.grad(loss_fn, graph=graph, graph_updates=False)(m)
    self.assertIsInstance(grads, Foo)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m.z, 0)

  def test_two_args(self):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(graph=True, graph_updates=True)
    def f(m1: Foo, m2: Foo):
      m1.z += 1
      y = jnp.sin(m1.x) * m1.y  # type: ignore
      return y, m2

    def f_fwd(m1: Foo, m2: Foo):
      y, m2 = f(m1, m2)
      res = (jnp.cos(m1.x), jnp.sin(m1.x), m1)  # type: ignore
      return (y, m2), res

    def f_bwd(res, g):
      (m1_g, m2_g), (y_g, _) = g
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
        loss_fn,
        argnums=(nnx.DiffState(0, ...), nnx.DiffState(1, ...)),
        graph=True,
        graph_updates=True,
    )(m1, m2)

    np.testing.assert_allclose(m1_grad['x'][...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(m1_grad['y'][...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m1.z, 1)
    np.testing.assert_allclose(m2_grad['x'][...], 4.0)  # type: ignore
    np.testing.assert_allclose(m2_grad['y'][...], 3.0)  # type: ignore

  @parameterized.parameters(True, False)
  def test_two_args_functional(self, graph):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(graph=graph, graph_updates=False)
    def f(m1: Foo, m2: Foo):
      m1.z += 1
      y = jnp.sin(m1.x) * m1.y  # type: ignore
      return y, nnx.clone(m2)

    def f_fwd(m1: Foo, m2: Foo):
      y, m2_out = f(m1, m2)
      res = (jnp.cos(m1.x), jnp.sin(m1.x), m1, m2)
      return (y, m2_out), res

    def f_bwd(res, g):
      y_g, m2_g = g
      cos_x, sin_x, m1, m2 = res

      m1_g = nnx.clone(m1)
      m1_g.x[...] = cos_x * y_g * m1.y
      m1_g.y[...] = sin_x * y_g

      return m1_g, m2_g

    f.defvjp(f_fwd, f_bwd)

    m1 = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)
    m2 = Foo(nnx.Param(jnp.array(3.0)), nnx.Param(jnp.array(4.0)), 0)

    def loss_fn(m1, m2):
      y, m2 = f(m1, m2)
      return y + m2.x * m2.y

    m1_grad, m2_grad = nnx.grad(
        loss_fn,
        argnums=(0, 1),
        graph=graph,
        graph_updates=False,
    )(m1, m2)

    self.assertIsInstance(m1_grad, Foo)
    self.assertIsInstance(m2_grad, Foo)
    np.testing.assert_allclose(m1_grad.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(m1_grad.y[...], jnp.sin(1.0))  # type: ignore
    self.assertEqual(m1.z, 0)
    np.testing.assert_allclose(m2_grad.x[...], 4.0)  # type: ignore
    np.testing.assert_allclose(m2_grad.y[...], 3.0)  # type: ignore

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_non_diff_args(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]
      z: int

    @nnx.custom_vjp(
      nondiff_argnums=(0, 2), graph=graph, graph_updates=graph_updates,
    )
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

    def f_bwd(a, b, res, g):
      cos_x, sin_x, m = res
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      if graph and graph_updates:
        (m_g,), out_g = g
        self.assertIsInstance(m_g, nnx.State)
        m_g['x'][...] = cos_x * out_g * m.y
        m_g['y'][...] = sin_x * out_g
        return (m_g,)
      else:
        out_g = g
        m_g = jax.tree.map(lambda x: x, m)
        m_g.x[...] = cos_x * out_g * m.y
        m_g.y[...] = sin_x * out_g
        return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)), 0)

    def loss_fn(m):
      a = 1
      b = 2
      return f(a, m, b)

    if graph and graph_updates:
      grads = nnx.grad(
          loss_fn, argnums=nnx.DiffState(0, ...), graph=True, graph_updates=True
      )(m)
      self.assertIsInstance(grads, nnx.State)
    else:
      grads = nnx.grad(
        loss_fn, graph=graph, graph_updates=graph_updates,
      )(m)
      self.assertIsInstance(grads, Foo)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)  # type: ignore
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))  # type: ignore
    if graph and graph_updates:
      self.assertEqual(m.z, 1)
    else:
      self.assertEqual(m.z, 0)

  def test_docs_example(self):
    import jax.numpy as jnp
    from flax import nnx

    class Foo(nnx.Module):
      def __init__(self, x, y):
        self.x = nnx.Param(x)
        self.y = nnx.Param(y)

    @nnx.custom_vjp(graph=True, graph_updates=True)
    def f(m: Foo):
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore

    def f_bwd(res, g):
      ins_g, out_g = g
      cos_x, sin_x, m = res
      tangent_m = nnx.State(dict(x=cos_x * out_g * m.y, y=sin_x * out_g))
      return (tangent_m,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
    grads = nnx.grad(f, graph=True, graph_updates=True)(m)

  @parameterized.parameters(True, False)
  def test_docs_example_functional(self, graph):
    import jax.numpy as jnp
    from flax import nnx

    class Foo(nnx.Module):
      def __init__(self, x, y):
        self.x = nnx.Param(x)
        self.y = nnx.Param(y)

    @nnx.custom_vjp(graph=graph, graph_updates=False)
    def f(m: Foo):
      return jnp.sin(m.x) * m.y  # type: ignore

    def f_fwd(m: Foo):
      return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)  # type: ignore

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      out_g = g
      m_g = nnx.clone(m)
      m_g.x[...] = cos_x * out_g * m.y
      m_g.y[...] = sin_x * out_g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(x=jnp.array(1.0), y=jnp.array(2.0))
    grads = nnx.grad(f, graph=graph, graph_updates=False)(m)
    self.assertIsInstance(grads, Foo)

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
      m.n[...] += 1
      y = x @ m.kernel + m.bias
      return y

    def linear_fwd(m: MyLinear, x: jax.Array):
      return linear(m, x), (m, x)

    def linear_bwd(res, g):
      m, x = res
      (m_g, _x_grad), outputs_g = g
      kernel_grad = outputs_g[None, :] * x[:, None]
      bias_grad = outputs_g
      x_grad = m.kernel @ outputs_g
      assert x_grad.shape == x.shape, 'Shape mismatch for x'
      assert m.kernel.shape == kernel_grad.shape, 'Shape mismatch for kernel'
      assert m.bias.shape == bias_grad.shape, 'Shape mismatch for bias'
      return (m_g, x_grad)

    if use_custom_vjp:
      linear = nnx.custom_vjp(linear, graph=True, graph_updates=True)
      linear.defvjp(linear_fwd, linear_bwd)

    @nnx.jit(graph=True, graph_updates=True)
    def loss_fn(x, mod):
      y = linear(mod, x)
      return y.mean()

    mod = MyLinear(10, 5, rngs=nnx.Rngs(0))
    self.assertEqual(mod.n[...], 0)
    x = jax.random.normal(jax.random.key(0), (10,))
    loss, grad = nnx.value_and_grad(loss_fn, graph=True, graph_updates=True)(x, mod)
    self.assertEqual(loss.shape, ())
    self.assertEqual(grad.shape, (10,))
    self.assertEqual(mod.n[...], 1)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_basic_call(self, graph, graph_updates):
    m1 = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(1, 1, rngs=nnx.Rngs(1))
    state = nnx.BatchStat(jnp.array(0.0))

    @nnx.custom_vjp(
      nondiff_argnums=(2,), graph=graph, graph_updates=graph_updates,
    )
    def f(m1: nnx.Linear, m2: nnx.Linear, state):
      y = m1.kernel * m2.kernel
      state[...] = jnp.array(-1.0)
      return y

    def f_fwd(m1, m2, state):
      y = f(m1, m2, state)
      return y, (m1, m2)

    def f_bwd(state, res, g):
      m1, m2 = res
      return g, g

    f.defvjp(f_fwd, f_bwd)

    y = f(m1, m2, state)

    self.assertEqual(state[...], -1.0)
    self.assertEqual(y.shape, (1, 1))

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_jax_example(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]

    @nnx.custom_vjp(graph=graph, graph_updates=graph_updates)
    def f(m: Foo):
      return jnp.sin(m.x) * m.y

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)
      return y, res

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      m_g = jax.tree.map(lambda x: x, m)
      m_g.x[...] = cos_x * g * m.y
      m_g.y[...] = sin_x * g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)))

    grads = nnx.grad(f, graph=graph, graph_updates=graph_updates)(m)

    self.assertIsInstance(grads, Foo)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_with_remat(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]

    @nnx.custom_vjp(graph=graph, graph_updates=graph_updates)
    @nnx.remat(graph=graph, graph_updates=graph_updates)
    def f(m: Foo):
      return jnp.sin(m.x) * m.y

    def f_fwd(m: Foo):
      y = f(m)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)
      return y, res

    def f_bwd(res, g):
      cos_x, sin_x, m = res
      m_g = jax.tree.map(lambda x: x, m)
      m_g.x[...] = cos_x * g * m.y
      m_g.y[...] = sin_x * g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def loss_fn(m):
      return f(m)

    grads = nnx.grad(loss_fn, graph=graph, graph_updates=graph_updates)(m)

    self.assertIsInstance(grads, Foo)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_non_diff_args(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]

    @nnx.custom_vjp(nondiff_argnums=(0, 2), graph=graph,
                    graph_updates=graph_updates)
    def f(a, m: Foo, b):
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      return jnp.sin(m.x) * m.y

    def f_fwd(a, m: Foo, b):
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      y = f(a, m, b)
      res = (jnp.cos(m.x), jnp.sin(m.x), m)
      return y, res

    def f_bwd(a, b, res, g):
      cos_x, sin_x, m = res
      self.assertEqual(a, 1)
      self.assertEqual(b, 2)
      m_g = jax.tree.map(lambda x: x, m)
      m_g.x[...] = cos_x * g * m.y
      m_g.y[...] = sin_x * g
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)))

    def loss_fn(m):
      a = 1
      b = 2
      return f(a, m, b)

    grads = nnx.grad(loss_fn, graph=graph, graph_updates=graph_updates)(m)

    self.assertIsInstance(grads, Foo)
    np.testing.assert_allclose(grads.x[...], jnp.cos(1.0) * 2.0)
    np.testing.assert_allclose(grads.y[...], jnp.sin(1.0))

  def test_tree_mode_diffstate_error(self):
    x_in_path = nnx.PathContains('x')
    diff_state = nnx.DiffState(0, x_in_path)
    with self.assertRaisesRegex(
      ValueError,
      r'`nondiff_argnums` cannot contain `DiffState` objects',
    ):
      nnx.custom_vjp(lambda m: m, nondiff_argnums=(diff_state,), graph=False)

  def test_grad_inconsistent_aliasing(self):
    v = nnx.Param(jnp.array(1.0))

    def f(v_diff, v_nondiff):
      return v_diff[...] + v_nondiff[...]

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
      nnx.grad(f, argnums=0, graph=True, graph_updates=False)(v, v)

  def test_custom_vjp_inconsistent_aliasing(self):
    v = nnx.Param(jnp.array(1.0))

    @nnx.custom_vjp(nondiff_argnums=(1,), graph=True, graph_updates=False)
    def f(v_diff, v_nondiff):
      return v_diff[...] + v_nondiff[...]

    def f_fwd(v_diff, v_nondiff):
      return f(v_diff, v_nondiff), ()

    def f_bwd(v_nondiff, res, g):
      return (nnx.clone(v_nondiff),)

    f.defvjp(f_fwd, f_bwd)

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
      f(v, v)

  def test_custom_vjp_diff_arg_mutation(self):
    n = 0
    @nnx.custom_vjp(graph=True, graph_updates=False)
    def f(m):
      m.x[...] += 1
      return m.x[...] * m.y[...]

    def f_fwd(m):
      return f(m), (m,)

    def f_bwd(res, g, updates_g):
      nonlocal n
      n += 1
      (m_up_g,) = updates_g
      self.assertIsInstance(m_up_g.x, nnx.Param)
      self.assertIsNone(m_up_g.y)
      (m,) = res
      m_g = nnx.clone(m)
      m_g.x[...] = g * m.y[...]
      m_g.y[...] = g * m.x[...]
      return (m_g,)

    f.defvjp(f_fwd, f_bwd)

    @dataclasses.dataclass
    class Foo(nnx.Module):
      x: nnx.Param[jax.Array]
      y: nnx.Param[jax.Array]

    m = Foo(nnx.Param(jnp.array(1.0)), nnx.Param(jnp.array(2.0)))
    
    g = nnx.grad(f)(m)
    self.assertEqual(n, 1)


class TestVjpJvp(parameterized.TestCase):

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_vjp_basic(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    def f(m: Foo, x):
      return jnp.sum(m.w * x)

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0, 3.0])))
    x = jnp.array([4.0, 5.0, 6.0])

    primals_out, vjp_fn = nnx.vjp(
      f, m, x, graph=graph, graph_updates=graph_updates,
    )

    np.testing.assert_allclose(primals_out, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)

    m_grad, x_grad = vjp_fn(jnp.ones_like(primals_out))
    self.assertIsInstance(m_grad, Foo)
    np.testing.assert_allclose(m_grad.w[...], x)
    np.testing.assert_allclose(x_grad, m.w[...])

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_vjp_has_aux(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    def f(m: Foo, x):
      y = jnp.sum(m.w * x)
      return y, {'input': x}

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0])))
    x = jnp.array([3.0, 4.0])

    primals_out, vjp_fn, aux = nnx.vjp(
      f, m, x, has_aux=True, graph=graph, graph_updates=graph_updates,
    )

    np.testing.assert_allclose(primals_out, 1.0 * 3.0 + 2.0 * 4.0)
    np.testing.assert_allclose(aux['input'], x)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_vjp_state_propagation(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]
      count: nnx.BatchStat[jax.Array]

    def f(m: Foo, x):
      m.count[...] += 1
      return jnp.sum(m.w * x)

    m = Foo(
      w=nnx.Param(jnp.array([1.0, 2.0])),
      count=nnx.BatchStat(jnp.array(0)),
    )
    x = jnp.array([3.0, 4.0])

    self.assertEqual(m.count[...], 0)
    primals_out, vjp_fn = nnx.vjp(
      f, m, x, graph=graph, graph_updates=graph_updates,
    )
    self.assertEqual(m.count[...], 1)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_vjp_matches_jax(self, graph, graph_updates):
    def f(w, x):
      return jnp.sum(w * x)

    w = jnp.array([1.0, 2.0, 3.0])
    x = jnp.array([4.0, 5.0, 6.0])

    jax_primals, jax_vjp_fn = jax.vjp(f, w, x)
    jax_grads = jax_vjp_fn(jnp.ones_like(jax_primals))

    nnx_primals, nnx_vjp_fn = nnx.vjp(
      f, w, x, graph=graph, graph_updates=graph_updates,
    )
    nnx_grads = nnx_vjp_fn(jnp.ones_like(nnx_primals))

    np.testing.assert_allclose(nnx_primals, jax_primals)
    for ng, jg in zip(nnx_grads, jax_grads):
      np.testing.assert_allclose(ng, jg)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_vjp_decorator(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    @nnx.vjp(graph=graph, graph_updates=graph_updates)
    def f(m: Foo, x):
      return jnp.sum(m.w * x)

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0])))
    x = jnp.array([3.0, 4.0])

    primals_out, vjp_fn = f(m, x)
    np.testing.assert_allclose(primals_out, 11.0)
    m_grad, x_grad = vjp_fn(jnp.ones_like(primals_out))
    np.testing.assert_allclose(m_grad.w[...], x)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_jvp_basic(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    def f(m: Foo, x):
      return jnp.sum(m.w * x)

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0, 3.0])))
    x = jnp.array([4.0, 5.0, 6.0])

    m_tangent = jax.tree.map(jnp.ones_like, m)
    x_tangent = jnp.ones_like(x)

    primals_out, tangent_out = nnx.jvp(
      f, (m, x), (m_tangent, x_tangent),
      graph=graph, graph_updates=graph_updates,
    )

    np.testing.assert_allclose(primals_out, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)
    expected_tangent = jnp.sum(jnp.ones(3) * x + m.w[...] * jnp.ones(3))
    np.testing.assert_allclose(tangent_out, expected_tangent)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_jvp_has_aux(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    def f(m: Foo, x):
      y = jnp.sum(m.w * x)
      return y, {'input': x}

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0])))
    x = jnp.array([3.0, 4.0])

    m_tangent = jax.tree.map(jnp.ones_like, m)
    x_tangent = jnp.ones_like(x)

    primals_out, tangent_out, aux = nnx.jvp(
      f, (m, x), (m_tangent, x_tangent), has_aux=True,
      graph=graph, graph_updates=graph_updates,
    )

    np.testing.assert_allclose(primals_out, 11.0)
    np.testing.assert_allclose(aux['input'], x)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_jvp_state_propagation(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]
      count: nnx.BatchStat[jax.Array]

    def f(m: Foo, x):
      m.count[...] += 1
      return jnp.sum(m.w * x)

    m = Foo(
      w=nnx.Param(jnp.array([1.0, 2.0])),
      count=nnx.BatchStat(jnp.array(0.0)),
    )
    x = jnp.array([3.0, 4.0])

    m_tangent = jax.tree.map(jnp.zeros_like, m)
    x_tangent = jnp.zeros_like(x)

    self.assertEqual(m.count[...], 0.0)
    primals_out, tangent_out = nnx.jvp(
      f, (m, x), (m_tangent, x_tangent),
      graph=graph, graph_updates=graph_updates,
    )
    self.assertEqual(m.count[...], 1.0)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_jvp_matches_jax(self, graph, graph_updates):
    def f(w, x):
      return jnp.sum(w * x)

    w = jnp.array([1.0, 2.0, 3.0])
    x = jnp.array([4.0, 5.0, 6.0])

    w_tangent = jnp.ones_like(w)
    x_tangent = jnp.ones_like(x)

    jax_primals, jax_tangents = jax.jvp(f, (w, x), (w_tangent, x_tangent))
    nnx_primals, nnx_tangents = nnx.jvp(
      f, (w, x), (w_tangent, x_tangent),
      graph=graph, graph_updates=graph_updates,
    )

    np.testing.assert_allclose(nnx_primals, jax_primals)
    np.testing.assert_allclose(nnx_tangents, jax_tangents)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_jvp_decorator(self, graph, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      w: nnx.Param[jax.Array]

    @nnx.jvp(graph=graph, graph_updates=graph_updates)
    def f(m: Foo, x):
      return jnp.sum(m.w * x)

    m = Foo(w=nnx.Param(jnp.array([1.0, 2.0])))
    x = jnp.array([3.0, 4.0])

    m_tangent = jax.tree.map(jnp.ones_like, m)
    x_tangent = jnp.ones_like(x)

    primals_out, tangent_out = f((m, x), (m_tangent, x_tangent))
    np.testing.assert_allclose(primals_out, 11.0)
    expected_tangent = jnp.sum(jnp.ones(2) * x + m.w[...] * jnp.ones(2))
    np.testing.assert_allclose(tangent_out, expected_tangent)


class TestScan(parameterized.TestCase):
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

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)

    @nnx.scan(in_axes=(nnx.Carry, 0, None), length=5)
    def forward_block(_, block: Block, x: jax.Array):
      return None, block(x)

    x = jnp.ones((1, 3))
    out, y = forward_block(None, module, x)

    assert y.shape == (5, 1, 3)
    assert out is None

  def test_broadcast_args(self):
    def scale_cumsum(carry, scale, x):
      carry = carry + x * scale
      return carry, carry

    final_carry, _ = nnx.scan(
        scale_cumsum,
        in_axes=(nnx.Carry, None, 0),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.array(0.0), jnp.array(2.0), jnp.arange(5.0))
    np.testing.assert_allclose(final_carry, 20.0)

  def test_no_carry_all_scanned(self):
    def double(x):
      return x * 2

    ys = nnx.scan(
        double, in_axes=0, out_axes=0, graph=False
    )(jnp.arange(5.0))
    np.testing.assert_allclose(ys, jnp.arange(5.0) * 2)

  def test_pytree_prefix_in_axes(self):
    def fn(carry, x):
      carry = carry + x['a'] + x['b']
      return carry, carry

    xs = {'a': jnp.arange(3.0), 'b': jnp.array(1.0)}
    final_carry, _ = nnx.scan(
        fn,
        in_axes=(nnx.Carry, {'a': 0, 'b': None}),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.array(0.0), xs)
    np.testing.assert_allclose(final_carry, 6.0)

  def test_nested_carry_rejected(self):
    with self.assertRaises(ValueError):
      nnx.scan(
          lambda x: x,
          in_axes=({'a': nnx.Carry},),
          out_axes=nnx.Carry,
          graph=False,
      )({'a': jnp.array(1.0)})

  @parameterized.parameters(True, False)
  def test_broadcast_out_axes_rejected1(self, graph):
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast output state'):
      nnx.scan(
          lambda c, x: (c, x),
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, None),
          graph=graph,
          graph_updates=False,
      )(jnp.array(0.0), jnp.arange(3.0))

  def test_none_broadcast_input(self):
    def fn(carry, _unused, x):
      carry = carry + x
      return carry, carry

    final_carry, _ = nnx.scan(
        fn,
        in_axes=(nnx.Carry, None, 0),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.array(0.0), None, jnp.arange(3.0))
    np.testing.assert_allclose(final_carry, 3.0)

  def test_none_nested_in_arg(self):
    def fn(carry, x):
      carry = carry + x['a']
      return carry, carry

    xs = {'a': jnp.arange(3.0), 'b': None}
    final_carry, _ = nnx.scan(
        fn,
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.array(0.0), xs)
    np.testing.assert_allclose(final_carry, 3.0)

  def test_nested_carry_in_out_axes_rejected(self):
    with self.assertRaises(ValueError):
      nnx.scan(
          lambda c, x: (c, x),
          in_axes=(nnx.Carry, 0),
          out_axes=({'a': nnx.Carry},),
          graph=False,
      )(jnp.array(0.0), jnp.arange(3.0))

  def test_carry_in_in_axes_only_rejected(self):
    with self.assertRaisesRegex(ValueError, 'If one of in_axes or out_axes has Carry'):
      nnx.scan(
          lambda c, x: (c + x,),
          in_axes=(nnx.Carry, 0),
          out_axes=(0,),
          graph=False,
      )(jnp.array(0.0), jnp.arange(3.0))

  def test_carry_in_out_axes_only_rejected(self):
    with self.assertRaisesRegex(ValueError, 'If one of in_axes or out_axes has Carry'):
      nnx.scan(
          lambda x: x,
          in_axes=(0,),
          out_axes=nnx.Carry,
          graph=False,
      )(jnp.arange(3.0))

  def test_non_tuple_carry_only(self):
    def f(carry):
      return carry + 1

    result = nnx.scan(
        f,
        in_axes=nnx.Carry,
        out_axes=nnx.Carry,
        length=5,
        graph=False,
    )(jnp.array(0))
    self.assertEqual(result, 5)

  def test_non_tuple_scan_only(self):
    def f(x):
      return x * 2

    ys = nnx.scan(
        f,
        in_axes=0,
        out_axes=0,
        graph=False,
    )(jnp.arange(5.0))
    np.testing.assert_allclose(ys, jnp.arange(5.0) * 2)

  @parameterized.parameters(True, False)
  def test_variables_in_scan(self, graph_updates):
    def block_init(din, dout, rngs):
      w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)))
      b = nnx.Param(jnp.zeros((dout,)))
      return w, b

    def block_forward(w, b, x):
      return nnx.gelu(x @ w + b[None])

    @nnx.split_rngs(splits=5)
    @nnx.scan(in_axes=0, out_axes=0, length=5, graph_updates=graph_updates)
    def create_block(rngs: nnx.Rngs):
      return block_init(3, 3, rngs)

    w, b = create_block(nnx.Rngs(0))

    assert w.shape == (5, 3, 3)
    assert b.shape == (5, 3)

    @nnx.scan(
      in_axes=(0, 0, nnx.Carry), out_axes=nnx.Carry,
      graph_updates=graph_updates,
    )
    def stack_forward(w, b, x):
      return block_forward(w, b, x)

    x = jnp.ones((1, 3))
    y = stack_forward(w, b, x)
    assert y.shape == (1, 3)

  def test_variables_as_carries_in_scan(self):
    w = nnx.Param(jax.random.normal(jax.random.key(0), (3, 3)))
    b = nnx.Param(jnp.zeros((3,)))
    count = nnx.BatchStat(0)

    def block_forward(w, b, x):
      return nnx.gelu(x @ w + b[None])

    @nnx.scan(
      in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0),
      graph=True, graph_updates=True,
    )
    def stack_forward(params, x):
      w, b, count = params
      y = block_forward(w, b, x)
      count[...] += 1
      return (w, b, count), y

    x = jnp.ones((5, 1, 3))
    (w, b, count), y = stack_forward((w, b, count), x)

    assert y.shape == (5, 1, 3)
    assert count[...] == 5

  @parameterized.parameters(True, False)
  def test_variables_broadcast_in_scan(self, graph):
    w = nnx.Param(jax.random.normal(jax.random.key(0), (3, 3)))
    b = nnx.Param(jnp.zeros((3,)))
    count = nnx.BatchStat(0)

    def block_forward(w, b, x):
      return nnx.gelu(x @ w + b[None])

    @nnx.scan(
      in_axes=(None, None, None, 0), out_axes=0,
      graph=graph, graph_updates=False
    )
    def stack_forward(w, b, count, x):
      y = block_forward(w, b, x)
      count[...] += 1
      return y

    x = jnp.ones((5, 1, 3))
    y = stack_forward(w, b, count, x)

    assert y.shape == (5, 1, 3)
    assert count[...] == 5

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

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    # assert module.node.shape == (2,)

    @nnx.scan(in_axes=(0, None), out_axes=0, length=5)
    def forward_block(block: Block, x: jax.Array):
      return block(x)

    x = jnp.ones((1, 3))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  @parameterized.parameters(True, False)
  def test_all_carry(self, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      n: nnx.BatchStat[int]

    foo = Foo(n=nnx.BatchStat(0))

    @nnx.scan(
      in_axes=nnx.Carry, out_axes=nnx.Carry, length=3,
      graph_updates=graph_updates,
    )
    def loop(foo: Foo):
      foo.n[...] += 1
      return foo

    foo2 = loop(foo)

    self.assertIs(foo2.n, foo.n)
    self.assertEqual(foo.n[...], 3)

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

  @parameterized.parameters(True, False)
  def test_all_carry_new_reference_error(self, graph_updates):
    class Foo(nnx.Module):
      def __init__(self, n: nnx.BatchStat[int]):
        self.n = n

    xs = jnp.arange(3)
    foo = Foo(n=nnx.BatchStat(0))

    @nnx.scan(
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        graph=True,
        graph_updates=graph_updates,
    )
    def loop(foo: Foo, x):
      x = x + 1
      foo = Foo(nnx.BatchStat(foo.n[...] + 1))  # new reference
      return foo, x

    msg = (
        'Carry references must be the same between iterations'
        if graph_updates
        else 'scan Variable identity must be preserved across iterations'
    )

    with self.assertRaisesRegex(ValueError, msg):
      loop(foo, xs)

  @parameterized.parameters(True, False)
  def test_all_scan(self, graph_updates):
    class Foo(nnx.Module):
      def __init__(self, n: nnx.BatchStat[jax.Array]):
        self.n = n

    xs = jnp.arange(3)
    foo = Foo(n=nnx.BatchStat(jnp.arange(3)))

    @nnx.scan(in_axes=0, out_axes=0, graph_updates=graph_updates)
    def loop(foo: Foo, x):
      x = x + 1
      foo.n[...] += 1
      return x

    ys = loop(foo, xs)

    np.testing.assert_allclose(ys, jnp.arange(1, 4))
    np.testing.assert_allclose(foo.n[...], jnp.arange(1, 4))

  def test_all_broadcast(self):
    @nnx.dataclass
    class Foo(nnx.Module):
      n: nnx.BatchStat[int]

    xs = jnp.array(1)
    foo = Foo(n=nnx.BatchStat(2))

    @nnx.scan(in_axes=None, out_axes=0, length=4)
    def loop(foo: Foo, x):
      return x + foo.n

    ys = loop(foo, xs)

    np.testing.assert_allclose(ys, 3)
    self.assertEqual(ys.shape, (4,))

  @parameterized.parameters(True, False)
  def test_input_output_carry_mismatch_error(self, graph_updates):
    with nnx.set_graph_updates(graph_updates):
      with self.assertRaisesRegex(
          ValueError,
          'If one of in_axes or out_axes has Carry, the other must also have'
          ' Carry',
      ):

        @nnx.scan(in_axes=0, out_axes=(nnx.Carry, 0))
        def loop(a, b):
          ...

      with self.assertRaisesRegex(
          ValueError,
          'If one of in_axes or out_axes has Carry, the other must also have'
          ' Carry',
      ):

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=0)
        def loop(a, b):
          ...

  @parameterized.parameters(True, False)
  def test_double_carry_error(self, graph_updates):
    with self.assertRaisesRegex(
      ValueError,
      'Found multiple Carry definitions',
    ):

      @nnx.scan(in_axes=(nnx.Carry, nnx.Carry), graph_updates=graph_updates)
      def loop(a, b):
        ...

  @parameterized.parameters(True, False)
  def test_broadcast_in_output_error(self, graph_updates):
    with self.assertRaisesRegex(
      ValueError,
      'Cannot broadcast output state',
    ):

      @nnx.scan(
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, None),
          graph_updates=graph_updates,
      )
      def loop(a, b):
        ...

  def test_broadcast_in_output_state_axes_error(self):
    with self.assertRaisesRegex(
      ValueError,
      'Cannot broadcast output state. Got StateAxes',
    ):

      @nnx.scan(
          in_axes=(nnx.Carry, 0),
          out_axes=(nnx.Carry, nnx.StateAxes({...: None})),
          graph=True,
          graph_updates=True,
      )
      def loop(a, b):
        ...

  @parameterized.parameters(
    (True, False), (False, False),
  )
  def test_scan_stateful(self, graph, graph_updates):
    count = nnx.Variable(jnp.array(0))

    @nnx.scan(graph=graph, graph_updates=graph_updates)
    def f(count, x):
      count[...] += 1
      return count, x + count[...]

    xs = jnp.arange(5)
    count_out, ys = f(count, xs)

    self.assertIs(count_out, count)
    self.assertEqual(count[...], 5)
    np.testing.assert_allclose(ys, jnp.array([1, 3, 5, 7, 9]))

  @parameterized.parameters(
    (True, False), (False, False),
  )
  def test_scan_carry_identity_error(self, graph, graph_updates):
    count = nnx.Variable(jnp.array(0))

    @nnx.scan(graph=graph, graph_updates=graph_updates)
    def f(count, x):
      new_count = nnx.Variable(count[...] + 1)
      return new_count, x

    with self.assertRaisesRegex(
      ValueError,
      'scan Variable identity must be preserved',
    ):
      f(count, jnp.arange(3))

  def test_tree_mode_custom_axes(self):
    @nnx.scan(in_axes=nnx.Carry, out_axes=nnx.Carry, length=3, graph=False)
    def loop(x):
      return x

    result = loop(jnp.array(1.0))
    np.testing.assert_allclose(result, jnp.array(1.0))

  @parameterized.parameters(True, False)
  def test_only_carry(self, graph_updates):
    class Foo(nnx.Module):
      def __init__(self):
        self.c = nnx.BatchStat(jnp.array(0))

    @nnx.scan(in_axes=(nnx.Carry,), length=5, graph_updates=graph_updates)
    def loop(foo: Foo) -> tuple[Foo, jax.Array]:
      foo.c[...] += 1
      return foo, foo.c[...]

    foo = Foo()
    foo2, cs = loop(foo)
    self.assertIs(foo2.c, foo.c)
    self.assertEqual(foo.c[...], 5)
    np.testing.assert_allclose(cs, jnp.arange(1, 6))

  @parameterized.parameters(True, False)
  def test_only_carry_functional(self, graph):
    class Foo(nnx.Module):
      def __init__(self):
        self.c = nnx.BatchStat(jnp.array(0))

    @nnx.scan(
        in_axes=None, out_axes=0, length=5, graph=graph, graph_updates=False
    )
    def loop(foo: Foo) -> tuple[Foo, jax.Array]:
      foo.c[...] += 1
      return foo.c[...]

    foo = Foo()
    cs = loop(foo)
    self.assertEqual(foo.c[...], 5)
    np.testing.assert_allclose(cs, jnp.arange(1, 6))

  def test_out_axes(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5, graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), out_axes=(nnx.Carry, 1, 2), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, x, x

    module = MLP(rngs=nnx.Rngs(0))

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    c, y1, y2 = module(x)

    assert c.shape == (1, 3)
    assert y1.shape == (1, 5, 3)
    assert y2.shape == (1, 3, 5)

  @parameterized.parameters(True, False)
  def test_out_axes_functional(self, graph_mode):
    class MLP(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, x, x

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      return MLP(rngs=rngs)

    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=(nnx.Carry, 1, 2), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    c, y1, y2 = forward(module, x)

    assert c.shape == (1, 3)
    assert y1.shape == (1, 5, 3)
    assert y2.shape == (1, 3, 5)

  def test_in_axes_simple(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.vmap(in_axes=(state_axes, 0), graph=True, graph_updates=True)
      def __init__(self, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), out_axes=nnx.Carry, graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    key = jax.random.split(jax.random.key(0), 5)
    module = MLP(key=key)

    x = jnp.ones((1, 3))
    y = module(x)

    assert y.shape == (1, 3)

  @parameterized.parameters(True, False)
  def test_in_axes_simple_functional(self, graph_mode):
    class MLP(nnx.Module):
      def __init__(self, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = nnx.gelu(x)
        return x

    @nnx.vmap(in_axes=0, graph=graph_mode, graph_updates=False)
    def create(key):
      return MLP(key=key)

    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry, graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    key = jax.random.split(jax.random.key(0), 5)
    module = create(key)

    x = jnp.ones((1, 3))
    y = forward(module, x)

    assert y.shape == (1, 3)

  def test_in_axes(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState, nnx.Intermediate): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry, 0), graph=True, graph_updates=True)
      def __call__(
        self, x: jax.Array, a: jax.Array
      ) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape
        x = x + a
        x = self.linear(x)
        x = nnx.gelu(x)
        self.sow(nnx.Intermediate, "data", x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    (y, out), intermediates = nnx.capture(module, nnx.Intermediate)(x, a)

    assert y.shape == (1, 3)
    assert out is None

    assert intermediates['data'][0].shape == (5, 1, 3)

  @parameterized.parameters(True, False)
  def test_in_axes_functional(self, graph_mode):
    class MLP(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array, a: jax.Array) -> tp.Tuple[jax.Array, None]:
        assert x.shape == a.shape
        x = x + a
        x = self.linear(x)
        x = nnx.gelu(x)
        self.sow(nnx.Intermediate, "data", x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      return MLP(rngs=rngs)

    @nnx.scan(in_axes=(0, nnx.Carry, 0), graph=graph_mode, graph_updates=False)
    def forward(module, x, a):
      return module(x, a)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    (y, out), intermediates = nnx.capture(forward, nnx.Intermediate)(module, x, a)

    assert y.shape == (1, 3)
    assert out is None
    assert intermediates['data'][0].shape == (5, 1, 3)

  def test_in_axes_broadcast(self):
    test = self
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry, 0, None), graph=True, graph_updates=True)
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

    self.assertEqual(module.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.shape, (5, 3))
    self.assertEqual(module.node.shape, (2,))

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    b = jnp.ones((1, 3))
    y, out = module(x, a, b)

    self.assertEqual(y.shape, (1, 3))
    self.assertIsNone(out)

  @parameterized.parameters(True, False)
  def test_in_axes_broadcast_functional(self, graph_mode):
    test = self

    class MLP(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      def __call__(self, x: jax.Array, a: jax.Array, b: jax.Array) -> tp.Tuple[jax.Array, None]:
        test.assertEqual(x.shape, a.shape)
        test.assertEqual(x.shape, b.shape)
        x = x + a + b
        x = self.linear(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      return MLP(rngs=rngs)

    @nnx.scan(in_axes=(0, nnx.Carry, 0, None), graph=graph_mode, graph_updates=False)
    def forward(module, x, a, b):
      return module(x, a, b)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    a = jnp.ones((5, 1, 3))
    b = jnp.ones((1, 3))
    y, out = forward(module, x, a, b)

    self.assertEqual(y.shape, (1, 3))
    self.assertIsNone(out)

  def test_complex(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = module(x)

    assert y.shape == (1, 3)

  @parameterized.parameters(True, False)
  def test_complex_functional(self, graph_mode):
    class MLP(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.BatchStat(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      m = MLP(rngs=rngs)
      m.set_attributes(deterministic=False, use_running_average=False)
      return m

    @nnx.scan(in_axes=(0, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    y, _ = forward(module, x)

    assert y.shape == (1, 3)

  def test_complex_view(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(0))
    new_module = nnx.view(module, deterministic=False, use_running_average=False)

    assert new_module.linear.kernel.shape == (5, 3, 3)
    assert new_module.linear.bias.shape == (5, 3)
    assert new_module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = new_module(x)

    assert y.shape == (1, 3)

  @parameterized.parameters(True, False)
  def test_complex_view_functional(self, graph_mode):
    class MLP(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      return MLP(rngs=rngs)

    @nnx.scan(in_axes=(0, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))
    new_module = nnx.view(module, deterministic=False, use_running_average=False)

    x = jnp.ones((1, 3))
    y, _ = forward(new_module, x)

    assert y.shape == (1, 3)

  def test_complex_broadcast_dropout(self):
    state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, only='params', graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.split_rngs(splits=5, only='params', graph=True, graph_updates=True)
      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(params=0, dropout=1))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = module(x)

    assert y.shape == (1, 3)

  @parameterized.parameters((False,))
  def test_complex_broadcast_dropout_functional(self, graph_mode):

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = nnx.gelu(self.dropout(self.bn(self.linear(x))))
        return x, None

    rngs = nnx.Rngs(params=0, dropout=1)
    dummy_module = Block(rngs)
    (module_axes, rngs_axes) = nnx.prefix((dummy_module, rngs), {(nnx.Param, 'params', 'dropout'): 0, ...: None}, graph=graph_mode)

    @nnx.with_rngs(split={'params': 5}, broadcast={'dropout': 5}, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=(rngs_axes,), out_axes=module_axes, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs: nnx.Rngs):
      return Block(rngs=rngs)

    module = create(rngs)

    @nnx.scan(in_axes=(module_axes, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      module = nnx.with_attributes(module, deterministic=False, use_running_average=False)
      return module(x)

    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = forward(module, x)

    assert y.shape == (1, 3)

  def test_complex_broadcast_dropout_view(self):
    state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})

    class MLP(nnx.Module):
      @nnx.split_rngs(splits=5, only='params', graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.split_rngs(splits=5, only='params', graph=True, graph_updates=True)
      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = MLP(rngs=nnx.Rngs(params=0, dropout=1))
    new_module = nnx.view(module, deterministic=False, use_running_average=False)

    assert new_module.linear.kernel.shape == (5, 3, 3)
    assert new_module.linear.bias.shape == (5, 3)
    assert new_module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, _ = new_module(x)

    assert y.shape == (1, 3)

  def test_complex_decorator(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class Block(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5, graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = Block(rngs=nnx.Rngs(0))
    module.set_attributes(deterministic=False, use_running_average=False)

    assert module.d == 3
    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)
    assert module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, out = module(x)

    assert y.shape == (1, 3)
    assert out is None

  @parameterized.parameters(True, False)
  def test_complex_decorator_functional(self, graph_mode):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      m = Block(rngs=rngs)
      m.set_attributes(deterministic=False, use_running_average=False)
      return m

    @nnx.scan(in_axes=(0, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    y, _ = forward(module, x)

    assert y.shape == (1, 3)

  @parameterized.parameters(True, False)
  def test_complex_broadcast_dropout_view_functional(self, graph_mode):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      m = Block(rngs=rngs)
      m.set_attributes(deterministic=False, use_running_average=False)
      return m

    @nnx.scan(in_axes=(0, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))

    x = jnp.ones((1, 3))
    y, _ = forward(module, x)

    assert y.shape == (1, 3)

  def test_complex_decorator_view(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class Block(nnx.Module):
      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(in_axes=(state_axes, state_axes), axis_size=5, graph=True, graph_updates=True)
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      @nnx.scan(in_axes=(state_axes, nnx.Carry), graph=True, graph_updates=True)
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    module = Block(rngs=nnx.Rngs(0))
    new_module = nnx.view(module, deterministic=False, use_running_average=False)

    assert new_module.d == 3
    assert new_module.linear.kernel.shape == (5, 3, 3)
    assert new_module.linear.bias.shape == (5, 3)
    assert new_module.node.shape == (2,)

    x = jnp.ones((1, 3))
    y, out = new_module(x)

    assert y.shape == (1, 3)
    assert out is None

  @parameterized.parameters(True, False)
  def test_complex_decorator_view_functional(self, graph_mode):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.d = 3
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.node = nnx.Variable(jnp.ones((2,)))

      def __call__(self, x: jax.Array) -> tp.Tuple[jax.Array, None]:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = nnx.gelu(x)
        return x, None

    @nnx.split_rngs(splits=5, graph=graph_mode, graph_updates=False)
    @nnx.vmap(in_axes=0, axis_size=5, graph=graph_mode, graph_updates=False)
    def create(rngs):
      return Block(rngs=rngs)

    @nnx.scan(in_axes=(0, nnx.Carry), graph=graph_mode, graph_updates=False)
    def forward(module, x):
      return module(x)

    module = create(nnx.Rngs(0))
    new_module = nnx.view(module, deterministic=False, use_running_average=False)

    assert new_module.d == 3

    x = jnp.ones((1, 3))
    y, _ = forward(new_module, x)

    assert y.shape == (1, 3)

  def test_scan_with_sharding(self):
    test = self
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})
    transform_metadata = {nnx.PARTITION_NAME: 'layers'}

    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(
          in_axes=(state_axes, state_axes),
          transform_metadata=transform_metadata,
          graph=True,
          graph_updates=True,
      )
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
          3,
          3,
          kernel_init=nnx.with_metadata(
            nnx.initializers.lecun_normal(), out_sharding=('din', 'dout')
          ),
          bias_init=nnx.with_metadata(
            nnx.initializers.zeros_init(), out_sharding=('dout',)
          ),
          rngs=rngs,
        )

      @nnx.scan(
          in_axes=(state_axes, nnx.Carry),
          transform_metadata=transform_metadata,
          graph=True,
          graph_updates=True,
      )
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        # test sharding layer axes is not present inside scan
        test.assertEqual(self.linear.kernel.shape, (3, 3))
        test.assertEqual(self.linear.kernel.out_sharding, ('din', 'dout'))
        test.assertEqual(self.linear.bias.shape, (3,))
        test.assertEqual(self.linear.bias.out_sharding, ('dout',))
        return x, None

    mesh = jax.make_mesh((1, 1, 1), ('layers', 'din', 'dout'), axis_types=(jax.sharding.AxisType.Auto,) * len(('layers', 'din', 'dout')))
    with jax.set_mesh(mesh):
      m = MLP(rngs=nnx.Rngs(0))

    # test sharding layers axes is set
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.out_sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.out_sharding, ('layers', 'dout'))

    x = jnp.ones((1, 3))
    with jax.set_mesh(mesh):
      y, out = m(x)

    # test sharding axes is preserved
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.out_sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.out_sharding, ('layers', 'dout'))

  @parameterized.parameters(True, False)
  def test_scan_with_sharding_functional(self, graph):
    test = self

    class MLP(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            3,
            3,
            kernel_init=nnx.with_metadata(
                nnx.initializers.lecun_normal(), out_sharding=('din', 'dout')
            ),
            bias_init=nnx.with_metadata(
                nnx.initializers.zeros_init(), out_sharding=('dout',)
            ),
            rngs=rngs,
        )

      def __call__(self, x: jax.Array):
        x = self.linear(x)
        return x

    mesh = jax.make_mesh(
        (1, 1, 1),
        ('layers', 'din', 'dout'),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )

    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        graph=graph,
        graph_updates=False,
    )
    @nnx.transform_metadata(
        in_axes=0,
        out_axes=0,
        partition='layers',
        graph=graph,
    )
    def init_fn(rngs):
      return MLP(rngs)

    with jax.set_mesh(mesh):
      m = init_fn(nnx.Rngs(0).split(5))

    # verify shapes outside
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.out_sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.out_sharding, ('layers', 'dout'))

    x = jnp.ones((1, 3))

    @nnx.scan(
        in_axes=(0, nnx.Carry),
        out_axes=nnx.Carry,
        graph=graph,
        graph_updates=False,
    )
    @nnx.transform_metadata(
        in_axes=(0, nnx.Carry),
        out_axes=nnx.Carry,
        partition='layers',
        graph=graph,
    )
    def call_fn(m, x):
      y = m(x)
      # test sharding layer axes is not present inside scan
      test.assertEqual(m.linear.kernel.shape, (3, 3))
      test.assertEqual(m.linear.kernel.out_sharding, ('din', 'dout'))
      test.assertEqual(m.linear.bias.shape, (3,))
      test.assertEqual(m.linear.bias.out_sharding, ('dout',))
      return y

    with jax.set_mesh(mesh):
      y = call_fn(m, x)

    # verify shapes after call
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.out_sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    self.assertEqual(m.linear.bias.out_sharding, ('layers', 'dout'))

  @parameterized.parameters(True, False)
  def test_cache_tracing_simple(self, graph_updates):
    n = 0
    x = jnp.arange(5)
    count = jnp.array(0)

    @nnx.scan(graph_updates=graph_updates)
    def f(count, x):
      nonlocal n
      n += 1
      return count + 1, x**2

    count, y = f(count, x)
    self.assertEqual(n, 1)
    self.assertEqual(count, 5)
    np.testing.assert_allclose(y, x**2)

    count, y = f(count, x)
    self.assertEqual(n, 1)
    self.assertEqual(count, 10)

  @parameterized.parameters(True, False)
  def test_cache_tracing_object(self, graph_updates):
    n = 0
    x = jnp.arange(5)
    count = jnp.array(0)

    class Foo(nnx.Pytree):
      def __init__(self, rngs: nnx.Rngs):
        self.x = nnx.Param(rngs.normal((3,)))

    @nnx.split_rngs(splits=5)
    @nnx.vmap(graph_updates=graph_updates)
    def create_foo(rngs: nnx.Rngs):
      return Foo(rngs)

    foo = create_foo(nnx.Rngs(0))
    self.assertEqual(foo.x.shape, (5, 3))

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), graph_updates=graph_updates)
    def f(count, x, foo):
      nonlocal n
      n += 1
      self.assertEqual(foo.x.shape, (3,))
      return count + 1, x**2

    count, y = f(count, x, foo)
    self.assertEqual(n, 1)
    self.assertEqual(count, 5)
    np.testing.assert_allclose(y, x**2)

    count, y = f(count, x, foo)
    self.assertEqual(n, 1)
    self.assertEqual(count, 10)

  def test_scan_broadcast_keys(self):
    params_key = jax.random.split(jax.random.key(0), 3)
    rngs = nnx.Rngs(params=params_key, dropout=1)
    state_axes = nnx.StateAxes({'params': 0, ...: None})

    @nnx.scan(
        in_axes=(nnx.Carry, state_axes),
        length=3,
        graph=True,
        graph_updates=True,
    )
    def f(_, rngs: nnx.Rngs):
      param_key = rngs.params()
      dropout_key = rngs.dropout()
      return (), (param_key, dropout_key)

    _, (param_keys, dropout_keys) = f((), rngs)

    assert jnp.not_equal(param_keys[0], param_keys[1])
    assert jnp.not_equal(param_keys[1], param_keys[2])
    assert jnp.equal(dropout_keys[0], dropout_keys[1])
    assert jnp.equal(dropout_keys[1], dropout_keys[2])

  @parameterized.parameters(True, False)
  def test_scan_broadcast_keys_functional(self, graph):
    rngs = nnx.Rngs(params=0, dropout=1).split({'params': 3}).broadcast({'dropout': 3})
    rngs_axes = nnx.prefix(rngs, {...: 0}, graph=graph)

    @nnx.scan(
        in_axes=(rngs_axes,), out_axes=0, graph=graph, graph_updates=False
    )
    def f(rngs):
      param_key = rngs.params()
      dropout_key = rngs.dropout()
      return param_key, dropout_key

    param_keys, dropout_keys = f(rngs)

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
        self.drop = nnx.Dropout(0.1, deterministic=False, rngs=rngs)
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

      @nnx.scan(
          in_axes=(state_axes, nnx.Carry, 1),
          out_axes=(nnx.Carry, 1),
          graph=True,
          graph_updates=True,
      )
      def unroll(cell: RNNCell, carry, x) -> tuple[jax.Array, jax.Array]:
        return cell(carry, x)

      _, y = unroll(cell, carry, x)
      return y

    x = jnp.ones((16, 10, 20))
    y = rnn_forward(cell, x)

  @parameterized.parameters(True, False)
  def test_rnn_example_functional(self, graph):
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

    cell_axes = nnx.prefix(cell, {'dropout': 0, ...: None}, graph=graph)

    def rnn_forward(cell: RNNCell, x: jax.Array):
      carry = cell.initial_state(x.shape[0])

      @nnx.with_rngs(broadcast={'dropout': 10}, graph=graph, graph_updates=False)
      @nnx.scan(
          in_axes=(cell_axes, nnx.Carry, 1),
          out_axes=(nnx.Carry, 1),
          graph=graph,
          graph_updates=False,
      )
      def unroll(cell: RNNCell, carry, x) -> tuple[jax.Array, jax.Array]:
        return cell(carry, x)

      _, y = unroll(cell, carry, x)
      return y

    x = jnp.ones((16, 10, 20))
    y = rnn_forward(cell, x)

  def test_carry_pytree_sow(self):
    class CarryAsPytree(nnx.Pytree):
      def __init__(self, data: jax.Array):
        self.data = data

    class Model(nnx.Module):
      def __init__(self, num_steps):
        self.fc = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
        self.num_steps = num_steps

      def _step(self, state):
        new_data = state.data + 1
        self.sow(nnx.Intermediate, "data", new_data)
        state.data = new_data
        return state

      def _step2(self, state: tuple[CarryAsPytree, jax.Array, CarryAsPytree]):
        out = self.fc(state[1])

        new_data1 = state[0].data + 1
        new_data2 = state[2].data + 1

        self.sow(nnx.Intermediate, "data1", new_data1)
        self.sow(nnx.Intermediate, "data2", new_data2)

        state[0].data = new_data1
        state[2].data = new_data2
        return (state[0], out, state[2])

      @nnx.jit(static_argnames=("method"), graph=True, graph_updates=True)
      def __call__(self, state, method):
        state_axes = nnx.StateAxes({nnx.Intermediate: 0, ...: nnx.Carry})
        state_final = nnx.scan(
          method,
          in_axes=(state_axes, nnx.Carry),
          out_axes=nnx.Carry,
          length=self.num_steps,
          graph=True,
          graph_updates=True,
        )(self, state)

        return state_final

    num_steps = 5
    model = Model(num_steps=num_steps)
    carry = CarryAsPytree(data=jnp.array(0.0))
    carry_final, intermediates = nnx.capture(model, nnx.Intermediate)(carry, method=Model._step)
    self.assertEqual(carry_final.data, num_steps)
    np.testing.assert_array_equal(
      intermediates['data'][0], 1.0 + jnp.arange(num_steps)
    )

    carry = (
      CarryAsPytree(data=jnp.array(0.0)),
      jnp.ones((10,)),
      CarryAsPytree(data=jnp.array(10.0))
    )

    carry_final, intermediates = nnx.capture(model, nnx.Intermediate)(carry, method=Model._step2)

    self.assertEqual(carry_final[0].data, num_steps)
    self.assertEqual(carry_final[2].data, 10 + num_steps)
    np.testing.assert_array_equal(
      intermediates['data1'][0], 1.0 + jnp.arange(num_steps)
    )
    np.testing.assert_array_equal(
      intermediates['data2'][0], 11.0 + jnp.arange(num_steps)
    )

  @parameterized.parameters(True, False)
  def test_carry_pytree_sow_functional(self, graph):
    class CarryAsPytree(nnx.Pytree):
      def __init__(self, data: jax.Array):
        self.data = data

    class Model(nnx.Module):
      def __init__(self, num_steps):
        self.fc = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
        self.num_steps = num_steps

      def _step(self, state):
        new_data = state.data + 1
        self.sow(nnx.Intermediate, "data", new_data)
        state.data = new_data
        return state

      def _step2(self, state: tuple[CarryAsPytree, jax.Array, CarryAsPytree]):
        out = self.fc(state[1])

        new_data1 = state[0].data + 1
        new_data2 = state[2].data + 1

        self.sow(nnx.Intermediate, "data1", new_data1)
        self.sow(nnx.Intermediate, "data2", new_data2)

        state[0].data = new_data1
        state[2].data = new_data2
        return (state[0], out, state[2])

      @nnx.jit(static_argnames=("method",), graph=graph, graph_updates=False)
      def __call__(self, state, method):
        state_axes = nnx.prefix(
            self, {nnx.Intermediate: 0, ...: None}, graph=graph
        )
        state_final = nnx.scan(
          method,
          in_axes=(state_axes, nnx.Carry),
          out_axes=nnx.Carry,
          length=self.num_steps,
          graph=graph,
          graph_updates=False,
        )(self, state)

        return state_final

    num_steps = 5
    model = Model(num_steps=num_steps)
    carry = CarryAsPytree(data=jnp.array(0.0))
    carry_final, intermediates = nnx.capture(model, nnx.Intermediate)(carry, method=Model._step)
    self.assertEqual(carry_final.data, num_steps)
    np.testing.assert_array_equal(
      intermediates['data'][0], 1.0 + jnp.arange(num_steps)
    )

    carry = (
      CarryAsPytree(data=jnp.array(0.0)),
      jnp.ones((num_steps, 10)),
      CarryAsPytree(data=jnp.array(10.0))
    )

    carry_final, intermediates = nnx.capture(model, nnx.Intermediate)(carry, method=Model._step2)

    self.assertEqual(carry_final[0].data, num_steps)
    self.assertEqual(carry_final[2].data, 10 + num_steps)
    np.testing.assert_array_equal(
      intermediates['data1'][0], 1.0 + jnp.arange(num_steps)
    )
    np.testing.assert_array_equal(
      intermediates['data2'][0], 11.0 + jnp.arange(num_steps)
    )

  def test_broadcast_variable_mutation(self):
    v = nnx.Variable(jnp.array(1))

    @nnx.scan(
      in_axes=(None, nnx.Carry, 0), graph=False, graph_updates=False,
    )
    def fn(v, carry, x):
      v[...] = v[...] + 1
      return carry + x, carry

    carry, ys = fn(v, jnp.array(0), jnp.arange(3))
    # v is broadcast (None axis), mutated each iteration: 1 -> 2 -> 3 -> 4
    self.assertEqual(v[...], 4)
    #   step 0: carry=0, x=0 -> carry=0, y=0
    #   step 1: carry=0, x=1 -> carry=1, y=0
    #   step 2: carry=1, x=2 -> carry=3, y=1
    np.testing.assert_allclose(carry, 3)
    np.testing.assert_allclose(ys, jnp.array([0, 0, 1]))

  def test_broadcast_out_axes_rejected(self):
    with self.assertRaisesRegex(ValueError, 'broadcast'):
      @nnx.scan(
        in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, None),
        graph=False, graph_updates=False,
      )
      def fn(carry, x):
        return carry + x, jnp.zeros(3)

      fn(jnp.array(0.0), jnp.arange(3.0))

  def test_scan_inconsistent_aliasing(self):
    v = nnx.Param(jnp.array(0.0))

    @nnx.scan(
      in_axes=(nnx.Carry, 0),
      out_axes=(nnx.Carry, 0),
      graph=True,
      graph_updates=False,
    )
    def f(carry, x):
      return carry, x[...]

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
      f(v, v)

  def test_scan_input_output_aliasing(self):
    v = nnx.Param(jnp.arange(5))

    @nnx.scan(in_axes=0, out_axes=0, graph=True, graph_updates=False)
    def f(carry):
      return carry

    with self.assertRaisesRegex(ValueError, 'does not support Variable aliasing'):
      f(v)

  def test_scan_carry_and_scan(self):
    def cumsum(carry, x):
      carry = carry + x
      return carry, carry

    final_carry, ys = nnx.scan(
        cumsum,
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.array(0.0), jnp.arange(5.0))
    np.testing.assert_allclose(final_carry, 10.0)
    np.testing.assert_allclose(ys, jnp.array([0., 1., 3., 6., 10.]))

  def test_scan_pytree_carry(self):
    def dict_scan(carry, x):
      carry = {'a': carry['a'] + x['a'], 'b': carry['b'] + x['b']}
      return carry, carry

    xs = {'a': jnp.arange(3.0), 'b': jnp.ones(3)}
    init = {'a': jnp.array(0.0), 'b': jnp.array(0.0)}
    final_carry, _ = nnx.scan(
        dict_scan,
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(init, xs)
    np.testing.assert_allclose(final_carry['a'], 3.0)
    np.testing.assert_allclose(final_carry['b'], 3.0)

  def test_scan_reverse(self):
    def cumsum(carry, x):
      carry = carry + x
      return carry, carry

    final_carry, _ = nnx.scan(
        cumsum,
        in_axes=(nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        reverse=True,
        graph=False,
    )(jnp.array(0.0), jnp.arange(5.0))
    np.testing.assert_allclose(final_carry, 10.0)

  def test_scan_axis_1(self):
    def cumsum(carry, x):
      carry = carry + x
      return carry, carry

    x = jnp.arange(10.0).reshape((2, 5))
    final_carry, ys = nnx.scan(
        cumsum,
        in_axes=(nnx.Carry, 1),
        out_axes=(nnx.Carry, 1),
        graph=False,
    )(jnp.zeros(2), x)
    np.testing.assert_allclose(final_carry, jnp.array([10.0, 35.0]))
    expected_ys = jnp.array([
        [0., 1., 3., 6., 10.],
        [5., 11., 18., 26., 35.]
    ])
    np.testing.assert_allclose(ys, expected_ys)

  def test_scan_axis_negative_1(self):
    def cumsum(carry, x):
      carry = carry + x
      return carry, carry

    x = jnp.arange(10.0).reshape((2, 5))
    final_carry, ys = nnx.scan(
        cumsum,
        in_axes=(nnx.Carry, -1),
        out_axes=(nnx.Carry, -1),
        graph=False,
    )(jnp.zeros(2), x)
    np.testing.assert_allclose(final_carry, jnp.array([10.0, 35.0]))
    expected_ys = jnp.array([
        [0., 1., 3., 6., 10.],
        [5., 11., 18., 26., 35.]
    ])
    np.testing.assert_allclose(ys, expected_ys)

  def test_scan_different_in_out_axes(self):
    def cumsum(carry, x):
      carry = carry + x
      return carry, carry

    x = jnp.arange(10.0).reshape((2, 5))
    final_carry, ys = nnx.scan(
        cumsum,
        in_axes=(nnx.Carry, 1),
        out_axes=(nnx.Carry, 0),
        graph=False,
    )(jnp.zeros(2), x)
    np.testing.assert_allclose(final_carry, jnp.array([10.0, 35.0]))
    expected_ys = jnp.array([
        [0., 5.],
        [1., 11.],
        [3., 18.],
        [6., 26.],
        [10., 35.]
    ])
    np.testing.assert_allclose(ys, expected_ys)


class TestRemat(parameterized.TestCase):
  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_remat_basic(self, graph, graph_updates):
    class RematLinear(nnx.Module):
      def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)

      @nnx.remat(graph=graph, graph_updates=graph_updates)
      def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)

    module = RematLinear(2, 3, nnx.Rngs(0))

    def loss_fn(module, x):
      y = module(x)
      return jnp.sum(y)

    grad_type = nnx.State if graph and graph_updates else RematLinear
    loss, grads = nnx.value_and_grad(
      loss_fn, graph=graph, graph_updates=graph_updates,
    )(module, jnp.ones((1, 2)))

    assert loss.shape == ()
    assert isinstance(grads, grad_type)

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, False),
  )
  def test_remat_variables(self, graph, graph_updates):
    rngs = nnx.Rngs(0)
    w = nnx.Param(jax.random.normal(rngs(), (2, 3)))
    b = nnx.Param(jax.random.normal(rngs(), (3,)))
    count = nnx.BatchStat(jnp.array(0))

    @nnx.remat(graph=graph, graph_updates=graph_updates)
    def linear(w, b, count, x):
      count[...] += 1
      return x @ w + b[None]

    def loss_fn(w, b, count, x):
      return jnp.sum(linear(w, b, count, x))

    x = jnp.ones((1, 2))
    loss, grads = nnx.value_and_grad(
      loss_fn, argnums=(0, 1), graph=graph, graph_updates=graph_updates,
    )(w, b, count, x)

    assert loss.shape == ()
    assert isinstance(grads, tuple)
    assert len(grads) == 2
    assert count[...] == 1

  def test_remat_with_scan_decorator(self):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class ScanLinear(nnx.Module):

      @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
      @nnx.vmap(
          in_axes=(state_axes, state_axes),
          axis_size=5,
          graph=True,
          graph_updates=True,
      )
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)

      @nnx.scan(
          in_axes=(state_axes, nnx.Carry),
          out_axes=nnx.Carry,
          graph=True,
          graph_updates=True,
      )
      @nnx.remat(graph=True, graph_updates=True)
      def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)

    m = ScanLinear(nnx.Rngs(0))

    assert m.linear.kernel.shape == (5, 3, 3)
    assert m.linear.bias.shape == (5, 3)

    y = m(jnp.ones((1, 3)))
    assert y.shape == (1, 3)

  @parameterized.parameters(True, False)
  def test_remat_with_scan_decorator_functional(self, graph):
    state_axes = nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None})

    class ScanLinear(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=5, graph=graph, graph_updates=False)
        @nnx.vmap(in_axes=0, axis_size=5, graph=graph, graph_updates=False)
        def create(rngs):
          return nnx.Linear(3, 3, rngs=rngs)

        self.linear = create(rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        @nnx.scan(
            in_axes=(0, nnx.Carry),
            out_axes=nnx.Carry,
            graph=graph,
            graph_updates=False,
        )
        @nnx.remat(graph=graph, graph_updates=False)
        def forward(linear, x):
          return linear(x)

        return forward(self.linear, x)

    m = ScanLinear(nnx.Rngs(0))

    assert m.linear.kernel.shape == (5, 3, 3)

    x = jnp.ones((2, 3))
    y = m(x)

    assert y.shape == (2, 3)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_remat_basic(self, graph, graph_updates):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.remat(graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    def loss_fn(model, x):
      y = forward(model, x)
      return jnp.sum(y)

    grads = nnx.grad(
      loss_fn, graph=graph, graph_updates=graph_updates,
    )(model, jnp.ones((1, 2)))
    assert grads.kernel.shape == (2, 3)
    assert grads.bias.shape == (3,)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_tree_mode_remat_stateful(self, graph, graph_updates):
    class Counter(nnx.Variable):
      pass

    class Linear(nnx.Module):
      def __init__(self, din, dout, *, rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.count = Counter(jnp.array(0))

      def __call__(self, x):
        self.count[...] += 1
        return x @ self.w + self.b[None]

    model = Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.remat(graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    y = forward(model, jnp.ones((1, 2)))
    assert y.shape == (1, 3)
    assert model.count[...] == 1


class TestVmap(parameterized.TestCase):
  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_vmap_basic(self, graph, graph_updates):
    class LinearEnsemble(nnx.Module):
      def __init__(self, num, *, rngs):
        self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))

    model = LinearEnsemble(5, rngs=nnx.Rngs(0))
    x = jnp.ones((2,))

    @nnx.vmap(in_axes=(0, None), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return x @ model.w

    y = forward(model, x)
    assert y.shape == (5, 3)

  def test_prefix_graph_node_error(self):
    m = nnx.Dict(a=nnx.Param(1))

    with self.assertRaisesRegex(
      ValueError, 'Graph nodes are not allowed as prefixes'
    ):
      nnx.vmap(lambda x: x, in_axes=m, graph=True, graph_updates=True)

  def test_prefix_mapping_tree_mode_error(self):
    axes = nnx.StateAxes({nnx.PathContains('a'): 0})

    with self.assertRaisesRegex(
      ValueError, 'cannot contain `StateAxes` objects'
    ):
      nnx.vmap(
        lambda x: x, in_axes=axes, graph=False, graph_updates=False
      )

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_vmap_stateful(self, graph, graph_updates):
    class Counter(nnx.Variable):
      pass

    class Linear(nnx.Module):
      def __init__(self, din, dout, *, rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.count = Counter(jnp.array(0))

      def __call__(self, x):
        self.count[...] += 1
        return x @ self.w

    model = Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(None, 0), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    x = jnp.ones((5, 2))
    y = forward(model, x)
    assert y.shape == (5, 3)
    assert model.count[...] == 1

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_vmap_variables(self, graph, graph_updates):
    rngs = nnx.Rngs(0)
    w = nnx.Param(jax.random.normal(rngs(), (5, 2, 3)))
    b = nnx.Param(jax.random.normal(rngs(), (5, 3)))

    @nnx.vmap(in_axes=(0, 0, 1), out_axes=1, graph=graph, graph_updates=graph_updates)
    def forward(w, b, x):
      return x @ w + b

    x = jax.random.uniform(rngs(), (2, 5))
    y = forward(w, b, x)
    assert y.shape == (3, 5)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_vmap_ensemble_forward(self, graph, graph_updates):
    class Linear(nnx.Module):
      def __init__(self, din, dout, *, rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

      def __call__(self, x):
        return x @ self.w + self.b[None]

    @nnx.vmap(in_axes=0, out_axes=0, graph=graph, graph_updates=graph_updates)
    def create_ensemble(keys):
      return Linear(2, 3, rngs=nnx.Rngs(keys))

    keys = jax.random.split(jax.random.key(0), 5)
    ensemble = create_ensemble(keys)

    assert ensemble.w.shape == (5, 2, 3)
    assert ensemble.b.shape == (5, 3)

    @nnx.vmap(in_axes=(0, None), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    x = jnp.ones((1, 2))
    y = forward(ensemble, x)
    assert y.shape == (5, 1, 3)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_vmap_replicate(self, graph, graph_updates):
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(None, 0), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    x = jnp.ones((5, 1, 2))
    y = forward(model, x)
    assert y.shape == (5, 1, 3)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic(self, graph, graph_updates):
    @nnx.split_rngs(splits=5)
    @nnx.vmap(in_axes=0, out_axes=0, axis_size=5, graph=graph, graph_updates=graph_updates)
    def create_block(rngs: nnx.Rngs):
      return nnx.Linear(2, 3, rngs=rngs)

    rngs = nnx.Rngs(0)

    block = create_block(rngs)

    self.assertEqual(block.kernel.shape, (5, 2, 3))
    self.assertEqual(rngs.default.count[...], 1)

    @nnx.vmap(in_axes=(0, 1), out_axes=1, graph=graph, graph_updates=graph_updates)
    def forward(block: nnx.Linear, x):
      self.assertEqual(block.kernel.shape, (2, 3))
      self.assertEqual(block.bias.shape, (3,))
      self.assertEqual(x.shape, (2,))
      return block(x)

    x = jax.random.uniform(rngs(), (2, 5))
    y = forward(block, x)

    self.assertEqual(y.shape, (3, 5))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic_variables(self, graph, graph_updates):
    @nnx.split_rngs(splits=5)
    @nnx.vmap(in_axes=0, out_axes=0, axis_size=5, graph=graph, graph_updates=graph_updates)
    def create_block(rngs: nnx.Rngs):
      w = nnx.Param(jax.random.normal(rngs(), (2, 3)))
      b = nnx.Param(jax.random.normal(rngs(), (3,)))
      return w, b

    rngs = nnx.Rngs(0)
    w, b = create_block(rngs)

    self.assertEqual(w.shape, (5, 2, 3))
    self.assertEqual(b.shape, (5, 3))
    self.assertEqual(rngs.default.count[...], 1)

    @nnx.vmap(in_axes=(0, 0, 1), out_axes=1, graph=graph, graph_updates=graph_updates)
    def forward(w, b, x):
      self.assertEqual(w.shape, (2, 3))
      self.assertEqual(b.shape, (3,))
      self.assertEqual(x.shape, (2,))
      return x @ w + b

    x = jax.random.uniform(rngs(), (2, 5))
    y = forward(w, b, x)

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
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key[...]

    backups = nnx.split_rngs(rngs, splits=5, graph=True, graph_updates=True)
    module = create_block(rngs)
    nnx.restore_rngs(backups)

    self.assertEqual(rngs.default.count[...], 1)
    self.assertEqual(rngs.default.key[...], initial_key)
    self.assertFalse(
        jnp.allclose(
            module.linear.kernel[0],
            module.linear.kernel[1],
        )
    )
    self.assertEqual(module.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.shape, (5, 3))

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(
        in_axes=(nnx.StateAxes({(nnx.Param, nnx.RngState): 0, ...: None}), 0),
        graph=True,
        graph_updates=True,
    )
    def forward_block(module, x):
      return module(x)

    backups = nnx.split_rngs(rngs, splits=5, graph=True, graph_updates=True)
    y = forward_block(module, x)
    nnx.restore_rngs(backups)

    self.assertEqual(y.shape, (5, 1, 3))
    self.assertEqual(rngs.default.count[...], 2)
    self.assertEqual(rngs.default.key[...], initial_key)

    y2 = forward_block(module, x)

    self.assertFalse(jnp.allclose(y, y2))

  @parameterized.parameters(True, False)
  def test_state_axes_functional(self, graph):

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key[...]

    vec_filter = (nnx.Param, nnx.RngState)
    unb_filter = ...

    abs_block = nnx.eval_shape(lambda: Block(nnx.Rngs(0)))
    model_axes = nnx.prefix(
        abs_block, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.with_rngs(split=5, graph=graph, graph_updates=False)
    @nnx.vmap(
        in_axes=0,
        out_axes=model_axes,
        graph=graph,
        graph_updates=False,
    )
    def create_block_functional(rngs):
      return Block(rngs)

    module = create_block_functional(rngs)

    self.assertEqual(rngs.default.count[...], 1)
    self.assertTrue(np.all(rngs.default.key[...] == initial_key))

    self.assertFalse(
        jnp.allclose(
            module.linear.kernel[0],
            module.linear.kernel[1],
        )
    )
    self.assertEqual(module.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.shape, (5, 3))

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=False,
    )
    def forward_block_functional(module, x):
      return module(x)

    y = forward_block_functional(module, x)

    self.assertEqual(y.shape, (5, 1, 3))
    self.assertEqual(rngs.default.count[...], 1)
    self.assertTrue(np.all(rngs.default.key[...] == initial_key))

    y2 = forward_block_functional(module, x)
    self.assertFalse(jnp.allclose(y, y2))

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

    @nnx.vmap(
        in_axes=(state_axes,),
        out_axes=state_axes,
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key[...]

    module = create_block(rngs.split(5))

    assert rngs.default.count[...] == 1
    assert rngs.default.key[...] == initial_key
    assert not jnp.allclose(
      module.linear.kernel[0],
      module.linear.kernel[1],
    )
    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(in_axes=(state_axes, 0), graph=True, graph_updates=True)
    def forward_block(module, x):
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.key[...] == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  @parameterized.parameters(True, False)
  def test_split_rngs_context_manager_functional(self, graph):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    vec_filter = (nnx.Param, nnx.RngState)
    unb_filter = ...

    abs_block = nnx.eval_shape(lambda: Block(nnx.Rngs(0)))
    model_axes = nnx.prefix(
        abs_block, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.vmap(
        in_axes=(0,), out_axes=model_axes, graph=graph, graph_updates=False
    )
    def create_block_functional(rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)

    module = create_block_functional(rngs.split(5))

    self.assertEqual(module.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.shape, (5, 3))
    self.assertFalse(
        jnp.allclose(module.linear.kernel[0], module.linear.kernel[1])
    )

    x = jnp.ones((5, 1, 3))

    model_axes = nnx.prefix(
        module, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=False,
    )
    def forward_block_functional(module, x):
      return module(x)

    y = forward_block_functional(module, x)

    self.assertEqual(y.shape, (5, 1, 3))

    y2 = forward_block_functional(module, x)

    self.assertFalse(jnp.allclose(y, y2))

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

    @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
    @nnx.vmap(
        in_axes=(state_axes,),
        out_axes=state_axes,
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key[...]

    module = create_block(rngs)

    assert rngs.default.count[...] == 1
    assert rngs.default.key[...] == initial_key
    assert not jnp.allclose(
      module.linear.kernel[0],
      module.linear.kernel[1],
    )
    assert module.linear.kernel.shape == (5, 3, 3)
    assert module.linear.bias.shape == (5, 3)

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(in_axes=(state_axes, 0), graph=True, graph_updates=True)
    def forward_block(module, x):
      self.assertEqual(x.shape, (1, 3))
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)
    assert rngs.default.key[...] == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  @parameterized.parameters(True, False)
  def test_split_rngs_decorator_functional(self, graph):
    graph_updates = False

    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x

    vec_filter = (nnx.Param, nnx.RngState)
    unb_filter = ...

    abs_block = nnx.eval_shape(lambda: Block(nnx.Rngs(0)))
    model_axes = nnx.prefix(
        abs_block, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.split_rngs(splits=5, graph=graph, graph_updates=graph_updates)
    @nnx.vmap(
        in_axes=(0,),
        out_axes=model_axes,
        graph=graph,
        graph_updates=graph_updates,
    )
    def create_block_functional(rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    initial_key = rngs.default.key[...]

    module = create_block_functional(rngs)

    if graph and graph_updates:
      self.assertEqual(rngs.default.count[...], 1)
      self.assertTrue(jnp.allclose(rngs.default.key[...], initial_key))

    self.assertEqual(module.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(module.linear.bias.shape, (5, 3))
    self.assertFalse(
        jnp.allclose(module.linear.kernel[0], module.linear.kernel[1])
    )

    x = jnp.ones((5, 1, 3))

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=graph_updates,
    )
    def forward_block_functional(module, x):
      return module(x)

    y = forward_block_functional(module, x)

    self.assertEqual(y.shape, (5, 1, 3))

    y2 = forward_block_functional(module, x)

    self.assertFalse(jnp.allclose(y, y2))

  def test_state_axes_simple(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, use_running_average=False, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    state_axes = nnx.StateAxes({(nnx.BatchStat, 'dropout'): 0, ...: None})

    @nnx.split_rngs(splits=5, only='dropout')
    @nnx.vmap(
        in_axes=(state_axes,),
        out_axes=state_axes,
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(params=0, dropout=1)
    module = create_block(rngs)

    assert module.linear.kernel.shape == (2, 3)
    assert module.bn.scale.shape == (3,)
    assert module.bn.mean.shape == (5, 3)

    @nnx.vmap(
        in_axes=(state_axes, 0), out_axes=0, graph=True, graph_updates=True
    )
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert y.shape == (5, 1, 3)

  @parameterized.parameters(True, False)
  def test_state_axes_simple_functional(self, graph):
    graph_updates = False

    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    rngs = nnx.Rngs(params=0, dropout=1).split({'dropout': 5})

    vec_filter = (nnx.BatchStat, 'dropout')
    unb_filter = ...

    abs_block = nnx.eval_shape(lambda: Block(nnx.Rngs(params=0, dropout=1)))
    model_axes = nnx.prefix(
        abs_block, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.vmap(
        in_axes=(None, 0),
        out_axes=model_axes,
        graph=graph,
        graph_updates=graph_updates,
    )
    def create_block_functional(params, dropout):
      return Block(nnx.Rngs(params=params, dropout=dropout))

    module = create_block_functional(rngs.params, rngs.dropout)

    self.assertEqual(module.linear.kernel.shape, (2, 3))
    self.assertEqual(module.bn.scale.shape, (3,))
    self.assertEqual(module.bn.mean.shape, (5, 3))

    initial_mean = module.bn.mean[...]

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=graph_updates,
    )
    def forward_block_functional(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block_functional(module, x)

    self.assertEqual(y.shape, (5, 1, 3))
    # Verify that updates were tracked and applied
    self.assertFalse(jnp.allclose(initial_mean, module.bn.mean[...]))

  def test_split_rngs_decorator_simple(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, use_running_average=False, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    state_axes = nnx.StateAxes({(nnx.BatchStat, 'dropout'): 0, ...: None})

    @nnx.split_rngs(splits=5, only='dropout', graph=True, graph_updates=True)
    @nnx.vmap(
        in_axes=(state_axes,),
        out_axes=state_axes,
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(params=0, dropout=1)

    module = create_block(rngs)

    assert module.linear.kernel.shape == (2, 3)
    assert module.bn.scale.shape == (3,)
    assert module.bn.mean.shape == (5, 3)
    assert module.dropout.rngs is not None
    self.assertEqual(module.dropout.rngs.key.shape, (5,))

    @nnx.vmap(
        in_axes=(state_axes, 0),
        out_axes=0,
        graph=True,
        graph_updates=True,
    )
    def forward_block(module: Block, x):
      assert module.dropout.rngs is not None
      self.assertEqual(module.dropout.rngs.key.shape, ())
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    assert module.dropout.rngs is not None
    self.assertEqual(module.dropout.rngs.key.shape, (5,))
    assert y.shape == (5, 1, 3)

  @parameterized.parameters(True, False)
  def test_split_rngs_decorator_simple_functional(self, graph):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, use_running_average=False, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    vec_filter = (nnx.BatchStat, 'dropout')
    unb_filter = ...

    rngs = nnx.Rngs(params=0, dropout=1)
    abs_block = nnx.eval_shape(lambda: Block(nnx.clone(rngs)))
    model_axes = nnx.prefix(
        abs_block, {vec_filter: 0, unb_filter: None}, graph=graph
    )
    rngs_axes = nnx.prefix(rngs, {'dropout': 0, ...: None}, graph=graph)

    @nnx.split_rngs(splits=5, only='dropout')
    @nnx.vmap(
        in_axes=(rngs_axes,),
        out_axes=model_axes,
        graph=graph,
        graph_updates=False,
    )
    def create_block_functional(rngs):
      return Block(rngs)

    module = create_block_functional(rngs)

    self.assertEqual(module.linear.kernel.shape, (2, 3))
    self.assertEqual(module.bn.scale.shape, (3,))
    self.assertEqual(module.bn.mean.shape, (5, 3))
    self.assertIsNotNone(module.dropout.rngs)
    self.assertEqual(module.dropout.rngs.key.shape, (5,))

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=False,
    )
    def forward_block_functional(module, x):
      self.assertIsNotNone(module.dropout.rngs)
      self.assertEqual(module.dropout.rngs.key.shape, ())
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block_functional(module, x)

    self.assertIsNotNone(module.dropout.rngs)
    self.assertEqual(module.dropout.rngs.key.shape, (5,))
    self.assertEqual(y.shape, (5, 1, 3))

  @parameterized.parameters(
      (True, True),
      (True, False),
      (False, False),
  )
  def test_state_axes_super_simple(self, graph, graph_updates):
    class Block(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.bn = nnx.BatchNorm(3, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    @nnx.split_rngs(splits=5, graph=graph, graph_updates=graph_updates)
    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        graph=graph,
        graph_updates=graph_updates,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    self.assertEqual(module.linear.kernel.shape, (5, 2, 3))
    self.assertEqual(module.bn.scale.shape, (5, 3))
    self.assertEqual(module.bn.mean.shape, (5, 3))

    @nnx.vmap(
        in_axes=(0, 0),
        out_axes=0,
        graph=graph,
        graph_updates=graph_updates,
    )
    def forward_block(module, x):
      return module(x)

    x = jnp.ones((5, 1, 2))
    y = forward_block(module, x)

    self.assertEqual(y.shape, (5, 1, 3))

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

    @nnx.split_rngs(splits=5, graph=True, graph_updates=True)
    @nnx.vmap(
        in_axes=(state_axes, 0), out_axes=0, graph=True, graph_updates=True
    )
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)
    initial_key = module.dropout.rngs.key[...]

    self.assertEqual(module.dropout.rngs.count[...], 0)
    self.assertEqual(module.linear.kernel.shape, (din, dout))
    self.assertEqual(module.linear.bias.shape, (dout,))

    x = jnp.ones((5, 1, din))

    y = forward_block(module, x)

    self.assertEqual(y.shape, (5, 1, dout))
    self.assertEqual(module.dropout.rngs.count[...], 1)

    self.assertFalse(jnp.allclose(y[0], y[1]))

    y2 = forward_block(module, x)

    self.assertFalse(jnp.allclose(y, y2))

    self.assertTrue(np.all(module.dropout.rngs.key[...] == initial_key))

  @parameterized.parameters(True, False)
  def test_replicate_functional(self, graph):
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

    vec_filter = nnx.RngState
    unb_filter = ...

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    self.assertEqual(module.dropout.rngs.count[...], 0)
    self.assertEqual(module.linear.kernel.shape, (din, dout))

    x = jnp.ones((5, 1, din))

    module = nnx.split_rngs(module, splits=5, graph=graph, graph_updates=False)
    model_axes = nnx.prefix(
        module, {vec_filter: 0, unb_filter: None}, graph=graph
    )

    @nnx.vmap(
        in_axes=(model_axes, 0),
        out_axes=0,
        graph=graph,
        graph_updates=False,
    )
    def forward_block_functional(module, x):
      return module(x)

    y = forward_block_functional(module, x)

    self.assertEqual(y.shape, (5, 1, dout))
    self.assertFalse(jnp.allclose(y[0], y[1]))

    y2 = forward_block_functional(module, x)

    self.assertFalse(jnp.allclose(y, y2))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_consistent_aliasing_inputs(self, graph, graph_updates):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.zeros((5, 5)))

    m = Foo()

    @nnx.vmap(in_axes=(0, 1), graph=graph, graph_updates=graph_updates)
    def f(m1, m2):
      pass

    error_msg = (
        'Inconsistent aliasing detected' if graph else 'Duplicate Param'
    )
    with self.assertRaisesRegex(ValueError, error_msg):
      f(m, m)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_consistent_aliasing_input_output(self, graph, graph_updates):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.zeros((2, 3)))

    m = Foo()

    @nnx.vmap(in_axes=0, out_axes=1, graph=graph, graph_updates=graph_updates)
    def f(m):
      return m

    error_msg = (
        'Inconsistent aliasing detected' if graph_updates else 'Duplicate Param'
    )
    with self.assertRaisesRegex(ValueError, error_msg):
      m2 = f(m)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_consistent_aliasing_shared(self, graph, graph_updates):
    class Shared(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.zeros((3, 3)))

    class Foo(nnx.Module):
      def __init__(self, shared: Shared):
        self.a = shared

    shared = Shared()
    m1 = Foo(shared)
    m2 = Foo(shared)

    @nnx.vmap(in_axes=(0, 1), graph=graph, graph_updates=graph_updates)
    def f(m1, m2):
      pass

    error_msg = (
        r'Inconsistent aliasing detected([\s\S]*)Param([\s\S]*)a:'
        r' 0([\s\S]*)a: 1'
        if graph else 'Duplicate Param'
    )
    with self.assertRaisesRegex(ValueError, error_msg):
      f(m1, m2)

  def test_equivalent_state_axes_mapping(self):
    m = nnx.Linear(3, 3, rngs=nnx.Rngs(0))

    sa1 = nnx.StateAxes({...: 0})
    sa2 = nnx.StateAxes({nnx.Param: 0})

    @nnx.vmap(in_axes=(0, sa1, sa2), graph=True, graph_updates=True)
    def f(m1, m2, m3):
      pass

    f(m, m, m)

  def test_equivalent_state_axes_mapping_functional(self):
    m = nnx.eval_shape(lambda: nnx.Linear(3, 3, rngs=nnx.Rngs(0)))

    sa1, sa2, sa3 = nnx.prefix((m, m, m), {...: 0}, graph=True)

    @nnx.vmap(out_axes=(sa1, sa2, sa3), axis_size=2, graph=True, graph_updates=False)
    def f():
      m = nnx.Linear(3, 3, rngs=nnx.Rngs(0))
      return m, m, m

    m1, m2, m3 = f()

    assert m1 is m2 and m2 is m3
    self.assertEqual(m1.kernel.shape, (2, 3, 3))

  def test_equivalent_state_sharding_mapping(self):
    m = nnx.Linear(4, 4, rngs=nnx.Rngs(0))

    mesh = jax.sharding.Mesh(jax.devices(), ('mp',))
    sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('mp')
    )

    sa1 = nnx.StateSharding({...: sharding})
    sa2 = nnx.StateSharding({nnx.Param: sharding})

    @nnx.jit(in_shardings=(sharding, sa1, sa2), graph=True, graph_updates=True)
    def f(m1, m2, m3):
      pass

    f(m, m, m)

    assert m.kernel.sharding == sharding

  def test_equivalent_state_sharding_mapping_functional(self):

    mesh = jax.sharding.Mesh(jax.devices(), ('mp',))
    m = nnx.eval_shape(lambda: nnx.Linear(4, 4, rngs=nnx.Rngs(0)))

    sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('mp')
    )

    sa1, sa2, sa3 = nnx.prefix((m, m, m), {...: sharding}, graph=True)

    @nnx.jit(out_shardings=(sa1, sa2, sa3), graph=True, graph_updates=False)
    def f():
      m = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
      return m, m, m

    with jax.set_mesh(mesh):
      m1, m2, m3 = f()

    assert m1 is m2 and m2 is m3
    assert m1.kernel.sharding == sharding

  @parameterized.parameters(True, False)
  def test_captured_module_in_return_error(self, graph_updates):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Variable(jnp.arange(4))

    m = Foo()

    @nnx.vmap(in_axes=0, out_axes=0, graph=True, graph_updates=graph_updates)
    def f(x):
      return m

    if graph_updates:
      error_regex = 'Cannot extract graph node from different trace level'
    else:
      error_regex = 'Cannot return captured Variable'

    with self.assertRaisesRegex(ValueError, error_regex):
      f(jnp.zeros((4,)))

  def test_vmap_and_cond_passthrough(self):
    class Broadcast(nnx.Variable[nnx.A]): ...

    class Vectorized(nnx.Variable[nnx.A]): ...

    class Env(nnx.Module):
      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.vmap(
        in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),),
        graph=True,
        graph_updates=True,
    )
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step[...] += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(
          is_even, increment, no_nothing, env, graph=True, graph_updates=True
      )

    f(env)

    np.testing.assert_array_equal(env.step[...], [1, 0, 1, 0, 1, 0, 1, 0])

  @parameterized.parameters(True, False)
  def test_vmap_and_cond_passthrough_functional_success(self, graph):
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

    env_axes = nnx.prefix(env, {Broadcast: None, Vectorized: 0}, graph=graph)

    @nnx.vmap(in_axes=(env_axes,), graph=graph, graph_updates=False)
    def f(env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step[...] += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(
          is_even,
          increment,
          no_nothing,
          env,
          graph=graph,
          graph_updates=False,
      )

    f(env)
    np.testing.assert_array_equal(env.step[...], [1, 0, 1, 0, 1, 0, 1, 0])

  def test_vmap_and_cond_passthrough_error(self):
    class Broadcast(nnx.Variable[nnx.A]): ...

    class Vectorized(nnx.Variable[nnx.A]): ...

    class Env(nnx.Module):
      def __init__(self):
        self.broadcast = Broadcast(jnp.array(1))
        self.index = Vectorized(jnp.arange(8))
        self.step = Vectorized(jnp.zeros((8,), jnp.uint32))

    env = Env()

    @nnx.vmap(
        in_axes=(nnx.StateAxes({Broadcast: None, Vectorized: 0}),),
        graph=True,
        graph_updates=True,
    )
    def f(env: Env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step[...] += 1
        env.broadcast[...] += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(
          is_even, increment, no_nothing, env, graph=True, graph_updates=True
      )

    with self.assertRaisesRegex(
      ValueError,
      r"at vmap.*'broadcast'.*got axis spec None but output was batched on"
      r' axis 0',
    ):
      f(env)

  @parameterized.parameters(True, False)
  def test_vmap_and_cond_passthrough_functional(self, graph):
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

    model_axes = nnx.prefix(env, {Broadcast: None, Vectorized: 0}, graph=graph)

    @nnx.vmap(
        in_axes=(model_axes,),
        out_axes=model_axes,
        graph=graph,
        graph_updates=False,
    )
    def f(env):
      self.assertEqual(env.step.shape, ())

      def increment(env: Env):
        env.step[...] += 1
        env.broadcast[...] += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(
          is_even,
          increment,
          no_nothing,
          env,
          graph=graph,
          graph_updates=False,
      )
      # Returning the module outputs the unmodified graph state (aliasing the input)
      # which triggers the ValueError when graph_updates=False
      return env

    with self.assertRaisesRegex(ValueError, r'Duplicate Broadcast'):
      f(env)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_example(self, graph, graph_updates):
    class Model(nnx.Module):
      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, use_running_average=False, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

    @nnx.vmap(in_axes=0, out_axes=0, graph=graph, graph_updates=graph_updates)
    def initialize_ensemble(key):
      rngs = nnx.Rngs(key)
      return Model(2, 3, rngs=rngs)

    keys = jax.random.split(jax.random.key(0), 5)
    ensemble = initialize_ensemble(keys)

    self.assertEqual(ensemble.linear.kernel.shape, (5, 2, 3))

    @nnx.vmap(in_axes=(0, None), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    x = jnp.ones((4, 2))
    y = forward(ensemble, x)
    self.assertEqual(y.shape, (5, 4, 3))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_example_with_vectorization(self, graph, graph_updates):
    class LinearEnsemble(nnx.Module):
      def __init__(self, num, rngs):
        self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))

    model = LinearEnsemble(5, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(0, None), out_axes=0, graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      self.assertEqual(model.w.shape, (2, 3))
      return jnp.dot(x, model.w)

    x = jnp.ones((4, 2))
    y = forward(model, x)

    self.assertEqual(y.shape, (5, 4, 3))

  def test_metadata_graph_updates(self):
    @nnx.compat.vmap(
        in_axes=(None,),
        out_axes=0,
        axis_size=5,
        transform_metadata={nnx.spmd.PARTITION_NAME: 'c'},
        graph_updates=True,
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

    mesh = jax.make_mesh(
        (1, 1, 1),
        ('a', 'b', 'c'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('a', 'b', 'c')),
    )
    with jax.set_mesh(mesh):
      m = create_block(nnx.Rngs(0))
    self.assertEqual(m.kernel.shape, (5, 16, 32))
    self.assertEqual(m.kernel.out_sharding, ('c', 'a', 'b'))

  @parameterized.parameters(True, False)
  def test_metadata_graph_updates_functional(self, graph):
    @nnx.vmap(
        in_axes=(None,),
        out_axes=0,
        axis_size=5,
        graph=graph,
        graph_updates=False,
    )
    @nnx.transform_metadata(
        in_axes=(None,),
        out_axes=0,
        partition='c',
        graph=graph,
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

    mesh = jax.make_mesh(
        (1, 1, 1),
        ('a', 'b', 'c'),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('a', 'b', 'c')),
    )
    with jax.set_mesh(mesh):
      m = create_block(nnx.Rngs(0))
    self.assertEqual(m.kernel.shape, (5, 16, 32))
    self.assertEqual(m.kernel.out_sharding, ('c', 'a', 'b'))

  def test_metadata_transform_metadata(self):
    @nnx.vmap(
        in_axes=(None,),
        out_axes=0,
        axis_size=5,
        graph_updates=False,
    )
    @nnx.transform_metadata(
        in_axes=(None,),
        out_axes=0,
        partition='c',
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

    mesh = jax.make_mesh((1, 1, 1), ('a', 'b', 'c'), axis_types=(jax.sharding.AxisType.Auto,) * len(('a', 'b', 'c')))
    with jax.set_mesh(mesh):
      m = create_block(nnx.Rngs(0))
    self.assertEqual(m.kernel.shape, (5, 16, 32))
    self.assertEqual(m.kernel.out_sharding, ('c', 'a', 'b'))

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

    @nnx.vmap(out_axes=state_axes, axis_size=5, graph=True, graph_updates=True)
    def create_block():
      return Model(2, 3, rngs=nnx.Rngs(0))

    model = create_block()

    self.assertEqual(model.linear.kernel.shape, (5, 2, 3))
    self.assertEqual(model.linear.bias.shape, (3, 5))
    self.assertEqual(model.bn.scale.shape, (5, 3))
    self.assertEqual(model.bn.mean.shape, (3, 5))
    self.assertEqual(model.bn.var.shape, (5, 3))
    self.assertEqual(model.bn.bias.shape, (3,))

  @parameterized.parameters(True, False)
  def test_state_axes_from_state_functional(self, graph):
    class Model(nnx.Module):

      def __init__(self, din, dout, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.bn = nnx.BatchNorm(dout, rngs=rngs)

    filter_0 = lambda path, var: (
        (path[0] == 'linear' and path[1] == 'kernel')
        or (path[0] == 'bn' and path[1] in ('scale', 'var'))
    )
    filter_1 = lambda path, var: (
        (path[0] == 'linear' and path[1] == 'bias')
        or (path[0] == 'bn' and path[1] == 'mean')
    )

    abs_model = nnx.eval_shape(lambda: Model(2, 3, rngs=nnx.Rngs(0)))
    model_axes = nnx.prefix(
        abs_model, {filter_0: 0, filter_1: 1, ...: None}, graph=graph
    )

    @nnx.vmap(
        out_axes=model_axes,
        axis_size=5,
        graph=graph,
        graph_updates=False,
    )
    def create_block_functional():
      return Model(2, 3, rngs=nnx.Rngs(0))

    model = create_block_functional()

    self.assertEqual(model.linear.kernel.shape, (5, 2, 3))
    self.assertEqual(model.linear.bias.shape, (3, 5))
    self.assertEqual(model.bn.scale.shape, (5, 3))
    self.assertEqual(model.bn.mean.shape, (3, 5))
    self.assertEqual(model.bn.var.shape, (5, 3))
    self.assertEqual(model.bn.bias.shape, (3,))

  @parameterized.parameters(True, False)
  def test_vmap_inconsistent_aliasing(self, graph_updates):
    v = nnx.Param(jnp.arange(3.0))

    @nnx.vmap(in_axes=(0, None), graph=True, graph_updates=graph_updates)
    def f(v_mapped, v_broadcast):
      return v_mapped[...] + v_broadcast[...]

    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing'):
      f(v, v)

  def test_variable_prefix_error(self):
    prefix = nnx.Variable(0)

    with self.assertRaisesRegex(ValueError, 'Variables prefixes are not supported'):
      @nnx.vmap(in_axes=(prefix,), graph=False)
      def f(v):
        ...

class TestPmap(parameterized.TestCase):
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
    @nnx.pmap(
        in_axes=(state_axes,),
        out_axes=state_axes,
        axis_size=1,
        graph=True,
        graph_updates=True,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)
    initial_key = module.dropout.rngs.key[...]

    assert module.dropout.rngs.count[0] == 0
    assert module.linear.kernel.shape == (1, 3, 10)
    assert module.linear.bias.shape == (1, 10)

    x = jnp.ones((1, 1, 3))

    @nnx.pmap(
        in_axes=(state_axes, 0), axis_size=1, graph=True, graph_updates=True
    )
    def forward_block(module, x):
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (1, 1, 10)
    assert module.dropout.rngs.count[0] == 1
    assert module.dropout.rngs.key[...] == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  @parameterized.parameters(True, False)
  def test_basic_single_functional(self, graph):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 10, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.elu(x)
        x = self.dropout(x)
        return x

    abs_block = nnx.eval_shape(lambda: Block(nnx.Rngs(0)))
    state_axes = nnx.prefix(abs_block, {(nnx.Param, nnx.RngState): 0, ...: None}, graph=graph)

    abs_rngs = nnx.eval_shape(lambda: nnx.Rngs(0))
    rngs_axes = nnx.prefix(abs_rngs, {nnx.RngState: 0, ...: None}, graph=graph)

    @nnx.with_rngs(split=1, graph=graph, graph_updates=False)
    @nnx.pmap(
        in_axes=(rngs_axes,), out_axes=state_axes, axis_size=1,
        graph=graph, graph_updates=False,
    )
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)
    initial_key = module.dropout.rngs.key[...]

    assert module.dropout.rngs.count[0] == 0
    assert module.linear.kernel.shape == (1, 3, 10)
    assert module.linear.bias.shape == (1, 10)

    x = jnp.ones((1, 1, 3))

    @nnx.pmap(in_axes=(state_axes, 0), axis_size=1, graph=graph, graph_updates=False)
    def forward_block(module, x):
      return module(x)

    y = forward_block(module, x)

    assert y.shape == (1, 1, 10)
    assert module.dropout.rngs.count[0] == 1
    assert module.dropout.rngs.key[...] == initial_key

    y2 = forward_block(module, x)

    assert not jnp.allclose(y, y2)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic_demo_single(self, graph, graph_updates):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(20, 20, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    @nnx.split_rngs(splits=1)
    @nnx.pmap(axis_size=1, graph=graph, graph_updates=graph_updates)
    def create_block(rngs: nnx.Rngs):
      return Block(rngs)

    @nnx.pmap(axis_size=1, graph=graph, graph_updates=graph_updates)
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = create_block(rngs)

    assert module.dropout.rngs.count[...] == 0
    assert module.linear.kernel.shape == (1, 20, 20)
    assert module.linear.bias.shape == (1, 20)

    x = jnp.ones((1, 10, 20))

    y = forward_block(module, x)

    assert y.shape == (1, 10, 20)
    assert module.dropout.rngs.count[...] == 1

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

  @parameterized.parameters(True, False)
  def test_replicate_single_functional(self, graph):
    din = 3
    dout = 10

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    state_axes = nnx.prefix(
        nnx.eval_shape(lambda: Block(nnx.Rngs(0))),
        {nnx.RngState: 0, ...: None},
        graph=graph,
    )

    @nnx.split_rngs(splits=1)
    @nnx.pmap(
        in_axes=(state_axes, 0),
        out_axes=0,
        axis_size=1,
        graph=graph,
        graph_updates=False,
    )
    def forward_block(module: Block, x):
      return module(x)

    rngs = nnx.Rngs(0)
    module = Block(rngs)
    initial_key = module.dropout.rngs.key[...]

    assert module.dropout.rngs.count[...] == 0
    assert module.linear.kernel.shape == (din, dout)
    assert module.linear.bias.shape == (dout,)

    x = jnp.ones((1, 5, din))

    y = forward_block(module, x)

    assert y.shape == (1, 5, dout)
    assert module.dropout.rngs.count[...] == 1

    y2 = forward_block(module, x)

    # dropout is working!
    assert not jnp.allclose(y, y2)

    assert module.dropout.rngs.key[...] == initial_key


class TestCond(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_basic(self, graph_updates: bool):
    class TimeStep(tp.NamedTuple):
      step: nnx.Variable[jax.Array]
      reward: nnx.Variable[jax.Array]

      @staticmethod
      def zero():
        return TimeStep(
            step=nnx.Variable(jnp.array(0)), reward=nnx.Variable(jnp.array(0.0))
        )

    @nnx.dataclass
    class Foo(nnx.Pytree):
      timestep: TimeStep = nnx.data()

      def update(self):
        def reward_2(self: Foo):
          self.timestep.step[...] += 1
          self.timestep.reward[...] = 2.0

        def reward_0(self: Foo):
          self.timestep.step[...] += 1
          self.timestep.reward[...] = 0.0

        nnx.cond(
            self.timestep.step % 2 == 0,
            reward_2,
            reward_0,
            self,
            graph_updates=graph_updates,
        )

    foo = Foo(timestep=TimeStep.zero())
    foo.update()
    self.assertEqual(foo.timestep.step[...], 1)
    self.assertEqual(foo.timestep.reward[...], 2.0)
    foo.update()
    self.assertEqual(foo.timestep.step[...], 2)
    self.assertEqual(foo.timestep.reward[...], 0.0)

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_basic_variable(self, graph, graph_updates):
    def collatz(x):
      def even(x):
        x[...] = x // 2

      def odd(x):
        x[...] = 3 * x + 1

      return nnx.cond(
        x % 2 == 0, even, odd, x,
        graph=graph, graph_updates=graph_updates,
      )

    x = nnx.Variable(jnp.array(8))
    collatz(x)
    self.assertEqual(x[...], 4)
    collatz(x)
    self.assertEqual(x[...], 2)
    collatz(x)
    self.assertEqual(x[...], 1)
    collatz(x)
    self.assertEqual(x[...], 4)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_cond_and_vmap(self, graph, graph_updates):
    class Env(nnx.Pytree):
      def __init__(self):
        self.index = nnx.Variable(jnp.arange(8))
        self.step = nnx.Variable(jnp.zeros((8,), jnp.uint32))

    env = Env()
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.vmap(in_axes=(0, None), out_axes=None, graph=graph, graph_updates=graph_updates)
    def f(env: Env, model: nnx.Linear):
      self.assertEqual(env.index.shape, ())

      def increment(env: Env):
        env.step[...] += 1

      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      nnx.cond(
        is_even, increment, no_nothing, env,
        graph=graph, graph_updates=graph_updates,
      )

    f(env, model)

    np.testing.assert_array_equal(
      env.step[...], np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint32)
    )

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_cond_different_variable_per_branch(self, graph, graph_updates):
    a = nnx.Variable(jnp.array(0))
    b = nnx.Variable(jnp.array(0))

    def update_a(a, b):
      a[...] += 1

    def update_b(a, b):
      b[...] += 10

    nnx.cond(
      True, update_a, update_b, a, b,
      graph=graph, graph_updates=graph_updates,
    )
    self.assertEqual(a[...], 1)
    self.assertEqual(b[...], 0)

    nnx.cond(
      False, update_a, update_b, a, b,
      graph=graph, graph_updates=graph_updates,
    )
    self.assertEqual(a[...], 1)
    self.assertEqual(b[...], 10)

  @parameterized.parameters(True, False)
  def test_cond_shared_references(self, graph_updates):

    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: nnx.Variable
      b: nnx.Variable

    v = nnx.Variable(jnp.array(0))
    m = Foo(a=v, b=v)

    def true_fn(m):
      m.a[...] += 1

    def false_fn(m):
      m.b[...] += 2

    nnx.cond(True, true_fn, false_fn, m, graph=True, graph_updates=graph_updates)
    np.testing.assert_allclose(m.a[...], 1)
    np.testing.assert_allclose(m.b[...], 1)
    nnx.cond(False, true_fn, false_fn, m, graph=True, graph_updates=graph_updates)
    np.testing.assert_allclose(m.a[...], 3)
    np.testing.assert_allclose(m.b[...], 3)

    with self.assertRaises(ValueError):
      nnx.cond(True, true_fn, false_fn, m, graph=False)

class TestSwitch(parameterized.TestCase):
  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_basic(self, graph, graph_updates):
    class RoundTable(nnx.Module):
      def __init__(self):
        self.next_index = 0
        self.linear = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
        self.linear.kernel[...] = jnp.identity(10)
        self.rounds_count = nnx.Variable(jnp.array(0))

      def __call__(self, x):
        def fn0(m, x):
          m.rounds_count[...] += 1
          return m.linear(x)
        def fn1(m, x):
          return m.linear(x) * 2
        def fn2(m, x):
          m.linear.kernel[...] = jnp.zeros((10, 10))
          return m.linear(x)

        y = nnx.switch(
          self.next_index, (fn0, fn1, fn2), self, x,
          graph=graph, graph_updates=graph_updates,
        )
        self.next_index = (self.next_index + 1) % 3
        return y

    model = RoundTable()
    x = jnp.ones((10,))
    np.testing.assert_array_equal(model(x), x)
    assert model.rounds_count[...] == 1
    assert model.next_index == 1
    np.testing.assert_array_equal(model(x), x * 2)
    assert model.rounds_count[...] == 1
    assert model.next_index == 2
    np.testing.assert_array_equal(model(x), jnp.zeros((10,)))
    assert model.rounds_count[...] == 1
    assert model.next_index == 0
    np.testing.assert_array_equal(model(x), jnp.zeros((10,)))
    assert model.rounds_count[...] == 2
    assert model.next_index == 1

  @parameterized.parameters(
    (True, False),
    (False, False),
  )
  def test_switch_variable(self, graph, graph_updates):
    def add_1(x):
      x[...] += 1

    def add_10(x):
      x[...] += 10

    def add_100(x):
      x[...] += 100

    x = nnx.Variable(jnp.array(0))
    nnx.switch(0, (add_1, add_10, add_100), x,
               graph=graph, graph_updates=graph_updates)
    self.assertEqual(x[...], 1)
    nnx.switch(1, (add_1, add_10, add_100), x,
               graph=graph, graph_updates=graph_updates)
    self.assertEqual(x[...], 11)
    nnx.switch(2, (add_1, add_10, add_100), x,
               graph=graph, graph_updates=graph_updates)
    self.assertEqual(x[...], 111)

  @parameterized.parameters(True, False)
  def test_switch_shared_references(self, graph_updates):
    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: nnx.Variable
      b: nnx.Variable

    v = nnx.Variable(jnp.array(0))
    m = Foo(a=v, b=v)

    def add_a(m):
      m.a[...] += 1

    def add_b(m):
      m.b[...] += 10

    nnx.switch(0, (add_a, add_b), m, graph=True, graph_updates=graph_updates)
    np.testing.assert_allclose(m.a[...], 1)
    np.testing.assert_allclose(m.b[...], 1)

    nnx.switch(1, (add_a, add_b), m, graph=True, graph_updates=graph_updates)
    np.testing.assert_allclose(m.a[...], 11)
    np.testing.assert_allclose(m.b[...], 11)

    with self.assertRaises(ValueError):
      nnx.switch(0, (add_a, add_b), m, graph=False)

class TestWhileLoop(parameterized.TestCase):
  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic(self, graph, graph_updates):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0),
      graph=graph, graph_updates=graph_updates)
    np.testing.assert_array_equal(y, x * 8)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_multiple_objects(self, graph, graph_updates):
    def fwd_fn(input):
      m1, (w2,), x, c = input
      y = m1(x) @ w2
      return m1, (w2,), y, c - 1.0

    m1 = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    m1.kernel[...] = jnp.identity(10) * 2
    w2 = nnx.Variable(jnp.identity(10) * 0.5)
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (m1, (w2,), x, 3.0),
      graph=graph, graph_updates=graph_updates)
    np.testing.assert_allclose(y, x)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_nested_module(self, graph, graph_updates):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    module = nnx.Sequential(module)
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0),
      graph=graph, graph_updates=graph_updates)
    np.testing.assert_array_equal(y, x * 8)

  @parameterized.parameters(True, False)
  def test_shared_module(self, graph_updates):
    m1 = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(10, 10, use_bias=False, rngs=nnx.Rngs(0))
    m2.kernel = m1.kernel
    module = nnx.Sequential(m1, m2)
    self.assertLen(jax.tree.leaves(nnx.compat.state(module)), 2)  # only m1 params

    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      m.layers[0].kernel[...] = jnp.zeros_like(m.layers[0].kernel[...])
      return m, y, c - 1.0

    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))
    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 2.0),
      graph=True, graph_updates=graph_updates)
    self.assertLen(jax.tree.leaves(nnx.compat.state(module)), 2)  # only m1 params
    np.testing.assert_array_equal(
      m1.kernel[...],
      jnp.zeros((10, 10)),
    )
    np.testing.assert_array_equal(
      m2.kernel[...],
      jnp.zeros((10, 10)),
    )
    np.testing.assert_array_equal(y, jnp.zeros((10,)))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_value_changed(self, graph, graph_updates):
    def fwd_fn(input):
      m, x, c = input
      m.kernel[...] = jnp.zeros_like(m.kernel)
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    _, y, _ = nnx.while_loop(
      lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0),
      graph=graph, graph_updates=graph_updates)
    np.testing.assert_array_equal(
      module.kernel[...],
      jnp.zeros((10, 10)),
    )
    np.testing.assert_array_equal(y, jnp.zeros((10,)))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_ref_changed(self, graph, graph_updates):
    def fwd_fn(input):
      m, x, c = input
      y = m(x)
      m.kernel = nnx.Param(jnp.zeros_like(m.kernel))
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    with self.assertRaises(ValueError):
      _, y, _ = nnx.while_loop(
        lambda input: input[-1] > 0, fwd_fn, (module, x, 2.0),
        graph=graph, graph_updates=graph_updates)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_structure_changed(self, graph, graph_updates):
    def fwd_fn(input):
      m, x, c = input
      m = nnx.Linear(10, 10, use_bias=False, rngs=nnx.Rngs(1))
      m.kernel[...] = jnp.identity(10) * 2
      y = m(x)
      return m, y, c - 1.0

    module = nnx.Linear(10, 10, use_bias=True, rngs=nnx.Rngs(0))
    x = 1e1 * jax.random.normal(jax.random.key(0), (10,))

    with self.assertRaises((ValueError, TypeError)):
      _, y, _ = nnx.while_loop(
        lambda input: input[-1] > 0, fwd_fn, (module, x, 3.0),
        graph=graph, graph_updates=graph_updates)

  @parameterized.parameters(True, False)
  def test_repeated_object(self, graph_updates):
    m = nnx.Linear(10, 10, rngs=nnx.Rngs(0))

    def body_fn(val):
      count, m, _ = val
      return count + 1, m, m

    count, m, _ = nnx.while_loop(
      lambda val: val[0] < 2,
      body_fn,
      (0, m, m),
      graph=True, graph_updates=graph_updates,
    )

  @parameterized.parameters(
      (True, True),
      (True, False),
      (False, False),
  )
  def test_immut_fori_loop(self, graph: bool, graph_updates: bool):
    def immut_fn(i, carry):

      def update_fn(p, c):
        c[...] += 1.0
        return c

      nnx.map(update_fn, carry, graph=graph)
      return carry

    model = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    zeros = nnx.map(
        lambda p, v: nnx.Variable(jnp.zeros_like(v)), model, graph=graph
    )
    res = nnx.fori_loop(
        0, 2, immut_fn, zeros, graph=graph, graph_updates=graph_updates
    )

    def assert_zeros(path, c):
      np.testing.assert_array_equal(c[...], jnp.full(c.shape, 2.0))
      return c

    nnx.map(assert_zeros, res, graph=graph)

    self.assertIs(zeros.kernel, res.kernel)
    self.assertIs(zeros.bias, res.bias)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_fori_loop_grad_accum(self, graph, graph_updates):
    accum = nnx.Variable(jnp.zeros((10, 10)))

    def accum_fn(i, accum):
      accum[...] += 1
      return accum

    accum = nnx.fori_loop(0, 3, accum_fn, accum,
                          graph=graph, graph_updates=graph_updates)
    np.testing.assert_array_equal(accum[...], jnp.full((10, 10), 3.0))

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_fori_loop_basic(self, graph, graph_updates):
    def fwd_fn(i, input):
      m, x = input
      m.kernel[...] = jnp.identity(10) * i
      return m, m(x)

    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (10,))

    _, y = nnx.fori_loop(2, 4, fwd_fn, (module, x),
                         graph=graph, graph_updates=graph_updates)
    np.testing.assert_array_equal(y, x * 2 * 3)

  @parameterized.parameters(True, False)
  def test_fori_loop_with_sharing(self, graph_updates):
    class A(nnx.Pytree):
      def __init__(self):
        self.params = nnx.Param(jnp.zeros((10,), dtype=int))

    class B(nnx.Pytree):
      def __init__(self, a: A):
        self.a = a

    class C(nnx.Pytree):
      def __init__(self, a: A):
        self.a = a

    class D(nnx.Pytree):
      def __init__(self):
        self.a = A()
        self.b = B(self.a)
        self.c = C(self.a)

    def increment(_, d: D) -> D:
      d.a.params[...] += 1
      return d

    @nnx.jit(graph=True, graph_updates=graph_updates)
    def rollout(d: D):
      nnx.fori_loop(0, 10, increment, d, graph=True, graph_updates=graph_updates)

    d = D()
    rollout(d)

    np.testing.assert_array_equal(
      d.a.params[...], np.full((10,), 10, dtype=int)
    )

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_loops_multiple_modules(self, graph, graph_updates):
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
    nnx.while_loop(lambda input: input[-1] > 0, while_loop_fn, (a, b, 2),
                   graph=graph, graph_updates=graph_updates)
    nnx.fori_loop(0, 2, fori_loop_fn, (a, b), graph=graph)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_while_loop_stateful(self, graph, graph_updates):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

    counter = Counter()
    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    x = jax.random.normal(jax.random.key(0), (10,))

    def body_fn(val):
      counter, module, x, i = val
      counter.count[...] += 1
      x = module(x)
      return counter, module, x, i - 1

    counter, module, y, _ = nnx.while_loop(
      lambda val: val[-1] > 0,
      body_fn,
      (counter, module, x, 3),
      graph=graph,
      graph_updates=graph_updates,
    )
    np.testing.assert_array_equal(counter.count[...], 3)
    np.testing.assert_array_equal(y, x * 8)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_while_loop_inside_jit(self, graph, graph_updates):
    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    x = jax.random.normal(jax.random.key(0), (10,))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(module, x):
      def body_fn(val):
        m, x, c = val
        return m, m(x), c - 1.0
      _, y, _ = nnx.while_loop(
        lambda val: val[-1] > 0,
        body_fn,
        (module, x, 3.0),
        graph=graph,
      )
      return y

    y = f(module, x)
    np.testing.assert_array_equal(y, x * 8)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_fori_loop_stateful(self, graph, graph_updates):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

    counter = Counter()
    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    x = jax.random.normal(jax.random.key(0), (10,))

    def body_fn(i, val):
      counter, module, x = val
      counter.count[...] += 1
      x = module(x)
      return counter, module, x

    counter, module, y = nnx.fori_loop(
      0, 3, body_fn, (counter, module, x),
      graph=graph, graph_updates=graph_updates,
    )
    np.testing.assert_array_equal(counter.count[...], 3)
    np.testing.assert_array_equal(y, x * 8)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_fori_loop_inside_jit(self, graph, graph_updates):
    module = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
    module.kernel[...] = jnp.identity(10) * 2
    x = jax.random.normal(jax.random.key(0), (10,))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(module, x):
      def body_fn(i, val):
        m, x = val
        return m, m(x)
      _, y = nnx.fori_loop(
        0, 3, body_fn, (module, x), graph=graph,
      )
      return y

    y = f(module, x)
    np.testing.assert_array_equal(y, x * 8)

class TestSplitMergeInputs(absltest.TestCase):
  def test_split_inputs(self):
    class StatefulLinear(nnx.Linear):
      def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        super().__init__(din, dout, rngs=rngs)
        self.counter = nnx.BatchStat(jnp.array(0, jnp.uint32))

      def __call__(self, x):
        self.counter[...] += 1
        return super().__call__(x)

    model = StatefulLinear(3, 4, rngs=nnx.Rngs(0))

    @general.split_inputs
    @jax.jit
    @general.merge_inputs
    def forward(model, x):
      return model(x)

    x = jnp.ones((2, 3))
    y = forward(model, x)

    self.assertEqual(model.counter[...], 1)

  def test_split_inputs_cond(self):
    class Counter(nnx.Linear):
      def __init__(self):
        self.count = nnx.BatchStat(jnp.array(0, jnp.uint32))

      def increment(self):
        self.count[...] += 1

    counter = Counter()

    @general.merge_inputs
    def increment(counter: Counter):
      counter.increment()

    @general.merge_inputs
    def no_nothing(counter: Counter):
      pass

    general.split_inputs(jax.lax.cond)(True, increment, no_nothing, counter)

    self.assertEqual(counter.count[...], 1)

    general.split_inputs(jax.lax.cond)(False, increment, no_nothing, counter)

    self.assertEqual(counter.count[...], 1)

  def test_split_inputs_vmap(self):
    class EnvState(nnx.Variable[nnx.A]):
      pass

    class Env(nnx.Pytree):
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
      self.assertEqual(env.index.shape, ())

      @general.merge_inputs
      def increment(env: Env):
        env.step[...] += 1

      @general.merge_inputs
      def no_nothing(env: Env):
        pass

      is_even = env.index % 2 == 0
      general.split_inputs(jax.lax.cond)(is_even, increment, no_nothing, env)

    f(env, model)

    np.testing.assert_array_equal(
      env.step[...], np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint32)
    )

class TestCheckify(parameterized.TestCase):
  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_basic(self, graph, graph_updates):

    @dataclasses.dataclass
    class Foo(nnx.Module):
      a: nnx.Param

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(m):
      y = jnp.sin(m.a)  # error
      return m.a + y

    m = Foo(a=nnx.Param(jnp.inf))
    err, out = nnx.checkify(
      f, errors=checkify.float_checks,
      graph=graph, graph_updates=graph_updates,
    )(m)

    with self.assertRaisesRegex(ValueError, 'nan generated by primitive: sin'):
      err.throw()

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_checkify_stateful(self, graph, graph_updates):
    count = nnx.Variable(jnp.array(0))

    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def f(c):
      c[...] += 1
      return c[...]

    err, out = nnx.checkify(
      f, graph=graph, graph_updates=graph_updates,
    )(count)
    self.assertEqual(count[...], 1)
    np.testing.assert_allclose(out, 1)


class TestMakeJaxpr(parameterized.TestCase):

  def test_make_jaxpr_graph_updates_error(self):
    m = nnx.Dict(a=nnx.Param(jnp.array(1)))

    def f(m):
      return m['a'][...]

    with self.assertRaisesRegex(
      ValueError, 'nnx.make_jaxpr does not support graph_updates=True.'
    ):
      nnx.make_jaxpr(f, graph=True, graph_updates=True)(m)

  @parameterized.parameters(True, False)
  def test_make_jaxpr_with_variable_update(self, graph):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = nnx.Variable(jnp.array(0))

      def __call__(self):
        self.count[...] += 1
        return self.count[...]

    m = Counter()
    jaxpr = nnx.make_jaxpr(lambda m: m(), graph=graph, graph_updates=False)(m)
    self.assertIsNotNone(jaxpr)
    self.assertEqual(m.count[...], 0)


class TestBoundMethodTransforms(parameterized.TestCase):
  def test_remat_with_bound_method_raises(self):
    class M(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.count = nnx.BatchStat(0)

      def block(self, x: jax.Array) -> jax.Array:
        self.count[...] += 1
        return self.linear(x)

    m = M(rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.remat(m.block)

  def test_jit_with_bound_method_raises(self):
    class M(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
      def apply(self, x: jax.Array, scale: int):
        return self.linear(x) * scale

    m = M(rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.jit(m.apply, static_argnums=1)

  def test_vmap_with_bound_method_raises(self):
    class M(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
      def __call__(self, x: jax.Array):
        return self.linear(x)

    m = M(rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.vmap(m.__call__, in_axes=(0,), out_axes=0)

  def test_eval_shape_with_bound_method_raises(self):
    class M(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
      def __call__(self, x: jax.Array):
        return self.linear(x)

    m = M(rngs=nnx.Rngs(0))
    x_spec = jax.ShapeDtypeStruct((1, 2), jnp.float32)
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.eval_shape(m.__call__, x_spec)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_grad_with_bound_method_raises(self, graph_mode, graph_updates):
    class M(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(jnp.array(1.0))
      def loss(self, s: float):
        return (self.w * s) ** 2

    m = M()
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.grad(m.loss, graph=graph_mode)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_value_and_grad_with_bound_method_raises(self, graph_mode, graph_updates):
    class TestModel(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 1, rngs=rngs)

      def loss_fn(self, x, y):
        pred = self.linear(x)
        return jnp.mean((pred - y) ** 2)

    model = TestModel(rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.value_and_grad(model.loss_fn, graph=graph_mode)

  def test_checkify_with_bound_method_raises(self):
    """Test that checkify raises error for bound methods."""
    class M(nnx.Module):
      def __call__(self, x: jax.Array):
        return x + 1

    m = M()
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.checkify(m.__call__)

  def test_pmap_with_bound_method_raises(self):
    """Test that pmap raises error for bound methods."""
    class M(nnx.Module):
      def __call__(self, x: jax.Array):
        return x + 1

    m = M()
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.pmap(m.__call__)

  def test_shard_map_with_bound_method_raises(self):
    """Test that shard_map raises error for bound methods."""
    class M(nnx.Module):
      def __call__(self, x: jax.Array):
        return x + 1

    m = M()
    mesh = jax.sharding.Mesh(jax.local_devices()[:1], ('data',))
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.shard_map(m.__call__, mesh=mesh, in_specs=None, out_specs=None)

  def test_custom_vjp_with_bound_method_raises(self):
    """Test that custom_vjp raises error for bound methods."""
    class M(nnx.Module):
      def __call__(self, x: jax.Array):
        return x + 1

    m = M()
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      nnx.custom_vjp(m.__call__)

  def test_scan_bound_method_raises(self):
    class M(nnx.Module):
      def __call__(self, x: jax.Array):
        return x + 1
    m = M()
    with self.assertRaisesRegex(ValueError, 'bound methods'):
      _ = nnx.scan(m.__call__, in_axes=(0,), out_axes=0)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_pmap_basic(self, graph, graph_updates):
    class LinearEnsemble(nnx.Module):
      def __init__(self, num, *, rngs):
        self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))

    model = LinearEnsemble(1, rngs=nnx.Rngs(0))
    x = jnp.ones((2,))

    @nnx.pmap(in_axes=(0, None), out_axes=0, axis_size=1,
              graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return x @ model.w

    y = forward(model, x)
    assert y.shape == (1, 3)

  @parameterized.parameters(
    (True, True), (True, False), (False, False),
  )
  def test_tree_mode_pmap_stateful(self, graph, graph_updates):
    class Counter(nnx.Variable):
      pass

    class Linear(nnx.Module):
      def __init__(self, din, dout, *, rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.count = Counter(jnp.array(0))

      def __call__(self, x):
        self.count[...] += 1
        return x @ self.w

    model = Linear(2, 3, rngs=nnx.Rngs(0))

    @nnx.pmap(in_axes=(None, 0), out_axes=0, axis_size=1,
              graph=graph, graph_updates=graph_updates)
    def forward(model, x):
      return model(x)

    x = jnp.ones((1, 2))
    y = forward(model, x)
    assert y.shape == (1, 3)
    assert model.count.get_value() == 1

  def test_tree_mode_pmap_split_merge(self):
    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 10, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        x = nnx.elu(x)
        x = self.dropout(x)
        return x

    rngs = nnx.Rngs(0)

    @nnx.split_rngs(splits=1, graph=False)
    @nnx.pmap(in_axes=0, out_axes=(None, 0, 0, None), axis_size=1, graph=False)
    def create_block(rngs):
      block = Block(rngs)
      graphdef, params_state, rng_state, rest_state = nnx.split(
          block, nnx.Param, nnx.RngState, ...,
      )
      return graphdef, params_state, rng_state, rest_state

    graphdef, params_state, rng_state, rest_state = create_block(rngs)

    assert rng_state.dropout.rngs.count[0] == 0
    assert params_state.linear.kernel.shape == (1, 3, 10)
    assert params_state.linear.bias.shape == (1, 10)

    x = jnp.ones((1, 1, 3))

    @nnx.pmap(in_axes=(0, 0, None, 0), axis_size=1, graph=False)
    def forward_block(params_state, rng_state, rest_state, x):
      return nnx.merge(graphdef, params_state, rng_state, rest_state)(x)

    y = forward_block(params_state, rng_state, rest_state, x)

    assert y.shape == (1, 1, 10)

    y2 = forward_block(params_state, rng_state, rest_state, x)

    assert not jnp.allclose(y, y2)

  def test_tree_mode_pmap_replicate(self):
    din = 3
    dout = 10

    class Block(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.dropout(nnx.relu(self.linear(x)))

    rngs = nnx.Rngs(0)
    module = Block(rngs)

    assert module.dropout.rngs.count[...] == 0
    assert module.linear.kernel.shape == (din, dout)
    assert module.linear.bias.shape == (dout,)

    module = nnx.split_rngs(module, splits=1, graph=False)
    graphdef, rng_state, rest_state = nnx.split(
        module, nnx.RngState, ...,
    )

    @nnx.pmap(in_axes=(0, None, 0), out_axes=0, axis_size=1, graph=False)
    def forward_block(rng_state, rest_state, x):
      module = nnx.merge(graphdef, rng_state, rest_state)
      y = module(x)
      return y

    x = jnp.ones((1, 5, din))

    y = forward_block(rng_state, rest_state, x)

    assert y.shape == (1, 5, dout)
    assert module.dropout.rngs.count[0] == 1

    y2 = forward_block(rng_state, rest_state, x)

    assert module.dropout.rngs.count[0] == 2
    assert not jnp.allclose(y, y2)


if __name__ == '__main__':
  absltest.main()

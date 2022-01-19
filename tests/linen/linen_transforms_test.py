# Copyright 2022 The Flax Authors.
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

"""Transforms tests."""

from functools import partial
from typing import Any, Tuple, Iterable, Callable, Sequence
import operator

from absl.testing import absltest
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import errors
from flax import linen as nn
from flax.core import freeze

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()



def tree_equals(x, y):
  return jax.tree_util.tree_all(
      jax.tree_multimap(operator.eq, x, y))


id_fn = lambda x: x


class TransformedMLP(nn.Module):
  features: Sequence[int]
  transform: Callable = id_fn

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      # JIT the Module (it's __call__ fn by default.)
      x = self.transform(nn.Dense)(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x


def decorated_MLP(transform: Callable = id_fn):
  class MLP(nn.Module):
    features: Sequence[int]

    @transform
    @nn.compact
    def __call__(self, inputs):
      x = inputs
      for i, feat in enumerate(self.features):
        # JIT the Module (it's __call__ fn by default.)
        x = nn.Dense(feat, name=f'layers_{i}')(x)
        if i != len(self.features) - 1:
          x = nn.relu(x)
      return x
  return MLP


class TransformTest(absltest.TestCase):

  def test_jit(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = TransformedMLP(features=[3, 4, 5])
    jit_model = TransformedMLP(features=[3, 4, 5], transform=nn.jit)
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = jit_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_jit_decorated(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = decorated_MLP()(features=[3, 4, 5])
    jit_model = decorated_MLP(nn.jit)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = jit_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_remat(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = TransformedMLP(features=[3, 4, 5])
    remat_model = TransformedMLP(features=[3, 4, 5], transform=nn.remat)
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = remat_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_remat_decorated(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = decorated_MLP()(features=[3, 4, 5])
    remat_model = decorated_MLP(nn.remat)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = remat_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_remat_kwargs(self):
    class ConditionalReLU(nn.Module):
      @nn.compact
      def __call__(self, input, apply_relu : bool = False):
        return nn.relu(input) if apply_relu else input
    key = random.PRNGKey(0)
    x = jnp.ones((4, 4)) * -1
    remat_model = nn.remat(ConditionalReLU)()
    p = remat_model.init(key, x)
    y = remat_model.apply(p, x, apply_relu=True)

    self.assertTrue(np.all(y == jnp.zeros_like(x)))

  def test_vmap(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key1, (5, 4, 4))

    def vmap(cls):
      return nn.vmap(cls,
                     in_axes=(0,),
                     variable_axes={'params': None},
                     split_rngs={'params': False})
    normal_model = TransformedMLP(features=[3, 4, 5])
    vmap_model = TransformedMLP(features=[3, 4, 5], transform=vmap)
    init_variables = normal_model.init(key2, x)
    # simulate vmap in python for comparison:
    y1 = jnp.vstack([normal_model.apply(init_variables, x2[i])[None, ...]
                     for i in np.arange(x2.shape[0])])
    y2 = vmap_model.apply(init_variables, x2)
    np.testing.assert_allclose(y1, y2, atol=1e-7)

  def test_vmap_decorated(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key1, (5, 4, 4))

    def vmap(fn):
      return nn.vmap(fn,
                     in_axes=(0,),
                     variable_axes={'params': None},
                     split_rngs={'params': False})
    normal_model = decorated_MLP()(features=[3, 4, 5])
    vmap_model = decorated_MLP(vmap)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    # simulate vmap in python for comparison:
    y1 = jnp.vstack([normal_model.apply(init_variables, x2[i])[None, ...]
                     for i in np.arange(x2.shape[0])])
    y2 = vmap_model.apply(init_variables, x2)
    np.testing.assert_allclose(y1, y2, atol=1e-7)

  def test_vmap_batchnorm(self):
    key1, key2 = random.split(random.PRNGKey(3), 2)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key1, (5, 4, 4))

    def vmap(cls):
      return nn.vmap(cls,
                     in_axes=(0,),
                     variable_axes={'params': None, 'batch_stats': None},
                     split_rngs={'params': False},
                     axis_name='batch')
    class MlpBn(nn.Module):
      axis_name: Any = None

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.BatchNorm(axis_name=self.axis_name, use_running_average=False)(x)
        return x

    normal_model = MlpBn()
    vmap_model = vmap(MlpBn)(axis_name='batch')
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x2.reshape((-1, 4)), mutable=['batch_stats'])[0]
    y1 = y1.reshape((5, 4, 3))
    y2 = vmap_model.apply(init_variables, x2, mutable=['batch_stats'])[0]
    np.testing.assert_allclose(y1, y2, atol=1e-6)

  def test_scan(self):
    class SimpleScan(nn.Module):
      @nn.compact
      def __call__(self, c, xs):
        LSTM = nn.scan(nn.LSTMCell,
                       variable_broadcast='params',
                       split_rngs={'params': False})
        return LSTM(name="lstm_cell")(c, xs)

    key1, key2 = random.split(random.PRNGKey(0), 2)
    xs = random.uniform(key1, (5, 3, 2))
    dummy_rng = random.PRNGKey(0)
    init_carry = nn.LSTMCell.initialize_carry(dummy_rng,
                                              xs.shape[1:-1],
                                              xs.shape[-1])
    model = SimpleScan()
    init_variables = model.init(key2, init_carry, xs)
    # simulate scan in python for comparison:
    c = init_carry
    ys = []
    lstmcell_variables = freeze({'params': init_variables['params']['lstm_cell']})
    for i in range(xs.shape[0]):
      c, y = nn.LSTMCell().apply(lstmcell_variables, c, xs[i])
      ys.append(y[None, ...])
    y1 = jnp.vstack(ys)

    c2, y2 = model.apply(init_variables, init_carry, xs)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    np.testing.assert_allclose(c[0], c2[0], atol=1e-7)
    np.testing.assert_allclose(c[1], c2[1], atol=1e-7)

  def test_scan_decorated(self):
    class SimpleScan(nn.Module):
      @partial(nn.scan,
               variable_broadcast='params',
               in_axes=(nn.broadcast, 0),
               split_rngs={'params': False})
      @nn.compact
      def __call__(self, c, b, xs):
        assert b.shape == (4,)
        return nn.LSTMCell(name="lstm_cell")(c, xs)

    key1, key2 = random.split(random.PRNGKey(0), 2)
    xs = random.uniform(key1, (4, 3, 2))
    b = jnp.ones((4,))
    dummy_rng = random.PRNGKey(0)
    init_carry = nn.LSTMCell.initialize_carry(dummy_rng,
                                              xs.shape[1:-1],
                                              xs.shape[-1])
    model = SimpleScan()
    init_variables = model.init(key2, init_carry, b, xs)
    # simulate scan in python for comparison:
    c = init_carry
    ys = []
    lstmcell_variables = freeze({'params': init_variables['params']['lstm_cell']})
    for i in range(xs.shape[0]):
      c, y = nn.LSTMCell().apply(lstmcell_variables, c, xs[i])
      ys.append(y[None, ...])
    y1 = jnp.vstack(ys)

    c2, y2 = model.apply(init_variables, init_carry, b, xs)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    np.testing.assert_allclose(c[0], c2[0], atol=1e-7)
    np.testing.assert_allclose(c[1], c2[1], atol=1e-7)

  def test_multiscope_lifting_simple(self):
    class Counter(nn.Module):
      @nn.compact
      def __call__(self):
        v = self.variable('counter', 'foo', lambda: jnp.array([0]))
        v.value += jnp.array([1])
        return v.value
    class Outer(nn.Module):
      @nn.compact
      def __call__(self, x):
        cntr = nn.jit(Counter)(name='cntr')()
        return x
    class Inner(nn.Module):
      outer_module: nn.Module
      @nn.compact
      def __call__(self, x):
        return self.outer_module(x)
    class Test(nn.Module):
      @nn.compact
      def __call__(self, x):
        outer_dense = nn.jit(Outer)(name='outer')
        # we share stateful outer module as arg to two different, transformed modules:
        inner = nn.jit(Inner)(outer_dense, name='inner1')
        inner2 = nn.jit(Inner)(outer_dense, name='inner2')
        res = inner(x) + inner2(x)
        return res

    x = jnp.ones((10, 10))
    rngs = random.PRNGKey(0)
    init_vars = Test(None).init(rngs, x)
    _, new_vars = Test(None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(init_vars['counter']['outer']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer']['cntr']['foo'],
                     jnp.array([4], jnp.int32))

  def test_multiscope_lifting_simple_decorator(self):
    class Counter(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self):
        v = self.variable('counter', 'foo', lambda: jnp.array([0]))
        v.value += jnp.array([1])
        return v.value
    class Outer(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        cntr = Counter(name='cntr')()
        return x
    class Inner(nn.Module):
      outer_module: nn.Module
      @nn.jit
      @nn.compact
      def __call__(self, x):
        return self.outer_module(x)
    class Test(nn.Module):
      @nn.compact
      def __call__(self, x):
        outer_dense = Outer(name='outer')
        # we share stateful outer module as arg to two different, transformed modules:
        inner = Inner(outer_dense, name='inner1')
        inner2 = Inner(outer_dense, name='inner2')
        res = inner(x) + inner2(x)
        return res

    x = jnp.ones((1, 1))
    rngs = random.PRNGKey(0)
    init_vars = Test(None).init(rngs, x)
    _, new_vars = Test(None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(init_vars['counter']['outer']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer']['cntr']['foo'],
                     jnp.array([4], jnp.int32))

  def test_multiscope_lifting_argtree(self):
    class Counter(nn.Module):
      @nn.compact
      def __call__(self):
        v = self.variable('counter', 'foo', lambda: jnp.array([0]))
        v.value += jnp.array([1])
        return v.value
    class Outer(nn.Module):
      @nn.compact
      def __call__(self, x):
        cntr = nn.jit(Counter)(name='cntr')()
        return x
    class Inner(nn.Module):
      outer_module: Sequence[nn.Module]
      @nn.compact
      def __call__(self, x):
        return self.outer_module[0](x) + self.outer_module[1](x)
    class Test(nn.Module):
      @nn.compact
      def __call__(self, x):
        outer_dense1 = nn.jit(Outer)(name='outer1')
        outer_dense2 = nn.jit(Outer)(name='outer2')
        # we share stateful outer module as arg to two different, transformed modules:
        inner1 = nn.jit(Inner)((outer_dense1, outer_dense2), name='inner1')
        inner2 = nn.jit(Inner)((outer_dense1, outer_dense2), name='inner2')
        res = inner1(x) + inner2(x)
        return res

    x = jnp.ones((1, 1))
    rngs = random.PRNGKey(0)
    init_vars = Test(None).init(rngs, x)
    _, new_vars = Test(None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(init_vars['counter']['outer1']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer1']['cntr']['foo'],
                     jnp.array([4], jnp.int32))
    self.assertEqual(init_vars['counter']['outer2']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer2']['cntr']['foo'],
                     jnp.array([4], jnp.int32))

  def test_multiscope_lifting_argtree_decorator(self):
    class Counter(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self):
        v = self.variable('counter', 'foo', lambda: jnp.array([0]))
        v.value += jnp.array([1])
        return v.value
    class Outer(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        cntr = nn.jit(Counter)(name='cntr')()
        return x
    class Inner(nn.Module):
      outer_module: Sequence[nn.Module]
      @nn.jit
      @nn.compact
      def __call__(self, x):
        return self.outer_module[0](x) + self.outer_module[1](x)
    class Test(nn.Module):
      @nn.compact
      def __call__(self, x):
        outer_dense1 = Outer(name='outer1')
        outer_dense2 = Outer(name='outer2')
        # we share stateful outer module as arg to two different, transformed modules:
        inner1 = Inner((outer_dense1, outer_dense2), name='inner1')
        inner2 = Inner((outer_dense1, outer_dense2), name='inner2')
        res = inner1(x) + inner2(x)
        return res

    x = jnp.ones((1, 1))
    rngs = random.PRNGKey(0)
    init_vars = Test(None).init(rngs, x)
    _, new_vars = Test(None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(init_vars['counter']['outer1']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer1']['cntr']['foo'],
                     jnp.array([4], jnp.int32))
    self.assertEqual(init_vars['counter']['outer2']['cntr']['foo'],
                     jnp.array([2], jnp.int32))
    self.assertEqual(new_vars['counter']['outer2']['cntr']['foo'],
                     jnp.array([4], jnp.int32))

  def test_multiscope_lifting_simple_decorator_w_named_call(self):
    # TODO: actually test jaxpr on a simpler module.
    with nn.override_named_call(True):
      class Counter(nn.Module):
        @nn.jit
        @nn.compact
        def __call__(self):
          v = self.variable('counter', 'foo', lambda: jnp.array([0]))
          v.value += jnp.array([1])
          return v.value
      class Outer(nn.Module):
        @nn.jit
        @nn.compact
        def __call__(self, x):
          cntr = Counter(name='cntr')()
          return x
      class Inner(nn.Module):
        outer_module: nn.Module
        @nn.jit
        @nn.compact
        def __call__(self, x):
          return self.outer_module(x)
      class Test(nn.Module):
        @nn.compact
        def __call__(self, x):
          outer_dense = Outer(name='outer')
          # we share stateful outer module as arg to two different, transformed modules:
          inner = Inner(outer_dense, name='inner1')
          inner2 = Inner(outer_dense, name='inner2')
          res = inner(x) + inner2(x)
          return res

      x = jnp.ones((1, 1))
      rngs = random.PRNGKey(0)
      init_vars = Test(None).init(rngs, x)
      _, new_vars = Test(None).apply(init_vars, x, mutable=['counter'])
      self.assertEqual(init_vars['counter']['outer']['cntr']['foo'],
                      jnp.array([2], jnp.int32))
      self.assertEqual(new_vars['counter']['outer']['cntr']['foo'],
                      jnp.array([4], jnp.int32))

  def test_vmapped_outer_module(self):
    class Outer(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        return nn.Dense(5)(x)
    class Inner(nn.Module):
      outer_module: nn.Module
      @partial(nn.vmap,
               in_axes=(0,),
               variable_axes={'params': 0},
               split_rngs={'params': True})
      @nn.jit
      @nn.compact
      def __call__(self, x):
        return self.outer_module(x)
    class Test(nn.Module):
      @nn.compact
      def __call__(self, x):
        outer_dense = Outer(name='outer')
        inner = Inner(outer_dense, name='inner1')
        inner2 = Inner(outer_dense, name='inner2')
        res = inner(x) + inner2(x)
        return res

    x = jnp.ones((3, 1, 2))
    rngs = random.PRNGKey(0)
    init_vars = Test(None).init(rngs, x)
    y = Test(None).apply(init_vars, x)
    self.assertEqual(
        init_vars['params']['outer']['Dense_0']['kernel'].shape,
        (3, 2, 5))
    self.assertEqual(
        init_vars['params']['outer']['Dense_0']['bias'].shape,
        (3, 5))
    self.assertEqual(y.shape, (3, 1, 5))

  def test_module_transform_with_setup(self):
    class Foo(nn.Module):
      def setup(self):
        self.test = self.param('test', nn.initializers.ones, ())

      def __call__(self, x):
        return x * self.test

    FooVmap = nn.vmap(Foo, in_axes=0, out_axes=0,
                      variable_axes={'params': 0}, split_rngs={'params': True})
    variables = FooVmap().init(random.PRNGKey(0), jnp.ones((4,)))
    self.assertEqual(variables['params']['test'].shape, (4,))


  def test_nested_module_args_vmap(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(3)(x)
    class B(nn.Module):
      A: nn.Module
      @nn.compact
      def __call__(self, x):
        return self.A(x)
    class C(nn.Module):
      B: nn.Module
      @partial(nn.vmap,
               variable_axes={'params': 0},
               split_rngs={'params': True})
      @nn.compact
      def __call__(self, x):
        return self.B(x)
    class D(nn.Module):
      @nn.compact
      def __call__(self, x):
        a = A()
        b = B(a)
        c = C(b)
        return c(x)

    key = random.PRNGKey(0)
    x = jnp.ones((10, 10))
    p = D().init(key, x)

    variable_shapes = jax.tree_map(jnp.shape, p)
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['kernel'],
        (10, 10, 3))
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['bias'],
        (10, 3))

  def test_nested_module_args_vmap_2(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(3)(x)
    class B(nn.Module):
      A: nn.Module
      @nn.compact
      def __call__(self, x):
        return self.A(x)
    class C(nn.Module):
      A: nn.Module
      B: nn.Module
      @partial(
          nn.vmap,
          variable_axes={'params': 0},
          split_rngs={'params': True})
      @nn.compact
      def __call__(self, x):
        return self.B(x) + self.A(x)
    class D(nn.Module):
      @nn.compact
      def __call__(self, x):
        a1 = A()
        a2 = A()
        b = B(a1)
        c = C(a2, b)
        return c(x)

    key = random.PRNGKey(0)
    x = jnp.ones((10, 10))
    p = D().init(key, x)

    variable_shapes = jax.tree_map(jnp.shape, p)
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['kernel'],
        (10, 10, 3))
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['bias'],
        (10, 3))
    self.assertEqual(
        variable_shapes['params']['A_1']['Dense_0']['kernel'],
        (10, 10, 3))
    self.assertEqual(
        variable_shapes['params']['A_1']['Dense_0']['bias'],
        (10, 3))

  def test_nested_setup_calls_count(self):
    D = 3
    N = 4
    setup_cntr = 0
    call_cntr = 0
    class Repeat(nn.Module):
      mdl_def: Any
      def setup(self):
        self.lyrs = [self.mdl_def() for _ in range(N)]
      @nn.remat  # we just use remat as a convenient test of transform logic
      def __call__(self, x):
        for lyr in self.lyrs:
          lyr(x)
        return x
    class Counter(nn.Module):
      def setup(self):
        nonlocal setup_cntr
        setup_cntr += 1
        self.dense = nn.Dense(2, use_bias=False)
      @nn.remat
      def __call__(self, x):
        nonlocal call_cntr
        call_cntr += 1
        return self.dense(x)

    def nested_repeat(mdl):
      for _ in range(D):
        mdl = partial(Repeat, mdl)
      return mdl()
    _ = nested_repeat(Counter).init(random.PRNGKey(0), jnp.ones((2,)))
    # setup_cntr == 128 due to 1 call in Counter.setup by _validate_setup
    # and 1 further "real" call.
    self.assertEqual(setup_cntr, 128)
    self.assertEqual(call_cntr, 64)

  def test_multimethod_setup_calls(self):
    cntr=0
    class A(nn.Module):
      def setup(self):
        nonlocal cntr
        cntr+=1
        self.d = nn.Dense(2)
      @nn.remat
      def foo(self, x):
        return self.d(x)
      @nn.remat
      def bar(self, x):
        return self.d(x)
    class B(nn.Module):
      def setup(self):
        self.a = A()
      def __call__(self, x):
        y1 = self.a.foo(x)
        y2 = self.a.bar(x)
        return y1, y2

    key = random.PRNGKey(0)
    x = jnp.ones((2,))
    (y1, y2), _ = B().init_with_output(key, x)
    np.testing.assert_array_equal(y1, y2)
    # cntr == 3 due to 1 call by _validate_setup
    # and two further "real" calls.
    self.assertEqual(cntr, 3)

  def test_toplevel_submodule_adoption_transform(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(3)(x)
    class B(nn.Module):
      A: nn.Module
      @nn.compact
      def __call__(self, x):
        return self.A(x)
    class C(nn.Module):
      A: nn.Module
      B: nn.Module
      @partial(
          nn.vmap,
          variable_axes={'params': 0},
          split_rngs={'params': True})
      @nn.compact
      def __call__(self, x):
        return self.B(x) + self.A(x)
    class Csimple(nn.Module):
      A: nn.Module
      B: nn.Module
      @nn.compact
      def __call__(self, x):
        return self.B(x) + self.A(x)
    class D(nn.Module):
      @nn.compact
      def __call__(self, x):
        a1 = A()
        a2 = A()
        b = B(a1)
        c = C(a2, b)
        return c(x)

    key = random.PRNGKey(0)
    x = jnp.ones((10, 10))
    p1 = D().init(key, x)
    y1 = D().apply(p1, x)

    a1 = A()
    a2 = A()
    b = B(a1)
    p2 = freeze({'params': {
        'A': p1['params']['A_0'],
        'B': {
            'A': p1['params']['A_1'],
        }
    }})

    # Test method wrapper transform.
    y2 = C(a2, b).apply(p2, x)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    # Test class transform.
    Ctrafo = nn.vmap(Csimple,
                     variable_axes={'params': 0},
                     split_rngs={'params': True})

    y3 = Ctrafo(a2, b).apply(p2, x)
    np.testing.assert_allclose(y1, y3, atol=1e-7)

  def test_toplevel_submodule_adoption_pytree_transform(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, c, x):
        counter = self.variable('counter', 'i', jnp.zeros, ())
        counter.value += 1
        x = nn.Dense(1)(x)
        return c, x

    class B(nn.Module):
      A: Any
      @nn.compact
      def __call__(self, c, x):
        return self.A['foo'](*self.A['bar'](c, x))

    a = A()
    As = {'foo': A(), 'bar': A()}
    b = nn.scan(B,
                in_axes=0,
                variable_carry='counter',
                variable_broadcast='params',
                split_rngs={'params': False})(As)

    key = random.PRNGKey(0)
    x = jnp.ones((10, 2))

    p = B(As).init(key, x, x)
    y, cntrs = b.apply(p, x, x, mutable='counter')
    ref_cntrs = freeze({
        'counter': {
            'A_bar': {
                'i': jnp.array(11.0),
            },
            'A_foo': {
                'i': jnp.array(11.0),
            },
        },
      })
    self.assertTrue(jax.tree_util.tree_all(
        jax.tree_multimap(
            lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
            cntrs, ref_cntrs)
        ))

  def test_partially_applied_module_constructor_transform(self):
    k = random.PRNGKey(0)
    x = jnp.ones((3,4,4))
    dense = partial(nn.Dense, use_bias=False)
    vmap_dense = nn.vmap(
        dense,
        variable_axes={'params':0},
        split_rngs={'params':True})(4)
    init_vars = vmap_dense.init(k, x)
    init_vars_shapes = jax.tree_map(jnp.shape, init_vars)
    ref_var_shapes = freeze({
        'params': {
            'kernel': (3, 4, 4),
        },
    })
    self.assertTrue(tree_equals(init_vars_shapes, ref_var_shapes))

  def test_partial_module_method(self):
    k = random.PRNGKey(0)
    x = jnp.ones((3,4,4))
    class Foo(nn.Module):

      @nn.compact
      def inner(self, x):
        return nn.Dense(2, use_bias=False)(x)

      def __call__(self, x):
        return nn.vmap(
            partial(Foo.inner),
            variable_axes={'params':0},
            split_rngs={'params':True})(self, x)

    init_vars = Foo().init(k, x)
    init_vars_shapes = jax.tree_map(jnp.shape, init_vars)
    ref_var_shapes = freeze({
        'params': {
          'Dense_0': {'kernel': (3, 4, 2)}
        },
    })
    self.assertTrue(tree_equals(init_vars_shapes, ref_var_shapes))

  def test_variable_in_args_transform(self):
    class Test(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        baz = self.variable('test', 'baz', jnp.zeros, x.shape)
        y = self.mutate_variable_in_method(x, baz)
        return y
      @nn.jit
      def mutate_variable_in_method(self, x, baz):
        baz.value += x
        return baz.value

    k = random.PRNGKey(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(variables['test']['baz'],
                               jnp.array([1.0,]), atol=1e-7)
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(variables['test']['baz'],
                               jnp.array([2.0,]), atol=1e-7)

  def test_module_instance_in_args_transform(self):
    class Inner(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        baz = self.variable('test', 'baz', jnp.zeros, x.shape)
        baz.value += x
        return baz.value

    class Test(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        inner = Inner(name="inner")
        y = self.call_instance_arg_in_method(x, inner)
        return y
      @nn.jit
      def call_instance_arg_in_method(self, x, inner):
        return inner(x)

    k = random.PRNGKey(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(variables['test']['inner']['baz'],
                                jnp.array([1.0,]), atol=1e-7)
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(variables['test']['inner']['baz'],
                                jnp.array([2.0,]), atol=1e-7)

  def test_module_instance_in_args_transform_nested(self):
    class Inner(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        baz = self.variable('test', 'baz', jnp.zeros, x.shape)
        baz.value += x
        return baz.value

    class Outer(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, inner, x):
        y = self.call_instance_arg_in_method(x, inner)
        return y
      @nn.jit
      def call_instance_arg_in_method(self, x, inner):
        return inner(x)

    class Test(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        inner = Inner(name="inner")
        outer = Outer(name="outer")
        return outer(inner, x)

    k = random.PRNGKey(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(variables['test']['inner']['baz'],
                                jnp.array([1.0,]), atol=1e-7)
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(variables['test']['inner']['baz'],
                                jnp.array([2.0,]), atol=1e-7)


  def test_nested_variable_passing(self):
    class NestedVarUser(nn.Module):
      somevar: nn.Variable
      @nn.jit
      @nn.compact
      def __call__(self, x):
        self.somevar.value += x
        return x
    class VarUser(nn.Module):
      somevar: nn.Variable
      @nn.jit
      @nn.compact
      def __call__(self, x):
        return NestedVarUser(self.somevar)(x)
    class VarPasser(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        baz = self.variable('test', 'baz', jnp.zeros, x.shape)
        y = VarUser(baz)(x)
        return y

    k = random.PRNGKey(0)
    x = jnp.ones((1,))
    variables = VarPasser().init(k, x)
    np.testing.assert_allclose(variables['test']['baz'],
                               jnp.array([1.0,]), atol=1e-7)
    y, variables = VarPasser().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(variables['test']['baz'],
                               jnp.array([2.0,]), atol=1e-7)

  def test_returned_module_warning(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return x
    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        f = self._helper()
        return f(x)
      def _helper(self):
        return Foo()
    with nn.override_named_call(True):
      b = Bar()
      with self.assertRaises(errors.TransformedMethodReturnValueError):
        b.apply({}, jnp.ones(2))

  def test_nowrap_named_call(self):
    with nn.override_named_call(True):
      class Foo(nn.Module):
        @nn.compact
        def __call__(self, x):
          return x
      class Bar(nn.Module):
        @nn.compact
        def __call__(self, x):
          f = self._helper()
          return f(x)
        # will fail without nowrap
        @nn.nowrap
        def _helper(self):
          return Foo()

      b = Bar()
      x = jnp.ones(2)
      y = b.apply({}, x)
      np.testing.assert_array_equal(x, y)

  def test_nowrap(self):
    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        return self._helper(x)
      @nn.nowrap
      def _helper(self, x):
        if len(nn.module._context.module_stack) > 2:
          raise ValueError('Module stack too deep.')
        return x

    b = Bar()
    b.apply({}, jnp.ones(2))

  def test_map_variables_tied_autoencoder(self):
    def trans(variables):
      return jax.tree_map(lambda x: x.T, variables)

    class TiedAutencoder(nn.Module):

      features: int
      latents: int

      @nn.compact
      def _call(self, x, decode):
        def f(self):
          return nn.Dense(self.features if decode else self.latents, use_bias=False)(x)

        if decode:
          map_fn = trans
        else:
          map_fn = lambda x: x
        return nn.map_variables(f, "params", map_fn, map_fn, mutable=True)(self)

      def encode(self, x):
        return self._call(x, False)

      def decode(self, x):
        return self._call(x, True)

      def __call__(self, x):
        return self.decode(self.encode(x))

    x = jnp.ones((2, 4))
    ae = TiedAutencoder(4, 5)
    variables = ae.init(random.PRNGKey(0), x)
    param_shapes = jax.tree_map(jnp.shape, variables["params"])
    self.assertEqual(param_shapes, {
      "Dense_0": {"kernel": (4, 5)}
    })


  def test_map_variables_bit_weights(self):
    class BitWeights(nn.Module):
      @nn.compact
      def __call__(self, x):
        def sign(x):
          return jax.tree_map(jnp.sign, x)
        BitDense = nn.map_variables(nn.Dense, "params", sign, init=True)
        return BitDense(4)(x)
    bw = BitWeights()
    x = jnp.ones((2, 4))
    y, variables = bw.init_with_output(random.PRNGKey(0), x)
    y_2 = bw.apply(variables, x)
    np.testing.assert_allclose(y, y_2)



  def test_remat_scan(self):
    class BigModel(nn.Module):
      @nn.compact
      def __call__(self, x):
        DenseStack = nn.remat_scan(nn.Dense, lengths=(100,))
        return DenseStack(8, name="dense_stack")(x)

    x = jnp.ones((2, 8))
    model = BigModel()
    variables = model.init(random.PRNGKey(0), x)
    param_shapes = jax.tree_map(jnp.shape, variables['params'])
    self.assertEqual(param_shapes["dense_stack"]["kernel"], (100, 8, 8))
    self.assertEqual(param_shapes["dense_stack"]["bias"], (100, 8))
    y = model.apply(variables, x)
    self.assertEqual(y.shape, (2, 8))


  def test_vjp(self):
    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('test', nn.initializers.zeros, ())
        self.variable('state', 'counter', lambda: 0)
        return p * x

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        y, bwd = nn.vjp(Bar.__call__, Bar(), x)
        params_grad, x_grad = bwd(jnp.ones(y.shape))
        return params_grad, x_grad

    x = jnp.ones((3,))
    params = Foo().init(random.PRNGKey(0), x)
    x_grad, params_grad = Foo().apply(params, x)
    self.assertEqual(params_grad, {
      'params': nn.FrozenDict({'test': 3.}),
    })
    np.testing.assert_allclose(x_grad, 0. * x)

  def test_jvp(self):
    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('test', nn.initializers.zeros, ())
        self.variable('state', 'counter', lambda: 0)
        return p * x

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        bar = Bar()
        vars_t = jax.tree_map(jnp.ones_like, bar.variables.get('params', {}))
        _, out_t = nn.jvp(Bar.__call__, bar, (x,), (jnp.zeros_like(x),), {'params': vars_t})
        return out_t

    x = jnp.ones((3,))
    params = Foo().init(random.PRNGKey(0), x)
    y_t = Foo().apply(params, x)
    np.testing.assert_allclose(y_t, jnp.ones_like(x))

  def test_complicated_alias_mutation(self):
    with nn.module.override_named_call(True):
      class A(nn.Module):
        b: nn.Module
        @nn.compact
        def __call__(self, x):
          return self.b(x)
      class B(nn.Module):
        c: nn.Module
        @nn.compact
        def __call__(self, x):
          y = C(name='outer_c')(x)
          z = self.c(x)
          return z
      class C(nn.Module):
        @nn.compact
        def __call__(self, x):
          initialized = self.has_variable('muts', 'v')
          v = self.variable('muts', 'v', lambda: jnp.zeros_like(x))
          if initialized:
            v.value += x
          return x

      a = A(b=B(c=C()))
      k = random.PRNGKey(0)
      x = jnp.ones((1,), jnp.float32)
      vs = a.init(k, x)
      y, vs_new = a.apply(vs, x, mutable=['muts',])
      np.testing.assert_array_equal(vs_new['muts']['b']['c']['v'],
                                    jnp.array([1.], jnp.float32))
      np.testing.assert_array_equal(vs_new['muts']['b']['outer_c']['v'],
                                    jnp.array([1.], jnp.float32))

  def test_custom_vjp(self):

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        def f(mdl, x):
          return mdl(x)

        def fwd(mdl, x):
          return nn.vjp(f, mdl, x)

        def bwd(vjp_fn, y_t):
          input_t, params_t = vjp_fn(y_t)
          params_t = jax.tree_map(jnp.sign, params_t)
          return input_t, params_t

        sign_grad = nn.custom_vjp(
            f, forward_fn=fwd, backward_fn=bwd)
        return sign_grad(nn.Dense(1), x).reshape(())
    x = jnp.ones((2,))
    variables = Foo().init(random.PRNGKey(0), x)
    grad = jax.grad(Foo().apply)(variables, x)
    for grad_leaf in jax.tree_leaves(grad):
      self.assertTrue(jnp.all(jnp.abs(grad_leaf) == 1.))

  def test_transform_with_setup_and_methods_on_submodules(self):
    # This is the archetypal example motivating the introduction of
    # SetupState as a triple-enum to handle multiple setup() calls
    # across transform boundaries and scope reuse.
    class Foo(nn.Module):
      def setup(self):
        self.inner = nn.Dense(2)
      def helper(self, x, m):
        return m(x)
      def __call__(self, x):
        return self.helper(x, self.inner)
    k = random.PRNGKey(0)
    x = jnp.ones((2,))
    with nn.module.override_named_call(True):
      vs_foo = Foo().init(k, x)

    class Bar(nn.Module):
      def setup(self):
        self.inner = nn.Dense(2)
      @nn.jit
      def helper(self, x, m):
        return m(x)
      @nn.jit
      def __call__(self, x):
        return self.helper(x, self.inner)
    vs_bar = Bar().init(k, x)
    self.assertTrue(tree_equals(
      jax.tree_map(jnp.shape, vs_foo),
      jax.tree_map(jnp.shape, vs_bar)))

  def test_transform_methods_on_submodules_still_reserve_names(self):
    class Foo(nn.Module):
      @nn.jit
      def helper(self, x, m):
        conflicting_a = nn.Dense(2, name="a")
        return m(x)
      @nn.jit
      @nn.compact
      def __call__(self, x):
        a = nn.Dense(2, name="a")
        return self.helper(x, a)
    k = random.PRNGKey(0)
    x = jnp.ones((2,))
    with self.assertRaises(errors.NameInUseError):
      vs = Foo().init(k, x)

  def test_transform_setup_still_reserve_names(self):
    class Identity(nn.Module):
      @nn.compact
      def __call__(self, x):
        return x
    class Test(nn.Module):
      def setup(self):
        self.sub = Identity()
        self.sub = Identity()
      @nn.jit
      def __call__(self, x):
        return x

    k = random.PRNGKey(0)
    x = jnp.array([1.])

    msg = 'Duplicate use of scope name: "sub"'
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      y = Test().init(k, x)

  def test_scan_of_setup_parameter(self):
    class Body(nn.Module):
      def setup(self):
        self.dense = nn.Dense(1)
        self.p = self.param('p', lambda k: jnp.ones((1,)))
      def __call__(self, x):
        return self.dense(x) + self.p, None
    scanbody = nn.scan(
      Body,
      variable_axes={'params': 0},
      split_rngs={'params': True},
      length=2)
    k = random.PRNGKey(0)
    x = jnp.ones((1,))
    vs = scanbody().init(k, x)
    y = scanbody().apply(vs, x)


if __name__ == '__main__':
  absltest.main()

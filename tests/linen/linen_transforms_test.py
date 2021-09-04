# Copyright 2021 The Flax Authors.
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
    xs = random.uniform(key1, (3, 2))
    dummy_rng = random.PRNGKey(0)
    init_carry = nn.LSTMCell.initialize_carry(dummy_rng,
                                              xs.shape[:1],
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
    xs = random.uniform(key1, (3, 2))
    b = jnp.ones((4,))
    dummy_rng = random.PRNGKey(0)
    init_carry = nn.LSTMCell.initialize_carry(dummy_rng,
                                              xs.shape[:1],
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
    nn.enable_named_call()
    try:
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
    finally:
      nn.disable_named_call()

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
    cntr = 0
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
        nonlocal cntr
        cntr += 1
        self.dense = nn.Dense(2, use_bias=False)
      @nn.remat
      def __call__(self, x):
        return self.dense(x)

    def nested_repeat(mdl):
      for _ in range(D):
        mdl = partial(Repeat, mdl)
      return mdl()
    _ = nested_repeat(Counter).init(random.PRNGKey(0), jnp.ones((2,)))
    self.assertEqual(cntr, 64)

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
    self.assertEqual(cntr, 2)

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

    print(jax.tree_map(jnp.shape, p1))
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
    nn.enable_named_call()
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
    nn.disable_named_call()

    b = Bar()
    with self.assertRaises(errors.TransformedMethodReturnValueError):
      b.apply({}, jnp.ones(2))

  def test_nowrap_named_call(self):
    nn.enable_named_call()
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
    nn.disable_named_call()

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


if __name__ == '__main__':
  absltest.main()

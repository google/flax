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

"""Transforms tests."""

from functools import partial
import operator
from typing import Any
from collections.abc import Callable, Sequence
import unittest

from absl.testing import absltest, parameterized
from flax import errors
from flax import linen as nn
from flax import serialization
from flax import struct
from flax.core import copy, freeze, AxisMetadata
from flax.linen.transforms import _HashableProxy
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

# pylint: disable=attribute-defined-outside-init,unused-variable,g-wrong-blank-lines,g-bare-generic


def tree_equals(x, y):
  return jax.tree_util.tree_all(jax.tree_util.tree_map(operator.eq, x, y))


def tree_allclose(x, y):
  return jax.tree_util.tree_all(
      jax.tree_util.tree_map(lambda x, y: np.all(np.isclose(x, y)), x, y)
  )


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


class TransformTest(parameterized.TestCase):

  def assert_keys_equal(self, key1, key2):
    self.assertEqual(key1.dtype, key2.dtype)
    np.testing.assert_array_equal(random.key_data(key1), random.key_data(key2))

  def test_jit(self):
    key1, key2 = random.split(random.key(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = TransformedMLP(features=[3, 4, 5])
    jit_model = TransformedMLP(features=[3, 4, 5], transform=nn.jit)
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = jit_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_jit_decorated(self):
    key1, key2 = random.split(random.key(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = decorated_MLP()(features=[3, 4, 5])
    jit_model = decorated_MLP(nn.jit)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = jit_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_jit_init_fn(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(2)(x)

      @nn.jit
      def init_with_output(self, rngs, *args, **kwargs):
        return super().init_with_output(rngs, *args, **kwargs)

    Foo().init_with_output(random.key(0), jnp.ones((2, 3)))


  def test_remat(self):
    key1, key2 = random.split(random.key(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = TransformedMLP(features=[3, 4, 5])
    remat_model = TransformedMLP(features=[3, 4, 5], transform=nn.remat)
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = remat_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_remat_decorated(self):
    key1, key2 = random.split(random.key(3), 2)
    x = random.uniform(key1, (4, 4))

    normal_model = decorated_MLP()(features=[3, 4, 5])
    remat_model = decorated_MLP(nn.remat)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    y1 = normal_model.apply(init_variables, x)
    y2 = remat_model.apply(init_variables, x)

    self.assertTrue(np.all(y1 == y2))

  def test_remat_kwargs(self):
    raise unittest.SkipTest('test breaks with grad')

    class ConditionalReLU(nn.Module):

      @nn.compact
      def __call__(self, input, apply_relu: bool = False):
        return nn.relu(input) if apply_relu else input

    key = random.key(0)
    x = jnp.ones((4, 4)) * -1
    remat_model = nn.remat(ConditionalReLU)()
    p = remat_model.init(key, x)
    y = remat_model.apply(p, x, apply_relu=True)

    self.assertTrue(np.all(y == jnp.zeros_like(x)))

    # This next line crashes with a concretization error
    _ = jax.grad(lambda x: remat_model.apply(p, x, apply_relu=True))(x)

  def test_remat_static_argnums(self):
    test = self

    class Foo(nn.Module):
      train_is_static: bool

      @nn.compact
      def __call__(self, inputs, train: bool):
        if self.train_is_static:
          test.assertTrue(isinstance(train, bool))
        else:
          test.assertTrue(isinstance(train, jnp.ndarray))

        return nn.Dense(3, use_bias=False)(inputs)

    # set train as a static argument
    FooRemat = nn.remat(Foo, static_argnums=(2,))
    foo = FooRemat(train_is_static=True)

    x = jnp.empty((1, 2))
    variables = foo.init(random.key(0), x, True)
    y = foo.apply(variables, x, False)
    self.assertEqual(y.shape, (1, 3))

    # set train as a non-static arguments
    FooRemat = nn.remat(Foo, static_argnums=())
    foo = FooRemat(train_is_static=False)

    variables = foo.init(random.key(0), x, True)
    y = foo.apply(variables, x, False)
    self.assertEqual(y.shape, (1, 3))

  def test_remat_decorator_static_argnums(self):
    test = self

    class FooTrainStatic(nn.Module):

      @partial(nn.remat, static_argnums=(2,))
      @nn.compact
      def __call__(self, inputs, train: bool):
        test.assertTrue(isinstance(train, bool))

        return nn.Dense(3, use_bias=False)(inputs)

    # set train as a static argument
    foo = FooTrainStatic()

    x = jnp.empty((1, 2))
    variables = foo.init(random.key(0), x, True)
    y = foo.apply(variables, x, False)
    self.assertEqual(y.shape, (1, 3))

    class FooTrainDynamic(nn.Module):

      @partial(nn.remat, static_argnums=())
      @nn.compact
      def __call__(self, inputs, train: bool):
        test.assertTrue(isinstance(train, jnp.ndarray))

        return nn.Dense(3, use_bias=False)(inputs)

    # set train as a non-static arguments
    foo = FooTrainDynamic()

    variables = foo.init(random.key(0), x, True)
    y = foo.apply(variables, x, False)
    self.assertEqual(y.shape, (1, 3))

  def test_vmap(self):
    key1, key2, key3 = random.split(random.key(3), 3)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key2, (5, 4, 4))

    def vmap(cls):
      return nn.vmap(
          cls,
          in_axes=(0,),
          variable_axes={'params': None},
          split_rngs={'params': False},
      )

    normal_model = TransformedMLP(features=[3, 4, 5])
    vmap_model = TransformedMLP(features=[3, 4, 5], transform=vmap)
    init_variables = normal_model.init(key3, x)
    # simulate vmap in python for comparison:
    y1 = jnp.vstack([
        normal_model.apply(init_variables, x2[i])[None, ...]
        for i in np.arange(x2.shape[0])
    ])
    y2 = vmap_model.apply(init_variables, x2)
    np.testing.assert_allclose(y1, y2, atol=1e-6)

  def test_vmap_decorated(self):
    key1, key2, key3 = random.split(random.key(3), 3)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key2, (5, 4, 4))

    def vmap(fn):
      return nn.vmap(
          fn,
          in_axes=(0,),
          variable_axes={'params': None},
          split_rngs={'params': False},
      )

    normal_model = decorated_MLP()(features=[3, 4, 5])
    vmap_model = decorated_MLP(vmap)(features=[3, 4, 5])
    init_variables = normal_model.init(key3, x)
    # simulate vmap in python for comparison:
    y1 = jnp.vstack([
        normal_model.apply(init_variables, x2[i])[None, ...]
        for i in np.arange(x2.shape[0])
    ])
    y2 = vmap_model.apply(init_variables, x2)
    np.testing.assert_allclose(y1, y2, atol=1e-6)

  def test_vmap_batchnorm(self):
    key1, key2, key3 = random.split(random.key(3), 3)
    x = random.uniform(key1, (4, 4))
    x2 = random.uniform(key2, (5, 4, 4))

    def vmap(cls):
      return nn.vmap(
          cls,
          in_axes=(0,),
          variable_axes={'params': None, 'batch_stats': None},
          split_rngs={'params': False},
          axis_name='batch',
      )

    class MlpBn(nn.Module):
      axis_name: Any = None

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.BatchNorm(axis_name=self.axis_name, use_running_average=False)(x)
        return x

    normal_model = MlpBn()
    vmap_model = vmap(MlpBn)(axis_name='batch')
    init_variables = normal_model.init(key3, x)
    y1 = normal_model.apply(
        init_variables, x2.reshape((-1, 4)), mutable=['batch_stats']
    )[0]
    y1 = y1.reshape((5, 4, 3))
    y2 = vmap_model.apply(init_variables, x2, mutable=['batch_stats'])[0]
    np.testing.assert_allclose(y1, y2, atol=1e-5)

  def test_scan(self):
    class SimpleScan(nn.Module):
      features: int

      @nn.compact
      def __call__(self, c, xs):
        LSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast='params',
            split_rngs={'params': False},
        )
        return LSTM(self.features, name='lstm_cell')(c, xs)

    key1, key2 = random.split(random.key(0), 2)
    xs = random.uniform(key1, (5, 3, 2))
    dummy_rng = random.key(0)
    init_carry = nn.LSTMCell(2).initialize_carry(dummy_rng, xs[0].shape)
    model = SimpleScan(2)
    init_variables = model.init(key2, init_carry, xs)
    # simulate scan in python for comparison:
    c = init_carry
    ys = []
    lstmcell_variables = freeze(
        {'params': init_variables['params']['lstm_cell']}
    )
    for i in range(xs.shape[0]):
      c, y = nn.LSTMCell(2).apply(lstmcell_variables, c, xs[i])
      ys.append(y[None, ...])
    y1 = jnp.vstack(ys)

    c2, y2 = model.apply(init_variables, init_carry, xs)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    np.testing.assert_allclose(c[0], c2[0], atol=1e-7)
    np.testing.assert_allclose(c[1], c2[1], atol=1e-7)

  def test_scan_decorated(self):
    class SimpleScan(nn.Module):
      features: int

      @partial(
          nn.scan,
          variable_broadcast='params',
          in_axes=(nn.broadcast, 0),
          split_rngs={'params': False},
      )
      @nn.compact
      def __call__(self, c, b, xs):
        assert b.shape == (4,)
        return nn.LSTMCell(self.features, name='lstm_cell')(c, xs)

    key1, key2 = random.split(random.key(0), 2)
    xs = random.uniform(key1, (4, 3, 2))
    b = jnp.ones((4,))
    dummy_rng = random.key(0)
    init_carry = nn.LSTMCell(2).initialize_carry(dummy_rng, xs[0].shape)
    model = SimpleScan(2)
    init_variables = model.init(key2, init_carry, b, xs)
    # simulate scan in python for comparison:
    c = init_carry
    ys = []
    lstmcell_variables = freeze(
        {'params': init_variables['params']['lstm_cell']}
    )
    for i in range(xs.shape[0]):
      c, y = nn.LSTMCell(2).apply(lstmcell_variables, c, xs[i])
      ys.append(y[None, ...])
    y1 = jnp.vstack(ys)

    c2, y2 = model.apply(init_variables, init_carry, b, xs)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    np.testing.assert_allclose(c[0], c2[0], atol=1e-7)
    np.testing.assert_allclose(c[1], c2[1], atol=1e-7)

  def test_scan_negative_axes(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, _, x):
        x = nn.Dense(4)(x)
        return None, x

    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        _, x = nn.scan(
            Foo,
            variable_broadcast='params',
            split_rngs=dict(params=False),
            in_axes=1,
            out_axes=-1,
        )()(None, x)
        return x

    y, variables = Bar().init_with_output(
        {'params': jax.random.PRNGKey(0)},
        jax.random.normal(jax.random.PRNGKey(1), shape=[1, 2, 3]),
    )
    params = variables['params']

    self.assertEqual(y.shape, (1, 4, 2))
    self.assertEqual(params['ScanFoo_0']['Dense_0']['kernel'].shape, (3, 4))
    self.assertEqual(params['ScanFoo_0']['Dense_0']['bias'].shape, (4,))

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
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    _, new_vars = Test(parent=None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(
        init_vars['counter']['outer']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer']['cntr']['foo'], jnp.array([4], jnp.int32)
    )

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
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    _, new_vars = Test(parent=None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(
        init_vars['counter']['outer']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer']['cntr']['foo'], jnp.array([4], jnp.int32)
    )

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
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    _, new_vars = Test(parent=None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(
        init_vars['counter']['outer1']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer1']['cntr']['foo'], jnp.array([4], jnp.int32)
    )
    self.assertEqual(
        init_vars['counter']['outer2']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer2']['cntr']['foo'], jnp.array([4], jnp.int32)
    )

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
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    _, new_vars = Test(parent=None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(
        init_vars['counter']['outer1']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer1']['cntr']['foo'], jnp.array([4], jnp.int32)
    )
    self.assertEqual(
        init_vars['counter']['outer2']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer2']['cntr']['foo'], jnp.array([4], jnp.int32)
    )

  def test_multiscope_lifting_simple_decorator_w_jit(self):
    # TODO: actually test jaxpr on a simpler module.
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

      @nn.jit
      @nn.compact
      def __call__(self, x):
        outer_dense = Outer(name='outer')
        # we share stateful outer module as arg to two different, transformed modules:
        inner = Inner(outer_dense, name='inner1')
        inner2 = Inner(outer_dense, name='inner2')
        res = inner(x) + inner2(x)
        return res

    x = jnp.ones((1, 1))
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    _, new_vars = Test(parent=None).apply(init_vars, x, mutable=['counter'])
    self.assertEqual(
        init_vars['counter']['outer']['cntr']['foo'], jnp.array([2], jnp.int32)
    )
    self.assertEqual(
        new_vars['counter']['outer']['cntr']['foo'], jnp.array([4], jnp.int32)
    )

  def test_vmapped_outer_module(self):
    class Outer(nn.Module):

      @nn.jit
      @nn.compact
      def __call__(self, x):
        return nn.Dense(5)(x)

    class Inner(nn.Module):
      outer_module: nn.Module

      @partial(
          nn.vmap,
          in_axes=(0,),
          variable_axes={'params': 0},
          split_rngs={'params': True},
      )
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
    rngs = random.key(0)
    init_vars = Test(parent=None).init(rngs, x)
    y = Test(parent=None).apply(init_vars, x)
    self.assertEqual(
        init_vars['params']['outer']['Dense_0']['kernel'].shape, (3, 2, 5)
    )
    self.assertEqual(
        init_vars['params']['outer']['Dense_0']['bias'].shape, (3, 5)
    )
    self.assertEqual(y.shape, (3, 1, 5))

  def test_module_transform_with_setup(self):
    class Foo(nn.Module):

      def setup(self):
        self.test = self.param('test', nn.initializers.ones_init(), ())

      def __call__(self, x):
        return x * self.test

    FooVmap = nn.vmap(
        Foo,
        in_axes=0,
        out_axes=0,
        variable_axes={'params': 0},
        split_rngs={'params': True},
    )
    variables = FooVmap().init(random.key(0), jnp.ones((4,)))
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

      @partial(
          nn.vmap, variable_axes={'params': 0}, split_rngs={'params': True}
      )
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

    key = random.key(0)
    x = jnp.ones((10, 10))
    p = D().init(key, x)

    variable_shapes = jax.tree_util.tree_map(jnp.shape, p)
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['kernel'], (10, 10, 3)
    )
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['bias'], (10, 3)
    )

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
          nn.vmap, variable_axes={'params': 0}, split_rngs={'params': True}
      )
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

    key = random.key(0)
    x = jnp.ones((10, 10))
    p = D().init(key, x)

    variable_shapes = jax.tree_util.tree_map(jnp.shape, p)
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['kernel'], (10, 10, 3)
    )
    self.assertEqual(
        variable_shapes['params']['A_0']['Dense_0']['bias'], (10, 3)
    )
    self.assertEqual(
        variable_shapes['params']['A_1']['Dense_0']['kernel'], (10, 10, 3)
    )
    self.assertEqual(
        variable_shapes['params']['A_1']['Dense_0']['bias'], (10, 3)
    )

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

    _ = nested_repeat(Counter).init(random.key(0), jnp.ones((2,)))
    # setup_cntr == 128 due to 1 call in Counter.setup by _validate_setup
    # and 1 further "real" call.
    self.assertEqual(setup_cntr, 128)
    self.assertEqual(call_cntr, 64)

  def test_multimethod_setup_calls(self):
    cntr = 0

    class A(nn.Module):

      def setup(self):
        nonlocal cntr
        cntr += 1
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

    key = random.key(0)
    x = jnp.ones((2,))
    (y1, y2), _ = B().init_with_output(key, x)
    np.testing.assert_array_equal(y1, y2)
    # cntr == 4 due to:
    # 1 call by _validate_setup
    # 1 call for the setup() outside transform boundary
    # and two further "real" calls in transform boundaries
    self.assertEqual(cntr, 4)

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
          nn.vmap, variable_axes={'params': 0}, split_rngs={'params': True}
      )
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

    key = random.key(0)
    x = jnp.ones((10, 10))
    p1 = D().init(key, x)
    y1 = D().apply(p1, x)

    a1 = A()
    a2 = A()
    b = B(a1)
    p2 = freeze({
        'params': {
            'A': p1['params']['A_0'],
            'B': {
                'A': p1['params']['A_1'],
            },
        }
    })

    # Test method wrapper transform.
    y2 = C(a2, b).apply(p2, x)
    np.testing.assert_allclose(y1, y2, atol=1e-7)
    # Test class transform.
    Ctrafo = nn.vmap(
        Csimple, variable_axes={'params': 0}, split_rngs={'params': True}
    )

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
    b = nn.scan(
        B,
        in_axes=0,
        variable_carry='counter',
        variable_broadcast='params',
        split_rngs={'params': False},
    )(As)

    key = random.key(0)
    x = jnp.ones((10, 2))

    p = B(As).init(key, x, x)
    y, cntrs = b.apply(p, x, x, mutable='counter')
    ref_cntrs = {
        'counter': {
            'A_bar': {
                'i': jnp.array(11.0),
            },
            'A_foo': {
                'i': jnp.array(11.0),
            },
        },
    }
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
                cntrs,
                ref_cntrs,
            )
        )
    )

  def test_partially_applied_module_constructor_transform(self):
    k = random.key(0)
    x = jnp.ones((3, 4, 4))
    dense = partial(nn.Dense, use_bias=False)
    vmap_dense = nn.vmap(
        dense, variable_axes={'params': 0}, split_rngs={'params': True}
    )(4)
    init_vars = vmap_dense.init(k, x)
    init_vars_shapes = jax.tree_util.tree_map(jnp.shape, init_vars)
    ref_var_shapes = {
        'params': {
            'kernel': (3, 4, 4),
        },
    }
    self.assertTrue(tree_equals(init_vars_shapes, ref_var_shapes))

  def test_partial_module_method(self):
    k = random.key(0)
    x = jnp.ones((3, 4, 4))

    class Foo(nn.Module):

      @nn.compact
      def inner(self, x):
        return nn.Dense(2, use_bias=False)(x)

      def __call__(self, x):
        return nn.vmap(
            partial(Foo.inner),
            variable_axes={'params': 0},
            split_rngs={'params': True},
        )(self, x)

    init_vars = Foo().init(k, x)
    init_vars_shapes = jax.tree_util.tree_map(jnp.shape, init_vars)
    ref_var_shapes = {
        'params': {'Dense_0': {'kernel': (3, 4, 2)}},
    }
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

    k = random.key(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(
        variables['test']['baz'],
        jnp.array([
            1.0,
        ]),
        atol=1e-7,
    )
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(
        variables['test']['baz'],
        jnp.array([
            2.0,
        ]),
        atol=1e-7,
    )

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
        inner = Inner(name='inner')
        y = self.call_instance_arg_in_method(x, inner)
        return y

      @nn.jit
      def call_instance_arg_in_method(self, x, inner):
        return inner(x)

    k = random.key(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(
        variables['test']['inner']['baz'],
        jnp.array([
            1.0,
        ]),
        atol=1e-7,
    )
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(
        variables['test']['inner']['baz'],
        jnp.array([
            2.0,
        ]),
        atol=1e-7,
    )

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
        inner = Inner(name='inner')
        outer = Outer(name='outer')
        return outer(inner, x)

    k = random.key(0)
    x = jnp.ones((1,))
    variables = Test().init(k, x)
    np.testing.assert_allclose(
        variables['test']['inner']['baz'],
        jnp.array([
            1.0,
        ]),
        atol=1e-7,
    )
    y, variables = Test().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(
        variables['test']['inner']['baz'],
        jnp.array([
            2.0,
        ]),
        atol=1e-7,
    )

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

    k = random.key(0)
    x = jnp.ones((1,))
    variables = VarPasser().init(k, x)
    np.testing.assert_allclose(
        variables['test']['baz'],
        jnp.array([
            1.0,
        ]),
        atol=1e-7,
    )
    y, variables = VarPasser().apply(variables, x, mutable=['test'])
    np.testing.assert_allclose(
        variables['test']['baz'],
        jnp.array([
            2.0,
        ]),
        atol=1e-7,
    )

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

      @nn.jit
      def _helper(self):
        return Foo()

    b = Bar()
    with self.assertRaises(errors.TransformedMethodReturnValueError):
      b.apply({}, jnp.ones(2))

  def test_returned_variable_warning(self):
    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        f = self._helper()
        return f(x)

      @nn.jit
      def _helper(self):
        return nn.Variable(None, None, None, False)

    b = Bar()
    with self.assertRaises(errors.TransformedMethodReturnValueError):
      b.apply({}, jnp.ones(2))

  def test_nowrap(self):
    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        return self._helper(x)

      @nn.nowrap
      def _helper(self, x):
        if len(nn.module._context.module_stack) > 2:  # pylint: disable=protected-access
          raise ValueError('Module stack too deep.')
        return x

    b = Bar()
    b.apply({}, jnp.ones(2))

  def test_map_variables_tied_autoencoder(self):
    def trans(variables):
      return jax.tree_util.tree_map(lambda x: x.T, variables)

    class TiedAutencoder(nn.Module):
      features: int
      latents: int

      @nn.compact
      def _call(self, x, decode):
        def f(self):
          return nn.Dense(
              self.features if decode else self.latents, use_bias=False
          )(x)

        if decode:
          map_fn = trans
        else:
          map_fn = lambda x: x
        return nn.map_variables(f, 'params', map_fn, map_fn, mutable=True)(self)

      def encode(self, x):
        return self._call(x, False)

      def decode(self, x):
        return self._call(x, True)

      def __call__(self, x):
        return self.decode(self.encode(x))

    x = jnp.ones((2, 4))
    ae = TiedAutencoder(4, 5)
    variables = ae.init(random.key(0), x)
    param_shapes = jax.tree_util.tree_map(jnp.shape, variables['params'])
    self.assertEqual(param_shapes, {'Dense_0': {'kernel': (4, 5)}})

  def test_map_variables_bit_weights(self):
    class BitWeights(nn.Module):

      @nn.compact
      def __call__(self, x):
        def sign(x):
          return jax.tree_util.tree_map(jnp.sign, x)

        BitDense = nn.map_variables(nn.Dense, 'params', sign, init=True)
        return BitDense(4)(x)

    bw = BitWeights()
    x = jnp.ones((2, 4))
    y, variables = bw.init_with_output(random.key(0), x)
    y_2 = bw.apply(variables, x)
    np.testing.assert_allclose(y, y_2)

  def test_remat_scan(self):
    class BigModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        DenseStack = nn.remat_scan(nn.Dense, lengths=(100,))
        return DenseStack(8, name='dense_stack')(x)

    x = jnp.ones((2, 8))
    model = BigModel()
    variables = model.init(random.key(0), x)
    param_shapes = jax.tree_util.tree_map(jnp.shape, variables['params'])
    self.assertEqual(param_shapes['dense_stack']['kernel'], (100, 8, 8))
    self.assertEqual(param_shapes['dense_stack']['bias'], (100, 8))
    y = model.apply(variables, x)
    self.assertEqual(y.shape, (2, 8))

  def test_vjp(self):
    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        p = self.param('test', nn.initializers.constant(0.5), ())
        self.variable('state', 'counter', lambda: 0)
        return p * x * y

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        z, bwd = nn.vjp(Bar.__call__, Bar(), x, y)
        return bwd(jnp.ones(z.shape))

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    params = Foo().init(random.key(0), x, y)
    params_grad, x_grad, y_grad = Foo().apply(params, x, y)
    self.assertEqual(
        params_grad,
        {
            'params': nn.FrozenDict({'test': 32.0}),
        },
    )
    np.testing.assert_allclose(x_grad, [2.0, 2.5, 3.0])
    np.testing.assert_allclose(y_grad, [0.5, 1.0, 1.5])

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
        vars_t = jax.tree_util.tree_map(
            jnp.ones_like, bar.variables.get('params', {})
        )
        _, out_t = nn.jvp(
            Bar.__call__, bar, (x,), (jnp.zeros_like(x),), {'params': vars_t}
        )
        return out_t

    x = jnp.ones((3,))
    params = Foo().init(random.key(0), x)
    y_t = Foo().apply(params, x)
    np.testing.assert_allclose(y_t, jnp.ones_like(x))

  def test_complicated_alias_mutation(self):
    class A(nn.Module):
      b: nn.Module

      @nn.jit
      @nn.compact
      def __call__(self, x):
        return self.b(x)

    class B(nn.Module):
      c: nn.Module

      @nn.jit
      @nn.compact
      def __call__(self, x):
        y = C(name='outer_c')(x)
        z = self.c(x)
        return z

    class C(nn.Module):

      @nn.jit
      @nn.compact
      def __call__(self, x):
        initialized = self.has_variable('muts', 'v')
        v = self.variable('muts', 'v', lambda: jnp.zeros_like(x))
        if initialized:
          v.value += x
        return x

    a = A(b=B(c=C()))
    k = random.key(0)
    x = jnp.ones((1,), jnp.float32)
    vs = a.init(k, x)
    y, vs_new = a.apply(
        vs,
        x,
        mutable=[
            'muts',
        ],
    )
    np.testing.assert_array_equal(
        vs_new['muts']['b']['c']['v'], jnp.array([1.0], jnp.float32)
    )
    np.testing.assert_array_equal(
        vs_new['muts']['b']['outer_c']['v'], jnp.array([1.0], jnp.float32)
    )

  def test_custom_vjp(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        def f(mdl, x):
          return mdl(x)

        def fwd(mdl, x):
          return nn.vjp(f, mdl, x)

        def bwd(vjp_fn, y_t):
          params_t, input_t = vjp_fn(y_t)
          params_t = jax.tree_util.tree_map(jnp.sign, params_t)
          return params_t, input_t

        sign_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return sign_grad(nn.Dense(1), x).reshape(())

    x = jnp.ones((2,))
    variables = Foo().init(random.key(0), x)
    grad = jax.grad(Foo().apply)(variables, x)
    for grad_leaf in jax.tree_util.tree_leaves(grad):
      self.assertTrue(jnp.all(jnp.abs(grad_leaf) == 1.0))

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

    k = random.key(0)
    x = jnp.ones((2,))
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
    self.assertTrue(
        tree_equals(
            jax.tree_util.tree_map(jnp.shape, vs_foo),
            jax.tree_util.tree_map(jnp.shape, vs_bar),
        )
    )

  def test_transform_methods_on_submodules_still_reserve_names(self):
    class Foo(nn.Module):

      @nn.jit
      def helper(self, x, m):
        conflicting_a = nn.Dense(2, name='a')
        return m(x)

      @nn.jit
      @nn.compact
      def __call__(self, x):
        a = nn.Dense(2, name='a')
        return self.helper(x, a)

    k = random.key(0)
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

    k = random.key(0)
    x = jnp.array([1.0])

    with self.assertRaises(errors.NameInUseError):
      y = Test().init(k, x)

  def test_transform_with_setup_and_methods_on_submodule_pytrees(self):
    class Foo(nn.Module):

      def setup(self):
        self.inners = [nn.Dense(2), nn.Dense(2)]

      def helper(self, x, ms):
        return ms[0](x) + ms[1](x)

      @nn.fold_rngs
      def __call__(self, x):
        return self.helper(x, self.inners)

    class JitFoo(nn.Module):

      def setup(self):
        self.inners = [nn.Dense(2), nn.Dense(2)]

      def helper(self, x, ms):
        return ms[0](x) + ms[1](x)

      @nn.jit
      def __call__(self, x):
        return self.helper(x, self.inners)

    k = random.key(0)
    x = jnp.ones((2,))

    vs_0 = Foo().init(k, x)
    vs_1 = JitFoo().init(k, x)

    self.assertTrue(tree_allclose(vs_0, vs_1))

  def test_transform_setup_still_reserve_names_pytrees(self):
    class Identity(nn.Module):

      @nn.compact
      def __call__(self, x):
        return x

    class Test(nn.Module):

      def setup(self):
        self.subs = [Identity(), Identity()]
        self.subs = [Identity(), Identity()]

      @nn.jit
      def __call__(self, x):
        return x

    k = random.key(0)
    x = jnp.array([1.0])

    msg = r'Could not create submodule "subs_0".*'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      y = Test().init(k, x)

  def test_scan_of_setup_parameter(self):
    class Body(nn.Module):

      def setup(self):
        self.dense = nn.Dense(1)
        self.p = self.param('p', lambda k: jnp.ones((1,)))

      def __call__(self, x):
        return self.dense(x) + self.p, None

    scanbody = nn.scan(
        Body, variable_axes={'params': 0}, split_rngs={'params': True}, length=2
    )
    k = random.key(0)
    x = jnp.ones((1,))
    vs = scanbody().init(k, x)
    y = scanbody().apply(vs, x)

  def test_multi_method_class_transform(self):
    class Foo(nn.Module):

      def setup(self):
        self.dense0 = nn.Dense(2)
        self.dense1 = nn.Dense(2)

      def method_0(self, x):
        return self.dense0(x), x

      def method_1(self, x, y):
        return self.dense1(x) + y, None

    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        ScanFoo = nn.scan(
            Foo,
            methods={
                'method_0': dict(
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    in_axes=nn.broadcast,
                    out_axes=0,
                    length=3,
                ),
                'method_1': dict(
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    in_axes=0,
                    length=3,
                ),
            },
        )
        sf = ScanFoo()
        y, ys = sf.method_0(x)
        z, _ = sf.method_1(y, ys)
        return z

    k = random.key(0)
    x = random.uniform(random.key(1), (2, 2))
    vs = Bar().init(k, x)
    y = Bar().apply(vs, x)

  def test_compact_aliasing_collision(self):
    class Foo(nn.Module):
      m1: nn.Module
      m2: nn.Module

      @nn.compact
      def __call__(self, x):
        x = self.m2(self.m1(x))
        return x

    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        dense = nn.Dense(2)
        x = nn.jit(Foo)(dense, dense)(x)
        return x

    k = random.key(0)
    x = jnp.zeros((2, 2))
    _ = Bar().init(k, x)

  def test_compact_aliasing_collision_arg_and_attrib(self):
    class Foo(nn.Module):
      m1: nn.Module

      @nn.compact
      def __call__(self, x, m2):
        x = m2(self.m1(x))
        return x

    class Bar(nn.Module):

      @nn.compact
      def __call__(self, x):
        dense = nn.Dense(2)
        x = nn.jit(Foo)(dense)(x, dense)
        return x

    k = random.key(0)
    x = jnp.zeros((2, 2))
    _ = Bar().init(k, x)

  def test_jit_with_setup_helpers(self):
    class Foo(nn.Module):

      def setup(self):
        self.a = nn.Dense(2)
        self.setup_helper()

      def setup_helper(self):
        self.b = nn.Dense(2)

      @nn.fold_rngs
      def __call__(self, x):
        return self.b(self.a(x))

    class JitFoo(nn.Module):

      def setup(self):
        self.a = nn.Dense(2)
        self.setup_helper()

      def setup_helper(self):
        self.b = nn.Dense(2)

      @nn.jit
      def __call__(self, x):
        return self.b(self.a(x))

    k = random.key(0)
    x = jnp.ones((2, 2))
    vs = JitFoo().init(k, x)
    y0 = JitFoo().apply(vs, x)
    vs = Foo().init(k, x)
    y1 = Foo().apply(vs, x)
    np.testing.assert_array_equal(y0, y1)

  def test_jit_kwargs(self):
    class Foo(nn.Module):

      @nn.jit
      def __call__(self, a: jax.Array, b: jax.Array):
        return a + b

    m = Foo()
    y = m.apply({}, jnp.array(1.0), b=jnp.array(2.0))

    np.testing.assert_array_equal(y, jnp.array(3.0))

  def test_jit_static_argnames(self):
    s = None

    class Foo(nn.Module):

      @partial(nn.jit, static_argnames=['b'])
      def __call__(self, a: jax.Array, b: str):
        nonlocal s
        s = b
        return a

    m = Foo()
    y = m.apply({}, jnp.array(1.0), b='hi')

    self.assertEqual(s, 'hi')
    np.testing.assert_array_equal(y, jnp.array(1.0))

  def test_jit_and_sow(self):
    class Inner(nn.Module):

      @nn.compact
      def __call__(self, x):
        self.sow('intermediates', 'loss', jnp.sum(x))
        return x + 1

    class Outer(nn.Module):

      def setup(self):
        self.inner = Inner()

      @nn.jit
      def __call__(self, x):
        return self.inner(x)

    m = Outer()
    x = jnp.ones((2, 2))
    vs = m.init(random.key(0), x)
    y, updates = m.apply(vs, x, mutable=['intermediates'])
    np.testing.assert_array_equal(
        updates['intermediates']['inner']['loss'], 4.0
    )
    np.testing.assert_array_equal(y, 2)

  def test_fold_rngs(self):
    class Foo(nn.Module):

      def __call__(self, use_jit: bool):
        def f(foo: Foo):
          return foo.make_rng('params')

        if use_jit:
          key = nn.jit(f)(self)
        else:
          key = nn.fold_rngs(f)(self)

        return key

    foo = Foo()
    key_jit = foo.apply({}, True, rngs={'params': random.key(0)})
    key_fold_rngs = foo.apply({}, False, rngs={'params': random.key(0)})

    self.assert_keys_equal(key_jit, key_fold_rngs)

  def test_same_key(self):

    class Block(nn.Module):

      @nn.jit
      @nn.compact
      def __call__(self, carry, inputs):
        # dump_rng_info(self)
        key = self.make_rng('params')
        # y = jax.random.uniform(self.make_rng('params'), (2,))
        return carry, key

    class Transformer(nn.Module):

      @nn.compact
      def __call__(self):
        num_blocks = 10
        carry, key = nn.scan(
            Block,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            # length=num_blocks,
        )()(None, jnp.arange(num_blocks))
        return key

    model = Transformer()
    keys1, _ = model.init_with_output(jax.random.key(1))
    keys2, _ = model.init_with_output(jax.random.key(1))
    keys3, _ = model.init_with_output(jax.random.key(1))
    keys4, _ = model.init_with_output(jax.random.key(1))

    self.assert_keys_equal(keys1, keys2)
    self.assert_keys_equal(keys2, keys3)
    self.assert_keys_equal(keys2, keys3)

  def test_jit_repr_hash(self):
    n = 0

    @partial(jax.jit, static_argnums=0)
    def f(obj):
      nonlocal n
      n += 1
      return None

    f(_HashableProxy.from_module(nn.Dense(10)))
    self.assertEqual(n, 1)
    f(_HashableProxy.from_module(nn.Dense(10)))
    self.assertEqual(n, 1)

    f(_HashableProxy.from_module(nn.Dense(20)))
    self.assertEqual(n, 2)
    f(_HashableProxy.from_module(nn.Dense(20)))
    self.assertEqual(n, 2)

  def test_jit_reuse(self):
    n = 0

    class Foo(nn.Module):

      @nn.jit
      def __call__(self, x):
        nonlocal n
        n += 1
        return x

    x = jnp.array(1.0)
    m = Foo()

    self.assertEqual(n, 0)

    y = m.apply({}, x)
    self.assertEqual(n, 1)
    y = m.apply({}, x)
    self.assertEqual(n, 1)

  def test_jit_recursive(self):
    n = 0

    class Foo(nn.Module):

      @partial(nn.jit, static_argnames='recurse_once')
      def __call__(self, x, *, recurse_once: bool = True):
        nonlocal n
        n += 1
        if recurse_once:
          x = self(x, recurse_once=False)
        return x + 1

    x = jnp.array(1.0)
    m = Foo()

    self.assertEqual(n, 0)

    y = m.apply({}, x)
    self.assertEqual(n, 2)
    y = m.apply({}, x)
    self.assertEqual(n, 2)

  @parameterized.named_parameters(('class', True), ('method', False))
  def test_jit_reuse_hash(self, jit_class: bool):
    n = 0

    class Foo(nn.Module):
      key: int

      def __call__(self, x):
        nonlocal n
        n += 1
        return x

    if jit_class:
      Foo = nn.jit(Foo)
    else:  # jit method
      Foo.__call__ = nn.jit(Foo.__call__)

    x = jnp.array(1.0)
    self.assertEqual(n, 0)

    y = Foo(1).apply({}, x)
    self.assertEqual(n, 1)
    y = Foo(1).apply({}, x)
    self.assertEqual(n, 1)
    y = Foo(2).apply({}, x)
    self.assertEqual(n, 2)
    y = Foo(2).apply({}, x)
    self.assertEqual(n, 2)

  @parameterized.named_parameters(('class', True), ('method', False))
  def test_jit_reuse_submodules(self, jit_class: bool):
    test = self
    n = 0
    key = None
    name = None

    class Foo(nn.Module):
      key: int

      def __call__(self, x):
        nonlocal n, key, name
        n += 1
        key = self.key
        name = self.name
        return x

    if jit_class:
      Foo = nn.jit(Foo)
    else:  # jit method
      Foo.__call__ = nn.jit(Foo.__call__)

    class Parent(nn.Module):

      @nn.compact
      def __call__(self, x):

        for i in range(3):
          m = Foo(i)
          y = m(x)
          test.assertEqual(key, i)
          test.assertEndsWith(name, f'Foo_{i}')
          test.assertEqual(n, i + 1)

    x = jnp.array(1.0)
    self.assertEqual(n, 0)

    y = Parent().apply({}, x)

  @parameterized.named_parameters(('class', True), ('method', False))
  def test_jit_stateful_submodules(self, jit_class: bool):
    n = 0

    class Foo(nn.Module):
      key: int

      @nn.compact
      def __call__(self, x):
        nonlocal n
        n += 1
        count = self.variable('counts', 'count', lambda: 0)
        if not self.is_initializing():
          count.value += 1
        return x

    if jit_class:
      Foo = nn.jit(Foo)
    else:  # jit method
      Foo.__call__ = nn.jit(Foo.__call__)

    class Parent(nn.Module):
      @nn.compact
      def __call__(self, x):
        for _ in range(3):
          m = Foo(0)
          x = m(x)
        return x

    m = Parent()
    x = jnp.array(1.0)
    counts = m.init({}, x)['counts']
    self.assertEqual(n, 1)

    y, updates = m.apply({'counts': counts}, x, mutable=['counts'])
    counts = updates['counts']
    self.assertEqual(n, 2)

    for count in jax.tree.leaves(counts):
      self.assertEqual(count, 1)

    y, updates = m.apply({'counts': counts}, x, mutable=['counts'])
    counts = updates['counts']
    self.assertEqual(n, 2)

    for count in jax.tree.leaves(counts):
      self.assertEqual(count, 2)

  @parameterized.named_parameters(('class', True), ('method', False))
  def test_jit_reuse_nested_submodules(self, jit_class: bool):
    test = self
    n = 0

    class Foo(nn.Module):
      key: int

      def __call__(self, x):
        return x

    class Parent(nn.Module):
      submodules: list[Foo]

      @nn.compact
      def __call__(self, x):
        nonlocal n
        n += 1
        for i, m in enumerate(self.submodules):
          x = m(x)

        return x

    if jit_class:
      Parent = nn.jit(Parent)
    else:  # jit method
      Parent.__call__ = nn.jit(Parent.__call__)

    x = jnp.array(1.0)
    self.assertEqual(n, 0)

    y = Parent([Foo(1), Foo(2)]).apply({}, x)
    self.assertEqual(n, 1)
    y = Parent([Foo(1), Foo(2)]).apply({}, x)
    self.assertEqual(n, 1)
    y = Parent([Foo(3), Foo(4)]).apply({}, x)
    self.assertEqual(n, 2)
    y = Parent([Foo(3), Foo(4)]).apply({}, x)
    self.assertEqual(n, 2)

  def test_jit_hashes_serializable_types(self):
    class Node:

      def __init__(self, a: int):
        self.a = a

      def __hash__(self):
        # test object is not being passed as static
        raise Exception('this should not be called')

      def __eq__(self, __value, /):
        raise Exception('this should not be called')

    def to_dict(node: Node):
      return {'a': node.a}

    def from_dict(node: Node, d: dict[str, Any]):
      node.a = d['a']
      return node

    serialization.register_serialization_state(Node, to_dict, from_dict)

    try:
      n = 0

      class Foo(nn.Module):
        node: Node

        @nn.jit
        @nn.compact
        def __call__(self, x):
          nonlocal n
          n += 1
          return self.node.a + nn.Dense(2)(x)

      m = Foo(Node(1))
      m.init_with_output(random.key(0), jnp.ones((2, 2)))
      self.assertEqual(n, 1)
      m.init_with_output(random.key(0), jnp.ones((2, 2)))
      self.assertEqual(n, 1)
    finally:
      del serialization._STATE_DICT_REGISTRY[Node]

  def test_while_loop(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        key_zero = random.key(0)
        key_zero = jnp.broadcast_to(key_zero, (2, *key_zero.shape))
        self.param('inc', lambda _: 1)
        self.put_variable('state', 'acc', 0)
        self.put_variable('state', 'rng_params', key_zero)
        self.put_variable('state', 'rng_loop', key_zero)

        def cond_fn(mdl, c):
          acc = mdl.get_variable('state', 'acc')
          return acc < x

        def body_fn(mdl, c):
          i = mdl.get_variable('state', 'acc')
          p_rng = mdl.make_rng('params')
          l_rng = mdl.make_rng('loop')
          mdl.put_variable(
              'state',
              'rng_params',
              mdl.get_variable('state', 'rng_params').at[i].set(p_rng),
          )
          mdl.put_variable(
              'state',
              'rng_loop',
              mdl.get_variable('state', 'rng_loop').at[i].set(l_rng),
          )
          inc = mdl.get_variable('params', 'inc')
          mdl.put_variable('state', 'acc', i + inc)
          return c

        return nn.while_loop(
            cond_fn,
            body_fn,
            self,
            (),
            carry_variables='state',
            split_rngs={'params': False, 'loop': True},
        )

    x = 2
    mdl = Foo()
    _, vars = mdl.apply(
        {},
        x,
        mutable=True,
        rngs={'params': random.key(1), 'loop': random.key(2)},
    )
    self.assertEqual(vars['state']['acc'], x)
    self.assertTrue(
      jnp.equal(vars['state']['rng_params'][0], vars['state']['rng_params'][1])
    )
    with jax.debug_key_reuse(False):
      self.assertFalse(
        jnp.equal(
          vars['state']['rng_loop'][0],
          vars['state']['rng_loop'][1],
        )
      )

  def test_cond(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, pred):
        self.variable('state', 'true_count', lambda: 0)
        self.variable('state', 'false_count', lambda: 0)

        def true_fn(mdl, x):
          mdl.variable('state', 'true_count').value += 1
          return nn.Dense(2, name='dense')(x)

        def false_fn(mdl, x):
          mdl.variable('state', 'false_count').value += 1
          return -nn.Dense(2, name='dense')(x)

        return nn.cond(pred, true_fn, false_fn, self, x)

  def test_switch(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, pred):
        self.variable('state', 'a_count', lambda: 0)
        self.variable('state', 'b_count', lambda: 0)
        self.variable('state', 'c_count', lambda: 0)

        def a_fn(mdl, x):
          mdl.variable('state', 'a_count').value += 1
          return nn.Dense(2, name='dense')(x)

        def b_fn(mdl, x):
          mdl.variable('state', 'b_count').value += 1
          return -nn.Dense(2, name='dense')(x)

        def c_fn(mdl, x):
          mdl.variable('state', 'c_count').value += 1
          return nn.Dense(2, name='dense')(x)

        return nn.switch(pred, [a_fn, b_fn, c_fn], self, x)

    x = jnp.ones((1, 3))
    foo = Foo()
    y1, vars = foo.init_with_output(random.key(0), x, 0)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 0, 'c_count': 0})
    y2, updates = foo.apply(vars, x, 1, mutable='state')
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 1, 'c_count': 0})
    np.testing.assert_allclose(y1, -y2)
    y3, updates = foo.apply(vars, x, 2, mutable='state')
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 1, 'c_count': 1})
    np.testing.assert_allclose(y1, y3)

  def test_switch_multihead(self):
    class Foo(nn.Module):

      def setup(self) -> None:
        self.heads = [
            nn.Sequential([nn.Dense(10), nn.Dense(7), nn.Dense(5)]),
            nn.Sequential([nn.Dense(11), nn.Dense(5)]),
            nn.Dense(5),
        ]

      @nn.compact
      def __call__(self, x, index):
        def head_fn(i):
          def fn(mdl, x):
            mdl.variable('state', f'{i}_count', lambda: -1).value += 1
            return mdl.heads[i](x)

          return fn

        branches = [head_fn(i) for i in range(len(self.heads))]

        if self.is_mutable_collection('params'):
          for branch in branches:
            _ = branch(self, x)

        return nn.switch(index, branches, self, x)

    x = jnp.ones((1, 3))
    foo = Foo()
    y1, vars = foo.init_with_output(random.key(0), x, 0)
    self.assertEqual(vars['state'], {'0_count': 1, '1_count': 0, '2_count': 0})
    y2, updates = foo.apply(vars, x, 1, mutable='state')
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'0_count': 1, '1_count': 1, '2_count': 0})
    y3, updates = foo.apply(vars, x, 2, mutable='state')
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'0_count': 1, '1_count': 1, '2_count': 1})

    self.assertEqual(
        vars['params']['heads_0']['layers_0']['kernel'].shape, (3, 10)
    )
    self.assertEqual(vars['params']['heads_0']['layers_0']['bias'].shape, (10,))
    self.assertEqual(
        vars['params']['heads_0']['layers_1']['kernel'].shape, (10, 7)
    )
    self.assertEqual(vars['params']['heads_0']['layers_1']['bias'].shape, (7,))
    self.assertEqual(
        vars['params']['heads_0']['layers_2']['kernel'].shape, (7, 5)
    )
    self.assertEqual(vars['params']['heads_0']['layers_2']['bias'].shape, (5,))

    self.assertEqual(
        vars['params']['heads_1']['layers_0']['kernel'].shape, (3, 11)
    )
    self.assertEqual(vars['params']['heads_1']['layers_0']['bias'].shape, (11,))
    self.assertEqual(
        vars['params']['heads_1']['layers_1']['kernel'].shape, (11, 5)
    )
    self.assertEqual(vars['params']['heads_1']['layers_1']['bias'].shape, (5,))

    self.assertEqual(vars['params']['heads_2']['kernel'].shape, (3, 5))
    self.assertEqual(vars['params']['heads_2']['bias'].shape, (5,))

  def test_lift_instance_error(self):
    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.checkpoint(nn.Dense(2))(x)

    with self.assertRaises(errors.TransformTargetError):
      Foo().init(random.key(0), jnp.zeros((2, 3)))

  def test_scan_compact_count(self):
    class Foo(nn.Module):
      num_layers: int = 5

      @nn.compact
      def __call__(self, x):
        def body_fn(mdl, x):
          return nn.Dense(features=x.shape[-1])(x), ()

        x, _ = nn.scan(
            body_fn,
            length=self.num_layers,
            variable_axes={'params': 0},
            split_rngs={'params': True},
        )(self, x)
        return x

    m = Foo()
    x = jnp.ones((3,))
    v = m.init(jax.random.key(0), x)
    self.assertEqual(v['params']['Dense_0']['kernel'].shape, (5, 3, 3))
    m.apply(v, x)

  def test_bound_methods_in_direct_transforms(self):
    class CondModel(nn.Module):

      def setup(self):
        self.dense = nn.Dense(3)

      def f1(self, arr):
        arr = self.dense(arr)
        return arr

      def f2(self, arr):
        _ = self.dense(arr)
        return arr

      def __call__(self, x):
        return nn.cond(x.sum() > 0, self.f1, self.f2, self, x)

    cond_model = CondModel()

    output, init_params = jax.jit(cond_model.init_with_output)(
        jax.random.key(0), x=jnp.ones(3)
    )

  def test_add_metadata_axis(self):
    vars_copy = None

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        nonlocal vars_copy
        kernel_init = nn.with_partitioning(
            nn.initializers.lecun_normal(), ('foo', 'bar')
        )
        vars_copy = self.variables
        return nn.Dense(
            4, kernel_init=kernel_init, use_bias=False, name='dense'
        )(x)

    class Test(nn.Module):

      @partial(
          nn.add_metadata_axis,
          variable_axes={'params': 0},
          metadata_params={nn.PARTITION_NAME: 'baz'},
      )
      @nn.compact
      def __call__(self, x):
        return Foo(name='foo')(x)

    k = random.key(0)
    x = jnp.ones((4, 4), dtype=jnp.float32)
    vs = Test().init(k, x)
    y = Test().apply(vs, x)
    outer_expect = jax.tree_util.tree_map(
        jnp.shape,
        freeze({
            'params': {
                'foo': {
                    'dense': {
                        'kernel': nn.Partitioned(
                            jnp.ones((4, 4)), names=('baz', 'foo', 'bar')
                        )
                    }
                }
            }
        }),
    )
    inner_expect = jax.tree_util.tree_map(
        jnp.shape,
        freeze({
            'params': {
                'dense': {
                    'kernel': nn.Partitioned(
                        jnp.ones((4, 4)), names=('foo', 'bar')
                    )
                }
            }
        }),
    )
    self.assertEqual(jax.tree_util.tree_map(jnp.shape, vs), outer_expect)
    self.assertEqual(jax.tree_util.tree_map(jnp.shape, vars_copy), inner_expect)

  def test_outer_setup_called_with_sharing_across_transforms(self):
    class A(nn.Module):

      def setup(self):
        self.foo = self.param('foo', nn.initializers.zeros, (2, 2), jnp.float32)

      def __call__(self, x):
        return self.foo

    class B(nn.Module):
      a: Any

      @nn.compact
      def __call__(self, x):
        return self.a(x)

    class C(nn.Module):

      def setup(self):
        self.a = A()
        self.b = nn.jit(B)(self.a)

      def __call__(self, x):
        b = self.b(x)
        a = self.a(x)
        return a + b

    k = random.key(0)
    x = random.randint(k, (2, 2), minval=0, maxval=10)
    vs = C().init(k, x)
    y = C().apply(vs, x)
    outer_expect = jax.tree_util.tree_map(
        jnp.shape, freeze({'params': {'a': {'foo': jnp.zeros((2, 2))}}})
    )
    self.assertEqual(jax.tree_util.tree_map(jnp.shape, vs), outer_expect)

  def test_grad_simple(self):
    class LearnScale(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        p = self.param('scale', nn.initializers.ones_init(), ())
        return jnp.sum(p * x * y)

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        x_grad, y_grad = nn.grad(
            lambda mdl, x, y: mdl(x, y), LearnScale(), x, y
        )
        return x_grad, y_grad

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Foo().init(random.key(0), x, y)

    x_grad, y_grad = Foo().apply(vs, x, y)
    self.assertTrue(tree_allclose(x_grad, y))
    self.assertTrue(tree_allclose(y_grad, x))

  def test_grad_simple_with_aux(self):
    class LearnScale(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        p = self.param('scale', nn.initializers.ones_init(), ())
        return jnp.sum(p * x * y), p

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        (x_grad, y_grad), aux = nn.grad(
            lambda mdl, x, y: mdl(x, y), LearnScale(), x, y, has_aux=True
        )
        return aux, x_grad, y_grad

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Foo().init(random.key(0), x, y)

    aux, x_grad, y_grad = Foo().apply(vs, x, y)
    self.assertTrue(tree_allclose(x_grad, y))
    self.assertTrue(tree_allclose(y_grad, x))
    self.assertTrue(tree_allclose(aux, vs['params']['LearnScale_0']['scale']))

  def test_value_and_grad_simple(self):
    class LearnScale(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        p = self.param('scale', nn.initializers.ones_init(), ())
        return jnp.sum(p * x * y)

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        z, (x_grad, y_grad) = nn.value_and_grad(
            lambda mdl, x, y: mdl(x, y), LearnScale(), x, y
        )
        return z, x_grad, y_grad

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Foo().init(random.key(0), x, y)

    z, x_grad, y_grad = Foo().apply(vs, x, y)
    self.assertTrue(tree_allclose(x_grad, y))
    self.assertTrue(tree_allclose(y_grad, x))

  def test_value_and_grad_simple_with_aux(self):
    class LearnScale(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        p = self.param('scale', nn.initializers.ones_init(), ())
        return jnp.sum(p * x * y), p

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        (z, aux), (x_grad, y_grad) = nn.value_and_grad(
            lambda mdl, x, y: mdl(x, y), LearnScale(), x, y, has_aux=True
        )
        return z, aux, x_grad, y_grad

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Foo().init(random.key(0), x, y)

    z, aux, x_grad, y_grad = Foo().apply(vs, x, y)
    self.assertTrue(tree_allclose(x_grad, y))
    self.assertTrue(tree_allclose(y_grad, x))
    self.assertTrue(tree_allclose(aux, vs['params']['LearnScale_0']['scale']))

  def test_value_and_grad_multiscope(self):
    class Foo(nn.Module):
      bar: nn.Module

      @nn.compact
      def __call__(self, x, y):
        def fn(self, x, y):
          qup = nn.Dense(y.shape[-1])
          delta = y - self.bar(qup(x))
          return jnp.sum(delta**2)

        z, (x_grad, y_grad) = nn.value_and_grad(fn, self, x, y)
        return z, x_grad, y_grad

    class Baz(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        bar = nn.Dense(y.shape[-1])
        return Foo(bar=bar)(x, y)

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Baz().init(random.key(0), x, y)
    z, x_grad, y_grad = Baz().apply(vs, x, y)

    def comparison_fn(x, y):
      w1 = vs['params']['Foo_0']['Dense_0']['kernel']
      w2 = vs['params']['Dense_0']['kernel']
      delta = y - jnp.dot(jnp.dot(x, w1), w2)
      return jnp.sum(delta**2)

    self.assertTrue(tree_allclose(comparison_fn(x, y), z))
    self.assertTrue(tree_allclose(jax.grad(comparison_fn, 0)(x, y), x_grad))
    self.assertTrue(tree_allclose(jax.grad(comparison_fn, 1)(x, y), y_grad))

  def test_value_and_grad_multiscope_adopted(self):
    class Foo(nn.Module):
      bar: nn.Module
      qup: nn.Module

      @nn.compact
      def __call__(self, x, y):
        def fn(self, x, y):
          delta = y - self.bar(self.qup(x))
          return jnp.sum(delta**2)

        z, (x_grad, y_grad) = nn.value_and_grad(fn, self, x, y)
        return z, x_grad, y_grad

    x = random.uniform(random.key(1), (4,))
    y = random.uniform(random.key(2), (4,))
    vs = Foo(bar=nn.Dense(4), qup=nn.Dense(4)).init(random.key(0), x, y)
    z, x_grad, y_grad = Foo(bar=nn.Dense(4), qup=nn.Dense(4)).apply(vs, x, y)

    def comparison_fn(x, y):
      w1 = vs['params']['qup']['kernel']
      w2 = vs['params']['bar']['kernel']
      delta = y - jnp.dot(jnp.dot(x, w1), w2)
      return jnp.sum(delta**2)

    self.assertTrue(tree_allclose(comparison_fn(x, y), z))
    self.assertTrue(tree_allclose(jax.grad(comparison_fn, 0)(x, y), x_grad))
    self.assertTrue(tree_allclose(jax.grad(comparison_fn, 1)(x, y), y_grad))

  def test_vmap_add_remove_axis_transforms(self):
    class BoxedData(struct.PyTreeNode, AxisMetadata):
      value: Any
      def unbox(self):
        return self.value
      def replace_boxed(self, val):
        return self.replace(value=val)
      def add_axis(self, index: int, params: dict[Any, Any]):
        value = jnp.mean(self.value, axis=index)
        return self.replace(value=value)
      def remove_axis(self, index: int, params: dict[Any, Any]):
        value_shape = list(self.value.shape)
        value_shape.insert(index, params['axis_size'])
        value = jnp.broadcast_to(self.value, value_shape)
        return self.replace(value=value)

    class Top(nn.Module):
      @nn.compact
      def __call__(self, x):
        VFoo = nn.vmap(
            Foo, in_axes=0, out_axes=0, variable_axes={'params':0, 'aux': 0},
            metadata_params={'axis_size': x.shape[0]},
        )
        vfoo = VFoo(name="vfoo")
        y = vfoo(x)
        y = vfoo(x)
        assert vfoo.variables['aux']['v'].value.shape == ()
        return y

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        if self.has_variable('aux', 'v'):
          assert self.variables['aux']['v'].value.shape == ()
        boxed_v = self.variable('aux', 'v', lambda: BoxedData(jnp.ones(())))
        assert self.variables['aux']['v'].value.shape == ()
        return x

    vs = Top().init(random.key(0), jnp.ones((2,5)))
    y = Top().apply(vs, jnp.ones((2, 5)))
    assert vs['aux']['vfoo']['v'].value.shape == ()

  def test_vjp_tracer_leak(self):
    class LearnScale(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('scale', nn.initializers.zeros, ())
        return p * x
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        y, bwd = nn.vjp(lambda mdl, x: mdl(x), LearnScale(), x)
        params_grad, x_grad = bwd(jnp.ones(y.shape))
        return y, params_grad, x_grad
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3))
    foo = Foo()
    with jax.check_tracer_leaks():
      params = foo.init(key, x)
      foo.apply(params, x)

  @parameterized.named_parameters(
      ('retracing scan', True), ('simple scan', False)
  )
  def test_jit_scan_retracing(self, retracing_scan: bool):
    num_blocks = 4
    num_patterns = 4
    features = 4
    trace_counts = [0, 0]

    class Block(nn.Module):
      def setup(self):
        self.dense = nn.Dense(features, use_bias=False)
      @nn.jit
      def __call__(self, x):
        nonlocal trace_counts
        trace_counts[1] += 1
        return self.dense(x)

    class BlockSequence(nn.Module):
      def setup(self):
        self.blocks = [Block() for _ in range(num_blocks)]
      @nn.jit
      def __call__(self, carry, inputs):
        nonlocal trace_counts
        trace_counts[0] += 1
        for block in self.blocks:
          carry = block(carry)
        return carry, inputs

    class Transformer(nn.Module):
      retracing_scan: bool = True
      def setup(self):
        self.scan = nn.scan(
            BlockSequence,
            variable_axes={'params': 0},
            split_rngs={'params': False},
            length=num_patterns,
            check_constancy_invariants=retracing_scan,
        )()
      def __call__(self, inputs):
        return self.scan(jnp.zeros_like(inputs), inputs)

    model = Transformer(retracing_scan=retracing_scan)
    _ = model.init(random.key(0), jnp.ones((num_patterns, features,)))
    self.assertEqual(trace_counts[0], 2 if retracing_scan else 1)
    self.assertEqual(trace_counts[1], 2 if retracing_scan else 1)


if __name__ == '__main__':
  absltest.main()

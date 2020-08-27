# Copyright 2020 The Flax Authors.
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

from absl.testing import absltest
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


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
                     variable_in_axes={'param': None},
                     variable_out_axes={'param': None},
                     split_rngs={'param': False})
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
                     variable_in_axes={'param': None},
                     variable_out_axes={'param': None},
                     split_rngs={'param': False})
    normal_model = decorated_MLP()(features=[3, 4, 5])
    vmap_model = decorated_MLP(vmap)(features=[3, 4, 5])
    init_variables = normal_model.init(key2, x)
    # simulate vmap in python for comparison:
    y1 = jnp.vstack([normal_model.apply(init_variables, x2[i])[None, ...]
                     for i in np.arange(x2.shape[0])])
    y2 = vmap_model.apply(init_variables, x2)
    np.testing.assert_allclose(y1, y2, atol=1e-7)

  def test_scan(self):
    class SimpleScan(nn.Module):
      @nn.compact
      def __call__(self, c, xs):
        LSTM = nn.scan(nn.LSTMCell,
                       variable_in_axes={'param': nn.broadcast},
                       variable_out_axes={'param': nn.broadcast},
                       split_rngs={'param': False})
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
    lstmcell_variables = freeze({'param': init_variables['param']['lstm_cell']})
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
               variable_in_axes={'param': nn.broadcast},
               variable_out_axes={'param': nn.broadcast},
               split_rngs={'param': False})
      @nn.compact
      def __call__(self, c, xs):
        return nn.LSTMCell(name="lstm_cell")(c, xs)

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
    lstmcell_variables = freeze({'param': init_variables['param']['lstm_cell']})
    for i in range(xs.shape[0]):
      c, y = nn.LSTMCell().apply(lstmcell_variables, c, xs[i])
      ys.append(y[None, ...])
    y1 = jnp.vstack(ys)

    c2, y2 = model.apply(init_variables, init_carry, xs)
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
               variable_in_axes={'param': 0},
               variable_out_axes={'param': 0},
               split_rngs={'param': True})
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
        init_vars['param']['outer']['Dense_0']['kernel'].shape,
        (3, 2, 5))
    self.assertEqual(
        init_vars['param']['outer']['Dense_0']['bias'].shape,
        (3, 5))
    self.assertEqual(y.shape, (3, 1, 5))


if __name__ == '__main__':
  absltest.main()

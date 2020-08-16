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

"""Tests for flax.linen."""

from absl.testing import absltest

import jax
from jax import random
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp
from typing import Any, Tuple, Iterable, Callable

from flax import linen as nn
from flax.linen import compact
from flax.core import Scope

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class DummyModule(nn.Module):
  @compact
  def __call__(self, x):
    bias = self.param('bias', initializers.ones, x.shape)
    return x + bias

class Dense(nn.Module):
  features: int
  @compact
  def __call__(self, x):
    kernel = self.param('kernel',
                        initializers.lecun_normal(),
                        (x.shape[-1], self.features))
    y = jnp.dot(x, kernel)
    return y


class ModuleTest(absltest.TestCase):

  def test_init_module(self):
    rngkey = jax.random.PRNGKey(0)
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    y = DummyModule(parent=scope)(x)
    params = scope.variables()['param']
    y2 = DummyModule(parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    onp.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_arg_module(self):
    rngkey = jax.random.PRNGKey(0)
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = Dense(3, parent=scope)(x)
    params = scope.variables()['param']
    y2 = Dense(3, parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    self.assertEqual(params['kernel'].shape, (10, 3))

  def test_util_fun(self):
    rngkey = jax.random.PRNGKey(0)
    class MLP(nn.Module):
      @compact
      def __call__(self, x):
        x = self._mydense(x)
        x = self._mydense(x)
        return x
      def _mydense(self, x):
        return Dense(3)(x)
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = MLP(parent=scope)(x)
    params = scope.variables()['param']
    y2 = MLP(parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'Dense_0': {'kernel': (10, 3)},
       'Dense_1': {'kernel': (3, 3)}})

  def test_nested_module_reuse(self):
    rngkey = jax.random.PRNGKey(0)
    class MLP(nn.Module):
      @compact
      def __call__(self, x):
        x = self._mydense(x)
        x = self._mydense(x)
        return x
      def _mydense(self, x):
        return Dense(3)(x)
    class Top(nn.Module):
      @compact
      def __call__(self, x):
        mlp = MLP()
        y = mlp(x)
        z = mlp(x)
        return y + z
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = Top(parent=scope)(x)
    params = scope.variables()['param']
    y2 = Top(parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'MLP_0':
        {'Dense_0': {'kernel': (10, 3)},
        'Dense_1': {'kernel': (3, 3)}}})

  def test_setup_dict_assignment(self):
    rngkey = jax.random.PRNGKey(0)
    class MLP(nn.Module):
      def setup(self):
        self.lyrs1 = {'a': Dense(3), 'b': Dense(3),}
        self.lyrs2 = [Dense(3), Dense(3)]
      def __call__(self, x):
        y = self.lyrs1['a'](x)
        z = self.lyrs1['b'](y)
        #w = self.lyrs2[0](x)
        return z
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = MLP(parent=scope)(x)
    params = scope.variables()['param']
    y2 = MLP(parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'lyrs1_a': {'kernel': (10, 3)},
      'lyrs1_b': {'kernel': (3, 3)}})

  def test_setup_cloning(self):
    class MLP(nn.Module):
      def setup(self):
        self.dense = Dense(3)
    scope = Scope({})
    MLPclone = MLP(parent=scope).clone()

  def test_submodule_attr(self):
    rngkey = jax.random.PRNGKey(0)
    class Inner(nn.Module):
      @compact
      def __call__(self):
        self.param('x', lambda rng: 40)

    class Outer(nn.Module):
      inner: nn.Module

      def __call__(self):
        return self.inner()

    class Wrapper(nn.Module):
      def setup(self):
        self.inner = Inner()
        self.outer = Outer(self.inner)

      def __call__(self):
        return self.outer()

    scope = Scope({'param': {}}, rngs={'param': rngkey})
    # Make sure this doesn't raise "Can't attach to remote parent"
    wrapper = Wrapper(parent=scope)
    wrapper()

    # Make sure that variables are registered at the level of the
    # Wrapper submodule, not the Outer submodule.
    self.assertEqual(40, scope.variables()['param']['inner']['x'])

  def test_param_in_setup(self):
    rngkey = jax.random.PRNGKey(0)
    class DummyModule(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    y = DummyModule(x.shape, parent=scope)(x)
    params = scope.variables()['param']
    y2 = DummyModule(x.shape, parent=scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    onp.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_init_outside_setup_without_compact(self):
    rngkey = jax.random.PRNGKey(0)
    class DummyModule(nn.Module):
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'must be initialized.*setup'):
      y = DummyModule(parent=scope)(x)

  def test_init_outside_call(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
      def foo(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'must be initialized.*setup'):
      y = Dummy(parent=scope).foo(x)

  def test_setup_call_var_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already in use'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_setup_var_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already in use'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_call_var_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already in use'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_setattr_name_var_disagreement(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('notbias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'notbias.*must equal.*bias'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_submodule_var_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = DummyModule()
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'name bias exists already'):
      y = Dummy(x.shape, parent=scope)(x)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      @compact
      def __call__(self, x):
        bias = DummyModule(name='bias')
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'name bias exists already'):
      y = Dummy(x.shape, parent=scope)(x)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = DummyModule()
      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_setattr_name_submodule_redundant(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = DummyModule(name='bias')
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'assign names via self'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_attr_param_name_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      bias: bool
      def setup(self):
        self.bias = self.param('bias', initializers.ones, (3, 3))
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'Name bias already in use'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_attr_submodule_name_collision(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      bias: bool
      def setup(self):
        self.bias = DummyModule(name='bias')
      def __call__(self, x):
        return self.bias(x)
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias exists already'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_only_one_compact_method(self):
    with self.assertRaisesRegex(RuntimeError, '@compact'):
      class Dummy(nn.Module):
        @compact
        def call1(self):
          pass
        @compact
        def call2(self):
          pass

  def test_only_one_compact_method_subclass(self):
    class Dummy(nn.Module):
      @nn.compact
      def __call__(self):
        pass
    class SubDummy(Dummy):
      @nn.compact
      def __call__(self):
        super().__call__()

    scope = Scope(variables={})

    subdummy = SubDummy(parent=scope)
    # Make sure the @compact annotation is valid on both base class and subclass, as long
    # as its on the same method.
    subdummy()

  def test_forgotten_compact_annotation(self):
    class Bar(nn.Module):
      # user forgot to add @compact
      def __call__(self, x):
        return nn.Dense(1)(x)
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        bar = Bar()
        x = bar(x)
        x = bar(x)
        return x
    with self.assertRaisesRegex(ValueError, '@compact'):
      Foo().init(random.PRNGKey(0), jnp.ones((1, 3)))

  def test_forgotten_compact_annotation_with_explicit_parent(self):
    class Bar(nn.Module):
      def __call__(self, x):
        return nn.Dense(1, parent=self)(x)

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        bar = Bar()
        x = bar(x)
        x = bar(x)
        return x

    with self.assertRaisesRegex(ValueError, '@compact'):
      Foo().init(random.PRNGKey(0), jnp.ones((1, 3)))

  def test_numpy_array_shape_class_args(self):
    class MLP(nn.Module):
      widths: Iterable
      @nn.compact
      def __call__(self, x):
        for width in self.widths[:-1]:
          x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.widths[-1])(x)
    test = MLP(onp.array([3, 3], onp.int32))
    params = test.init({'param': random.PRNGKey(42)}, jnp.ones((3, 3)))
    _ = test.apply(params, jnp.ones((3, 3)))

if __name__ == '__main__':
  absltest.main()

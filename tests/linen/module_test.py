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
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp
from typing import Any, Tuple

from flax import linen as nn
from flax.core import Scope

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class DummyModule(nn.Module):
  def __call__(self, x):
    bias = self.param('bias', initializers.ones, x.shape)
    return x + bias

class Dense(nn.Module):
  features: int
  def __call__(self, x):
    kernel = self.param('kernel',
                        initializers.lecun_normal(),
                        (x.shape[-1], self.features))
    y = jnp.dot(x, kernel)
    return y

rngkey = jax.random.PRNGKey(0)


class ModuleTest(absltest.TestCase):

  def test_init_module(self):
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    y = DummyModule(scope)(x)
    params = scope.variables()['param']
    y2 = DummyModule(scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    onp.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_arg_module(self):
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = Dense(scope, 3)(x)
    params = scope.variables()['param']
    y2 = Dense(scope.rewound(), 3)(x)
    onp.testing.assert_allclose(y, y2)
    self.assertEqual(params['kernel'].shape, (10, 3))

  def test_util_fun(self):
    class MLP(nn.Module):
      def __call__(self, x):
        x = self.mydense(x)
        x = self.mydense(x)
        return x
      def mydense(self, x):
        return Dense(self, 3)(x)
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = MLP(scope)(x)
    params = scope.variables()['param']
    y2 = MLP(scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'Dense_0': {'kernel': (10, 3)},
       'Dense_1': {'kernel': (3, 3)}})

  def test_nested_module_reuse(self):
    class MLP(nn.Module):
      def __call__(self, x):
        x = self.mydense(x)
        x = self.mydense(x)
        return x
      def mydense(self, x):
        return Dense(self, 3)(x)
    class Top(nn.Module):
      def __call__(self, x):
        mlp = MLP(self)
        y = mlp(x)
        z = mlp(x)
        return y + z
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = Top(scope)(x)
    params = scope.variables()['param']
    y2 = Top(scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'MLP_0':
        {'Dense_0': {'kernel': (10, 3)},
        'Dense_1': {'kernel': (3, 3)}}})

  def test_setup_dict_assignment(self):
    class MLP(nn.Module):
      def setup(self):
        self.lyrs1 = {'a': Dense(None, 3), 'b': Dense(None, 3),}
        self.lyrs2 = [Dense(None, 3), Dense(None, 3)]
      def __call__(self, x):
        y = self.lyrs1['a'](x)
        z = self.lyrs1['b'](y)
        #w = self.lyrs2[0](x)
        return z
    x = jnp.ones((10,))
    scope = Scope({}, {'param': rngkey})
    y = MLP(scope)(x)
    params = scope.variables()['param']
    y2 = MLP(scope.rewound())(x)
    onp.testing.assert_allclose(y, y2)
    param_shape = jax.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
      {'lyrs1_a': {'kernel': (10, 3)},
      'lyrs1_b': {'kernel': (3, 3)}})

  def test_setup_cloning(self):
    class MLP(nn.Module):
      def setup(self):
        self.dense = Dense(None, 3)
    scope = Scope({})
    MLPclone = MLP(scope).clone()

  def test_submodule_attr(self):
    class Inner(nn.Module):
      def __call__(self):
        self.param('x', lambda rng: 40)

    class Outer(nn.Module):
      inner: nn.Module

      def __call__(self):
        return self.inner()

    class Wrapper(nn.Module):
      def setup(self):
        self.inner = Inner(self)
        self.outer = Outer(self, self.inner)

      def __call__(self):
        return self.outer()

    scope = Scope({'param': {}}, rngs={'param': rngkey})
    # Make sure this doesn't raise "Can't attach to remote parent"
    wrapper = Wrapper(scope)
    wrapper()

    # Make sure that variables are registered at the level of the
    # Wrapper submodule, not the Outer submodule.
    self.assertEqual(40, scope.variables()['param']['inner']['x'])

  def test_multi_module(self):
    class DummyMultiModule(nn.MultiModule):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    y = DummyMultiModule(scope, x.shape)(x)
    params = scope.variables()['param']
    y2 = DummyMultiModule(scope.rewound(), x.shape)(x)
    onp.testing.assert_allclose(y, y2)
    onp.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_multi_module_init_outside_setup(self):
    class DummyMultiModule(nn.MultiModule):
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'must initialize.*setup'):
      y = DummyMultiModule(scope)(x)

  def test_init_outside_call(self):
    class Dummy(nn.Module):
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
      def foo(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias.*__call__'):
      y = Dummy(scope).foo(x)

  def test_init_in_setup(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    y = Dummy(scope, x.shape)(x)

  def test_setup_call_var_collision(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already in use'):
      y = Dummy(scope, x.shape)(x)

  def test_setup_var_collision(self):
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
      y = Dummy(scope, x.shape)(x)

  def test_call_var_collision(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        pass
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already in use'):
      y = Dummy(scope, x.shape)(x)

  def test_setattr_name_var_disagreement(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('notbias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'notbias.*must equal.*bias'):
      y = Dummy(scope, x.shape)(x)

  def test_submodule_var_collision(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = DummyModule(None)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'name bias exists already'):
      y = Dummy(scope, x.shape)(x)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        bias = DummyModule(self, name='bias')
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'name bias exists already'):
      y = Dummy(scope, x.shape)(x)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = DummyModule(None)
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias already'):
      y = Dummy(scope, x.shape)(x)

  def test_setattr_name_submodule_redundant(self):
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = DummyModule(self, name='bias')
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'assign names via self'):
      y = Dummy(scope, x.shape)(x)

  def test_attr_param_name_collision(self):
    class Dummy(nn.Module):
      bias: bool
      def setup(self):
        self.bias = self.param('bias', initializers.ones, (3, 3))
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'Name bias already in use'):
      y = Dummy(scope, x.shape)(x)

  def test_attr_submodule_name_collision(self):
    class Dummy(nn.Module):
      bias: bool
      def setup(self):
        self.bias = DummyModule(self, name='bias')
      def __call__(self, x):
        return self.bias(x)
    x = jnp.array([1.])
    scope = Scope({}, {'param': rngkey})
    with self.assertRaisesRegex(ValueError, 'bias exists already'):
      y = Dummy(scope, x.shape)(x)

if __name__ == '__main__':
  absltest.main()


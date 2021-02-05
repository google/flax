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

"""Tests for flax.linen."""

import dataclasses

from absl.testing import absltest

import jax
from jax import random
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np
from typing import Any, Tuple, Iterable, Callable

from flax import linen as nn
from flax.errors import CallCompactUnboundModuleError
from flax.linen import compact
from flax.core import Scope, freeze

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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = DummyModule(parent=scope)(x)
    params = scope.variables()['params']
    y2 = DummyModule(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    np.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_arg_module(self):
    rngkey = jax.random.PRNGKey(0)
    x = jnp.ones((10,))
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dense(3, parent=scope)(x)
    params = scope.variables()['params']
    y2 = Dense(3, parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = MLP(parent=scope)(x)
    params = scope.variables()['params']
    y2 = MLP(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Top(parent=scope)(x)
    params = scope.variables()['params']
    y2 = Top(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = MLP(parent=scope)(x)
    params = scope.variables()['params']
    y2 = MLP(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
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

    scope = Scope({'params': {}}, rngs={'params': rngkey}, mutable=['params'])
    # Make sure this doesn't raise "Can't attach to remote parent"
    wrapper = Wrapper(parent=scope)
    wrapper()

    # Make sure that variables are registered at the level of the
    # Wrapper submodule, not the Outer submodule.
    self.assertEqual(40, scope.variables()['params']['inner']['x'])

  def test_param_in_setup(self):
    rngkey = jax.random.PRNGKey(0)
    class DummyModule(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
      def __call__(self, x):
        return x + self.bias
    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = DummyModule(x.shape, parent=scope)(x)
    params = scope.variables()['params']
    y2 = DummyModule(x.shape, parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    np.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_init_outside_setup_without_compact(self):
    rngkey = jax.random.PRNGKey(0)
    class DummyModule(nn.Module):
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias
    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    with self.assertRaisesRegex(ValueError, 'notbias.*must equal.*bias'):
      y = Dummy(x.shape, parent=scope)(x)

  def test_setattr_name_var_disagreement_allowed_in_lists(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.biases = [
          self.param(f'bias_{i}', initializers.ones, self.xshape)
          for i in range(4)]
      def __call__(self, x):
        return x + self.biases[0]

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.]))

  def test_setattr_name_var_disagreement_allowed_in_dicts(self):
    rngkey = jax.random.PRNGKey(0)
    class Dummy(nn.Module):
      xshape: Tuple[int]
      def setup(self):
        self.biases = {
          # NOTE that keys still must be strings. This is to make a possible
          # future transition to automatically derived parameter names when assigned
          # as a dict easier (like we currently have with submodules).
          # See a bit of discussion here: https://github.com/google/flax/issues/705#issuecomment-738761853 
          str(i): self.param(f'bias_{i}', initializers.ones, self.xshape)
          for i in range(4)}
      def __call__(self, x):
        return x + self.biases['0']

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.]))

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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    with self.assertRaisesRegex(ValueError, 'In setup, assign names of Modules '
        'via self.<name> and not using keyword argument name="<name>"'):
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
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
    test = MLP(np.array([3, 3], np.int32))
    params = test.init({'params': random.PRNGKey(42)}, jnp.ones((3, 3)))
    _ = test.apply(params, jnp.ones((3, 3)))

  def test_get_local_methods(self):
    class Base:
      @staticmethod
      def bar(x):
        return x
      @classmethod
      def baz(cls, x):
        return x
      def bleep(self, x):
        return x
    class Derived1(Base):
      @staticmethod
      def bar2(x):
        return x
      @classmethod
      def baz2(cls, x):
        return x
      def bloop(self, x):
        return x
    class Derived2(Derived1):
      pass

    self.assertEqual(nn.module._get_local_method_names(Base), ('bleep',))
    self.assertEqual(nn.module._get_local_method_names(Derived1), ('bloop',))
    self.assertEqual(
        nn.module._get_local_method_names(Derived1, exclude=('bloop',)), ())
    self.assertEqual(nn.module._get_local_method_names(Derived2), ())

  def test_inheritance_dataclass_attribs(self):
    class Test(nn.Module):
      bar: int
      def __call__(self, x):
        return x
    class Test2(Test):
      baz: int
      def __call__(self, x):
        return x
    class Test3(Test):
      baz: int
      def __call__(self, x):
        return x

    key = random.PRNGKey(0)
    x = jnp.ones((5,))
    test1 = Test(bar=4)
    test2 = Test2(bar=4, baz=2)
    test3 = Test3(bar=4, baz=2)
    self.assertEqual(test1.init_with_output(key, x), (x, freeze({})))
    self.assertEqual(test2.init_with_output(key, x), (x, freeze({})))
    self.assertEqual(test3.init_with_output(key, x), (x, freeze({})))
    self.assertTrue(hasattr(test1, 'bar'))
    self.assertTrue(hasattr(test1, 'name'))
    self.assertTrue(hasattr(test1, 'parent'))
    self.assertTrue(hasattr(test2, 'bar'))
    self.assertTrue(hasattr(test2, 'baz'))
    self.assertTrue(hasattr(test2, 'name'))
    self.assertTrue(hasattr(test2, 'parent'))
    self.assertTrue(hasattr(test3, 'bar'))
    self.assertTrue(hasattr(test3, 'baz'))
    self.assertTrue(hasattr(test3, 'name'))
    self.assertTrue(hasattr(test3, 'parent'))
    self.assertEqual(
        list(Test.__dataclass_fields__.keys()),
        ['bar', 'parent', 'name'])
    self.assertEqual(
        list(Test2.__dataclass_fields__.keys()),
        ['bar', 'baz', 'parent', 'name'])
    self.assertEqual(
        list(Test3.__dataclass_fields__.keys()),
        ['bar', 'baz', 'parent', 'name'])

  def test_get_suffix_value_pairs(self):
    for x in [(), [], {}, None, 0, set()]:
      self.assertEqual(
          nn.module._get_suffix_value_pairs(x), [('', x)])
    self.assertEqual(
        nn.module._get_suffix_value_pairs(
            {'a': 1, 'b': 2}), [('_a', 1), ('_b', 2)])
    self.assertEqual(
        nn.module._get_suffix_value_pairs(
            [1, 2, 3]), [('_0', 1), ('_1', 2), ('_2', 3)])
    x1 = [nn.Dense(10), nn.relu, nn.Dense(10)]
    y1 = nn.module._get_suffix_value_pairs(x1)
    self.assertEqual(y1, [('_0', x1[0]), ('_1', x1[1]), ('_2', x1[2])])
    x2 = {'a': 1, 'b': {'c': nn.Dense(10), 'd': nn.relu}}
    y2 = nn.module._get_suffix_value_pairs(x2)
    self.assertEqual(y2,
        [('_a', 1), ('_b_c', x2['b']['c']), ('_b_d', x2['b']['d'])])

  def test_mixed_list_assignment_in_setup(self):
    class Test(nn.Module):
      def setup(self):
        self.layers = [nn.Dense(10), nn.relu, nn.Dense(10)]
      def __call__(self, x):
        for lyr in self.layers:
          x = lyr(x)
        return x
    x = random.uniform(random.PRNGKey(0), (5,5))
    variables = Test().init(random.PRNGKey(0), jnp.ones((5,5)))
    y = Test().apply(variables, x)
    m0 = variables['params']['layers_0']['kernel']
    m1 = variables['params']['layers_2']['kernel']
    self.assertTrue(jnp.all(y == jnp.dot(nn.relu(jnp.dot(x, m0)), m1)))

  def test_module_is_hashable(self):
    module_a = nn.Dense(10)
    module_a_2 = nn.Dense(10)
    module_b = nn.Dense(5)
    self.assertEqual(hash(module_a), hash(module_a_2))
    self.assertNotEqual(hash(module_a), hash(module_b))

  def test_module_with_scope_is_not_hashable(self):
    module_a = nn.Dense(10, parent=Scope({}))
    with self.assertRaisesWithLiteralMatch(AssertionError, 'Can\'t call __hash__ on modules that hold variables.'):
      hash(module_a)

  def test_module_trace(self):
    class MLP(nn.Module):
      act: Callable = nn.relu
      sizes: Iterable[int] = (3, 2)

      @nn.compact
      def __call__(self, x):
        for size in self.sizes:
          x = nn.Dense(size)(x)
          x = self.act(x)
        return repr(self)
    mlp = MLP()
    expected_trace = (
"""MLP(
    # attributes
    act = relu
    sizes = (3, 2)
    # children
    Dense_0 = Dense(
        # attributes
        features = 3
        use_bias = True
        dtype = float32
        precision = None
        kernel_init = init
        bias_init = zeros
    )
    Dense_1 = Dense(
        # attributes
        features = 2
        use_bias = True
        dtype = float32
        precision = None
        kernel_init = init
        bias_init = zeros
    )
)""")
    x = jnp.ones((1, 2))
    trace, variables = mlp.init_with_output(random.PRNGKey(0), x)
    self.assertEqual(trace, expected_trace)
    trace = mlp.apply(variables, x)
    self.assertEqual(trace, expected_trace)


  def test_call_unbound_compact_module_methods(self):
    dense = Dense(3)
    with self.assertRaises(CallCompactUnboundModuleError):
      dense(jnp.ones((1, )))


  def test_call_unbound_has_variable(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.has_variable('bar', 'baz')

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, "variable.*unbound module"):
      empty.foo()


  def test_call_unbound_make_rng(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.make_rng('bar')

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, "RNGs.*unbound module"):
      empty.foo()


  def test_call_unbound_variables(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.variables

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, "variables.*unbound module"):
      empty.foo()


  def test_call_unbound_noncompact_module_methods(self):
    class EmptyModule(nn.Module):
      foo: int = 3

      def bar(self):
        return self.foo

    empty = EmptyModule()
    # It's fine to call methods of unbound methods that don't depend on
    # attributes defined during `setup`
    self.assertEqual(empty.bar(), 3)


  def test_call_unbound_noncompact_module_methods(self):
    class EmptyModule(nn.Module):
      foo: int = 3

      def bar(self):
        return self.foo

    empty = EmptyModule()
    # It's fine to call methods of unbound methods that don't depend on
    # attributes defined during `setup`
    self.assertEqual(empty.bar(), 3)


  def test_module_with_attrs(self):
    class Foo(nn.Module):
      bar: nn.Dense = dataclasses.field(init=False)

      def setup(self):
        self.bar = nn.Dense(3)

      def __call__(self, x):
        return self.bar(x)

    foo = Foo()
    x = jnp.ones((2,))
    variables = foo.init(random.PRNGKey(0), x)
    self.assertEqual(variables['params']['bar']['kernel'].shape, (2, 3))

  def test_module_frozen(self):
    class Foo(nn.Module):
      bar: nn.Dense = dataclasses.field(init=False)

      def setup(self):
        self.i = 1

      def __call__(self):
        self.i = 2

    foo = Foo()
    with self.assertRaisesWithLiteralMatch(TypeError, "Module instance is frozen outside of setup method."):
      foo.init(random.PRNGKey(0))

  def test_module_lazy_getattr_setup(self):
    class A(nn.Module):
      def setup(self):
        self.d = nn.Dense(2)
      def __call__(self, x):
        return self.d(x)

    class B(nn.Module):
      def setup(self):
        self.a = A()
      def __call__(self, x):
        y1 = self.a.d(x)
        y2 = self.a(x)
        return y1, y2

    key = random.PRNGKey(0)
    x = jnp.ones((2,))

    (y1, y2), p = B().init_with_output(key, x)
    np.testing.assert_array_equal(y1, y2)

  def test_module_lazy_dir_setup(self):
    class A(nn.Module):
      def setup(self):
        self.d = nn.Dense(2)
      def __call__(self, x):
        return self.d(x)

    class B(nn.Module):
      def setup(self):
        self.a = A()
      def __call__(self, x):
        assert 'd' in dir(self.a)
        y1 = self.a.d(x)
        y2 = self.a(x)
        return y1, y2

    key = random.PRNGKey(0)
    x = jnp.ones((2,))
    _ = B().init_with_output(key, x)
  
  def test_module_unbound_getattr(self):
    class A(nn.Module):
      def setup(self):
        b = B()
        b.c  # B is unbound because it is not yet assigned to an attribute.
        self.b = b
      
      def __call__(self):
        pass
    
    class B(nn.Module):
      def setup(self):
        self.c = nn.Dense(2)

    with self.assertRaisesWithLiteralMatch(AttributeError, "'B' object has no attribute 'c'"):
      A().init(random.PRNGKey(0))

  def test_unbound_setup_call(self):
    setup_called = False

    class A(nn.Module):
      def setup(self):
        nonlocal setup_called
        setup_called = True
      
      def test(self):
        pass

    A().test()
    self.assertFalse(setup_called)


if __name__ == '__main__':
  absltest.main()

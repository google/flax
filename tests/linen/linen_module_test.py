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

"""Tests for flax.linen."""

import contextlib
import copy
import dataclasses
import enum
import functools
import gc
import inspect
import operator
import sys
from tempfile import TemporaryDirectory
from typing import (
  Any,
  Generic,
  NamedTuple,
  TypeVar,
  get_type_hints,
)
from collections.abc import Callable, Mapping, Sequence
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import random
from jax.nn import initializers

from flax import config, errors, struct
from flax import linen as nn
from flax.core import FrozenDict, Scope, freeze
from flax.linen import compact

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def tree_equals(x, y):
  return jax.tree_util.tree_all(jax.tree_util.tree_map(operator.eq, x, y))


@contextlib.contextmanager
def set_config(option: str, value: bool):
  old_value = getattr(config, option)
  try:
    config.update(option, value)
    yield None
  finally:
    config.update(option, old_value)


class DummyModule(nn.Module):
  @compact
  def __call__(self, x):
    bias = self.param('bias', initializers.ones, x.shape)
    return x + bias


class Dense(nn.Module):
  features: int

  @compact
  def __call__(self, x):
    kernel = self.param(
      'kernel', initializers.lecun_normal(), (x.shape[-1], self.features)
    )
    y = jnp.dot(x, kernel)
    return y


class IdentityModule(nn.Module):
  def __call__(self, x):
    return x


class RaisesModule(nn.Module):
  def __call__(self):
    assert False


class ModuleTest(absltest.TestCase):
  def test_init_module(self):
    rngkey = jax.random.key(0)
    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = DummyModule(parent=scope)(x)
    params = scope.variables()['params']
    y2 = DummyModule(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    np.testing.assert_allclose(y, jnp.array([2.0]))
    self.assertEqual(params, {'bias': jnp.array([1.0])})

  def test_lazy_init(self):
    class Foo(nn.Module):
      @compact
      def __call__(self, x):
        k = self.param(
          'kernel', nn.initializers.lecun_normal(), (x.shape[-1], x.shape[-1])
        )
        return x @ k

    # provide a massive input message which would OOM if any compute ops were actually executed
    variables = Foo().lazy_init(
      random.key(0),
      jax.ShapeDtypeStruct((1024 * 1024 * 1024, 128), jnp.float32),
    )
    self.assertEqual(variables['params']['kernel'].shape, (128, 128))

  def test_lazy_init_fails_on_data_dependence(self):
    class Foo(nn.Module):
      @compact
      def __call__(self, x):
        k = self.param('kernel', lambda _: x)
        return x * k

    with self.assertRaises(errors.LazyInitError):
      Foo().lazy_init(random.key(0), jax.ShapeDtypeStruct((8, 4), jnp.float32))

  def test_arg_module(self):
    rngkey = jax.random.key(0)
    x = jnp.ones((10,))
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dense(3, parent=scope)(x)
    params = scope.variables()['params']
    y2 = Dense(3, parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    self.assertEqual(params['kernel'].shape, (10, 3))

  def test_util_fun(self):
    rngkey = jax.random.key(0)

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
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(
      param_shape,
      {'Dense_0': {'kernel': (10, 3)}, 'Dense_1': {'kernel': (3, 3)}},
    )

  def test_nested_module_reuse(self):
    rngkey = jax.random.key(0)

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
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(
      param_shape,
      {
        'MLP_0': {
          'Dense_0': {'kernel': (10, 3)},
          'Dense_1': {'kernel': (3, 3)},
        }
      },
    )

  def test_setup_dict_assignment(self):
    rngkey = jax.random.key(0)

    class MLP(nn.Module):
      def setup(self):
        self.lyrs1 = {
          'a': Dense(3),
          'b': Dense(3),
        }
        self.lyrs2 = [Dense(3), Dense(3)]

      def __call__(self, x):
        y = self.lyrs1['a'](x)
        z = self.lyrs1['b'](y)
        return z

    x = jnp.ones((10,))
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = MLP(parent=scope)(x)
    params = scope.variables()['params']
    y2 = MLP(parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(
      param_shape,
      {'lyrs1_a': {'kernel': (10, 3)}, 'lyrs1_b': {'kernel': (3, 3)}},
    )

  def test_setup_dict_nonstring_keys(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = {(1, 2): nn.Dense(2)}  # Tuple as key.

      @nn.compact
      def __call__(self, x):
        return self.a[(1, 2)](x)

    foo = Foo()
    x = jnp.ones(shape=(1, 3))
    params = foo.init(random.key(0), x)['params']
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(
      param_shape, {'a_(1, 2)': {'kernel': (3, 2), 'bias': (2,)}}
    )

  def test_setup_cloning(self):
    class MLP(nn.Module):
      def setup(self):
        self.dense = Dense(3)

    scope = Scope({})
    unused_clone = MLP(parent=scope).clone()

  def test_submodule_attr(self):
    rngkey = jax.random.key(0)

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
    rngkey = jax.random.key(0)

    class DummyModuleWithoutCompact(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = DummyModuleWithoutCompact(x.shape, parent=scope)(x)
    params = scope.variables()['params']
    y2 = DummyModuleWithoutCompact(x.shape, parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    np.testing.assert_allclose(y, jnp.array([2.0]))
    self.assertEqual(params, {'bias': jnp.array([1.0])})

  def test_init_outside_setup_without_compact(self):
    rngkey = jax.random.key(0)

    class DummyModuleWithoutCompact(nn.Module):
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    with self.assertRaisesRegex(ValueError, 'must be initialized.*setup'):
      unused_y = DummyModuleWithoutCompact(parent=scope)(x)

  def test_init_outside_call(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias

      def foo(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    with self.assertRaisesRegex(ValueError, 'must be initialized.*setup'):
      unused_y = Dummy(parent=scope).foo(x)

  def test_setup_call_var_collision(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      @compact
      def __call__(self, x):
        unused_bias = self.param('bias', initializers.ones, x.shape)
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_call_var_collision(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_setup_var_collision(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = self.param('bias', initializers.ones, self.xshape)

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_setattr_name_var_disagreement_allowed_in_lists(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.biases = [
          self.param(f'bias_{i}', initializers.ones, self.xshape)
          for i in range(4)
        ]

      def __call__(self, x):
        return x + self.biases[0]

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.0]))

  def test_setattr_name_var_disagreement_allowed_in_dicts(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.biases = {
          # NOTE: Keys still must be strings. This is to make a possible
          # future transition to automatically derived parameter names when
          # assigned as a dict easier (like we currently have with
          # submodules). See a bit of discussion here:
          # https://github.com/google/flax/issues/705#issuecomment-738761853
          str(i): self.param(f'bias_{i}', initializers.ones, self.xshape)
          for i in range(4)
        }

      def __call__(self, x):
        return x + self.biases['0']

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.0]))

  def test_submodule_var_collision_with_scope(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = DummyModule()

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])

    with self.assertRaises(errors.NameInUseError):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_submodule_var_collision_with_submodule(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      @compact
      def __call__(self, x):
        unused_bias = DummyModule(name='bias')
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])

    msg = 'Could not create submodule "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_submodule_var_collision_with_params(self):
    rngkey = jax.random.key(0)

    class Dummy(nn.Module):
      xshape: tuple[int, ...]

      def setup(self):
        self.bias = DummyModule()

      @compact
      def __call__(self, x):
        unused_bias = self.param('bias', initializers.ones, self.xshape)
        return x + self.bias

    x = jnp.array([1.0])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])

    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_attr_empty_container(self):
    class Foo(nn.Module):
      bar: Mapping[str, Any]

      @compact
      def __call__(self):
        pass

    Foo({'a': ()}).apply({})

  def test_multiple_compact_methods(self):
    """Test that multiple methods with the @compact decorator can be used.

    NOTE: in the near future we might want to have compact methods reset the
    autoname_cursor such that Dense would be reused in the second method.
    """

    class MultipleCompactMethods(nn.Module):
      @compact
      def __call__(self, x):
        x = nn.Dense(1)(x)
        return self.method(x)

      @compact
      def method(self, x):
        x = nn.Dense(1)(x)
        return x

    m = MultipleCompactMethods()
    variables = m.init(random.key(0), jnp.ones((1, 1)))
    params = variables['params']
    self.assertIn('Dense_0', params)
    self.assertIn('Dense_1', params)

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
    # Make sure the @compact annotation is valid on both base class and
    # subclass, as long as its on the same method.
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

    msg = (
      r'Submodule Dense must be defined in `setup\(\)` or in a method '
      'wrapped in `@compact`'
    )
    with self.assertRaisesRegex(errors.AssignSubModuleError, msg):
      Foo().init(random.key(0), jnp.ones((1, 3)))

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

    msg = (
      r'Submodule Dense must be defined in `setup\(\)` or in a method '
      'wrapped in `@compact`'
    )
    with self.assertRaisesRegex(errors.AssignSubModuleError, msg):
      Foo().init(random.key(0), jnp.ones((1, 3)))

  def test_numpy_array_shape_class_args(self):
    class MLP(nn.Module):
      widths: Sequence[int]

      @nn.compact
      def __call__(self, x):
        for width in self.widths[:-1]:
          x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.widths[-1])(x)

    test = MLP(np.array([3, 3], np.int32))
    params = test.init({'params': random.key(42)}, jnp.ones((3, 3)))
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
      nn.module._get_local_method_names(Derived1, exclude=('bloop',)), ()
    )
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

    class Test4(Test2):
      def __call__(self, x):
        return x

    key = random.key(0)
    x = jnp.ones((5,))
    test1 = Test(bar=4)
    test2 = Test2(bar=4, baz=2)
    test3 = Test3(bar=4, baz=2)
    test4 = Test4(bar=5, baz=3)
    self.assertEqual(test1.init_with_output(key, x), (x, freeze({})))
    self.assertEqual(test2.init_with_output(key, x), (x, freeze({})))
    self.assertEqual(test3.init_with_output(key, x), (x, freeze({})))
    self.assertEqual(test4.init_with_output(key, x), (x, freeze({})))
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
    self.assertTrue(hasattr(test4, 'bar'))
    self.assertTrue(hasattr(test4, 'baz'))
    self.assertTrue(hasattr(test4, 'name'))
    self.assertTrue(hasattr(test4, 'parent'))
    self.assertEqual(
      list(Test.__dataclass_fields__.keys()), ['bar', 'parent', 'name']
    )
    self.assertEqual(
      list(Test2.__dataclass_fields__.keys()),
      ['bar', 'baz', 'parent', 'name'],
    )
    self.assertEqual(
      list(Test3.__dataclass_fields__.keys()),
      ['bar', 'baz', 'parent', 'name'],
    )
    self.assertEqual(
      list(Test4.__dataclass_fields__.keys()),
      ['bar', 'baz', 'parent', 'name'],
    )

  def test_get_suffix_value_pairs(self):
    for x in [(), [], {}, None, 0, set()]:
      self.assertEqual(nn.module._get_suffix_value_pairs(x), [('', x)])
    self.assertEqual(
      nn.module._get_suffix_value_pairs({'a': 1, 'b': 2}),
      [('_a', 1), ('_b', 2)],
    )
    self.assertEqual(
      nn.module._get_suffix_value_pairs([1, 2, 3]),
      [('_0', 1), ('_1', 2), ('_2', 3)],
    )
    x1 = [nn.Dense(10), nn.relu, nn.Dense(10)]
    y1 = nn.module._get_suffix_value_pairs(x1)
    self.assertEqual(y1, [('_0', x1[0]), ('_1', x1[1]), ('_2', x1[2])])
    x2 = {'a': 1, 'b': {'c': nn.Dense(10), 'd': nn.relu}}
    y2 = nn.module._get_suffix_value_pairs(x2)
    self.assertEqual(
      y2, [('_a', 1), ('_b_c', x2['b']['c']), ('_b_d', x2['b']['d'])]
    )

  def test_mixed_list_assignment_in_setup(self):
    class Test(nn.Module):
      def setup(self):
        self.layers = [nn.Dense(10), nn.relu, nn.Dense(10)]

      def __call__(self, x):
        for lyr in self.layers:
          x = lyr(x)
        return x

    x = random.uniform(random.key(0), (5, 5))
    variables = Test().init(random.key(0), jnp.ones((5, 5)))
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

  def test_module_custom_hash(self):
    class Test(nn.Module):
      x: int = 3
      y: int = 5

      def __hash__(self):
        return 42 + self.x

    module_a = Test(1, 2)
    module_a_2 = Test(1, 5)
    module_b = Test(2, 2)
    self.assertEqual(hash(module_a), hash(module_a_2))
    self.assertNotEqual(hash(module_a), hash(module_b))

  def test_module_with_scope_is_not_hashable(self):
    module_a = nn.Dense(10, parent=Scope({}))
    msg = "Can't call __hash__ on modules that hold variables."
    with self.assertRaisesWithLiteralMatch(TypeError, msg):
      hash(module_a)

  def test_module_trace(self):
    class MLP(nn.Module):
      act: Callable = nn.relu
      sizes: Sequence[int] = (3, 2)

      @nn.compact
      def __call__(self, x):
        for size in self.sizes:
          x = nn.Dense(size)(x)
          x = self.act(x)
        return repr(self)

    mlp = MLP()
    expected_trace = """MLP(
    # attributes
    act = relu
    sizes = (3, 2)
    # children
    Dense_0 = Dense(
        # attributes
        features = 3
        use_bias = True
        dtype = None
        param_dtype = float32
        precision = None
        kernel_init = init
        bias_init = zeros
        dot_general = None
        dot_general_cls = None
    )
    Dense_1 = Dense(
        # attributes
        features = 2
        use_bias = True
        dtype = None
        param_dtype = float32
        precision = None
        kernel_init = init
        bias_init = zeros
        dot_general = None
        dot_general_cls = None
    )
)"""
    x = jnp.ones((1, 2))
    trace, variables = mlp.init_with_output(random.key(0), x)
    self.assertEqual(trace, expected_trace)
    trace = mlp.apply(variables, x)
    self.assertEqual(trace, expected_trace)

  def test_default_params_rng_equivalence(self):
    class Model(nn.Module):
      @nn.compact
      def __call__(self, x, add_dropout=False, add_noise=False):
        x = nn.Dense(16)(x)
        x = nn.Dropout(0.5)(x, deterministic=not add_dropout)
        if add_noise:
          x += jax.random.normal(self.make_rng('params'))
        return x

    model = Model()
    key0, key1, key2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(key0, (10, 8))

    with self.assertRaisesRegex(
      ValueError,
      'First argument passed to an init function should be a ``jax.PRNGKey``',
    ):
      model.init({'params': 'test'}, x)
    with self.assertRaisesRegex(
      errors.InvalidRngError,
      'RNGs should be of shape \\(2,\\) or PRNGKey in Module Model, but rngs are: test',
    ):
      model.init('test', x)
    # should not throw an error, since nn.Dropout will get an RNG key from the 'params' stream
    model.init(key1, x, add_dropout=True)

    v = model.init({'params': key1}, x)
    v2 = model.init(key1, x)
    jax.tree_util.tree_map(np.testing.assert_allclose, v, v2)

    for add_dropout, add_noise in [[True, False], [False, True], [True, True]]:
      out = model.apply(
        v,
        x,
        add_dropout=add_dropout,
        add_noise=add_noise,
        rngs={'params': key2},
      )
      out2 = model.apply(
        v, x, add_dropout=add_dropout, add_noise=add_noise, rngs=key2
      )
      np.testing.assert_allclose(out, out2)

    with self.assertRaisesRegex(
      ValueError,
      'The ``rngs`` argument passed to an apply function should be a ``jax.PRNGKey``',
    ):
      model.apply(v, x, rngs={'params': 'test'})
    with self.assertRaisesRegex(
      errors.InvalidRngError,
      'RNGs should be of shape \\(2,\\) or PRNGKey in Module Model, but rngs are: test',
    ):
      model.apply(v, x, rngs='test')

  def test_module_apply_method(self):
    class Foo(nn.Module):
      not_callable: int = 1

      @nn.compact
      def __call__(self):
        pass

      def test(self):
        pass

    # We can use both instance and class methods in apply.
    Foo().apply({}, method=Foo.test)
    Foo().apply({}, method=Foo().test)

    # We also use a function that is not in the provided Module, although it
    # should have a first argument representing an instance of the Module (Foo
    # in this case).
    x = Foo().apply({}, method=lambda foo_instance: foo_instance)
    self.assertEqual(type(x), type(Foo()))

    # This is not allowed.
    msg = 'Cannot call apply()'
    with self.assertRaisesRegex(errors.ApplyModuleInvalidMethodError, msg):
      Foo().apply({}, method=lambda: True)

    # string method names are also allowed.
    Foo().apply({}, method='test')
    # test same for init.
    Foo().init({}, method='test')

    # non-existent attribute names will yield AttributeError.
    with self.assertRaisesRegex(AttributeError, 'allowed_apply_fn'):
      Foo().apply({}, method='allowed_apply_fn')
      # test same for init.
      Foo().init({}, method='allowed_apply_fn')

    # attributes which are not callables yield TypeError.
    with self.assertRaisesRegex(
      TypeError, "'Foo.not_callable' must be a callable"
    ):
      Foo().apply({}, method='not_callable')
      # test same for init.
      Foo().init({}, method='not_callable')

  def test_module_apply_method_submodule(self):
    class Foo(nn.Module):
      bar: nn.Module

      @nn.compact
      def __call__(self, x):
        return self.bar(x)

    foo = Foo(nn.Dense(3))
    variables = foo.init(jax.random.PRNGKey(0), jnp.zeros(3))

    foo.apply(variables, jnp.zeros(3), method='bar')

  def test_call_unbound_compact_module_methods(self):
    dense = Dense(3)
    msg = r'Can\'t call compact methods on unbound modules'
    with self.assertRaisesRegex(errors.CallCompactUnboundModuleError, msg):
      dense(jnp.ones((1,)))

  def test_call_unbound_has_variable(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.has_variable('bar', 'baz')

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, 'variable.*unbound module'):
      empty.foo()

  def test_call_unbound_make_rng(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.make_rng('bar')

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, 'RNGs.*unbound module'):
      empty.foo()

  def test_call_unbound_variables(self):
    class EmptyModule(nn.Module):
      def foo(self):
        self.variables

    empty = EmptyModule()
    with self.assertRaisesRegex(ValueError, 'variables.*unbound module'):
      empty.foo()

  def test_call_unbound_noncompact_module_methods(self):
    class EmptyModule(nn.Module):
      foo: int = 3

      def bar(self):
        return self.foo

    empty = EmptyModule()
    # It's fine to call methods of unbound methods that don't depend on
    # attributes defined during `setup`.
    self.assertEqual(empty.bar(), 3)

  def test_call_unbound_noncompact_module_methods_depending_on_setup(self):
    class EmptyModule(nn.Module):
      def setup(self):
        self.foo = 2

      def bar(self):
        return self.foo

    empty = EmptyModule()
    msg = r'"EmptyModule" object has no attribute "foo"'
    with self.assertRaisesRegex(AttributeError, msg):
      empty.bar()

  def test_module_with_attrs(self):
    class Foo(nn.Module):
      bar: nn.Dense = dataclasses.field(init=False)

      def setup(self):
        self.bar = nn.Dense(3)

      def __call__(self, x):
        return self.bar(x)

    foo = Foo()
    x = jnp.ones((2,))
    variables = foo.init(random.key(0), x)
    self.assertEqual(variables['params']['bar']['kernel'].shape, (2, 3))

  def test_noncompact_module_frozen(self):
    class Foo(nn.Module):
      def setup(self):
        self.i = 1  # This is allowed (for assigning submodules).

      def __call__(self):
        self.i = 2  # This is not allowed.

    msg = (
      "Can't set i=2 for Module of type Foo: Module instance is frozen "
      'outside of setup method.'
    )
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.key(0))

  def test_compact_module_frozen(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self):
        self.i = 2

    msg = (
      "Can't set i=2 for Module of type Foo: Module instance is frozen "
      'outside of setup method.'
    )
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.key(0))

  def test_submodule_frozen(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self):
        dense = nn.Dense(10)
        dense.features = 20  # <--- This is not allowed

    msg = (
      "Can't set features=20 for Module of type Dense: Module instance "
      'is frozen outside of setup method.'
    )
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.key(0))

  def test_module_call_not_implemented(self):
    class Foo(nn.Module):
      pass

    msg = '"Foo" object has no attribute "__call__"'
    with self.assertRaisesRegex(AttributeError, msg):
      Foo().init(random.key(0))

  def test_is_mutable_collection(self):
    class EmptyModule(nn.Module):
      def __call__(self):
        return self.is_mutable_collection('test')

    empty = EmptyModule()
    self.assertTrue(empty.apply({}, mutable=['test'])[0])
    self.assertFalse(empty.apply({}, mutable=False))

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

    key = random.key(0)
    x = jnp.ones((2,))

    (y1, y2), unused_vars = B().init_with_output(key, x)
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

    key = random.key(0)
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

    msg = '"B" object has no attribute "c"'
    with self.assertRaisesRegex(AttributeError, msg):
      A().init(random.key(0))

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

  def test_module_pass_as_attr(self):
    class A(nn.Module):
      def setup(self):
        self.b = B(nn.Dense(2))

      def __call__(self, x):
        return self.b(x)

    class B(nn.Module):
      foo: Any

      def __call__(self, x):
        return self.foo(x)

    variables = A().init(random.key(0), jnp.ones((1,)))
    var_shapes = jax.tree_util.tree_map(jnp.shape, variables)
    ref_var_shapes = {
      'params': {
        'b': {
          'foo': {
            'bias': (2,),
            'kernel': (1, 2),
          }
        },
      },
    }
    self.assertTrue(tree_equals(var_shapes, ref_var_shapes))

  def test_module_pass_in_closure(self):
    a = nn.Dense(2)

    class B(nn.Module):
      def setup(self):
        self.foo = a

      def __call__(self, x):
        return self.foo(x)

    variables = B().init(random.key(0), jnp.ones((1,)))
    var_shapes = jax.tree_util.tree_map(jnp.shape, variables)
    ref_var_shapes = {
      'params': {
        'foo': {
          'bias': (2,),
          'kernel': (1, 2),
        }
      },
    }
    self.assertTrue(tree_equals(var_shapes, ref_var_shapes))
    self.assertIsNone(a.name)

  def test_toplevel_submodule_adoption(self):
    class Encoder(nn.Module):
      n_layers: int
      ch: int

      def setup(self):
        self.layers = [nn.Dense(self.ch) for _ in range(self.n_layers)]

      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
          x = nn.relu(x)
        return x

    class Model(nn.Module):
      encoder: nn.Module
      n_out: int

      def setup(self):
        self.dense_out = nn.Dense(self.n_out)

      def __call__(self, x):
        x = self.encoder(x)
        return self.dense_out(x)

    # Define model.
    encoder = Encoder(n_layers=1, ch=8)
    model = Model(encoder=encoder, n_out=5)

    # Initialize.
    key = jax.random.key(0)
    x = random.uniform(key, (4, 4))

    variables = model.init(key, x)
    y = model.apply(variables, x)
    self.assertEqual(y.shape, (4, 5))

    var_shapes = jax.tree_util.tree_map(jnp.shape, variables)
    ref_var_shapes = {
      'params': {
        'dense_out': {
          'bias': (5,),
          'kernel': (8, 5),
        },
        'encoder': {
          'layers_0': {
            'bias': (8,),
            'kernel': (4, 8),
          },
        },
      },
    }
    self.assertTrue(tree_equals(var_shapes, ref_var_shapes))

  def test_toplevel_submodule_adoption_pytree(self):
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

    unused_a = A()
    a_pytree = {'foo': A(), 'bar': A()}
    b = B(a_pytree)

    key = random.key(0)
    x = jnp.ones((2, 2))

    params = B(a_pytree).init(key, x, x)
    unused_y, counters = b.apply(params, x, x, mutable='counter')
    ref_counters = {
      'counter': {
        'A_bar': {
          'i': jnp.array(2.0),
        },
        'A_foo': {
          'i': jnp.array(2.0),
        },
      },
    }
    self.assertTrue(
      jax.tree_util.tree_all(
        jax.tree_util.tree_map(
          lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
          counters,
          ref_counters,
        )
      )
    )

  def test_toplevel_submodule_adoption_sharing(self):
    dense = functools.partial(nn.Dense, use_bias=False)

    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        return dense(2)(x)

    class B(nn.Module):
      a: nn.Module

      @nn.compact
      def __call__(self, x):
        return dense(2)(x) + self.a(x)

    class C(nn.Module):
      a: nn.Module
      b: nn.Module

      @nn.compact
      def __call__(self, x):
        return dense(2)(x) + self.b(x) + self.a(x)

    key = random.key(0)
    x = jnp.ones((2, 2))
    a = A()
    b = B(a)
    c = C(a, b)
    p = c.init(key, x)
    var_shapes = jax.tree_util.tree_map(jnp.shape, p)
    ref_var_shapes = {
      'params': {
        'Dense_0': {
          'kernel': (2, 2),
        },
        'a': {
          'Dense_0': {
            'kernel': (2, 2),
          },
        },
        'b': {
          'Dense_0': {
            'kernel': (2, 2),
          },
        },
      },
    }
    self.assertTrue(tree_equals(var_shapes, ref_var_shapes))

  def test_toplevel_named_submodule_adoption(self):
    dense = functools.partial(nn.Dense, use_bias=False)

    class A(nn.Module):
      def setup(self):
        self.dense = dense(4)

      def __call__(self, x):
        return self.dense(x)

    class B(nn.Module):
      a: A

      def setup(self):
        self.proj = dense(6)

      def __call__(self, x):
        return self.proj(self.a(x))

    a = A(name='foo')
    b = B(a=a)
    k = jax.random.key(0)
    x = jnp.zeros((5, 5))
    init_vars = b.init(k, x)
    var_shapes = jax.tree_util.tree_map(jnp.shape, init_vars)
    if config.flax_preserve_adopted_names:
      ref_var_shapes = {
        'params': {
          'foo': {
            'dense': {
              'kernel': (5, 4),
            },
          },
          'proj': {
            'kernel': (4, 6),
          },
        },
      }
    else:
      ref_var_shapes = {
        'params': {
          'a': {
            'dense': {
              'kernel': (5, 4),
            },
          },
          'proj': {
            'kernel': (4, 6),
          },
        },
      }
    self.assertTrue(tree_equals(var_shapes, ref_var_shapes))

  def test_toplevel_submodule_pytree_adoption_sharing(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        counter = self.variable('counter', 'i', jnp.zeros, ())
        counter.value += 1
        x = nn.Dense(1)(x)
        return x

    class B(nn.Module):
      A: Any

      @nn.compact
      def __call__(self, x):
        return self.A['foo'](x) + self.A['bar'](x) + self.A['baz'](x)

    key = random.key(0)
    x = jnp.ones((2, 2))

    a = A()
    a_pytree = {'foo': a, 'bar': a, 'baz': a}
    b = B(a_pytree)

    params = b.init(key, x)
    _, counters = b.apply(params, x, mutable='counter')
    ref_counters = {
      'counter': {
        'A_bar': {
          'i': jnp.array(6.0),
        },
      },
    }
    self.assertTrue(tree_equals(counters, ref_counters))

  def test_inner_class_def(self):
    class X(nn.Module):
      class Hyper(struct.PyTreeNode):
        a: int

      hyper: Hyper

      @nn.compact
      def __call__(self, x):
        return x + 1

    self.assertIsInstance(X.Hyper(a=1), X.Hyper)

  def test_sow(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x, **sow_args):
        self.sow('intermediates', 'h', x, **sow_args)
        self.sow('intermediates', 'h', 2 * x, **sow_args)
        return 3 * x

    variables = Foo().init(random.key(0), 1)
    # During init we should not collect intermediates by default...
    self.assertNotIn('intermediates', variables)
    # ...unless we override mutable.
    variables = Foo().init(random.key(0), 1, mutable=True)
    self.assertEqual(variables, {'intermediates': {'h': (1, 2)}})

    _, state = Foo().apply({}, 1, mutable=['intermediates'])
    self.assertEqual(state, {'intermediates': {'h': (1, 2)}})
    _, state = Foo().apply(
      {},
      1,
      init_fn=lambda: 0,
      reduce_fn=lambda a, b: a + b,
      mutable=['intermediates'],
    )
    self.assertEqual(state, {'intermediates': {'h': 3}})
    self.assertEqual(Foo().apply({}, 1), 3)

  def test_capture_intermediates(self):
    class Bar(nn.Module):
      def test(self, x):
        return x + 1

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return Bar().test(x) + 1

    _, state = Foo().apply({}, 1, capture_intermediates=True)
    self.assertEqual(state, {'intermediates': {'__call__': (3,)}})
    fn = lambda mdl, _: isinstance(mdl, Bar)
    _, state = Foo().apply({}, 1, capture_intermediates=fn)
    self.assertEqual(state, {'intermediates': {'Bar_0': {'test': (2,)}}})

  def test_perturb(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(10)(x)
        x = self.perturb('before_multiply', x)
        x = 4 * x
        x = self.perturb('after_multiply', x)
        return x

    def loss(params, perturbations, inputs, targets):
      variables = {'params': params, 'perturbations': perturbations}
      preds = Foo().apply(variables, inputs)
      return jnp.square(preds - targets).mean()

    x = jax.random.uniform(jax.random.key(1), shape=(10,))
    y = jax.random.uniform(jax.random.key(2), shape=(10,))
    variables = Foo().init(jax.random.key(0), x)
    intm_grads = jax.grad(loss, argnums=1)(
      variables['params'], variables['perturbations'], x, y
    )
    # activation * 4 so reverse gradient also * 4
    self.assertTrue(
      all(intm_grads['after_multiply'] * 4 == intm_grads['before_multiply'])
    )

  def test_perturb_setup(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = nn.Dense(10)

      def __call__(self, x):
        x = self.a(x)
        x = self.perturb('before_multiply', x)
        x = 4 * x
        x = self.perturb('after_multiply', x)
        return x

    def loss(params, perturbations, inputs, targets):
      variables = {'params': params, 'perturbations': perturbations}
      preds = Foo().apply(variables, inputs)
      return jnp.square(preds - targets).mean()

    x = jax.random.uniform(jax.random.key(1), shape=(10,))
    y = jax.random.uniform(jax.random.key(2), shape=(10,))
    variables = Foo().init(jax.random.key(0), x)
    intm_grads = jax.grad(loss, argnums=1)(
      variables['params'], variables['perturbations'], x, y
    )
    # activation * 4 so reverse gradient also * 4
    self.assertTrue(
      all(intm_grads['after_multiply'] * 4 == intm_grads['before_multiply'])
    )

  def test_perturb_noop(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(10)(x)
        x = self.perturb('before_multiply', x)
        x = 4 * x
        x = self.perturb('after_multiply', x)
        return x

    x = jax.random.uniform(jax.random.key(1), shape=(10,))
    module = Foo()
    variables = module.init(jax.random.key(0), x)
    params = variables['params']
    perturbations = variables['perturbations']

    # check no error if perturbations is not passed
    module.apply({'params': params}, x)

    # check errors if perturbations is passed but empty
    with self.assertRaisesRegex(ValueError, 'Perturbation collection'):
      module.apply({'params': params, 'perturbations': {}}, x)

    # check no error if perturbations is passed and not empty
    module.apply({'params': params, 'perturbations': perturbations}, x)

  def test_functional_apply(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = nn.Dense(3)
        self.b = nn.Dense(1)

    def f(foo, x):
      x = foo.a(x)
      return foo.b(x)

    foo = Foo()
    x = jnp.ones((4,))
    f_init = nn.init_with_output(f, foo)
    f_apply = nn.apply(f, foo)
    y1, variables = f_init(random.key(0), x)
    y2 = f_apply(variables, x)
    self.assertEqual(y1, y2)

  def test_bind(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = nn.Dense(3)
        self.b = nn.Dense(1)

    def f(foo, x):
      x = foo.a(x)
      return foo.b(x)

    foo = Foo()
    x = jnp.ones((4,))
    f_init = nn.init_with_output(f, foo)
    y1, variables = f_init(random.key(0), x)
    y2 = f(foo.bind(variables), x)
    self.assertEqual(y1, y2)

  def test_bind_stateful(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = nn.Dense(3)
        self.bn = nn.BatchNorm()
        self.b = nn.Dense(1)

    def f(foo, x):
      x = foo.a(x)
      x = foo.bn(x, use_running_average=False)
      return foo.b(x)

    foo = Foo()
    x = jnp.ones((4,))
    f_init = nn.init_with_output(f, foo)
    y1, variables = f_init(random.key(0), x)
    foo_b = foo.bind(variables, mutable='batch_stats')
    y2 = f(foo_b, x)
    y3, new_state = nn.apply(f, foo, mutable='batch_stats')(variables, x)
    self.assertEqual(y1, y2)
    self.assertEqual(y2, y3)
    bs_1 = new_state['batch_stats']
    bs_2 = foo_b.variables['batch_stats']
    for x, y in zip(
      jax.tree_util.tree_leaves(bs_1), jax.tree_util.tree_leaves(bs_2)
    ):
      np.testing.assert_allclose(x, y)

  def test_unbind(self):
    class Foo(nn.Module):
      def setup(self):
        self.encoder = nn.Dense(4)
        self.decoder = nn.Dense(2)

      def __call__(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    foo = Foo()
    x = jnp.ones((2,))

    variables = foo.init(random.key(0), x)
    encoder, encoder_vars = foo.bind(variables).encoder.unbind()
    decoder, decoder_vars = foo.bind(variables).decoder.unbind()

    self.assertIsInstance(encoder, nn.Dense)
    self.assertEqual(encoder.features, 4)
    self.assertIsInstance(decoder, nn.Dense)
    self.assertEqual(decoder.features, 2)

    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda v1, v2: (v1 == v2).all(),
                variables['params']['encoder'],
                encoder_vars['params'],
            )
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda v1, v2: (v1 == v2).all(),
                variables['params']['decoder'],
                decoder_vars['params'],
            )
        )
    )

  def test_bind_unbind_equality(self):
    class Foo(nn.Module):
      sub_module: Any

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(2)(x)
        return self.sub_module(x)

    sub_module = Foo(nn.Dense(3))
    module = Foo(sub_module)
    x = jnp.ones((1, 2))
    variables = module.init(jax.random.PRNGKey(0), x)

    bound_module = module.bind(variables)
    self.assertTrue((module.apply(variables, x) == bound_module(x)).all())
    new_module, new_variables = bound_module.unbind()
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda v1, v2: (v1 == v2).all(), variables, new_variables
            )
        )
    )
    self.assertEqual(module, new_module)

  def test_passing_mutable_variables(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(2)(x)

    x = jnp.ones((3,))
    variables = Foo().init(random.key(0), x)
    y = Foo().apply(variables, x)
    self.assertEqual(y.shape, (2,))

  def test_super_compact(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(4)(x)

    class Bar(Foo):
      @nn.compact
      def __call__(self, x):
        y = super().__call__(x)
        return nn.Dense(3)(y)

    k = random.key(0)
    x = jnp.ones((4, 7))

    variables = Bar().init(k, x)
    shapes = jax.tree_util.tree_map(np.shape, variables['params'])
    self.assertEqual(
      shapes,
      {
        'Dense_0': {'kernel': (7, 4), 'bias': (4,)},
        'Dense_1': {'kernel': (4, 3), 'bias': (3,)},
      },
    )
    y = Bar().apply(variables, x)
    self.assertEqual(y.shape, (4, 3))

  def test_super_setup(self):
    class Foo(nn.Module):
      def setup(self):
        self.a = nn.Dense(4)

    class Bar(Foo):
      def setup(self):
        super().setup()
        self.b = nn.Dense(3)

      def __call__(self, x):
        y = self.a(x)
        return self.b(y)

    k = random.key(0)
    x = jnp.ones((4, 7))

    variables = Bar().init(k, x)
    y = Bar().apply(variables, x)
    self.assertEqual(y.shape, (4, 3))

  def test_freeze_attr(self):
    class Foo(NamedTuple):
      a: int
      b: int

    self.assertEqual(nn.module._freeze_attr([1, 2]), (1, 2))
    xs = nn.module._freeze_attr(Foo(1, 2))
    self.assertEqual(xs, (1, 2))
    self.assertEqual(
      type(xs), Foo
    )  # equality test for NamedTuple doesn't check class!

  def test_generic_multiple_inheritance(self):
    T = TypeVar('T')

    class MyComponent(nn.Module, Generic[T]):
      pass

    class MyModule(nn.Module):
      submodule: MyComponent[jnp.ndarray]

    class MyComponent2(Generic[T], nn.Module):
      pass

    class MyModule2(nn.Module):
      submodule: MyComponent2[jnp.ndarray]

  def test_jit_rng_equivalance(self):
    model = nn.fold_rngs(nn.Dense)(1, use_bias=False)
    jit_model = nn.jit(nn.Dense)(1, use_bias=False)
    param = model.init(random.key(0), np.ones((1, 1)))['params']['kernel']
    param_2 = jit_model.init(random.key(0), np.ones((1, 1)))['params']['kernel']
    self.assertEqual(param, param_2)

  def test_rng_reuse_after_rewind(self):
    class C(nn.Module):
      @nn.compact
      def __call__(self):
        # Some module that has dropouts in it, in general,
        # it does more than just dropout!
        return self.make_rng('dropout')

    class A(nn.Module):
      @nn.compact
      def __call__(self):
        # Some module that has dropouts in it, in general,
        # it does more than just dropout!
        return C()()

    class B(nn.Module):
      @nn.compact
      def __call__(self):
        a = A()
        x0 = a()
        x1 = a()
        return jnp.all(x0 == x1)

    k = random.key(0)
    rng_equals = B().apply({}, rngs={'dropout': k})
    self.assertFalse(rng_equals)

  def test_module_get_put_has_variable(self):
    class A(nn.Module):
      @nn.compact
      def __call__(self, x):
        self.put_variable('test_col', 'a', x)
        assert self.has_variable('test_col', 'a')
        return self.get_variable('test_col', 'a')

    class B(nn.Module):
      def __call__(self, x):
        self.put_variable('test_col', 'a', x)
        assert self.has_variable('test_col', 'a')
        return self.get_variable('test_col', 'a')

    class C(nn.Module):
      def setup(self):
        self.put_variable(
          'test_col',
          'a',
          jnp.ones(
            2,
          ),
        )
        assert self.has_variable('test_col', 'a')

      def __call__(self):
        return self.get_variable('test_col', 'a')

    x = jnp.ones((2,))

    y, vs = A().apply({}, x, mutable=['test_col'])
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(x, vs['test_col']['a'])

    y, vs = B().apply({}, x, mutable=['test_col'])
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(x, vs['test_col']['a'])

    y, vs = C().apply({}, mutable=['test_col'])
    np.testing.assert_array_equal(y, jnp.ones((2,)))
    np.testing.assert_array_equal(y, vs['test_col']['a'])

  def test_generic_module(self):
    # See https://github.com/google/flax/issues/1899
    T = TypeVar('T')

    class C(nn.Module, Generic[T]):
      def f(self, t: T) -> T:
        return t

    class D(nn.Module):
      def setup(self):
        unused_c = C[Any]()

      def __call__(self) -> None:
        pass

    rngs = {}
    D().init(rngs)

  def test_modifying_attribs_in_post_init(self):
    class Foo(nn.Module):
      love: int = 99

      def __post_init__(self):
        self.hate = 100 - self.love
        super().__post_init__()

    foo = Foo()
    self.assertEqual(foo.love, 99)
    self.assertEqual(foo.hate, 1)

    class Bar(nn.Module):
      love: int = 99

      def __post_init__(self):
        self.love = 101
        super().__post_init__()

    bar = Bar()
    self.assertEqual(bar.love, 101)

  def test_has_rng(self):
    class Foo(nn.Module):
      def __call__(self):
        return self.has_rng('bar')

    foo = Foo()
    with self.assertRaisesRegex(ValueError, 'RNGs.*unbound module'):
      foo()
    k = random.key(0)
    self.assertTrue(foo.apply({}, rngs={'bar': k}))
    self.assertFalse(foo.apply({}, rngs={'baz': k}))

  def test_is_initializing(self):
    class Foo(nn.Module):
      def __call__(self):
        return self.is_initializing()

    foo = Foo()
    k = random.key(0)
    self.assertTrue(foo.init_with_output(k)[0])
    self.assertFalse(foo.apply({}))

  def test_throws_invalid_instance_module_error(self):
    class B(nn.Module):
      @nn.compact
      def __call__(self, x):
        return x

    k = random.key(0)
    x = random.uniform(random.key(1), (2,))

    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.init(k, x)  # B is module class, not B() a module instance
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.init_with_output(k, x)
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.apply(
        {}, x
      )  # similar issue w. apply called on class instead of instance.
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.bind(
        {}, x
      )  # similar issue w. apply called on class instead of instance.

  def test_throws_incorrect_post_init_override_error(self):
    class A(nn.Module):
      x: float

      def __post_init__(self):
        self.x_square = self.x**2

      @nn.compact
      def __call__(self, input):
        return input + 3

    r = A(x=3)

    with self.assertRaises(errors.IncorrectPostInitOverrideError):
      r.init(jax.random.key(2), jnp.ones(3))

  def test_deepcopy_unspecified_parent(self):
    parent_parameter = inspect.signature(DummyModule).parameters['parent']
    unspecified_parent = parent_parameter.default

    self.assertIs(unspecified_parent, copy.copy(unspecified_parent))

    self.assertIs(unspecified_parent, copy.deepcopy(unspecified_parent))

  def test_type_hints(self):
    class Network(nn.Module):
      layers: int

    type_hints = get_type_hints(Network)
    self.assertEqual(type_hints['layers'], int)

  def test_incorrect_property(self):
    class Foo(nn.Module):
      @property
      def prop(self):
        return self.non_existent

      def __call__(self):
        return self.prop

    foo = Foo()
    with self.assertRaisesRegex(
      errors.DescriptorAttributeError, 'Trying to access a property that'
    ):
      foo.apply({})

  def test_custom_descriptor(self):
    class Descriptor:
      def __get__(self, obj, objtype=None):
        return 10

    class Foo(nn.Module):
      prop = Descriptor()

      def __call__(self):
        return self.prop

    foo = Foo()
    res = foo.apply({})
    self.assertEqual(res, 10)

  def test_custom_descriptor_error(self):
    class Descriptor:
      def __get__(self, obj, objtype=None):
        return obj.non_existent

    class Foo(nn.Module):
      prop = Descriptor()

      def __call__(self):
        return self.prop

    foo = Foo()
    with self.assertRaisesRegex(
      errors.DescriptorAttributeError, 'Trying to access a property that'
    ):
      foo.apply({})

  def test_nested_external_modules(self):
    class Baz(nn.Module):
      a: int

      def setup(self):
        self.b = self.param('b', lambda k: 2)

      def __call__(self, x):
        return x + self.a * self.b

    class Bar(nn.Module):
      baz: Baz

      def __call__(self, x):
        return self.baz(x)

    class Foo(nn.Module):
      def setup(self):
        self.bar = Bar(baz=Baz(a=1))

      def __call__(self, x):
        return self.bar.baz(x)

    module = Foo()
    y, variables = module.init_with_output(jax.random.key(0), 1)
    self.assertEqual(y, 3)

  def test_getattribute_triggers_setup(self):
    class B(nn.Module):
      def setup(self):
        self.p1 = self.param('p1', lambda k: jnp.ones((2,)))

      def fn1(self, x):
        return self.p1 + x

    class A(nn.Module):
      b: nn.Module

      def __call__(self, x):
        return self.b.fn1(x)

    a = A(b=B())
    k = random.key(0)
    x = jnp.zeros((2,))
    vs = nn.init(lambda a, x: a(x), a)(k, x)
    y = nn.apply(lambda a, x: a.b.fn1(x), a)(vs, x)
    np.testing.assert_array_equal(y, jnp.ones((2,)))

  def test_nested_sequential_in_call(self):
    class Foo(nn.Module):
      def setup(self):
        self.seq = nn.Sequential([nn.Dense(10) for i in range(10)])

      def __call__(self, x):
        # try calling only the first layer
        return self.seq.layers[0](x)

    module = Foo()
    variables = module.init(jax.random.key(0), jnp.ones((1, 10)))

  def test_setup_called_bounded_submodules(self):
    module = nn.Sequential(
      [
        nn.Sequential(
          [
            nn.Dense(2),
            nn.relu,
            nn.Dense(2),
          ]
        ),
        nn.relu,
        nn.Dense(2),
      ]
    )
    x = jnp.ones((1, 3))
    variables = module.init(jax.random.key(0), x)
    bound_module = module.bind(variables)

    self.assertIsNotNone(bound_module.layers[0].layers[0].scope)
    self.assertIsNotNone(bound_module.layers[0].layers[2].scope)
    self.assertIsNotNone(bound_module.layers[2].scope)

  def test_call_bounded_toplevel_mutable(self):
    class Bar(nn.Module):
      a: int

      def setup(self):
        self.b = self.param('b', lambda k: 1)

      def __call__(self, x):
        return x + self.a * self.b

    class Foo(nn.Module):
      bars: Sequence[Bar]

      def __call__(self, x):
        for bar in self.bars:
          x = bar(x)
        return x

    module = Foo(bars=[])
    module.bars = [Bar(a=1)]

    variables = module.init(jax.random.key(0), jnp.ones(()))
    bound_module = module.bind(variables)

    bar1 = bound_module.bars[0]
    self.assertIsNotNone(bar1.scope)

  def test_nested_init(self):
    class Baz(nn.Module):
      a: int

      def setup(self):
        self.b = self.param('b', lambda k: jnp.ones(()))

      def __call__(self, x):
        return x + self.a * self.b

    class Bar(nn.Module):
      baz: Baz

      def setup(self):
        a = 1

      def __call__(self, x):
        return self.baz(x)

    class Foo(nn.Module):
      def setup(self):
        self.bar: Bar = Bar(baz=Baz(a=1))

      def __call__(self, x):
        # y = self.bar(x)
        y, bar_vars = self.bar.init_with_output(jax.random.key(0), x)
        return y, bar_vars

    # create foo
    module = Foo()

    # run foo
    (y, bar_vars), variables = module.init_with_output(
      jax.random.key(0), jnp.ones(())
    )

    self.assertIn('params', bar_vars)

  def test_nested_shared(self):
    class Shared(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(1)(x)

    class Unshared(nn.Module):
      shared: nn.Module

      def __call__(self, x):
        return self.shared(x)

    class Super(nn.Module):
      a: nn.Module
      b: nn.Module

      def run_a(self, x):
        return self.a(x)

      def run_b(self, x):
        return self.b(x)

      def __call__(self, x):
        return self.a(x) + self.b(x)

    sh = Shared()
    a = Unshared(shared=sh)
    b = Unshared(shared=sh)
    module = Super(a=a, b=b)

    rng = jax.random.key(0)
    params = module.init(rng, jnp.ones(1))['params']

    module.apply({'params': params}, jnp.ones(1))  # works as expected
    module.apply(
      {'params': params}, jnp.ones(1), method='run_a'
    )  # works as expected
    module.apply(
      {'params': params}, jnp.ones(1), method='run_b'
    )  # ScopeParamNotFoundError: Could not find parameter named "kernel" in scope "/b/shared/Dense_0"

  def test_repr(self):
    class Base1(nn.Module):
      a: int

    class Base2(nn.Module):
      b: str

    class Foo(Base2, Base1):
      c: float

    module = Foo(a=1, b='ok', c=3.0)
    str_rep = repr(module)

    self.assertIn('a = 1', str_rep)
    self.assertIn("b = 'ok'", str_rep)
    self.assertIn('c = 3.0', str_rep)

  def test_repr_should_not_cause_setup(self):
    class MLP(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(1)(x)
        return repr(self)

    class Foo(nn.Module):
      a: float
      b: MLP

    scope = Scope({})
    module = Foo(parent=scope, a=1, b=MLP(parent=scope))
    str_rep = repr(module)
    self.assertIn('a = 1', str_rep)

    self.assertEqual(module._state.setup_called, nn.module.SetupState.NEW)
    # repr() on a module should not cause inadvertent setup of submodules
    # i.e. module.b._state.setup_called should remain nn.module.SetupState.NEW
    # and not nn.module.SetupState.DONE
    self.assertEqual(module.b._state.setup_called, nn.module.SetupState.NEW)

  def test_kw_only(self):
    def create_kw_layers():
      class BaseLayer(nn.Module, kw_only=True):
        base_multiplier: int | None = -1

      class ChildLayer(BaseLayer):
        child_multiplier: int  # Don't want to have to set a default argument!

        def __call__(self, x):
          return x * self.child_multiplier * self.base_multiplier

      return BaseLayer, ChildLayer

    if tuple(sys.version_info)[:3] < (3, 10, 0):
      with self.assertRaisesRegex(TypeError, 'not available before Py 3.10'):
        BaseLayer, ChildLayer = create_kw_layers()
    else:
      BaseLayer, ChildLayer = create_kw_layers()
      with self.assertRaisesRegex(TypeError, 'positional argument'):
        _ = BaseLayer(2)
      # Like in Python dataclass, `kw_only` is not inherited, so ChildLayer can
      # take positional arg. It takes BaseLayer's default kwargs though.
      np.testing.assert_equal(ChildLayer(8)(np.ones(10)), -8 * np.ones(10))

  def test_positional_cannot_be_kw_only(self):
    class Foo(nn.Module):
      a: int

    Foo(1)  # ok
    Foo(a=1)  # ok
    with self.assertRaisesRegex(
      TypeError, r'takes 2 positional arguments but 3 were'
    ):
      Foo(1, None)
    Foo(a=1, parent=None)  # type: ignore[call-arg]

  def test_module_path_empty(self):
    rngkey = jax.random.key(0)
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    m1 = DummyModule(parent=scope)

    self.assertEqual(m1.path, ())

    scope = Scope({}, {'params': rngkey}, mutable=['params'], path=['root'])
    m2 = DummyModule(parent=scope)

    self.assertEqual(m2.path, ('root',))

    m3 = DummyModule(parent=scope.rewound())

    self.assertEqual(m3.path, ('root',))

  def test_module_path_unbound_module_error(self):
    m1 = DummyModule()
    with self.assertRaisesRegex(ValueError, 'unbound module'):
      _ = m1.path

  def test_module_path_in_nested_module(self):
    module_paths = []
    debug_paths = []

    class A(nn.Module):
      def setup(self):
        self.b1 = B()
        self.b2 = B()
        self.c1 = C()

        module_paths.append(self.path)
        debug_paths.append(self.scope.debug_path)

      def __call__(self, x):
        return self.b1(x) + self.b2(x) + self.c1(x)

    class B(nn.Module):
      def setup(self):
        self.c1 = nn.remat(nn.remat(C))()
        self.c2 = C()

        module_paths.append(self.path)
        debug_paths.append(self.scope.debug_path)

      def __call__(self, x):
        return self.c1(x) + self.c2(x)

    class C(nn.Module):
      def setup(self):
        super().setup()
        if self.scope.__class__.__name__ != 'TestScope':
          module_paths.append(self.path)
          debug_paths.append(self.scope.debug_path)

      @nn.compact
      def __call__(self, x):
        return x

    a = A()
    k = random.key(0)
    x = random.uniform(random.key(42), (2,))
    _ = a.init(k, x)
    expected_module_paths = [
      (),
      ('b1',),
      ('b1', 'c1'),
      ('b1', 'c2'),
      ('b2',),
      ('b2', 'c1'),
      ('b2', 'c2'),
      ('c1',),
    ]
    expected_debug_paths = [
      (),
      ('b1',),
      ('b1', 'remat(remat(c1))'),
      ('b1', 'c2'),
      ('b2',),
      ('b2', 'remat(remat(c1))'),
      ('b2', 'c2'),
      ('c1',),
    ]

    self.assertEqual(module_paths, expected_module_paths)
    self.assertEqual(debug_paths, expected_debug_paths)

  def test_intercept_methods(self):
    mod = IdentityModule(parent=None)
    x = jnp.ones([])
    call_count = []

    def add_one_interceptor(f, args, kwargs, context):
      call_count.append(None)
      self.assertLen(dataclasses.fields(context), 3)
      self.assertIs(context.module, mod)
      self.assertEqual(context.method_name, '__call__')
      self.assertEqual(context.orig_method(3), 3)
      self.assertEqual(args, (x,))
      self.assertEmpty(kwargs)
      y = f(*args, **kwargs)
      return y + 1

    y1 = mod(x)
    with nn.intercept_methods(add_one_interceptor):
      y2 = mod(x)
    y3 = mod(x)

    self.assertLen(call_count, 1)
    self.assertEqual(y1, 1)
    self.assertEqual(y2, 2)
    self.assertEqual(y3, 1)

  def test_intercept_methods_compact(self):
    class CompactModule(nn.Module):
      @compact
      def __call__(self, x):
        return nn.Dense(2)(x)

    mod = CompactModule()
    x = jnp.ones(shape=(1, 3))
    variables = mod.init(jax.random.key(0), x)
    call_modules = []

    def log_interceptor(f, args, kwargs, context):
      call_modules.append(context.module)
      self.assertLen(dataclasses.fields(context), 3)
      self.assertEqual(context.method_name, '__call__')
      self.assertEqual(args, (x,))
      self.assertEmpty(kwargs)
      return f(*args, **kwargs)

    with nn.intercept_methods(log_interceptor):
      _ = mod.apply(variables, x)

    self.assertLen(call_modules, 2)
    self.assertIsInstance(call_modules[0], CompactModule)
    self.assertIsInstance(call_modules[1], nn.Dense)

  def test_intercept_methods_setup(self):
    class SetupModule(nn.Module):
      def setup(self):
        self.layer = nn.Dense(2)

      def __call__(self, x):
        return self.layer(x)

    mod = SetupModule()
    x = jnp.ones(shape=(1, 3))
    variables = mod.init(jax.random.key(0), x)
    call_modules = []
    log = []

    def log_interceptor(f, args, kwargs, context):
      call_modules.append(context.module)
      log.append((context.method_name, args, kwargs))
      return f(*args, **kwargs)

    with nn.intercept_methods(log_interceptor):
      _ = mod.apply(variables, x)

    self.assertLen(call_modules, 3)
    self.assertIsInstance(call_modules[0], SetupModule)
    self.assertIsInstance(call_modules[1], SetupModule)
    self.assertIsInstance(call_modules[2], nn.Dense)
    self.assertEqual(
      log, [('setup', (), {}), ('__call__', (x,), {}), ('__call__', (x,), {})]
    )

  def test_intercept_methods_calling_underlying_optional(self):
    def do_nothing_interceptor(f, args, kwargs, context):
      del f, context
      self.assertEmpty(args)
      self.assertEmpty(kwargs)

    m = RaisesModule()
    with nn.intercept_methods(do_nothing_interceptor):
      m()

    with self.assertRaises(AssertionError):
      m()

    with nn.intercept_methods(do_nothing_interceptor):
      m()

  def test_intercept_methods_run_in_lifo_order(self):
    def op_interceptor(op):
      def _interceptor(f, args, kwargs, context):
        del context
        y = f(*args, **kwargs)
        return op(y)

      return _interceptor

    mod = IdentityModule(parent=None)
    x = 7
    with (
      nn.intercept_methods(op_interceptor(lambda a: a + 1)),
      nn.intercept_methods(op_interceptor(lambda a: a**2)),
    ):
      y = mod(x)

    self.assertEqual(y, (x**2) + 1)

    with (
      nn.intercept_methods(op_interceptor(lambda a: a**2)),
      nn.intercept_methods(op_interceptor(lambda a: a + 1)),
    ):
      y = mod(x)

    self.assertEqual(y, (x + 1) ** 2)

  def test_intercept_methods_subclasses(self):
    class Foo(IdentityModule):
      def __call__(self, x):  # pylint: disable=useless-parent-delegation
        return super().__call__(x)

    class Bar(Foo):
      def __call__(self, x):  # pylint: disable=useless-parent-delegation
        return super().__call__(x)

    bar = Bar(parent=None)
    x = jnp.ones([])
    called = []

    def record_interceptor(f, args, kwargs, context):
      called.append(None)
      self.assertIs(context.module, bar)
      self.assertEqual(context.method_name, '__call__')
      self.assertEqual(args, (x,))
      self.assertEmpty(kwargs)
      return f(*args, **kwargs)

    with nn.intercept_methods(record_interceptor):
      bar(x)

    # Bar.__call__, Foo.__call__ and IdenityModule.__call__
    self.assertLen(called, 3)

  def test_intercept_methods_nested_module(self):
    class Foo(nn.Module):
      def __call__(self, x):
        return x

    class Bar(nn.Module):
      sub: nn.Module

      def __call__(self, x):
        return self.sub(x)

    foo = Foo()
    bar = Bar(sub=foo)
    x = jnp.ones([])
    called = []

    def record_interceptor(f, args, kwargs, context):
      called.append(context.module)
      self.assertEqual(context.method_name, '__call__')
      self.assertEqual(args, (x,))
      self.assertEmpty(kwargs)
      return f(*args, **kwargs)

    with nn.intercept_methods(record_interceptor):
      bar(x)

    # bar.__call__ and foo.__call__
    self.assertLen(called, 2)
    self.assertIs(called[0], bar)
    self.assertIs(called[1], foo)

  def test_cloudpickle_class(self):
    import cloudpickle

    class MyModule(nn.Module):
      pass

    a = MyModule()

    UnpickledMyModule = cloudpickle.loads(cloudpickle.dumps(MyModule))
    b = UnpickledMyModule()

  def test_cloudpickle_module(self):
    from cloudpickle import cloudpickle_fast

    class NNModuleWithProperty(nn.Module):
      a: int
      b: str

      @property
      def my_property(self):
        return self.b * self.a

    m = NNModuleWithProperty(a=2, b='ok')

    with TemporaryDirectory() as tmpdir:
      filename = f'{tmpdir}/module.pkl'
      with open(filename, 'wb') as f:
        cloudpickle_fast.dump(m, f)

      with open(filename, 'rb') as f:
        obj_loaded = cloudpickle_fast.load(f)

    self.assertEqual(obj_loaded.a, 2)
    self.assertEqual(obj_loaded.b, 'ok')
    self.assertEqual(obj_loaded.my_property, 'okok')

  def test_module_paths(self):
    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(3)(x)
        x = nn.Dense(4)(x)
        return x

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = Bar()(x)
        x = nn.Dense(5)(x)
        return x

    x = jnp.ones((1, 2))
    m = Foo()
    module_paths = m.module_paths(random.key(0), x)

    # assert all module are unbounded
    for module in module_paths.values():
      self.assertIsNone(module.scope)

    # test paths
    self.assertIn('', module_paths)
    self.assertEqual(type(module_paths['']), Foo)
    self.assertIn('Dense_0', module_paths)
    self.assertEqual(type(module_paths['Dense_0']), nn.Dense)
    self.assertIn('Bar_0', module_paths)
    self.assertEqual(type(module_paths['Bar_0']), Bar)
    self.assertIn('Bar_0/Dense_0', module_paths)
    self.assertEqual(type(module_paths['Bar_0/Dense_0']), nn.Dense)
    self.assertIn('Bar_0/Dense_1', module_paths)
    self.assertEqual(type(module_paths['Bar_0/Dense_1']), nn.Dense)

  def test_init_apply_default_rng(self):
    class SubModel(nn.Module):
      @nn.compact
      def __call__(self, x, apply_dropout):
        x = nn.Dense(8)(x)
        x = nn.Dropout(0.8)(x, deterministic=not apply_dropout)
        p = self.param(
          'parameter', lambda key, shape: jax.random.normal(key, shape), x.shape
        )
        noise = jax.random.normal(self.make_rng('noise'), x.shape)
        return x * p + noise

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x, apply_dropout):
        x = nn.Dense(16)(x)
        x = SubModel()(x, apply_dropout)
        x = nn.Dropout(0.5)(x, deterministic=not apply_dropout)
        v = self.variable(
          'var_collection',
          'variable',
          lambda shape: jax.random.normal(self.make_rng('var_rng'), shape),
          x.shape,
        )
        noise = jax.random.normal(self.make_rng('noise'), x.shape)
        return x * v.value + noise

    key0, key1, key2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(key0, (10, 4))
    model = Model()

    # test init equality
    default_variables = model.init({'params': key1}, x, apply_dropout=False)
    rngs = {'params': key1, 'var_rng': key1, 'noise': key1}
    explicit_variables = model.init(rngs, x, apply_dropout=False)
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda v1, v2: (v1 == v2).all(),
                default_variables,
                explicit_variables,
            )
        )
    )

    # test init inequality
    for rng_name in ('params', 'var_rng'):
      rngs[rng_name] = key2
      explicit_variables = model.init(rngs, x, apply_dropout=False)
      self.assertFalse(
          jax.tree_util.tree_all(
              jax.tree_util.tree_map(
                  lambda v1, v2: (v1 == v2).all(),
                  default_variables,
                  explicit_variables,
              )
          )
      )
      rngs[rng_name] = key1

    # test apply equality
    default_out = model.apply(
      default_variables, x, apply_dropout=True, rngs={'params': key1}
    )
    rngs = {'dropout': key1, 'noise': key1}
    explicit_out = model.apply(
      default_variables, x, apply_dropout=True, rngs=rngs
    )
    np.testing.assert_allclose(default_out, explicit_out)

    # test apply inequality
    for rng_name in ('dropout', 'noise'):
      rngs[rng_name] = key2
      explicit_out = model.apply(
        default_variables, x, apply_dropout=True, rngs=rngs
      )
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(default_out, explicit_out, atol=1e-1)
      rngs[rng_name] = key1

  def test_default_make_rng(self):
    class SubModel(nn.Module):
      @nn.compact
      def __call__(self, x):
        noise = jax.random.normal(self.make_rng(), x.shape)
        return x + noise

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = SubModel()(x)
        noise = jax.random.normal(self.make_rng(), x.shape)
        return x + noise

    key0, key1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(key0, (10, 4))
    default_out = Model().apply({}, x, rngs={'params': key1})

    class SubModel(nn.Module):
      @nn.compact
      def __call__(self, x):
        noise = jax.random.normal(self.make_rng('params'), x.shape)
        return x + noise

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = SubModel()(x)
        noise = jax.random.normal(self.make_rng('params'), x.shape)
        return x + noise

    explicit_out = Model().apply({}, x, rngs={'params': key1})
    np.testing.assert_allclose(default_out, explicit_out)

  def test_default_rng_error(self):
    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.Dense(2)(x)

    model = Model()
    with self.assertRaisesRegex(
      errors.InvalidRngError, 'Dense_0 needs PRNG for "params"'
    ):
      model.init({'other_rng_stream': jax.random.key(0)}, jnp.ones((1, 3)))

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        return x + jax.random.normal(self.make_rng(), x.shape)

    model = Model()
    with self.assertRaisesRegex(
      errors.InvalidRngError, 'None needs PRNG for "params"'
    ):
      model.init({'other_rng_stream': jax.random.key(0)}, jnp.ones((1, 3)))

  def test_compact_name_scope(self):
    class Foo(nn.Module):
      @nn.compact_name_scope
      def up(self, x):
        return nn.Dense(3)(x)

      @nn.compact_name_scope
      def down(self, x):
        return nn.Dense(3)(x)

      @nn.compact
      def __call__(self, x):
        return self.up(x) + self.down(x) + nn.Dense(3)(x)

    m = Foo()
    x = jnp.ones((1, 2))

    self.assertEqual(set(m._compact_name_scope_methods), {'up', 'down'})

    variables = m.init(random.key(0), x)
    params = variables['params']

    self.assertIn('Dense_0', params)
    self.assertIn('down', params)
    self.assertIn('up', params)
    self.assertIn('Dense_0', params['down'])
    self.assertIn('Dense_0', params['up'])

    y = m.apply(variables, x)
    y_up = m.apply(variables, x, method='up')
    y_down = m.apply(variables, x, method='down')

    assert y.shape == (1, 3)
    assert y_up.shape == (1, 3)
    assert y_down.shape == (1, 3)

  def test_compact_name_scope_outside_compact(self):
    class Foo(nn.Module):
      @nn.compact_name_scope
      def up(self, x):
        return nn.Dense(3)(x)

      @nn.compact_name_scope
      def down(self, x):
        return nn.Dense(3)(x)

      def __call__(self, x):
        return self.up(x) + self.down(x)

    m = Foo()
    x = jnp.ones((1, 2))

    self.assertEqual(set(m._compact_name_scope_methods), {'up', 'down'})

    variables = m.init(random.key(0), x)
    params = variables['params']

    self.assertIn('down', params)
    self.assertIn('up', params)
    self.assertIn('Dense_0', params['down'])
    self.assertIn('Dense_0', params['up'])

    y = m.apply(variables, x)
    y_up = m.apply(variables, x, method='up')
    y_down = m.apply(variables, x, method='down')

    assert y.shape == (1, 3)
    assert y_up.shape == (1, 3)
    assert y_down.shape == (1, 3)

  def test_split_basic(self):
    class Foo(nn.Module):
      a: int

    m = Foo(a=1)

    m = m.bind({})

    moduledef, variables, rngs = m.split()
    moduledef

  def test_split_basic_with_variables(self):
    m = nn.Dense(3)
    variables = m.init(random.key(0), jnp.ones((1, 2)))

    m = m.bind(variables)

    moduledef, variables, rngs = m.split()

    self.assertIn('params', variables)
    self.assertIn('kernel', variables['params'])

  def test_split_with_submodules(self):
    m = nn.Sequential([nn.Dense(3)])
    variables = m.init(random.key(0), jnp.ones((1, 2)))

    m = m.bind(variables)

    moduledef, variables, rngs = m.split()
    moduledef.state

    self.assertIn('params', variables)
    self.assertIn('layers_0', variables['params'])
    self.assertIn('kernel', variables['params']['layers_0'])


class LeakTests(absltest.TestCase):
  def test_tracer_leaks(self):
    model = nn.Sequential([nn.Dense(50)])

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def sample_from_prior(rng, inp):
      params = model.init(rng, np.zeros((10, 50)))
      out = model.apply(params, inp)
      del params
      return out

    # disable manual gc.collect call in jax leak checker
    # so that we can test tracer leaks in ref-cycles.  This is a
    # reasonable proxy for transiently leaked memory during
    # eager execution.
    with patch.object(gc, 'collect', return_value=0):
      with jax.checking_leaks():
        for i in range(5):
          rngs = jax.random.split(jax.random.key(23), 100)
          out = sample_from_prior(rngs, np.ones((4, 50)))
          out.block_until_ready()
          del out, rngs


class RelaxedNamingTests(absltest.TestCase):
  def test_relaxed_adoption(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('p', nn.initializers.zeros, x.shape)
        return x + p

    class Bar(nn.Module):
      sub: nn.Module

      def __call__(self, x):
        return self.sub(x)

    with set_config('flax_preserve_adopted_names', True):
      foo = Foo(name='foo')
      bar = Bar(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('foo' in vs['params'], 'relaxed naming failure')
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('sub' in vs['params'], 'old policy naming failure')
      y = bar.apply(vs, x)

  def test_class_optional_adoption_name_preservation(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('p', nn.initializers.zeros, x.shape)
        return x + p

    class Bar1(nn.Module):
      sub: nn.Module
      preserve_adopted_names = True

      def __call__(self, x):
        return self.sub(x)

    class Bar2(nn.Module):
      sub: nn.Module
      preserve_adopted_names = False

      def __call__(self, x):
        return self.sub(x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar1(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('foo' in vs['params'], 'adoption naming failure')
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', True):
      foo = Foo(name='foo')
      bar = Bar2(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('sub' in vs['params'], 'adoption naming failure')
      y = bar.apply(vs, x)

  def test_nested_class_optional_adoption_name_preservation(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('p', nn.initializers.zeros, x.shape)
        return x + p

    class Bar(nn.Module):
      sub: nn.Module
      preserve_adopted_names = True

      def __call__(self, x):
        return self.sub(x)

    class Baz(nn.Module):
      sub: nn.Module
      preserve_adopted_names = True

      def __call__(self, x):
        return self.sub(x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar(sub=foo, name='bar')
      baz = Baz(sub=bar)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = baz.init(k, x)
      self.assertTrue('bar' in vs['params'], 'adoption naming failure')
      self.assertTrue('foo' in vs['params']['bar'], 'adoption naming failure')
      y = baz.apply(vs, x)

  def test_relaxed_adoption_still_conflict_checks(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('p', nn.initializers.zeros, x.shape)
        return x + p

    class Bar(nn.Module):
      sub1: nn.Module
      sub2: nn.Module

      def __call__(self, x):
        return self.sub(x)

    with set_config('flax_preserve_adopted_names', True):
      foo1 = Foo(name='foo')
      foo2 = Foo(name='foo')
      bar = Bar(sub1=foo1, sub2=foo2)
      k = random.key(0)
      x = jnp.zeros((1,))
      with self.assertRaises(errors.NameInUseError):
        vs = bar.init(k, x)

  def test_relaxed_adoption_unnamed_adoptee(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        p = self.param('p', nn.initializers.zeros, x.shape)
        return x + p

    class Bar(nn.Module):
      sub: nn.Module

      def __call__(self, x):
        return self.sub(x)

    with set_config('flax_preserve_adopted_names', True):
      foo = Foo(name=None)
      bar = Bar(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('sub' in vs['params'], 'relaxed naming failure')
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar(sub=foo)
      k = random.key(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue('sub' in vs['params'], 'old policy naming failure')
      y = bar.apply(vs, x)

  def test_relaxed_python_conflict(self):
    class Foo(nn.Module):
      dummy = 0

      @nn.compact
      def __call__(self, x):
        p = self.param('dummy', nn.initializers.zeros, x.shape)
        return x + p

    foo = Foo(name='foo')
    k = random.key(0)
    x = jnp.zeros((1,))
    vs = foo.init(k, x)

  def test_relaxed_intercollection_conflict(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        v1 = self.variable('col1', 'v', lambda x: jnp.zeros(x), x.shape)
        v2 = self.variable('col2', 'v', lambda x: jnp.zeros(x), x.shape)
        return x + v1.value + v2.value

    foo = Foo(name='foo')
    k = random.key(0)
    x = jnp.zeros((1,))
    vs = foo.init(k, x)

  def test_relaxed_intercollection_conflict_set(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        v1 = self.variable('col1', 'v', lambda x: jnp.zeros(x), x.shape)
        v2 = self.variable('col2', 'v', lambda x: jnp.zeros(x), x.shape)
        v3 = self.variable('col1', 'v', lambda x: jnp.zeros(x), x.shape)
        return x + v1.value + v2.value + v3.value

    foo = Foo(name='foo')
    k = random.key(0)
    x = jnp.zeros((1,))
    with self.assertRaises(errors.NameInUseError):
      vs = foo.init(k, x)

  def test_internal_deep_clone(self):
    class Child(nn.Module):
      @nn.compact
      def __call__(self, x):
        w = self.param('w', nn.initializers.zeros, (5, x.shape[1]))
        return x @ w

    class Parent(nn.Module):
      num_layers: int
      child_template: Child

      @nn.compact
      def __call__(self, x):
        for i in range(self.num_layers):
          x = self.child_template.clone(
            parent=self, _deep_clone=True, name=None
          )(x)
        return x

    model = Parent(num_layers=2, child_template=Child())
    x = jnp.ones((32, 5))
    variables = model.init(jax.random.key(0), x)
    output = model.apply(variables, x)
    self.assertTrue(
      variables['params']['Child_0']['w'].shape
      == variables['params']['Child_1']['w'].shape
    )

  def test_copy_method(self):
    class Parent(nn.Module):
      @nn.compact
      def __call__(self, x):
        child = nn.Dense(
          2,
        )
        x = child(x)
        x = child.copy()(x)
        return x

    model = Parent()
    x = jnp.ones((2, 2))
    variables = model.init(jax.random.key(0), x)
    output = model.apply(variables, x)
    self.assertTrue(
      variables['params']['Dense_0']['kernel'].shape
      == variables['params']['Dense_1']['kernel'].shape
    )

  def test_copy_from_template(self):
    class Child(nn.Module):
      @nn.compact
      def __call__(self, x):
        w = self.param('w', nn.initializers.zeros, (5, x.shape[1]))
        return x @ w

    class Parent(nn.Module):
      num_layers: int
      child_template: Child

      @nn.compact
      def __call__(self, x):
        for i in range(self.num_layers):
          x = self.child_template.copy()(x)
        for i in range(self.num_layers):
          x = self.child_template.copy(name=f'next_layer_{i}')(x)
        return x

    model = Parent(num_layers=2, child_template=Child())
    x = jnp.ones((32, 5))
    variables = model.init(jax.random.key(0), x)
    output = model.apply(variables, x)
    self.assertTrue(
      variables['params']['Child_0']['w'].shape
      == variables['params']['Child_1']['w'].shape
    )
    self.assertIn('Child_0', variables['params'])
    self.assertIn('Child_1', variables['params'])
    self.assertIn('next_layer_0', variables['params'])
    self.assertIn('next_layer_1', variables['params'])
    self.assertNotIn('child_template', variables['params'])

  def test_nonstring_keys_in_dict_on_module(self):
    class MyEnum(str, enum.Enum):
      a = 'a'
      b = 'b'

    class MyModule(nn.Module):
      config: dict[MyEnum, int]

      def __call__(self, inputs):
        return inputs

    module = MyModule(config={MyEnum.a: 1, MyEnum.b: 2})
    variables = module.init(jax.random.key(0), jnp.zeros([0]))


class FrozenDictTests(absltest.TestCase):
  def test_frozendict_flag(self):
    with set_config('flax_return_frozendict', True):
      x = jnp.zeros((2, 3))
      layer = nn.Dense(5)
      params = layer.init(random.key(0), x)
      self.assertTrue(isinstance(params, FrozenDict))

    with set_config('flax_return_frozendict', False):
      x = jnp.zeros((2, 3))
      layer = nn.Dense(5)
      params = layer.init(random.key(0), x)
      self.assertTrue(isinstance(params, dict))


class ShareScopeTest(absltest.TestCase):
  def test_basic(self):
    class DenseLoRA(nn.Module):
      inner: nn.Dense
      rank: int

      def setup(self):
        nn.share_scope(self, self.inner)

      @nn.compact
      def __call__(self, x: jax.Array):
        din, dout = x.shape[-1], self.inner.features
        A = self.param('A', nn.zeros_init(), (din, self.rank))
        B = self.param('B', nn.zeros_init(), (self.rank, dout))
        return self.inner(x) + x @ A @ B

    dense_lora = DenseLoRA(nn.Dense(10), rank=2)

    params = dense_lora.init(random.key(0), jnp.ones((1, 5)))['params']

    self.assertIn('kernel', params)
    self.assertIn('bias', params)
    self.assertIn('A', params)
    self.assertIn('B', params)

  def test_child_scope(self):
    class DenseLoRA(nn.Module):
      rank: int

      def setup(self):
        self.child = nn.Dense(10)
        nn.share_scope(self, self.child)

      @nn.compact
      def __call__(self, x: jax.Array):
        din, dout = x.shape[-1], self.child.features
        A = self.param('A', nn.zeros_init(), (din, self.rank))
        B = self.param('B', nn.zeros_init(), (self.rank, dout))
        return self.child(x) + x @ A @ B

    dense_lora = DenseLoRA(rank=2)

    params = dense_lora.init(random.key(0), jnp.ones((1, 5)))['params']

    self.assertIn('kernel', params)
    self.assertIn('bias', params)
    self.assertIn('A', params)
    self.assertIn('B', params)

  def test_in_compact(self):
    class DenseLoRA(nn.Module):
      rank: int

      def setup(self):
        self.child = nn.Dense(10)
        nn.share_scope(self, self.child)

      @nn.compact
      def __call__(self, x: jax.Array):
        din, dout = x.shape[-1], self.child.features
        A = self.param('A', nn.zeros_init(), (din, self.rank))
        B = self.param('B', nn.zeros_init(), (self.rank, dout))
        return self.child(x) + x @ A @ B

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x: jax.Array):
        return DenseLoRA(rank=2)(x)

    model = Model()

    params = model.init(random.key(0), jnp.ones((1, 5)))['params']

    self.assertIn('kernel', params['DenseLoRA_0'])
    self.assertIn('bias', params['DenseLoRA_0'])
    self.assertIn('A', params['DenseLoRA_0'])
    self.assertIn('B', params['DenseLoRA_0'])

  def test_adopt_child_name(self):
    class DenseLoRA(nn.Module):
      inner: nn.Dense
      rank: int

      def setup(self):
        nn.share_scope(self, self.inner)

      @nn.compact
      def __call__(self, x: jax.Array):
        din, dout = x.shape[-1], self.inner.features
        A = self.param('A', nn.zeros_init(), (din, self.rank))
        B = self.param('B', nn.zeros_init(), (self.rank, dout))
        return self.inner(x) + x @ A @ B

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x: jax.Array):
        return DenseLoRA(nn.Dense(10), rank=2)(x)

    model = Model()

    params = model.init(random.key(0), jnp.ones((1, 5)))['params']

    self.assertIn('kernel', params['Dense_0'])
    self.assertIn('bias', params['Dense_0'])
    self.assertIn('A', params['Dense_0'])
    self.assertIn('B', params['Dense_0'])

  def test_other_scope_is_none(self):
    class DenseLoRA(nn.Module):
      inner: nn.Dense
      rank: int

      def setup(self):
        nn.share_scope(self, self.inner)

      @nn.compact
      def __call__(self, x: jax.Array):
        din, dout = x.shape[-1], self.inner.features
        A = self.param('A', nn.zeros_init(), (din, self.rank))
        B = self.param('B', nn.zeros_init(), (self.rank, dout))
        return self.inner(x) + x @ A @ B

    class Model(nn.Module):
      def setup(self):
        # here Dense doesn't have a scope yet
        self.dense_lora = DenseLoRA(nn.Dense(10), rank=2)

      @nn.compact
      def __call__(self, x: jax.Array):
        return self.dense_lora(x)

    model = Model()

    params = model.init(random.key(0), jnp.ones((1, 5)))['params']

    self.assertIn('kernel', params['dense_lora'])
    self.assertIn('bias', params['dense_lora'])
    self.assertIn('A', params['dense_lora'])
    self.assertIn('B', params['dense_lora'])

  def test_external_grandchild_scope_correct(self):
    class GrandChild(nn.Module):
      @nn.compact
      def __call__(self):
        return nn.Dense(50)(jnp.zeros(10))

    class Child(nn.Module):
      child: GrandChild

      @nn.compact
      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.child(*args, **kwargs)

    class Parent(nn.Module):
      main_child: Child

      def setup(self):
        nn.share_scope(self, self.main_child)

      @nn.compact
      def __call__(self, *args: Any, **kwargs: Any) -> Any:
        nn.Dense(10)(jnp.zeros(10))
        r = self.main_child(*args, **kwargs)
        return r

    params = Parent(Child(GrandChild())).init(jax.random.key(0))
    self.assertNotIn('main_child', params['params'])
    self.assertIn('child', params['params'])

if __name__ == '__main__':
  absltest.main()

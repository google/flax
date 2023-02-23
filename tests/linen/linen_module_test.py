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

"""Tests for flax.linen."""

import contextlib
import copy
import dataclasses
import functools
import gc
import inspect
import operator
from typing import (Any, Callable, Generic, Mapping, NamedTuple, Sequence,
                    Tuple, TypeVar, get_type_hints)

from absl.testing import absltest
from flax import config
from flax import errors
from flax import linen as nn
from flax import struct
from flax.core import Scope, freeze, FrozenDict, tracers
from flax.linen import compact
from flax.configurations import use_regular_dict
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

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
    kernel = self.param('kernel', initializers.lecun_normal(),
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

  def test_lazy_init(self):

    class Foo(nn.Module):
      @compact
      def __call__(self, x):
        k = self.param("kernel", nn.initializers.lecun_normal(), (x.shape[-1], x.shape[-1]))
        return x @ k
    # provide a massive input message which would OOM if any compute ops were actually executed
    variables = Foo().lazy_init(random.PRNGKey(0), jax.ShapeDtypeStruct((1024 * 1024 * 1024, 128), jnp.float32))
    self.assertEqual(variables["params"]["kernel"].shape, (128, 128))

  def test_lazy_init_fails_on_data_dependence(self):
    class Foo(nn.Module):
      @compact
      def __call__(self, x):
        k = self.param("kernel", lambda _: x)
        return x * k

    with self.assertRaises(errors.LazyInitError):
      Foo().lazy_init(random.PRNGKey(0), jax.ShapeDtypeStruct((8, 4), jnp.float32))

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
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(param_shape, {
        'Dense_0': {
            'kernel': (10, 3)
        },
        'Dense_1': {
            'kernel': (3, 3)
        }
    })

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
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(param_shape, {
        'MLP_0': {
            'Dense_0': {
                'kernel': (10, 3)
            },
            'Dense_1': {
                'kernel': (3, 3)
            }
        }
    })

  def test_setup_dict_assignment(self):
    rngkey = jax.random.PRNGKey(0)

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
    self.assertEqual(param_shape, {
        'lyrs1_a': {
            'kernel': (10, 3)
        },
        'lyrs1_b': {
            'kernel': (3, 3)
        }
    })

  def test_setup_dict_nonstring_keys(self):

    class Foo(nn.Module):

      def setup(self):
        self.a = {(1, 2): nn.Dense(2)}  # Tuple as key.

      @nn.compact
      def __call__(self, x):
        return self.a[(1, 2)](x)

    foo = Foo()
    x = jnp.ones(shape=(1, 3))
    params = foo.init(random.PRNGKey(0), x)['params']
    param_shape = jax.tree_util.tree_map(jnp.shape, params)
    self.assertEqual(param_shape,
                     {'a_(1, 2)': {
                         'kernel': (3, 2),
                         'bias': (2,)
                     }})

  def test_setup_cloning(self):

    class MLP(nn.Module):

      def setup(self):
        self.dense = Dense(3)

    scope = Scope({})
    unused_clone = MLP(parent=scope).clone()

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

    class DummyModuleWithoutCompact(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = DummyModuleWithoutCompact(x.shape, parent=scope)(x)
    params = scope.variables()['params']
    y2 = DummyModuleWithoutCompact(x.shape, parent=scope.rewound())(x)
    np.testing.assert_allclose(y, y2)
    np.testing.assert_allclose(y, jnp.array([2.]))
    self.assertEqual(params, {'bias': jnp.array([1.])})

  def test_init_outside_setup_without_compact(self):
    rngkey = jax.random.PRNGKey(0)

    class DummyModuleWithoutCompact(nn.Module):

      def __call__(self, x):
        bias = self.param('bias', initializers.ones, x.shape)
        return x + bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    with self.assertRaisesRegex(ValueError, 'must be initialized.*setup'):
      unused_y = DummyModuleWithoutCompact(parent=scope)(x)

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
      unused_y = Dummy(parent=scope).foo(x)

  def test_setup_call_var_collision(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      @compact
      def __call__(self, x):
        unused_bias = self.param('bias', initializers.ones, x.shape)
        return x + self.bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_call_var_collision(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      @compact
      def __call__(self, x):
        bias = self.param('bias', initializers.ones, self.xshape)
        bias = self.param('bias', initializers.ones, self.xshape)
        return x + bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_setup_var_collision(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = self.param('bias', initializers.ones, self.xshape)

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_setattr_name_var_disagreement_allowed_in_lists(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.biases = [
            self.param(f'bias_{i}', initializers.ones, self.xshape)
            for i in range(4)
        ]

      def __call__(self, x):
        return x + self.biases[0]

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.]))

  def test_setattr_name_var_disagreement_allowed_in_dicts(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

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

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])
    y = Dummy(x.shape, parent=scope)(x)
    self.assertEqual(y, jnp.array([2.]))

  def test_submodule_var_collision_with_scope(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)
        self.bias = DummyModule()

      def __call__(self, x):
        return x + self.bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])

    if config.flax_relaxed_naming:
      with self.assertRaises(errors.NameInUseError):
        unused_y = Dummy(x.shape, parent=scope)(x)
    else:
      msg = 'Duplicate use of scope name: "bias"'
      with self.assertRaisesWithLiteralMatch(ValueError, msg):
        unused_y = Dummy(x.shape, parent=scope)(x)

  def test_submodule_var_collision_with_submodule(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = self.param('bias', initializers.ones, self.xshape)

      @compact
      def __call__(self, x):
        unused_bias = DummyModule(name='bias')
        return x + self.bias

    x = jnp.array([1.])
    scope = Scope({}, {'params': rngkey}, mutable=['params'])

    msg = 'Could not create submodule "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(x.shape, parent=scope)(x)

  def test_submodule_var_collision_with_params(self):
    rngkey = jax.random.PRNGKey(0)

    class Dummy(nn.Module):
      xshape: Tuple[int, ...]

      def setup(self):
        self.bias = DummyModule()

      @compact
      def __call__(self, x):
        unused_bias = self.param('bias', initializers.ones, self.xshape)
        return x + self.bias

    x = jnp.array([1.])
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

  @absltest.skipIf(config.flax_relaxed_naming, "relaxed naming")
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
    msg = 'Could not create param "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(True, parent=scope)(x)

  @absltest.skipIf(config.flax_relaxed_naming, "relaxed naming")
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
    msg = 'Could not create submodule "bias" in Module Dummy: Name in use'
    with self.assertRaisesRegex(errors.NameInUseError, msg):
      unused_y = Dummy(True, parent=scope)(x)

  def test_only_one_compact_method(self):
    msg = 'Only one method per class can be @compact'
    with self.assertRaisesRegex(errors.MultipleMethodsCompactError, msg):

      class MultipleCompactMethods(nn.Module):

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

    msg = (r'Submodule Dense must be defined in `setup\(\)` or in a method '
           'wrapped in `@compact`')
    with self.assertRaisesRegex(errors.AssignSubModuleError, msg):
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

    msg = (r'Submodule Dense must be defined in `setup\(\)` or in a method '
           'wrapped in `@compact`')
    with self.assertRaisesRegex(errors.AssignSubModuleError, msg):
      Foo().init(random.PRNGKey(0), jnp.ones((1, 3)))

  def test_numpy_array_shape_class_args(self):

    class MLP(nn.Module):
      widths: Sequence[int]

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

    class Test4(Test2):

      def __call__(self, x):
        return x

    key = random.PRNGKey(0)
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
        list(Test.__dataclass_fields__.keys()), ['bar', 'parent', 'name'])
    self.assertEqual(
        list(Test2.__dataclass_fields__.keys()),
        ['bar', 'baz', 'parent', 'name'])
    self.assertEqual(
        list(Test3.__dataclass_fields__.keys()),
        ['bar', 'baz', 'parent', 'name'])
    self.assertEqual(
        list(Test4.__dataclass_fields__.keys()),
        ['bar', 'baz', 'parent', 'name'])

  def test_get_suffix_value_pairs(self):
    for x in [(), [], {}, None, 0, set()]:
      self.assertEqual(nn.module._get_suffix_value_pairs(x), [('', x)])
    self.assertEqual(
        nn.module._get_suffix_value_pairs({
            'a': 1,
            'b': 2
        }), [('_a', 1), ('_b', 2)])
    self.assertEqual(
        nn.module._get_suffix_value_pairs([1, 2, 3]), [('_0', 1), ('_1', 2),
                                                       ('_2', 3)])
    x1 = [nn.Dense(10), nn.relu, nn.Dense(10)]
    y1 = nn.module._get_suffix_value_pairs(x1)
    self.assertEqual(y1, [('_0', x1[0]), ('_1', x1[1]), ('_2', x1[2])])
    x2 = {'a': 1, 'b': {'c': nn.Dense(10), 'd': nn.relu}}
    y2 = nn.module._get_suffix_value_pairs(x2)
    self.assertEqual(y2, [('_a', 1), ('_b_c', x2['b']['c']),
                          ('_b_d', x2['b']['d'])])

  def test_mixed_list_assignment_in_setup(self):

    class Test(nn.Module):

      def setup(self):
        self.layers = [nn.Dense(10), nn.relu, nn.Dense(10)]

      def __call__(self, x):
        for lyr in self.layers:
          x = lyr(x)
        return x

    x = random.uniform(random.PRNGKey(0), (5, 5))
    variables = Test().init(random.PRNGKey(0), jnp.ones((5, 5)))
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
    msg = 'Can\'t call __hash__ on modules that hold variables.'
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
        dot_general = dot_general
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
        dot_general = dot_general
    )
)"""
    x = jnp.ones((1, 2))
    trace, variables = mlp.init_with_output(random.PRNGKey(0), x)
    self.assertEqual(trace, expected_trace)
    trace = mlp.apply(variables, x)
    self.assertEqual(trace, expected_trace)

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
    with self.assertRaisesRegex(AttributeError, "allowed_apply_fn"):
      Foo().apply({}, method='allowed_apply_fn')
      # test same for init.
      Foo().init({}, method='allowed_apply_fn')

    # attributes which are not callables yield TypeError.
    with self.assertRaisesRegex(TypeError, "'Foo.not_callable' must be a callable"):
      Foo().apply({}, method='not_callable')
      # test same for init.
      Foo().init({}, method='not_callable')

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
    variables = foo.init(random.PRNGKey(0), x)
    self.assertEqual(variables['params']['bar']['kernel'].shape, (2, 3))

  def test_noncompact_module_frozen(self):

    class Foo(nn.Module):

      def setup(self):
        self.i = 1  # This is allowed (for assigning submodules).

      def __call__(self):
        self.i = 2  # This is not allowed.

    msg = ('Can\'t set i=2 for Module of type Foo: Module instance is frozen '
           'outside of setup method.')
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.PRNGKey(0))

  def test_compact_module_frozen(self):

    class Foo(nn.Module):

      @nn.compact
      def __call__(self):
        self.i = 2

    msg = ('Can\'t set i=2 for Module of type Foo: Module instance is frozen '
           'outside of setup method.')
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.PRNGKey(0))

  def test_submodule_frozen(self):

    class Foo(nn.Module):

      @nn.compact
      def __call__(self):
        dense = nn.Dense(10)
        dense.features = 20  # <--- This is not allowed

    msg = ('Can\'t set features=20 for Module of type Dense: Module instance '
           'is frozen outside of setup method.')
    with self.assertRaisesRegex(errors.SetAttributeFrozenModuleError, msg):
      Foo().init(random.PRNGKey(0))

  def test_module_call_not_implemented(self):

    class Foo(nn.Module):
      pass

    msg = '"Foo" object has no attribute "__call__"'
    with self.assertRaisesRegex(AttributeError, msg):
      Foo().init(random.PRNGKey(0))

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

    key = random.PRNGKey(0)
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

    msg = '"B" object has no attribute "c"'
    with self.assertRaisesRegex(AttributeError, msg):
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

  @use_regular_dict()
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

    variables = A().init(random.PRNGKey(0), jnp.ones((1,)))
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

  @use_regular_dict()
  def test_module_pass_in_closure(self):
    a = nn.Dense(2)

    class B(nn.Module):

      def setup(self):
        self.foo = a

      def __call__(self, x):
        return self.foo(x)

    variables = B().init(random.PRNGKey(0), jnp.ones((1,)))
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

  @use_regular_dict()
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
    key = jax.random.PRNGKey(0)
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

  @use_regular_dict()
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

    key = random.PRNGKey(0)
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
                counters, ref_counters)))

  @use_regular_dict()
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

    key = random.PRNGKey(0)
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

  @use_regular_dict()
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
    k = jax.random.PRNGKey(0)
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

  @use_regular_dict()
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

    key = random.PRNGKey(0)
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

    variables = Foo().init(random.PRNGKey(0), 1)
    # During init we should not collect intermediates by default...
    self.assertNotIn('intermediates', variables)
    # ...unless we override mutable.
    variables = Foo().init(random.PRNGKey(0), 1, mutable=True)
    self.assertEqual(variables, {'intermediates': {'h': (1, 2)}})

    _, state = Foo().apply({}, 1, mutable=['intermediates'])
    self.assertEqual(state, {'intermediates': {'h': (1, 2)}})
    _, state = Foo().apply({},
                           1,
                           init_fn=lambda: 0,
                           reduce_fn=lambda a, b: a + b,
                           mutable=['intermediates'])
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

    x = jax.random.uniform(jax.random.PRNGKey(1), shape=(10,))
    y = jax.random.uniform(jax.random.PRNGKey(2), shape=(10,))
    variables = Foo().init(jax.random.PRNGKey(0), x)
    intm_grads = jax.grad(
        loss, argnums=1)(variables['params'], variables['perturbations'], x, y)
    # activation * 4 so reverse gradient also * 4
    self.assertTrue(
        all(intm_grads['after_multiply'] * 4 == intm_grads['before_multiply']))

  def test_perturb_noop(self):

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(10)(x)
        x = self.perturb('before_multiply', x)
        x = 4 * x
        x = self.perturb('after_multiply', x)
        return x

    x = jax.random.uniform(jax.random.PRNGKey(1), shape=(10,))
    module = Foo()
    variables = module.init(jax.random.PRNGKey(0), x)
    params = variables['params']
    perturbations = variables['perturbations']

    # check no error if perturbations is not passed
    module.apply({'params': params}, x)

    # check errors if perturbations is passed but empty
    with self.assertRaisesRegex(errors.ScopeCollectionNotFound, 'Tried to access'):
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
    y1, variables = f_init(random.PRNGKey(0), x)
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
    y1, variables = f_init(random.PRNGKey(0), x)
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
    y1, variables = f_init(random.PRNGKey(0), x)
    foo_b = foo.bind(variables, mutable='batch_stats')
    y2 = f(foo_b, x)
    y3, new_state = nn.apply(f, foo, mutable='batch_stats')(variables, x)
    self.assertEqual(y1, y2)
    self.assertEqual(y2, y3)
    bs_1 = new_state['batch_stats']
    bs_2 = foo_b.variables['batch_stats']
    for x, y in zip(
        jax.tree_util.tree_leaves(bs_1), jax.tree_util.tree_leaves(bs_2)):
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

    variables = foo.init(random.PRNGKey(0), x)
    encoder, encoder_vars = foo.bind(variables).encoder.unbind()
    decoder, decoder_vars = foo.bind(variables).decoder.unbind()

    self.assertIsInstance(encoder, nn.Dense)
    self.assertEqual(encoder.features, 4)
    self.assertIsInstance(decoder, nn.Dense)
    self.assertEqual(decoder.features, 2)

    np.testing.assert_equal(variables['params']['encoder'], encoder_vars['params'])
    np.testing.assert_equal(variables['params']['decoder'], decoder_vars['params'])

  def test_passing_mutable_variables(self):

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(2)(x)

    x = jnp.ones((3,))
    variables = Foo().init(random.PRNGKey(0), x)
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

    k = random.PRNGKey(0)
    x = jnp.ones((4, 7))

    variables = Bar().init(k, x)
    shapes = jax.tree_util.tree_map(np.shape, variables['params'])
    self.assertEqual(
        shapes, {
            'Dense_0': {
                'kernel': (7, 4),
                'bias': (4,)
            },
            'Dense_1': {
                'kernel': (4, 3),
                'bias': (3,)
            },
        })
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

    k = random.PRNGKey(0)
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
    self.assertEqual(type(xs),
                     Foo)  # equality test for NamedTuple doesn't check class!

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
    model = nn.Dense(1, use_bias=False)
    jit_model = nn.jit(nn.Dense)(1, use_bias=False)
    param = model.init(random.PRNGKey(0), np.ones((1, 1)))['params']['kernel']
    param_2 = jit_model.init(random.PRNGKey(0), np.ones(
        (1, 1)))['params']['kernel']
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
        return jnp.alltrue(x0 == x1)

    k = random.PRNGKey(0)
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
        self.put_variable('test_col', 'a', jnp.ones(2,))
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
    k = random.PRNGKey(0)
    self.assertTrue(foo.apply({}, rngs={'bar': k}))
    self.assertFalse(foo.apply({}, rngs={'baz': k}))

  def test_is_initializing(self):

    class Foo(nn.Module):

      def __call__(self):
        return self.is_initializing()

    foo = Foo()
    k = random.PRNGKey(0)
    self.assertTrue(foo.init_with_output(k)[0])
    self.assertFalse(foo.apply({}))

  def test_throws_invalid_instance_module_error(self):

    class B(nn.Module):

      @nn.compact
      def __call__(self, x):
        return x

    k = random.PRNGKey(0)
    x = random.uniform(random.PRNGKey(1), (2,))

    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.init(k, x)  # B is module class, not B() a module instance
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.init_with_output(k, x)
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.apply({},
              x)  # similar issue w. apply called on class instead of instance.
    with self.assertRaises(errors.InvalidInstanceModuleError):
      B.bind({},
             x)  # similar issue w. apply called on class instead of instance.

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
      r.init(jax.random.PRNGKey(2), jnp.ones(3))

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
    with self.assertRaisesRegex(errors.DescriptorAttributeError,
                                'Trying to access a property that'):
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
    with self.assertRaisesRegex(errors.DescriptorAttributeError,
                                'Trying to access a property that'):
      foo.apply({})

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
          rngs = jax.random.split(jax.random.PRNGKey(23), 100)
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
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("foo" in vs['params'], "relaxed naming failure")
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar(sub=foo)
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("sub" in vs['params'], "old policy naming failure")
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
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("foo" in vs['params'], "adoption naming failure")
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', True):
      foo = Foo(name='foo')
      bar = Bar2(sub=foo)
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("sub" in vs['params'], "adoption naming failure")
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
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = baz.init(k, x)
      self.assertTrue("bar" in vs['params'], "adoption naming failure")
      self.assertTrue("foo" in vs['params']['bar'], "adoption naming failure")
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
      k = random.PRNGKey(0)
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
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("sub" in vs['params'], "relaxed naming failure")
      y = bar.apply(vs, x)

    with set_config('flax_preserve_adopted_names', False):
      foo = Foo(name='foo')
      bar = Bar(sub=foo)
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = bar.init(k, x)
      self.assertTrue("sub" in vs['params'], "old policy naming failure")
      y = bar.apply(vs, x)

  def test_relaxed_python_conflict(self):

    class Foo(nn.Module):
      dummy = 0
      @nn.compact
      def __call__(self, x):
        p = self.param('dummy', nn.initializers.zeros, x.shape)
        return x + p

    with set_config('flax_relaxed_naming', True):
      foo = Foo(name='foo')
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = foo.init(k, x)

    with set_config('flax_relaxed_naming', False):
      foo = Foo(name='foo')
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      with self.assertRaises(errors.NameInUseError):
        vs = foo.init(k, x)

  def test_relaxed_intercollection_conflict(self):

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        v1 = self.variable('col1', 'v', lambda x: jnp.zeros(x), x.shape)
        v2 = self.variable('col2', 'v', lambda x: jnp.zeros(x), x.shape)
        return x + v1.value + v2.value

    with set_config('flax_relaxed_naming', True):
      foo = Foo(name='foo')
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      vs = foo.init(k, x)

    with set_config('flax_relaxed_naming', False):
      foo = Foo(name='foo')
      k = random.PRNGKey(0)
      x = jnp.zeros((1,))
      with self.assertRaises(errors.NameInUseError):
        vs = foo.init(k, x)


class FrozenDictTests(absltest.TestCase):

  def test_frozendict_flag(self):

    with set_config('flax_return_frozendict', True):
      x = jnp.zeros((2,3))
      layer = nn.Dense(5)
      params = layer.init(random.PRNGKey(0), x)
      self.assertTrue(isinstance(params, FrozenDict))

    with set_config('flax_return_frozendict', False):
      x = jnp.zeros((2,3))
      layer = nn.Dense(5)
      params = layer.init(random.PRNGKey(0), x)
      self.assertTrue(isinstance(params, dict))


if __name__ == '__main__':
  absltest.main()

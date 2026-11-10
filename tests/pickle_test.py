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

"""Tests for pickling flax objects."""

import pickle

from absl.testing import absltest, parameterized
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.errors import FlaxError, ScopeVariableNotFoundError


class MLP(nn.Module):
  # Defined at module level so stdlib pickle can serialize it by reference.
  hidden: int
  out: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.hidden)(x)
    x = nn.relu(x)
    return nn.Dense(self.out)(x)


class ErrorrsTest(absltest.TestCase):
  def test_exception_can_be_pickled(self):
    # tests the new __reduce__ method fixes bug reported in issue #4000
    ex = ScopeVariableNotFoundError('varname', 'collection', 'scope')
    pickled_ex = pickle.dumps(ex)
    unpicked_ex = pickle.loads(pickled_ex)
    self.assertIsInstance(unpicked_ex, FlaxError)
    self.assertIn('varname', str(unpicked_ex))
    self.assertIn('#flax.errors.ScopeVariableNotFoundError', str(unpicked_ex))
    self.assertNotIn('#flax.errors.FlaxError', str(unpicked_ex))


class ModulePickleTest(parameterized.TestCase):
  def assert_roundtrip_identical(self, module, pickler):
    """Round-trips ``module`` through ``pickler`` and checks behavior matches."""
    x = jnp.ones((2, 4))
    # Initialize before dumping so the module has been bound at least once:
    # internal state from init/apply must not leak into the payload (#1481).
    params = module.init(jax.random.PRNGKey(0), x)
    restored = pickler.loads(pickler.dumps(module))
    # The restored module must come back unbound.
    self.assertIsNone(restored.scope)
    y_original = module.apply(params, x)
    y_restored = restored.apply(params, x)
    np.testing.assert_array_equal(
      np.asarray(y_original), np.asarray(y_restored)
    )
    restored_params = restored.init(jax.random.PRNGKey(0), x)
    jax.tree_util.tree_map(
      np.testing.assert_array_equal, params, restored_params
    )
    return restored

  def test_dense_roundtrip_pickle(self):
    # Stdlib pickle serializes functions by reference, so field values must
    # be importable top-level functions (zeros/ones are, lecun_normal is not).
    module = nn.Dense(
      features=3,
      kernel_init=nn.initializers.ones_init(),
      bias_init=nn.initializers.zeros_init(),
    )
    restored = self.assert_roundtrip_identical(module, pickle)
    self.assertEqual(module, restored)

  def test_dense_roundtrip_cloudpickle(self):
    # cloudpickle serializes the default initializer closures by value, so
    # the default Dense round-trips. No object-equality check: the restored
    # initializers are equivalent copies, not identical function objects.
    self.assert_roundtrip_identical(nn.Dense(features=3), cloudpickle)

  def test_pickle_default_initializer_limitation(self):
    # Known limitation: the default kernel initializer (lecun_normal) is a
    # closure created inside jax.nn.initializers.variance_scaling, which
    # stdlib pickle cannot serialize by reference. This is the main reason
    # cloudpickle support (#1475) matters. If this test starts failing, the
    # limitation was lifted and the docs/tests should be updated.
    with self.assertRaises((AttributeError, pickle.PicklingError)):
      pickle.dumps(nn.Dense(features=3))

  @parameterized.named_parameters(
    ('pickle', pickle), ('cloudpickle', cloudpickle)
  )
  def test_nested_module_roundtrip(self, pickler):
    module = MLP(hidden=8, out=2)
    restored = self.assert_roundtrip_identical(module, pickler)
    self.assertEqual(module, restored)

  def test_cloudpickle_locally_defined_module(self):
    # Modules defined in a local scope (e.g. a notebook cell) can only be
    # serialized by value, which cloudpickle supports and stdlib pickle
    # does not. This is the use case fixed by PR #1475.
    class LocalDense(nn.Module):
      features: int

      @nn.compact
      def __call__(self, x):
        return nn.Dense(self.features)(x)

    self.assert_roundtrip_identical(LocalDense(features=3), cloudpickle)


if __name__ == '__main__':
  absltest.main()

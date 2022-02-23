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

from absl.testing import absltest

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np
from typing import Any, Tuple

from flax import linen as nn
from flax.core import Scope

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

class Dummy(nn.Module):
  @nn.compact
  def __call__(self):
    self.param('foo', lambda rng: 1)

class ModuleTopLevelTest(absltest.TestCase):
  pass
  # def test_toplevel_immutable(self):
  #   d = Dummy(parent=None)
  #   with self.assertRaisesRegex(BaseException, "orphaned module"):
  #     d()

  # def test_toplevel_initialized_requires_rng(self):
  #   with self.assertRaisesRegex(BaseException, "missing 1 required.*rngs"):
  #     d = Dummy(parent=None).initialized()

  # def test_toplevel_initialized_with_rng(self):
  #   d = Dummy(parent=None).initialized(rngs={'params': random.PRNGKey(0)})
  #   self.assertEqual(d.variables.param.foo, 1)

  # def test_toplevel_initialized_frozen(self):
  #   d = Dummy(parent=None).initialized(rngs={'params': random.PRNGKey(0)})
  #   with self.assertRaisesRegex(BaseException, "Can't set value"):
  #     d.variables.param.foo = 2

  # def test_toplevel_initialized_has_new_scope(self):
  #   d = Dummy(parent=None)
  #   # initializing should make a copy and not have any effect
  #   # on `d` itself.
  #   d_initialized = d.initialized(rngs={'params': random.PRNGKey(0)})
  #   # ... make sure that indeed `d` has no scope.
  #   self.assertIsNone(d.scope)

  # def test_can_only_call_initialized_once(self):
  #   d = Dummy(parent=None)
  #   d = d.initialized(rngs={'params': random.PRNGKey(0)})
  #   with self.assertRaises(BaseException):
  #     d.initialized(rngs={'params': random.PRNGKey(0)})


if __name__ == '__main__':
  absltest.main()

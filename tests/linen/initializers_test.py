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

"""Tests for flax.linen.initializers."""

from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax.linen import initializers

import jax
from jax import random
import jax.numpy as jnp

import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class InitializersTest(parameterized.TestCase):

  @parameterized.parameters(
    {
      'builder_fn': initializers.zeros_init,
      'params_shape': (2, 3),
      'expected_params': jnp.zeros((2, 3)),
    }, {
      'builder_fn': initializers.ones_init,
      'params_shape': (3, 2),
      'expected_params': jnp.ones((3, 2)),
    })
  def test_call_builder(self, builder_fn, params_shape, expected_params):
    params = builder_fn()(random.PRNGKey(42), params_shape, jnp.float32)
    np.testing.assert_allclose(params, expected_params)

  @parameterized.parameters(
    {
      'builder_fn': initializers.zeros_init,
      'expected_params': jnp.zeros((2, 5)),
    }, {
      'builder_fn': initializers.ones_init,
      'expected_params': jnp.ones((2, 5)),
    })
  def test_kernel_builder(self, builder_fn, expected_params):
    layer = nn.Dense(5, kernel_init=builder_fn())
    params = layer.init(random.PRNGKey(42), jnp.empty((3, 2)))['params']
    np.testing.assert_allclose(params['kernel'], expected_params)


if __name__ == '__main__':
  absltest.main()
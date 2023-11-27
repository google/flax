# Copyright 2023 The Flax Authors.
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

"""Tests for flax.linen.activation."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax import random

from flax import linen as nn

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class ActivationTest(parameterized.TestCase):
  def test_prelu(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5))
    act = nn.PReLU()
    y, _ = act.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)

  def test_geglu_high_dimension(self):
    rng = random.key(0)
    x = jnp.ones((4, 7, 6, 5, 2))
    act = nn.GeGLU()
    y, _ = act.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)

  def test_geglu_with_dim_expansion(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5))
    act = nn.GeGLU(10)
    expected_shape = (4, 6, 10)
    y, _ = act.init_with_output(rng, x)
    self.assertEqual(y.shape, expected_shape)

  def test_geglu_with_dim_contraction(self):
    rng = random.key(0)
    x = jnp.ones((4, 6, 5, 7))
    act = nn.GeGLU(3)
    expected_shape = (4, 6, 5, 3)
    y, _ = act.init_with_output(rng, x)
    self.assertEqual(y.shape, expected_shape)

  def test_geglu_result(self):
    rng = random.key(0)
    x = jnp.array([[0.123,0.234], [0.456,0.789]])
    act = nn.GeGLU()
    expected_result = jnp.array([[0.00024275, -0.00208032],
                                [0.00336634, -0.02307648]])
    y, _ = act.init_with_output(rng, x)
    for row in range(expected_result.shape[0]):
      for col in range(expected_result.shape[1]):
        self.assertAlmostEqual(y[row,col], expected_result[row,col])

  def test_geglu_with_dim_expansion_result(self):
    rng = random.key(0)
    x = jnp.array([[0.123,0.234], [0.456,0.789]])
    act = nn.GeGLU(3)
    expected_result = jnp.array([[-0.02157649, -0.00018928, -0.01176354],
                                [-0.08777858, 0.00258885, -0.18744925]])
    y, _ = act.init_with_output(rng, x)
    for row in range(expected_result.shape[0]):
      for col in range(expected_result.shape[1]):
        self.assertAlmostEqual(y[row,col], expected_result[row,col])

  def test_geglu_with_dim_contraction_result(self):
    rng = random.key(0)
    x = jnp.array([[0.123,0.234], [0.456,0.789]])
    act = nn.GeGLU(1)
    expected_result = jnp.array([[0.00224223], [0.0307451]])
    y, _ = act.init_with_output(rng, x)
    for row in range(expected_result.shape[0]):
      for col in range(expected_result.shape[1]):
        self.assertAlmostEqual(y[row,col], expected_result[row,col])

if __name__ == '__main__':
  absltest.main()

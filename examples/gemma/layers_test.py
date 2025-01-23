# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for transformer layers."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import layers
import jax.numpy as jnp
import numpy as np


class EinsumTest(parameterized.TestCase):
  @parameterized.parameters(
      dict(
          inputs_shape=(1, 4),
          params_shape=(3, 2, 4, 3),
          eqn='TD,SNDH->STNH',
          expected_shape=(3, 1, 2, 3),
      ),
      dict(
          inputs_shape=(1, 2, 4),
          params_shape=(2, 4, 8),
          eqn='ANH,NHD->AD',
          expected_shape=(1, 8),
      ),
  )
  def test_einsum(self, inputs_shape, params_shape, eqn, expected_shape):
    einsum = layers.Einsum(eqn, params_shape, rngs=nnx.Rngs(params=0))
    output = einsum(
        jnp.ones(inputs_shape),
    )
    self.assertEqual(output.shape, expected_shape)

  @parameterized.parameters(
      dict(
          shape=(1, 4),
      ),
      dict(
          shape=(2, 5, 4, 7),
      ),
  )
  def test_shape(self, shape):
    einsum = layers.Einsum('ij->ji', shape, rngs=nnx.Rngs(params=0))
    self.assertEqual(einsum.shape, shape)


class RMSNormTest(parameterized.TestCase):
  @parameterized.parameters(dict(x=[0.1, 0.2], expected=[0.6324429, 1.2648858]))
  def test_rmsnorm(self, x, expected):
    x = jnp.array([x])
    rmsnorm = layers.RMSNorm(x.shape[-1], rngs=nnx.Rngs(params=0))
    output = rmsnorm(x)
    np.testing.assert_array_equal(output, jnp.array([expected]))


if __name__ == '__main__':
  absltest.main()

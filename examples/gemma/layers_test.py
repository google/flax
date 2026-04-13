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


class RMSNormTest(parameterized.TestCase):
  @parameterized.parameters(dict(x=[0.1, 0.2], expected=[0.6324429, 1.2648858]))
  def test_rmsnorm(self, x, expected):
    x = jnp.array([x])
    rmsnorm = layers.RMSNorm(x.shape[-1], rngs=nnx.Rngs(params=0))
    output = rmsnorm(x)
    np.testing.assert_array_equal(output, jnp.array([expected]))


if __name__ == '__main__':
  absltest.main()

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

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from flax import nnx


class TestInitializers(parameterized.TestCase):

  @parameterized.product(
    shape=[(2, 3), (4,), ()],
    dtype=[jnp.float32, jnp.float16],
  )
  def test_zeros_init(self, shape, dtype):
    init_fn = nnx.initializers.zeros_init()
    result = init_fn(jax.random.key(0), shape, dtype)
    self.assertEqual(result.shape, shape)
    self.assertEqual(result.dtype, dtype)
    np.testing.assert_array_equal(result, np.zeros(shape))

  @parameterized.product(
    shape=[(2, 3), (4,), ()],
    dtype=[jnp.float32, jnp.float16],
  )
  def test_ones_init(self, shape, dtype):
    init_fn = nnx.initializers.ones_init()
    result = init_fn(jax.random.key(0), shape, dtype)
    self.assertEqual(result.shape, shape)
    self.assertEqual(result.dtype, dtype)
    np.testing.assert_array_equal(result, np.ones(shape))

  def test_zeros_init_with_linear(self):
    layer = nnx.Linear(
        2, 3,
        kernel_init=nnx.initializers.zeros_init(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(layer.kernel.shape, (2, 3))
    self.assertEqual(layer.kernel.dtype, jnp.float32)
    np.testing.assert_array_equal(layer.kernel[...], np.zeros((2, 3)))
    np.testing.assert_array_equal(layer.bias[...], np.zeros((3,)))
    output = layer(jnp.ones((1, 2)))
    np.testing.assert_array_equal(output, np.zeros((1, 3)))

  def test_ones_init_with_linear(self):
    layer = nnx.Linear(
        3, 5,
        kernel_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(layer.kernel.shape, (3, 5))
    self.assertEqual(layer.kernel.dtype, jnp.float32)
    np.testing.assert_array_equal(layer.kernel[...], np.ones((3, 5)))
    np.testing.assert_array_equal(layer.bias[...], np.zeros((5,)))
    # output = ones_input(1,3) @ ones_kernel(3,5) + zeros_bias = [[3,3,3,3,3]]
    output = layer(jnp.ones((1, 3)))
    np.testing.assert_array_equal(output, np.full((1, 5), 3.0))


if __name__ == '__main__':
  absltest.main()

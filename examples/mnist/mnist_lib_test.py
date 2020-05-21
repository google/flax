# Copyright 2020 The Flax Authors.
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

# Lint as: python3
"""Tests for flax.examples.mnist.mnist_lib."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

import mnist_lib

class MnistTest(absltest.TestCase):
  """Test cases for Mnist."""

  def test_cnn(self):
    rng = jax.random.PRNGKey(0)
    output, init_params = mnist_lib.CNN.init_by_shape(
        rng, [((1, 28, 28, 1), jnp.float32)])

    self.assertEqual((1, 10), output.shape)

    # Assert number of layers.
    self.assertEqual(4, len(init_params))

    # Assert shape of each layer.
    self.assertEqual((32,), init_params['Conv_0']['bias'].shape)
    self.assertEqual((3, 3, 1, 32), init_params['Conv_0']['kernel'].shape)

    self.assertEqual((64,), init_params['Conv_1']['bias'].shape)
    self.assertEqual((3, 3, 32, 64), init_params['Conv_1']['kernel'].shape)

    self.assertEqual((256,), init_params['Dense_2']['bias'].shape)
    self.assertEqual((3136, 256), init_params['Dense_2']['kernel'].shape)

    self.assertEqual((10,), init_params['Dense_3']['bias'].shape)
    self.assertEqual((256, 10), init_params['Dense_3']['kernel'].shape)

if __name__ == '__main__':
  absltest.main()

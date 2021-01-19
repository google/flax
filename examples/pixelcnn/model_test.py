# Copyright 2021 The Flax Authors.
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
"""Tests for PixelCNN Modules."""

import pixelcnn
from flax import linen as nn
from absl.testing import absltest
from absl.testing import parameterized

import numpy.testing as onp_testing

from jax import random
import jax.numpy as np
from jax.config import config
config.enable_omnistaging()


class ModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = random.PRNGKey(0)
    self.x = np.arange(24).reshape(1, 4, 3, 2)


  def get_weightnorm(self, params):
    return [params[k] for k in ('direction', 'scale', 'bias')]


  def assert_mean_and_variance(self, out):
    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var(out, (0, 1, 2)), 1., atol=1e-5)


  def test_conv(self):
    model = pixelcnn.ConvWeightNorm(features=4, kernel_size=(3, 2))
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 2, 2, 4))
    self.assert_mean_and_variance(out)


  def test_conv_down(self):
    model = pixelcnn.ConvDown(features=4)
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']['ConvWeightNorm_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))
    self.assert_mean_and_variance(out)


  def test_conv_down_right(self):
    model = pixelcnn.ConvDownRight(features=4)
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']['ConvWeightNorm_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))
    self.assert_mean_and_variance(out)


  def test_conv_transpose(self):
    model = pixelcnn.ConvTranspose(features=4, kernel_size = (3, 2))
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 6, 4, 4))
    self.assert_mean_and_variance(out)


  def test_conv_transpose_down(self):
    model = pixelcnn.ConvTransposeDown(features=4)
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']["ConvWeightNorm_0"]["weightnorm_params"]
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))


  def test_conv_transpose_down_right(self):
    model = pixelcnn.ConvTransposeDownRight(features=4)
    out, variables = model.init_with_output(self.rng, self.x)
    params = variables['params']['ConvWeightNorm_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(params)

    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))


  def test_pcnn_shape(self):
    x = random.normal(self.rng, (2, 4, 4, 3))
    model = pixelcnn.PixelCNNPP(depth=0, features=2, dropout_p=0)
    out, _ = model.init_with_output(self.rng, x)
    self.assertEqual(out.shape, (2, 4, 4, 100))


if __name__ == '__main__':
  absltest.main()

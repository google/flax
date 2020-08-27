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
    super(absltest.TestCase, self).setUp()
    rng = random.PRNGKey(0)
    self.x = np.arange(24).reshape(1, 4, 3, 2)
    self.init_input = np.zeros((1, 4, 3, 2), np.float32)
    self.rng_dict = {'param': rng, 'weightnorm_params': rng}


  def init_module(self, module_fn):
    return module_fn().init(self.rng_dict, self.init_input)['param']


  def apply_module(self, module_fn, params):
    return module_fn().apply(params, self.x, rngs=self.rng_dict)


  def get_weightnorm(self, params):
    return [params[k] for k in ('direction', 'scale', 'bias')]


  def assert_out_close_to_zero(self, out):
    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    # onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var(out, (0, 1, 2)), 1., atol=1e-5)


  def test_conv(self):
    module_fn = lambda: pixelcnn.Conv(features=4, kernel_size=(3, 2))
    params = self.init_module(module_fn)
    out = self.apply_module(module_fn, params)
    weightnorm_params = params['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(weightnorm_params)

    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 2, 2, 4))
    self.assert_close_to_zero(out)


  def test_conv_down(self):
    module_fn = lambda: pixelcnn.ConvDown(features=4)
    params = self.init_module(module_fn)
    out = self.apply_module(module_fn, params)
    weightnorm_params = params['Conv_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(weightnorm_params)

    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))
    self.assert_out_close_to_zero(out)


  def test_conv_down_right(self):
    module_fn = lambda: pixelcnn.ConvDownRight(features=4)
    params = self.init_module(module_fn)
    out = self.apply_module(module_fn, params)
    weightnorm_params = params['Conv_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(weightnorm_params)

    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))
    self.assert_out_close_to_zero(out)


  def test_conv_transpose(self):
    module_fn = lambda: pixelcnn.ConvTranspose(features=4, kernel_size = (3, 2))
    params = self.init_module(module_fn)
    out = self.apply_module(module_fn, params)
    weightnorm_params = params['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(weightnorm_params)

    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 6, 4, 4))
    self.assert_out_close_to_zero(out)
    

  def test_conv_transpose_down(self):
    module_fn = lambda: pixelcnn.ConvTransposeDown(features=4)
    params = self.init_module(module_fn)
    out = self.apply_module(module_fn, params)
    weightnorm_params = params['Conv_0']['weightnorm_params']
    direction, scale, bias = self.get_weightnorm(weightnorm_params)

    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))
    self.assert_out_close_to_zero(out)


  def test_conv_transpose_down_right(self):
    rng=random.PRNGKey(0)
    x=np.arange(24).reshape(1, 4, 3, 2)
    conv_module=pixelcnn.ConvTransposeDownRight.partial(features = 4)
    out, initial_params=conv_module.init(rng, x)
    model=nn.Model(conv_module, initial_params)
    params=model.params['Conv_0']['weightnorm_params']
    direction, scale, bias=[params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))

  def test_pcnn_shape(self):
    rng=random.PRNGKey(0)
    x=random.normal(rng, (2, 4, 4, 3))
    conv_module=pixelcnn.PixelCNNPP.partial(
        depth = 0, features = 2, dropout_p = 0)
    out, initial_params=conv_module.init(rng, x)
    model=nn.Model(conv_module, initial_params)
    self.assertEqual(out.shape, (2, 4, 4, 100))


if __name__ == '__main__':
  googletest.main()

# Lint as: python3
"""Tests for model."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy.testing as onp_testing

from jax import random
import jax.numpy as np

from flax import nn

import pixelcnn


class ModelTest(absltest.TestCase):

  def test_conv(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.Conv.partial(features=4, kernel_size=(3, 2))
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 2, 2, 4))

    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var (out, (0, 1, 2)), 1., atol=1e-5)

  def test_conv_down(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.ConvDown.partial(features=4)
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['Conv_0']['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))

    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var (out, (0, 1, 2)), 1., atol=1e-5)

  def test_conv_down_right(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.ConvDownRight.partial(features=4)
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['Conv_0']['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 4, 3, 4))

    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var (out, (0, 1, 2)), 1., atol=1e-5)

  def test_conv_transpose(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.ConvTranspose.partial(features=4,
                                                   kernel_size=(3, 2))
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (3, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 6, 4, 4))

    # Weightnorm should ensure that, at initialization time, the outputs of the
    # module have mean 0 and variance 1 over the non-feature dimensions.
    onp_testing.assert_allclose(np.mean(out, (0, 1, 2)), 0., atol=1e-5)
    onp_testing.assert_allclose(np.var (out, (0, 1, 2)), 1., atol=1e-5)

  def test_conv_transpose_down(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.ConvTransposeDown.partial(features=4)
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['Conv_0']['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (2, 3, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))

  def test_conv_transpose_down_right(self):
    rng = random.PRNGKey(0)
    x = np.arange(24).reshape(1, 4, 3, 2)
    conv_module = pixelcnn.ConvTransposeDownRight.partial(features=4)
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    params = model.params['Conv_0']['weightnorm_params']
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    self.assertEqual(direction.shape, (2, 2, 2, 4))
    self.assertEqual(scale.shape, (4,))
    self.assertEqual(bias.shape, (4,))
    self.assertEqual(out.shape, (1, 8, 6, 4))

  def test_pcnn_shape(self):
    rng = random.PRNGKey(0)
    x = random.normal(rng, (2, 4, 4, 3))
    conv_module = pixelcnn.PixelCNNPP.partial(depth=0, features=2, dropout_p=0)
    out, initial_params = conv_module.init(rng, x)
    model = nn.Model(conv_module, initial_params)
    self.assertEqual(out.shape, (2, 4, 4, 100))


if __name__ == '__main__':
  googletest.main()

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

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax implementation of PixelCNN++

Based on the paper

  PixelCNN++: Improving the PixelCNN with discretized logistic mixture
  likelihood and other modifications

published at ICLR '17 (https://openreview.net/forum?id=BJrFC6ceg).
"""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Iterable, Tuple, Optional, Union

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as onp


class PixelCNNPP(nn.Module):
  """PixelCNN++ module."""
  depth: int = 5
  features: int = 160
  logistic_components: int = 10
  dropout_p: float = 0.5

  @nn.compact
  def __call__(self, images, *, train):
    # Special convolutional and resnet blocks which allow information flow
    # downwards and to the right.
    conv_down = partial(ConvDown, features=self.features)
    conv_down_right = partial(ConvDownRight, features=self.features)

    dropout = partial(
      nn.Dropout, rate=self.dropout_p, deterministic=not train)

    res_down = partial(ResDown, dropout=dropout)
    res_down_right = partial(ResDownRight, dropout=dropout)

    # Conv Modules which halve or double the spatial dimensions
    halve_down = partial(conv_down, strides=(2, 2))
    halve_down_right = partial(conv_down_right, strides=(2, 2))

    double_down = partial(ConvTransposeDown, features=self.features)
    double_down_right = partial(ConvTransposeDownRight, features=self.features)

    # Add channel of ones to distinguish image from padding later on
    images = jnp.pad(images, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)

    # Stack of `(down, down_right)` pairs, where information flows downwards
    # through `down` and downwards and to the right through `down_right`.
    # We refer to the building of the stack as the 'forward pass' and the
    # undoing of the stack as the 'reverse pass'.
    stack = []

    # -------------------------- FORWARD PASS ----------------------------------
    down = shift_down(conv_down(kernel_size=(2, 3))(images))
    down_right = (
        shift_down(conv_down(kernel_size=(1, 3))(images))
        + shift_right(conv_down_right(kernel_size=(2, 1))(images)))

    stack.append((down, down_right))
    for _ in range(self.depth):
      down, down_right = res_down()(down), res_down_right()(down_right, down)
      stack.append((down, down_right))

    # Resize spatial dims 32 x 32  -->  16 x 16
    down, down_right = halve_down()(down), halve_down_right()(down_right)
    stack.append((down, down_right))

    for _ in range(self.depth):
      down, down_right = res_down()(down), res_down_right()(down_right, down)
      stack.append((down, down_right))

    # Resize spatial dims 16 x 16  -->  8 x 8
    down, down_right = halve_down()(down), halve_down_right()(down_right)
    stack.append((down, down_right))

    for _ in range(self.depth):
      down, down_right = res_down()(down), res_down_right()(down_right, down)
      stack.append((down, down_right))

    # The stack now contains (in order from last appended):
    #
    #   Number of layers     Spatial dims
    #   depth + 1             8 x  8
    #   depth + 1            16 x 16
    #   depth + 1            32 x 32

    # -------------------------- REVERSE PASS ----------------------------------
    down, down_right = stack.pop()

    for _ in range(self.depth):
      down_fwd, down_right_fwd = stack.pop()
      down = res_down()(down, down_fwd)
      down_right = res_down_right()(
          down_right, jnp.concatenate((down, down_right_fwd), -1))

    # Resize spatial dims 8 x 8  -->  16 x 16
    down, down_right = double_down()(down), double_down_right()(down_right)

    for _ in range(self.depth + 1):
      down_fwd, down_right_fwd = stack.pop()
      down = res_down()(down, down_fwd)
      down_right = res_down_right()(
          down_right, jnp.concatenate((down, down_right_fwd), -1))

    # Resize spatial dims 16 x 16  -->  32 x 32
    down, down_right = double_down()(down), double_down_right()(down_right)

    for _ in range(self.depth + 1):
      down_fwd, down_right_fwd = stack.pop()
      down = res_down()(down, down_fwd)
      down_right = res_down_right()(
          down_right, jnp.concatenate((down, down_right_fwd), -1))

    assert not stack

    # Note init_scale=0.1 on this layer was not in the original implementation,
    # but seems to make training more stable.
    return ConvOneByOne(10 * self.logistic_components,
                        init_scale=0.1)(nn.elu(down_right))


def concat_elu(x):
  return nn.elu(jnp.concatenate((x, -x), -1))


def spatial_pad(pad_vertical, pad_horizontal, operand):
  """Wrapper around lax.pad which pads spatial dimensions (horizontal and
  vertical) with zeros, without any interior padding."""
  zero = (0, 0, 0)
  return lax.pad(operand, 0.,
                 (zero, pad_vertical + (0,), pad_horizontal + (0,), zero))


shift_down = partial(spatial_pad, (1, -1), (0, 0))
shift_right = partial(spatial_pad, (0, 0), (1, -1))


# Weightnorm utils
def _l2_normalize(v):
  """Normalize a convolution kernel direction over the in_features and spatial
  dimensions."""
  return v / jnp.sqrt(jnp.sum(jnp.square(v), (0, 1, 2)))


def _make_kernel(direction, scale):
  """Maps weightnorm parameterization (direction, scale) to standard
  parameterization. The direction has shape (spatial..., in_features,
  out_features), scale has shape (out_features,)."""
  return scale * _l2_normalize(direction)


# 2D convolution Modules with weightnorm
class ConvWeightNorm(nn.Module):
  """2D convolution Modules with weightnorm."""
  features: int
  kernel_size: Tuple[int, int]
  strides: Optional[Tuple[int, int]] = None
  padding: Union[str, Iterable[Iterable[int]]] = 'VALID'
  transpose: bool = False
  init_scale: float = 1.
  dtype: Any = jnp.float32
  precision: Any = None

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    strides = self.strides or (1,) * (inputs.ndim - 2)

    if self.transpose:
      conv = partial(lax.conv_transpose, strides=strides, padding=self.padding,
                     precision=self.precision)
    else:
      conv = partial(lax.conv_general_dilated, window_strides=strides,
                     padding=self.padding,
                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                     precision=self.precision)

    in_features = inputs.shape[-1]
    kernel_shape = self.kernel_size + (in_features, self.features)

    def initializer(key):
      # A weightnorm initializer generating a (direction, scale, bias) tuple.
      direction = nn.initializers.normal()(key, kernel_shape, self.dtype)
      unnormed_out = conv(inputs, _l2_normalize(direction))
      mean = jnp.mean(unnormed_out, (0, 1, 2))
      var = jnp.std(unnormed_out, (0, 1, 2))
      return dict(
          direction=direction, scale=self.init_scale / var, bias=-mean / var)

    params = self.param('weightnorm_params', initializer)
    direction, scale, bias = [params[k] for k in ('direction', 'scale', 'bias')]
    return conv(inputs, _make_kernel(direction, scale)) + bias


ConvOneByOne = partial(ConvWeightNorm, kernel_size=(1, 1))
ConvTranspose = partial(ConvWeightNorm, transpose=True)


class ConvDown(nn.Module):
  """Convolution with padding so that information cannot flow upwards."""
  features: int
  kernel_size: Tuple[int, int] = (2, 3)
  strides: Optional[Tuple[int, int]] = None
  init_scale: float = 1.

  @nn.compact
  def __call__(self, inputs):
    k_h, k_w = self.kernel_size
    assert k_w % 2 == 1, 'kernel width must be odd.'
    padding = ((k_h - 1, 0),          # Vertical padding
               (k_w // 2, k_w // 2))  # Horizontal padding

    return ConvWeightNorm(
        self.features, self.kernel_size, self.strides, padding,
        init_scale=self.init_scale)(inputs)


class ConvDownRight(nn.Module):
  """Convolution with padding so that information cannot flow left/upwards."""
  features: Any
  kernel_size: Tuple[int, int] = (2, 2)
  strides: Optional[Tuple[int, int]] = None
  init_scale: float = 1.0

  @nn.compact
  def __call__(self, inputs):
    k_h, k_w = self.kernel_size
    padding = ((k_h - 1, 0),  # Vertical padding
               (k_w - 1, 0))  # Horizontal padding

    return ConvWeightNorm(
        self.features, self.kernel_size, self.strides, padding,
        init_scale=self.init_scale)(inputs)


class ConvTransposeDown(nn.Module):
  """Transpose convolution with output slicing so that information cannot flow
  upwards.  Strides are (2, 2) by default which implies the spatial dimensions
  of the output shape are double those of the input shape.
  """
  features: Any
  kernel_size: Tuple[int, int] = (2, 3)
  strides: Optional[Tuple[int, int]] = (2, 2)

  @nn.compact
  def __call__(self, inputs):
    _, k_w = self.kernel_size
    out_h, out_w = onp.multiply(self.strides, inputs.shape[1:3])
    return ConvTranspose(self.features, self.kernel_size, self.strides)(inputs)[
        :, :out_h, (k_w - 1) // 2:out_w + (k_w - 1) // 2, :]

class ConvTransposeDownRight(nn.Module):
  """Transpose convolution with output slicing so that information cannot flow.

  to the left or upwards. Strides are (2, 2) by default which implies the
  spatial dimensions of the output shape are double those of the input shape.
  """
  features: Any
  kernel_size: Tuple[int, int] = (2, 2)
  strides: Optional[Tuple[int, int]] = (2, 2)

  @nn.compact
  def __call__(self, inputs):
    out_h, out_w = onp.multiply(self.strides, inputs.shape[1:3])
    return ConvTranspose(self.features, self.kernel_size,
                         self.strides)(inputs)[:, :out_h, :out_w]


# Resnet modules
class GatedResnet(nn.Module):
  conv_module: Callable[..., Any]
  dropout: Callable[..., Any]
  nonlinearity: Callable[..., Any] = concat_elu

  @nn.compact
  def __call__(self, inputs, aux=None):
    c = inputs.shape[-1]
    y = self.conv_module(c)(self.nonlinearity(inputs))
    if aux is not None:
      y = self.nonlinearity(y + ConvOneByOne(c)(self.nonlinearity(aux)))

    y = self.dropout()(y)

    # Set init_scale=0.1 so that the res block is close to the identity at
    # initialization.
    a, b = jnp.split(self.conv_module(2 * c, init_scale=0.1)(y), 2, axis=-1)
    return inputs + a * nn.sigmoid(b)


ResDown = partial(GatedResnet, conv_module=ConvDown)
ResDownRight = partial(GatedResnet, conv_module=ConvDownRight)


# Logistic mixture distribution utils
def conditional_params_from_outputs(theta, img):
  """Maps an image `img` and the PixelCNN++ convnet output `theta` to
  conditional parameters for a mixture of k logistics over each pixel.

  Returns a tuple `(means, inverse_scales, logit_weights)` where `means` and
  `inverse_scales` are the conditional means and inverse scales of each mixture
  component (for each pixel-channel) and `logit_weights` are the logits of the
  mixture weights (for each pixel). These have the following shapes:

    means.shape == inv_scales.shape == (batch..., k, h, w, c)
    logit_weights.shape == (batch..., k, h, w)

  Args:
    theta: outputs of PixelCNN++ neural net with shape
      (batch..., h, w, (1 + 3 * c) * k)
    img: an image with shape (batch..., h, w, c)

  Returns:
    The tuple `(means, inverse_scales, logit_weights)`.
  """
  *batch, h, w, c = img.shape
  assert theta.shape[-1] % (3 * c + 1) == 0
  k = theta.shape[-1] // (3 * c + 1)

  logit_weights, theta = theta[..., :k], theta[..., k:]
  assert theta.shape[-3:] == (h, w, 3 * c * k)

  # Each of m, s and t must have shape (batch..., k, h, w, c), we effectively
  # spread the last dimension of theta out into c, k, 3, move the k dimension to
  # after batch and split along the 3 dimension.
  m, s, t = jnp.moveaxis(
      jnp.reshape(theta,
                  tuple(batch) + (h, w, c, k, 3)), (-2, -1), (-4, 0))
  assert m.shape[-4:] == (k, h, w, c)
  t = jnp.tanh(t)

  # Add a mixture dimension to images
  img = jnp.expand_dims(img, -4)

  # Ensure inv_scales cannot be zero (zeros cause nans in sampling)
  inv_scales = jnp.maximum(nn.softplus(s), 1e-7)

  # now condition the means for the last 2 channels (assuming c == 3)
  mean_red = m[..., 0]
  mean_green = m[..., 1] + t[..., 0] * img[..., 0]
  mean_blue = m[..., 2] + t[..., 1] * img[..., 0] + t[..., 2] * img[..., 1]
  means = jnp.stack((mean_red, mean_green, mean_blue), axis=-1)
  return means, inv_scales, jnp.moveaxis(logit_weights, -1, -3)


def logprob_from_conditional_params(images, means, inv_scales, logit_weights):
  """Compute log-likelihoods.

  Computes the log-likelihoods of images given the conditional logistic mixture
  parameters produced by `conditional_params_from_outputs`. The 8-bit pixel
  values are assumed to be scaled so that they are in the discrete set

    {-1, -1 + 1/127.5, -1 + 2/127.5, ..., 1 - 1/127.5, 1}
  """
  # Add a 'mixture' dimension to images.
  images = jnp.expand_dims(images, -4)

  # Calculate log probabilities under all mixture components.
  all_logprobs = discretized_logistic_logpmf(images, means, inv_scales)

  # Sum over the channel dimension because mixture components are shared
  # across channels.
  logprobs = jnp.sum(all_logprobs, -1)

  # Normalize the mixture weights.
  log_mix_coeffs = logit_weights - logsumexp(logit_weights, -3, keepdims=True)

  # Finally marginalize out mixture components and sum over pixels.
  return jnp.sum(logsumexp(log_mix_coeffs + logprobs, -3), (-2, -1))


def discretized_logistic_logpmf(images, means, inv_scales):
  """Compute log-probabilities for each mixture component, pixel and channel."""
  # Compute the difference between the logistic cdf half a level above and half
  # a level below the image value.
  centered = images - means

  # Where images == 1 we use log(1 - cdf(images - 1 / 255))
  top = -jnp.logaddexp(0, (centered - 1 / 255) * inv_scales)

  # Where images == -1 we use log(cdf(images + 1 / 255))
  bottom = -jnp.logaddexp(0, -(centered + 1 / 255) * inv_scales)

  # Elsewhere we use log(cdf(images + 1 / 255) - cdf(images - 1 / 255))
  mid = log1mexp(inv_scales / 127.5) + top + bottom

  return jnp.where(images == 1, top, jnp.where(images == -1, bottom, mid))


@jax.custom_jvp
def log1mexp(x):
  """Accurate computation of log(1 - exp(-x)) for x > 0."""

  # Method from
  # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  return jnp.where(x > jnp.log(2), jnp.log1p(-jnp.exp(-x)),
                   jnp.log(-jnp.expm1(-x)))


# log1mexp produces NAN gradients for small inputs because the derivative of the
# log1p(-exp(-eps)) branch has a zero divisor (1 + -jnp.exp(-eps)), and NANs in
# the derivative of one branch of a where cause NANs in the where's vjp, even
# when the NAN branch is not taken. See
# https://github.com/google/jax/issues/1052. We work around this by defining a
# custom jvp.
log1mexp.defjvps(lambda t, _, x: t / jnp.expm1(x))

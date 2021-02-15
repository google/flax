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

"""Normalization modules for Flax."""

from . import base

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


_no_init = lambda rng, shape: ()


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(base.Module):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  BatchNorm Module."""

  def apply(self,
            x,
            batch_stats=None,
            use_running_average=False,
            axis=-1,
            momentum=0.99,
            epsilon=1e-5,
            dtype=jnp.float32,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones,
            axis_name=None,
            axis_index_groups=None):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      batch_stats: a `flax.nn.Collection` used to store an exponential moving
        average of the batch statistics (default: None).
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of
        the batch statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  if True, bias (beta) is added.
      scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For example,
        `[[0, 1], [2, 3]]` would independently batch-normalize over the examples
        on the first two and last two devices. See `jax.lax.psum` for more details.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    axis = axis if isinstance(axis, tuple) else (axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)
    if self.is_stateful() or batch_stats:
      ra_mean = self.state('mean', reduced_feature_shape,
                           initializers.zeros, collection=batch_stats)
      ra_var = self.state('var', reduced_feature_shape,
                          initializers.ones, collection=batch_stats)
    else:
      ra_mean = None
      ra_var = None

    if use_running_average:
      if ra_mean is None:
        raise ValueError('when use_running_averages is True '
                         'either use a stateful context or provide batch_stats')
      mean, var = ra_mean.value, ra_var.value
    else:
      mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
      mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
      if axis_name is not None and not self.is_initializing():
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=axis_name,
                axis_index_groups=axis_index_groups), 2)
      var = mean2 - lax.square(mean)

      if ra_mean and not self.is_initializing():
        ra_mean.value = momentum * ra_mean.value + (1 - momentum) * mean
        ra_var.value = momentum * ra_var.value + (1 - momentum) * var

    y = x - mean.reshape(feature_shape)
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * self.param(
          'scale', reduced_feature_shape, scale_init).reshape(feature_shape)
    y = y * mul
    if bias:
      y = y + self.param(
          'bias', reduced_feature_shape, bias_init).reshape(feature_shape)
    return jnp.asarray(y, dtype)


class LayerNorm(base.Module):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.
  """

  def apply(self,
            x,
            epsilon=1e-6,
            dtype=jnp.float32,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones):
    """Applies layer normalization on the input.

    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Args:
      x: the inputs
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * jnp.asarray(self.param('scale', (features,), scale_init),
                              dtype)
    y = (x - mean) * mul
    if bias:
      y = y + jnp.asarray(self.param('bias', (features,), bias_init), dtype)
    return jnp.asarray(y, dtype)


class GroupNorm(base.Module):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Group normalization (arxiv.org/abs/1803.08494)."""

  def apply(self,
            x,
            num_groups=32,
            group_size=None,
            epsilon=1e-6,
            dtype=jnp.float32,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    This op is similar to batch normalization, but statistics are shared across
    equally-sized groups of channels and not shared across batch dimension.
    Thus, group normalization does not depend on the batch composition and does
    not require maintaining internal state for storing statistics.

    The user should either specify the total number of channel groups or the
    number of channels per group.

    Args:
      x: the input of shape N...C, where N is a batch dimension and C is a
        channels dimensions. `...` represents an arbitrary number of extra
        dimensions that are used to accumulate statistics over.
      num_groups: the total number of channel groups. The default value of 32 is
        proposed by the original group normalization paper.
      group_size: the number of channels in a group.
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    x = jnp.asarray(x, jnp.float32)
    if ((num_groups is None and group_size is None) or
        (num_groups is not None and group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')

    if group_size is not None:
      channels = x.shape[-1]
      if channels % group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, group_size))
      num_groups = channels // group_size

    input_shape = x.shape
    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

    x = x.reshape(group_shape)

    reduction_axis = [d for d in range(1, x.ndim - 2)] + [x.ndim - 1]

    mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(x), axis=reduction_axis,
                               keepdims=True)
    var = mean_of_squares - jnp.square(mean)

    x = (x - mean) * lax.rsqrt(var + epsilon)

    x = x.reshape(input_shape)

    feature_shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
    if scale:
      x = x * self.param('scale', feature_shape, scale_init)
    if bias:
      x = x + self.param('bias', feature_shape, bias_init)

    return x.astype(dtype)

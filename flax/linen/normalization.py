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

from typing import (Any, Callable, Optional, Tuple)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax.linen.module import Module, compact, merge_param


PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?

_no_init = lambda rng, shape: ()


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(Module):
  """BatchNorm Module.

  Attributes:
    use_running_average: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
  """
  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """Normalizes the input using batch statistics.

    NOTE:
    During initialization (when parameters are mutable) the running average
    of the batch statistics will not be updated. Therefore, the inputs
    fed during initialization don't need to match that of the actual input
    distribution and the reduction axis (set with `axis_name`) does not have
    to exist.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_average = merge_param(
        'use_running_average', self.use_running_average, use_running_average)
    x = jnp.asarray(x, jnp.float32)
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

    # see NOTE above on initialization behavior
    initializing = self.is_mutable_collection('params')

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            reduced_feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32),
                           reduced_feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
      mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
      if self.axis_name is not None and not initializing:
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups), 2)
      var = mean2 - lax.square(mean)

      if not initializing:
        ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

    y = x - mean.reshape(feature_shape)
    mul = lax.rsqrt(var + self.epsilon)
    if self.use_scale:
      scale = self.param('scale',
                         self.scale_init,
                         reduced_feature_shape).reshape(feature_shape)
      mul = mul * scale
    y = y * mul
    if self.use_bias:
      bias = self.param('bias',
                        self.bias_init,
                        reduced_feature_shape).reshape(feature_shape)
      y = y + bias
    return jnp.asarray(y, self.dtype)


class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).
  Operates on the last axis of the input data.

  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  """
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + self.epsilon)
    if self.use_scale:
      mul = mul * jnp.asarray(
          self.param('scale', self.scale_init, (features,)),
          self.dtype)
    y = (x - mean) * mul
    if self.use_bias:
      y = y + jnp.asarray(
          self.param('bias', self.bias_init, (features,)),
          self.dtype)
    return jnp.asarray(y, self.dtype)


class GroupNorm(Module):
  """Group normalization (arxiv.org/abs/1803.08494).
  
    This op is similar to batch normalization, but statistics are shared across
    equally-sized groups of channels and not shared across batch dimension.
    Thus, group normalization does not depend on the batch composition and does
    not require maintaining internal state for storing statistics.
    The user should either specify the total number of channel groups or the
    number of channels per group.

    Attributes:
      num_groups: the total number of channel groups. The default value of 32 is
        proposed by the original group normalization paper.
      group_size: the number of channels in a group.
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
  """
  num_groups: int = 32
  group_size: Optional[int] = None
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @compact
  def __call__(self, x):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    Args:
      x: the input of shape N...C, where N is a batch dimension and C is a
        channels dimensions. `...` represents an arbitrary number of extra
        dimensions that are used to accumulate statistics over.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')
    num_groups = self.num_groups

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, self.group_size))
      num_groups = channels // self.group_size

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError('Number of groups ({}) does not divide the number'
                       ' of channels ({}).'.format(num_groups, channels))

    input_shape = x.shape
    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)
    x = x.reshape(group_shape)

    reduction_axis = [d for d in range(1, x.ndim - 2)] + [x.ndim - 1]
    mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(x), axis=reduction_axis,
                               keepdims=True)
    var = mean_of_squares - jnp.square(mean)
    x = (x - mean) * lax.rsqrt(var + self.epsilon)
    x = x.reshape(input_shape)

    feature_shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
    if self.use_scale:
      x = x * self.param('scale', self.scale_init, feature_shape)
    if self.use_bias:
      x = x + self.param('bias', self.bias_init, feature_shape)

    return x.astype(self.dtype)

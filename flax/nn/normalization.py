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

"""Normalization modules for Flax."""

from . import base

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


_no_init = lambda rng, shape: ()


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(base.Module):
  """BatchNorm Module."""

  def apply(self,
            x,
            batch_stats=None,
            use_running_average=False,
            axis=-1,
            momentum=0.99,
            epsilon=1e-5,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones,
            axis_name=None):
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
      bias:  if True, bias (beta) is added.
      scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).

    Returns:
      Normalized inputs (this same shape as inputs).
    """
    if batch_stats is None and base.is_stateful():
      batch_stats = base.get_state()
    axis = axis if isinstance(axis, tuple) else (axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduction_axis = tuple([i for i in range(x.ndim) if i not in axis])

    if use_running_average:
      if batch_stats is None:
        raise ValueError('batch_stats should be provided if '
                         'use_running_averages is True')
      mean, var = batch_stats.retrieve()
    else:
      mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
      if axis_name is not None and not self.is_initializing():
        axis_size = lax.psum(1., axis_name=axis_name)
        mean = lax.psum(mean, axis_name=axis_name) / axis_size

      mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=True)
      if axis_name is not None and not self.is_initializing():
        mean2 = lax.psum(mean2, axis_name=axis_name) / axis_size
      var = mean2 - lax.square(mean)

      if batch_stats:
        if self.is_initializing():
          shape = mean.shape
          batch_stats.store((jnp.zeros(shape), jnp.ones(shape)))
        else:
          ra_mean, ra_var = batch_stats.retrieve(default=(0, 0))
          ra_mean = momentum * ra_mean + (1 - momentum) * mean
          ra_var = momentum * ra_var + (1 - momentum) * var
          batch_stats.store((ra_mean, ra_var))

    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * self.param('scale', feature_shape, scale_init)
    y = y * mul
    if bias:
      y = y + self.param('bias', feature_shape, bias_init)
    return y


class LayerNorm(base.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.
  """

  def apply(self,
            x,
            epsilon=1e-6,
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
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    mul = lax.rsqrt(var + epsilon)
    if scale:
      mul = mul * self.param('scale', (features,), scale_init)
    y = (x - mean) * mul
    if bias:
      y = y + self.param('bias', (features,), bias_init)
    return y

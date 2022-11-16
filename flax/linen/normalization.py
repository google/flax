# Copyright 2022 The Flax Authors.
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

from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
from flax.linen.dtypes import canonicalize_dtype

from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?

Axes = Union[int, Any]


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(x: Array, axes: Optional[Axes],
                   dtype: Optional[Dtype],
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - mean and variance are computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics
      are always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)

  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var


def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array]):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if use_scale:
    scale = mdl.param('scale', scale_init, reduced_feature_shape,
                      param_dtype).reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if use_bias:
    bias = mdl.param('bias', bias_init, reduced_feature_shape,
                     param_dtype).reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class BatchNorm(Module):
  """BatchNorm Module.

  Usage Note:
  If we define a model with BatchNorm, for example::

    BN = nn.BatchNorm(use_running_average=False, momentum=0.9, epsilon=1e-5,
                      dtype=jnp.float32)

  The initialized variables dict will contain in addition to a 'params'
  collection a separate 'batch_stats' collection that will contain all the
  running statistics for all the BatchNorm layers in a model::

    vars_initialized = BN.init(key, x)  # {'params': ..., 'batch_stats': ...}

  We then update the batch_stats during training by specifying that the
  `batch_stats` collection is mutable in the `apply` method for our module.::

    vars_in = {'params': params, 'batch_stats': old_batch_stats}
    y, mutated_vars = BN.apply(vars_in, x, mutable=['batch_stats'])
    new_batch_stats = mutated_vars['batch_stats']

  During eval we would define BN with `use_running_average=True` and use the
  batch_stats collection from training to set the statistics.  In this case
  we are not mutating the batch statistics collection, and needn't mark it
  mutable::

    vars_in = {'params': params, 'batch_stats': training_batch_stats}
    y = BN.apply(vars_in, x)

  Attributes:
    use_running_average: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
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
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
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
    During initialization (when `self.is_initializing()` is `True`) the running
    average of the batch statistics will not be updated. Therefore, the inputs
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
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32),
                           feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      mean, var = _compute_stats(
          x, reduction_axes,
          dtype=self.dtype,
          axis_name=self.axis_name if not self.is_initializing() else None,
          axis_index_groups=self.axis_index_groups)

      if not self.is_initializing():
        ra_mean.value = self.momentum * ra_mean.value + (1 -
                                                         self.momentum) * mean
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

    return _normalize(
        self, x, mean, var, reduction_axes, feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)


class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  LayerNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
  """
  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  reduction_axes: Axes = -1
  feature_axes: Axes = -1
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(x, self.reduction_axes, self.dtype,
                               self.axis_name, self.axis_index_groups)

    return _normalize(
        self, x, mean, var, self.reduction_axes, self.feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)


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
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is
        linear (also e.g. nn.relu), this can be disabled since the scaling will
        be done by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
        This is only needed if the model is subdivided across devices, i.e. the
        array being normalized is sharded across devices within a pmap.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over
        the examples on the first two and last two devices. See `jax.lax.psum`
        for more details.
  """
  num_groups: Optional[int] = 32
  group_size: Optional[int] = None
  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

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
    reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    feature_axes = (-1,)

    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified. If `group_size` is to be specified, '
                       'pass `num_groups=None` as argument to override '
                       'the default `num_groups` value of 32.')

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, self.group_size))
      num_groups = channels // self.group_size
    else:
      num_groups = self.num_groups
      assert isinstance(num_groups, int)

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError('Number of groups ({}) does not divide the number'
                       ' of channels ({}).'.format(num_groups, channels))

    group_size = x.shape[-1] // num_groups
    group_shape = x.shape[:-1] + (num_groups, group_size)

    def broadcast_stat(stat):
      stat = jnp.broadcast_to(stat[..., None],
                              (x.shape[0], num_groups, group_size))
      return stat.reshape((x.shape[0], num_groups * group_size))

    mean, var = _compute_stats(
        x.reshape(group_shape), reduction_axes, self.dtype, self.axis_name,
        self.axis_index_groups)
    mean = broadcast_stat(mean)
    var = broadcast_stat(var)

    return _normalize(
        self, x, mean, var, reduction_axes[:-1], feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)

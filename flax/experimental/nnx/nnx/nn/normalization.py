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

import typing as tp

import jax
import jax.numpy as jnp
from jax import lax

from flax.experimental import nnx
from flax.experimental.nnx.nnx import rnglib
from flax.experimental.nnx.nnx.module import Module, first_from
from flax.experimental.nnx.nnx.nn import dtypes, initializers
from flax.typing import (
  Array,
  Dtype,
  Initializer,
  Axes,
)


def _canonicalize_axes(rank: int, axes: Axes) -> tp.Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, tp.Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(
  x: Array,
  axes: Axes,
  dtype: tp.Optional[Dtype],
  axis_name: tp.Optional[str] = None,
  axis_index_groups: tp.Any = None,
  use_mean: bool = True,
  use_fast_variance: bool = True,
  mask: tp.Optional[Array] = None,
):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - If `use_fast_variance` is `True`, mean and variance are computed using
    Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
    XLA fusion.
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics are
      always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over. Note,
      this is only used for pmap and shard map. For SPMD jit, you do not need to
      manually synchronize. Just make sure that the axes are correctly annotated
      and XLA:SPMD will insert the necessary collectives.
    axis_index_groups: Optional axis indices.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute the
      variance without subtracting the mean.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    mask: Binary array of shape broadcastable to `inputs` tensor, indicating
      the positions for which the mean and variance should be computed.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)
  axes = _canonicalize_axes(x.ndim, axes)

  def maybe_distributed_mean(*xs, mask=None):
    mus = tuple(x.mean(axes, where=mask) for x in xs)
    if axis_name is None:
      return mus if len(xs) > 1 else mus[0]
    else:
      # In the distributed case we stack multiple arrays to speed comms.
      if len(xs) > 1:
        reduced_mus = lax.pmean(
          jnp.stack(mus, axis=0),
          axis_name,
          axis_index_groups=axis_index_groups,
        )
        return tuple(reduced_mus[i] for i in range(len(xs)))
      else:
        return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

  if use_mean:
    if use_fast_variance:
      mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
      # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
      # to floating point round-off errors.
      var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
    else:
      mu = maybe_distributed_mean(x, mask=mask)
      var = maybe_distributed_mean(
        _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
      )
  else:
    var = maybe_distributed_mean(_abs_sq(x), mask=mask)
    mu = jnp.zeros_like(var)
  return mu, var


def _normalize(
  x: Array,
  mean: Array,
  var: Array,
  scale: tp.Optional[Array],
  bias: tp.Optional[Array],
  reduction_axes: Axes,
  feature_axes: Axes,
  dtype: tp.Optional[Dtype],
  epsilon: float,
):
  """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    epsilon: Normalization epsilon.

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
  if scale is not None:
    scale = scale.reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if bias is not None:
    bias = bias.reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class BatchNorm(Module):
  """BatchNorm Module.

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
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  def __init__(
    self,
    num_features: int,
    *,
    use_running_average: tp.Optional[bool] = None,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 1e-5,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)
    self.mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
    self.var = nnx.BatchStat(jnp.ones(feature_shape, jnp.float32))

    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = nnx.Param(None)

    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = nnx.Param(None)

    self.num_features = num_features
    self.use_running_average = use_running_average
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(
    self,
    x,
    use_running_average: tp.Optional[bool] = None,
    *,
    mask: tp.Optional[jax.Array] = None,
  ):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = first_from(
      use_running_average,
      self.use_running_average,
      error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

    if use_running_average:
      mean, var = self.mean.value, self.var.value
    else:
      mean, var = _compute_stats(
        x,
        reduction_axes,
        dtype=self.dtype,
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
      )

      self.mean.value = (
        self.momentum * self.mean.value + (1 - self.momentum) * mean
      )
      self.var.value = (
        self.momentum * self.var.value + (1 - self.momentum) * var
      )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value,
      self.bias.value,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.epsilon,
    )


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
      use_fast_variance: If true, use a faster, but less numerically stable,
          calculation for the variance.
  """

  def __init__(
    self,
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    reduction_axes: Axes = -1,
    feature_axes: Axes = -1,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)

    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = nnx.Param(None)

    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = nnx.Param(None)

    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.feature_axes = feature_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, *, mask: tp.Optional[jax.Array] = None):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(
      x,
      self.reduction_axes,
      self.dtype,
      self.axis_name,
      self.axis_index_groups,
      use_fast_variance=self.use_fast_variance,
      mask=mask,
    )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value,
      self.bias.value,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.epsilon,
    )


class RMSNorm(Module):
  """RMS Layer normalization (https://arxiv.org/abs/1910.07467).

  RMSNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
  standard deviation of the activations, RMSNorm does not re-center at all
  and instead normalizes by the root mean square of the activations.

  Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
          (also e.g. nn.relu), this can be disabled since the scaling will be done
          by the next layer.
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
      use_fast_variance: If true, use a faster, but less numerically stable,
          calculation for the variance.
  """

  def __init__(
    self,
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_scale: bool = True,
    scale_init: Initializer = initializers.ones,
    reduction_axes: Axes = -1,
    feature_axes: Axes = -1,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)

    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = nnx.Param(None)

    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_scale = use_scale
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.feature_axes = feature_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, mask: tp.Optional[jax.Array] = None):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(
      x,
      self.reduction_axes,
      self.dtype,
      self.axis_name,
      self.axis_index_groups,
      use_mean=False,
      use_fast_variance=self.use_fast_variance,
      mask=mask,
    )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value,
      None,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.epsilon,
    )

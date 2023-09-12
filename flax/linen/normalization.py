# Copyright 2023 The Flax Authors.
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

import dataclasses
import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from flax.linen.transforms import map_variables
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


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


def _compute_stats(
    x: Array,
    axes: Axes,
    dtype: Optional[Dtype],
    axis_name: Optional[str] = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
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
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute the
      variance without subtracting the mean.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.

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

  def maybe_distributed_mean(*xs):
    mus = tuple(x.mean(axes) for x in xs)
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
      mu, mu2 = maybe_distributed_mean(x, _abs_sq(x))
      # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
      # to floating point round-off errors.
      var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
    else:
      mu = maybe_distributed_mean(x)
      var = maybe_distributed_mean(_abs_sq(x - jnp.expand_dims(mu, axes)))
  else:
    var = maybe_distributed_mean(_abs_sq(x))
    mu = jnp.zeros_like(var)
  return mu, var


def _z_normalize_and_scale(
    mdl: Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
):
  """Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

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
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])

  mean = jnp.expand_dims(mean, reduction_axes)
  var = jnp.expand_dims(var, reduction_axes)
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if use_scale:
    scale = mdl.param(
        'scale', scale_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if use_bias:
    bias = mdl.param(
        'bias', bias_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


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
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
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
  use_fast_variance: bool = True

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
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = merge_param(
        'use_running_average', self.use_running_average, use_running_average
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable(
        'batch_stats',
        'mean',
        lambda s: jnp.zeros(s, jnp.float32),
        feature_shape,
    )
    ra_var = self.variable(
        'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
    )

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      mean, var = _compute_stats(
          x,
          reduction_axes,
          dtype=self.dtype,
          axis_name=self.axis_name if not self.is_initializing() else None,
          axis_index_groups=self.axis_index_groups,
          use_fast_variance=self.use_fast_variance,
      )

      if not self.is_initializing():
        ra_mean.value = (
            self.momentum * ra_mean.value + (1 - self.momentum) * mean
        )
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

    return _z_normalize_and_scale(
        self,
        x,
        mean,
        var,
        reduction_axes,
        feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
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
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
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
  use_fast_variance: bool = True

  @compact
  def __call__(self, x):
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
    )

    return _z_normalize_and_scale(
        self,
        x,
        mean,
        var,
        self.reduction_axes,
        self.feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
    )


class RMSNorm(Module):
  """RMS Layer normalization (https://arxiv.org/abs/1910.07467).

  RMSNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
  standard deviation of the activations, RMSNorm does not re-center at all
  and instead normalizes by the root mean square of the activations.

  Example::
    >>> import jax.numpy as jnp
    >>> import jax
    >>> import flax.linen as nn
    ...
    >>> x = jax.random.uniform(jax.random.key(0), (2, 3))
    >>> layer = nn.RMSNorm()
    >>> variables = layer.init(jax.random.key(1), x)
    >>> y = layer.apply(variables, x)

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
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
  """

  epsilon: float = 1e-6
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_scale: bool = True
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
    mean, var = _compute_stats(
        x,
        self.reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_mean=False,
    )

    return _z_normalize_and_scale(
        self,
        x,
        mean,
        var,
        self.reduction_axes,
        self.feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        False,
        self.use_scale,
        initializers.zeros,
        self.scale_init,
    )


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
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
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
  use_fast_variance: bool = True

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

    if (self.num_groups is None and self.group_size is None) or (
        self.num_groups is not None and self.group_size is not None
    ):
      raise ValueError(
          'Either `num_groups` or `group_size` should be '
          'specified. If `group_size` is to be specified, '
          'pass `num_groups=None` as argument to override '
          'the default `num_groups` value of 32.'
      )

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError(
            'Number of channels ({}) is not multiple of the '
            'group size ({}).'.format(channels, self.group_size)
        )
      num_groups = channels // self.group_size
    else:
      num_groups = self.num_groups
      assert isinstance(num_groups, int)

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError(
          'Number of groups ({}) does not divide the number'
          ' of channels ({}).'.format(num_groups, channels)
      )

    group_size = x.shape[-1] // num_groups
    group_shape = x.shape[:-1] + (num_groups, group_size)

    mean, var = _compute_stats(
        x.reshape(group_shape),
        reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
    )
    mean = jnp.repeat(mean, group_size, axis=-1)
    var = jnp.repeat(var, group_size, axis=-1)

    return _z_normalize_and_scale(
        self,
        x,
        mean,
        var,
        reduction_axes[:-1],
        feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
    )


class SpectralNorm(Module):
  """Spectral normalization (https://arxiv.org/abs/2006.10108).

  Spectral normalization normalizes the weight params so that the spectral
  norm of the matrix is equal to 1.

  Example::
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.SpectralNorm(nn.Dense(2))(x)
        x = nn.Dense(2)(x)
        return x
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, 3))
    variables = Foo.init(jax.random.PRNGKey(1), x)
    y = Foo.apply(variables, x)

  Attributes:
    layer: Module instance that you want to wrap in SpectralNorm
    n_steps: How many steps of power iteration to perform to approximate the
      singular value of the input.
    epsilon: A small float added to l2-normalization to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    error_on_non_matrix: Spectral normalization is only defined on matrices.
      By default, this module will return scalars unchanged and flatten
      higher-order tensors in their leading dimensions. Setting this flag to
      True will instead throw errors in those cases.
  """

  # TODO: fix docstring

  layer: Module
  n_steps: int = 1
  epsilon: float = 1e-12
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  error_on_non_matrix: bool = False

  @compact
  def __call__(self, x, update_stats: bool):
    # update_stats = merge_param('update_stats', self.update_stats, update_stats)
    # TODO: fix docstring
    # x: the input to be normalized.
    # use_running_average: if true, the statistics stored in batch_stats will be
    #   used instead of computing the batch statistics on the input.
    #

    # TODO: figure out if there's a more elegant and robust way of unpacking a module instance and re-initializing it after it's been parametrized by map_variables
    # TODO: or figure out how to elegantly pass in a module class with it's args into the SpectralNorm constructor
    layer_class = type(self.layer)
    layer_kwargs = {
        f.name: (
            getattr(self.layer, f.name)
            if f.name != 'name'
            else f'{self.name}({getattr(self.layer, f.name)})'
        )
        for f in dataclasses.fields(self.layer)
    }
    return map_variables(
        layer_class,
        'params',
        trans_in_fn=lambda vs: jax.tree_map(
            functools.partial(
                self.spectral_normalize, update_stats=update_stats
            ),
            vs,
        ),
        init=True,
        mutable=True,
    )(**layer_kwargs)(x)

  def spectral_normalize(self, x, update_stats):
    # TODO: add docstring
    """update_stats: A boolean defaulting to True. Regardless of this arg, this
    function will return the normalized input. When
    `update_stats` is True, the internal state of this object will also be
    updated to reflect the input value. When `update_stats` is False the
    internal stats will remain unchanged."""
    value = jnp.asarray(x)
    value_shape = value.shape

    # Skip and return value if input is scalar, vector or if number of power iterations is less than 1
    if value.ndim <= 1 or self.n_steps < 1:
      return value
    # Handle higher-order tensors.
    elif value.ndim > 2:
      if self.error_on_non_matrix:
        raise ValueError(
            f'Input is {value.ndim}D but error_on_non_matrix is True'
        )
      else:
        value = jnp.reshape(value, (-1, value.shape[-1]))

    # TODO: check if you need to make a separate u and sigma collection for each param in the same layer
    # (e.g. transformer layer would need a separate u and sigma for query, key and value param)
    # u0 = jax.random.normal(
    #     jax.random.PRNGKey(0), (1, value.shape[-1]), self.param_dtype
    # )
    # TODO: maybe just change u_var and sigma_var to self.param instead of self.variable
    u_var = self.variable(
        'batch_stats',
        'u',
        jax.random.normal,
        self.make_rng(
            'spectral_norm'
        ),  # TODO: figure out how to not require an RNG during apply (only needed in init), or how to derive an rng key from the one that's passed down to init
        (1, value.shape[-1]),
        self.param_dtype,
    )
    u0 = u_var.value
    sigma_var = self.variable(
        'batch_stats', 'sigma', jnp.ones, (), self.param_dtype
    )

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      v0 = _l2_normalize(
          jnp.matmul(u0, value.transpose([1, 0])), eps=self.epsilon
      )
      u0 = _l2_normalize(jnp.matmul(v0, value), eps=self.epsilon)

    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]

    value /= sigma
    value_bar = value.reshape(value_shape)

    if update_stats:
      u_var.value = u0
      sigma_var.value = sigma

    dtype = canonicalize_dtype(x, u0, v0, sigma, dtype=self.dtype)
    return jnp.asarray(value_bar, dtype)

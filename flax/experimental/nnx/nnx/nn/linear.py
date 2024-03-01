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
from __future__ import annotations

import typing as tp
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from flax.experimental import nnx
from flax.experimental.nnx.nnx import rnglib, variables
from flax.experimental.nnx.nnx.module import Module
from flax.experimental.nnx.nnx.nn import dtypes, initializers
from flax.typing import (
  Array,
  Dtype,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)

Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, tp.Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: tp.Sequence[int] | int) -> tuple[int, ...]:
  if isinstance(x, tp.Iterable):
    return tuple(x)
  else:
    return (x,)


class LinearGeneral(Module):
  """A linear transformation with flexible axes.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> # equivalent to `nn.Linear(features=4)`
    >>> layer = nn.LinearGeneral(features=4)
    >>> # output features (4, 5)
    >>> layer = nn.LinearGeneral(features=(4, 5))
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    >>> jax.tree_map(jnp.shape, params)
    {'params': {'bias': (4, 5), 'kernel': (3, 4, 5)}}
    >>> # apply transformation on the the second and last axes
    >>> layer = nn.LinearGeneral(features=(4, 5), axis=(1, -1))
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3, 6, 7)))
    >>> jax.tree_map(jnp.shape, params)
    {'params': {'bias': (4, 5), 'kernel': (3, 7, 4, 5)}}

  Attributes:
    features: int or tuple with number of output features.
    axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    batch_dims: tuple with batch axes.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
  """

  def __init__(
    self,
    in_features: Size | tp.Sequence[Size],
    out_features: Size | tp.Sequence[Size],
    *,
    axis: Axis | tp.Sequence[Axis] = -1,
    batch_axis: tp.Mapping[Axis, Size] = MappingProxyType({}),
    use_bias: bool = True,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = initializers.zeros_init(),
    precision: PrecisionLike = None,
    # Deprecated. Will be removed.
    dot_general: DotGeneralT | None = None,
    dot_general_cls: tp.Any = None,
    rngs: rnglib.Rngs,
  ):
    self.in_features = _canonicalize_tuple(in_features)
    self.out_features = _canonicalize_tuple(out_features)
    self.axis = _canonicalize_tuple(axis)
    self.batch_axis = MappingProxyType(batch_axis)
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.precision = precision
    self.dot_general = dot_general
    self.dot_general_cls = dot_general_cls

    if len(self.in_features) != len(self.axis):
      raise ValueError(
        'in_features and axis must have the same length. '
        f'Got {self.in_features} and {self.axis}.'
      )

    if batch_axis:
      batch_dims = tuple(batch_axis.keys())
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError(
          'batch_dims %s must be consecutive leading '
          'dimensions starting from 0.' % str(batch_dims)
        )

    n_batch_axis = len(self.batch_axis)
    n_in_features = len(self.in_features)
    n_out_features = len(self.out_features)

    def kernel_init_wrap(rng, shape, dtype):
      flat_shape = (
        np.prod(shape[:n_batch_axis])
        * np.prod(shape[n_batch_axis : n_in_features + n_batch_axis]),
        np.prod(shape[-n_out_features:]),
      )
      flat_shape = jax.tree_map(int, flat_shape)
      kernel = self.kernel_init(rng, flat_shape, dtype)
      if isinstance(kernel, variables.VariableMetadata):
        kernel.raw_value = jnp.reshape(kernel.raw_value, shape)
      else:
        kernel = jnp.reshape(kernel, shape)

      return kernel

    batch_shape = tuple(self.batch_axis.values())
    kernel_shape = (
      *batch_shape,
      *self.in_features,
      *self.out_features,
    )
    self.kernel = nnx.Param(
      kernel_init_wrap(rngs.params(), kernel_shape, self.param_dtype)
    )

    if self.use_bias:

      def bias_init_wrap(rng, shape, dtype):
        flat_shape = (int(np.prod(shape)),)
        bias = self.bias_init(rng, flat_shape, dtype)
        if isinstance(bias, variables.VariableMetadata):
          bias.raw_value = jnp.reshape(bias.raw_value, shape)
        else:
          bias = jnp.reshape(bias, shape)
        return bias

      bias_shape = (*batch_shape, *self.out_features)
      self.bias = nnx.Param(
        bias_init_wrap(rngs.params(), bias_shape, self.param_dtype)
      )
    else:
      self.bias = nnx.Param(None)

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    ndim = inputs.ndim
    n_batch_dims = len(self.batch_axis)
    axis = _normalize_axes(self.axis, ndim)
    batch_axis = _normalize_axes(tuple(self.batch_axis.keys()), ndim)
    n_axis = len(axis)

    # batch and non-contracting dims of input with 1s for batch dims.
    expanded_batch_shape = tuple(
      inputs.shape[ax] if ax in batch_axis else 1
      for ax in range(inputs.ndim)
      if ax not in axis
    )
    kernel = self.kernel.value
    bias = self.bias.value

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))

    inputs, kernel, bias = dtypes.promote_dtype(
      inputs, kernel, bias, dtype=self.dtype
    )

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    out = dot_general(
      inputs,
      kernel,
      ((axis, contract_ind), (batch_axis, batch_ind)),
      precision=self.precision,
    )
    # dot_general output has shape [batch_dims/group_dims] + [feature_dims]
    if bias is not None:
      # expand bias shape to broadcast bias over batch dims.
      bias = jnp.reshape(bias, (*expanded_batch_shape, *self.out_features))
      out += bias
    return out


class Linear(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = initializers.zeros_init(),
    dot_general: DotGeneralT = lax.dot_general,
    rngs: rnglib.Rngs,
  ):
    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
    else:
      self.bias = nnx.Param(None)

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.dot_general = dot_general

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.kernel.value
    bias = self.bias.value

    inputs, kernel, bias = dtypes.promote_dtype(
      inputs, kernel, bias, dtype=self.dtype
    )
    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Conv(Module):
  """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`
      (default: 1). Convolution with input dilation `d` is equivalent to
      transposed convolution with stride `d`.
    kernel_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: tp.Union[None, int, tp.Sequence[int]] = 1,
    *,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    mask_fn: tp.Optional[tp.Callable[[Array], Array]] = None,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = initializers.zeros_init(),
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel = nnx.Param(kernel_init(kernel_key, kernel_shape, param_dtype))

    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = nnx.Param(None)

    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.mask_fn = mask_fn
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated

  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    assert isinstance(self.kernel_size, tuple)
    kernel_size = self.kernel_size

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tp.Tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: tp.List[tp.Tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    # One shared convolutional kernel for all pixels in the output.
    assert self.in_features % self.feature_group_count == 0

    kernel = self.kernel.value

    if self.mask_fn is not None:
      kernel = self.mask_fn(kernel)

    bias = self.bias.value

    inputs, kernel, bias = dtypes.promote_dtype(
      inputs, kernel, bias, dtype=self.dtype
    )

    y = self.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y


default_embed_init = initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)


class Embed(Module):
  """Embedding Module.

  A parameterized function from integers [0, ``num_embeddings``) to
  ``features``-dimensional vectors. This ``Module`` will create an ``embedding``
  matrix with shape ``(num_embeddings, features)``. When calling this layer,
  the input values will be used to 0-index into the ``embedding`` matrix.
  Indexing on a value greater than or equal to ``num_embeddings`` will result
  in ``nan`` values. When ``num_embeddings`` equals to 1, it will
  broadcast the ``embedding`` matrix to input shape with ``features``
  dimension appended.

  Attributes:
    num_embeddings: number of embeddings / vocab size.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: same as embedding).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
  """

  def __init__(
    self,
    num_embeddings: int,
    features: int,
    *,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    embedding_init: Initializer = default_embed_init,
    rngs: rnglib.Rngs,
  ):
    self.embedding = nnx.Param(
      embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
    )

    self.num_embeddings = num_embeddings
    self.features = features
    self.dtype = dtype or self.embedding.value.dtype
    self.param_dtype = param_dtype
    self.embedding_init = embedding_init

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
        Values in the input array must be integers.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    # Use take because fancy indexing numpy arrays with JAX indices does not
    # work correctly.
    (embedding,) = dtypes.promote_dtype(
      self.embedding.value, dtype=self.dtype, inexact=False
    )
    if self.num_embeddings == 1:
      return jnp.where(
        jnp.broadcast_to(inputs[..., None], inputs.shape + (self.features,))
        == 0,
        embedding,
        jnp.nan,
      )
    return jnp.take(embedding, inputs, axis=0)

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    query, embedding = dtypes.promote_dtype(
      query, self.embedding.value, dtype=self.dtype
    )
    return jnp.dot(query, embedding.T)

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

"""Recurrent neural network modules.

THe RNNCell modules can be scanned using lifted transforms. For more information
see: https://flax.readthedocs.io/en/latest/advanced_topics/lift.html.
"""

import abc
from functools import partial   # pylint: disable=g-importing-member
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from flax.linen.activation import sigmoid
from flax.linen.activation import tanh
from flax.linen.dtypes import promote_dtype
from flax.linen import initializers
from flax.linen.linear import Conv
from flax.linen.linear import default_kernel_init
from flax.linen.linear import Dense
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import Module
from jax import numpy as jnp
from jax import random
import numpy as np

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


class RNNCellBase(Module):
  """RNN cell base class."""

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    raise NotImplementedError


class LSTMCell(RNNCellBase):
  r"""LSTM cell.

  The mathematical definition of the cell is as follows

  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  where x is the input, h is the output of the previous time step, and c is
  the memory.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry, inputs):
    r"""A long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      kernel_init=self.kernel_init,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype)
    i = self.gate_fn(dense_i(name='ii')(inputs) + dense_h(name='hi')(h))
    f = self.gate_fn(dense_i(name='if')(inputs) + dense_h(name='hf')(h))
    g = self.activation_fn(dense_i(name='ig')(inputs) + dense_h(name='hg')(h))
    o = self.gate_fn(dense_i(name='io')(inputs) + dense_h(name='ho')(h))
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class DenseParams(Module):
  """Dummy module for creating parameters matching `flax.linen.Dense`."""

  features: int
  use_bias: bool = True
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

  @compact
  def __call__(self, inputs: Array) -> Tuple[Array, Array]:
    k = self.param(
        'kernel', self.kernel_init, (inputs.shape[-1], self.features),
        self.param_dtype)
    if self.use_bias:
      b = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
    else:
      b = None
    return k, b


class OptimizedLSTMCell(RNNCellBase):
  r"""More efficient LSTM Cell that concatenates state components before matmul.

  The parameters are compatible with `LSTMCell`. Note that this cell is often
  faster than `LSTMCell` as long as the hidden size is roughly <= 2048 units.

  The mathematical definition of the cell is the same as `LSTMCell` and as
  follows

  .. math::

      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  where x is the input, h is the output of the previous time step, and c is
  the memory.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init()).
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry: Tuple[Array, Array],
               inputs: Array) -> Tuple[Tuple[Array, Array], Array]:
    r"""An optimized long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell, initialized using
        `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step. All
        dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]

    def _concat_dense(inputs: Array,
                      params: Mapping[str, Tuple[Array, Array]],
                      use_bias: bool = True) -> Array:
      # Concatenates the individual kernels and biases, given in params, into a
      # single kernel and single bias for efficiency before applying them using
      # dot_general.
      kernels, biases = zip(*params.values())
      kernel = jnp.concatenate(kernels, axis=-1)
      if use_bias:
        bias = jnp.concatenate(biases, axis=-1)
      else:
        bias = None
      inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
      y = jnp.dot(inputs, kernel)
      if use_bias:
        # This assert is here since mypy can't infer that bias cannot be None
        assert bias is not None
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

      # Split the result back into individual (i, f, g, o) outputs.
      split_indices = np.cumsum([kernel.shape[-1] for kernel in kernels[:-1]])
      ys = jnp.split(y, split_indices, axis=-1)
      return dict(zip(params.keys(), ys))

    # Create params with the same names/shapes as `LSTMCell` for compatibility.
    dense_params_h = {}
    dense_params_i = {}
    for component in ['i', 'f', 'g', 'o']:
      dense_params_i[component] = DenseParams(
          features=hidden_features, use_bias=False,
          param_dtype=self.param_dtype,
          kernel_init=self.kernel_init, bias_init=self.bias_init,
          name=f'i{component}')(inputs)
      dense_params_h[component] = DenseParams(
          features=hidden_features, use_bias=True,
          param_dtype=self.param_dtype,
          kernel_init=self.recurrent_kernel_init, bias_init=self.bias_init,
          name=f'h{component}')(h)
    dense_h = _concat_dense(h, dense_params_h, use_bias=True)
    dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

    i = self.gate_fn(dense_h['i'] + dense_i['i'])
    f = self.gate_fn(dense_h['f'] + dense_i['f'])
    g = self.activation_fn(dense_h['g'] + dense_i['g'])
    o = self.gate_fn(dense_h['o'] + dense_i['o'])

    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.

    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class GRUCell(RNNCellBase):
  r"""GRU cell.

  The mathematical definition of the cell is as follows

  .. math::

      \begin{array}{ll}
      r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
      z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
      n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
      h' = (1 - z) * n + z * h \\
      \end{array}

  where x is the input and h, is the output of the previous time step.

  Attributes:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry, inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the GRU cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    r = self.gate_fn(dense_i(name='ir')(inputs) + dense_h(name='hr')(h))
    z = self.gate_fn(dense_i(name='iz')(inputs) + dense_h(name='hz')(h))
    # add bias because the linear transformations aren't directly summed.
    n = self.activation_fn(dense_i(name='in')(inputs) +
                           r * dense_h(name='hn', use_bias=True)(h))
    new_h = (1. - z) * n + z * h
    return new_h, new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)


class ConvLSTMCell(RNNCellBase):
  r"""A convolutional LSTM cell.

  The implementation is based on xingjian2015convolutional.
  Given x_t and the previous state (h_{t-1}, c_{t-1})
  the core computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where * denotes the convolution operator;
  i_t, f_t, o_t are input, forget and output gate activations,
  and g_t is a vector of cell updates.

  Notes:
    Forget gate initialization:
      Following jozefowicz2015empirical we add 1.0 to b_f
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  kernel_size: Sequence[int]
  strides: Optional[Sequence[int]] = None
  padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME'
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry, inputs):
    """Constructs a convolutional LSTM.

    Args:
      carry: the hidden state of the Conv2DLSTM cell,
        initialized using `Conv2DLSTM.initialize_carry`.
      inputs: input data with dimensions (batch, spatial_dims..., features).
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    input_to_hidden = partial(Conv,
                              features=4*self.features,
                              kernel_size=self.kernel_size,
                              strides=self.strides,
                              padding=self.padding,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              name='ih')

    hidden_to_hidden = partial(Conv,
                               features=4*self.features,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding=self.padding,
                               use_bias=self.use_bias,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype,
                               name='hh')

    gates = input_to_hidden()(inputs) + hidden_to_hidden()(h)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = sigmoid(f + 1)
    new_c = f * c + sigmoid(i) * jnp.tanh(g)
    new_h = sigmoid(o) * jnp.tanh(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the input_shape + (features,).
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + size
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)

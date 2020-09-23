# Copyright 2020 The Flax Authors.
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

THe RNNCell modules are designed to fit in with the scan function in JAX::

  _, initial_params = LSTMCell.init(rng_1, time_series[0])
  model = nn.Model(LSTMCell, initial_params)
  carry = LSTMCell.initialize_carry(rng_2, (batch_size,), memory_size)
  carry, y = jax.lax.scan(model, carry, time_series)

"""

import abc
from functools import partial
from typing import (Any, Callable, Sequence, Optional, Tuple, Union)

from .module import Module, compact
from . import activation
from . import initializers
from . import linear

from jax import numpy as jnp
from jax import random


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


class RNNCellBase(Module):
  """RNN cell base class."""

  @staticmethod
  @abc.abstractmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    pass


class LSTMCell(RNNCellBase):
  r"""LSTM cell.
    the mathematical definition of the cell is as follows
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

  Args:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros)
  """
  gate_fn: Callable = activation.sigmoid
  activation_fn: Callable = activation.tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = linear.default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros

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
    dense_h = partial(linear.Dense,
                      features=hidden_features,
                      use_bias=True,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(linear.Dense,
                      features=hidden_features,
                      use_bias=False,
                      kernel_init=self.kernel_init)
    i = self.gate_fn(dense_i(name='ii')(inputs) + dense_h(name='hi')(h))
    f = self.gate_fn(dense_i(name='if')(inputs) + dense_h(name='hf')(h))
    g = self.activation_fn(dense_i(name='ig')(inputs) + dense_h(name='hg')(h))
    o = self.gate_fn(dense_i(name='io')(inputs) + dense_h(name='ho')(h))
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.

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

  the mathematical definition of the cell is as follows
  .. math::
      \begin{array}{ll}
      r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
      z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
      n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
      h' = (1 - z) * n + z * h
      \end{array}
  where x is the input and h, is the output of the previous time step.

  Args:
    gate_fn: activation function used for gates (default: sigmoid)
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros)
  """
  gate_fn: Callable = activation.sigmoid
  activation_fn: Callable = activation.tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      linear.default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros

  @compact
  def __call__(self, carry, inputs):
    """Gated recurrent unit (GRU) cell.

    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(linear.Dense,
                      features=hidden_features,
                      use_bias=False,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(linear.Dense,
                      features=hidden_features,
                      use_bias=True,
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
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.

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


class ConvLSTM(RNNCellBase):
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

  Args:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
  """

  features: int
  kernel_size: Sequence[int]
  strides: Optional[Sequence[int]] = None
  padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME'
  use_bias: bool = True
  dtype: Dtype = jnp.float32

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
    input_to_hidden = partial(linear.Conv,
                              features=4*self.features,
                              kernel_size=self.kernel_size,
                              strides=self.strides,
                              padding=self.padding,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              name='ih')

    hidden_to_hidden = partial(linear.Conv,
                               features=4*self.features,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding=self.padding,
                               use_bias=self.use_bias,
                               dtype=self.dtype,
                               name='hh')

    gates = input_to_hidden()(inputs) + hidden_to_hidden()(h)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = activation.sigmoid(f + 1)
    new_c = f * c + activation.sigmoid(i) * jnp.tanh(g)
    new_h = activation.sigmoid(o) * jnp.tanh(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros):
    """initialize the RNN cell carry.

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

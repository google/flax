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

"""DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Recurrent neural network modules.

THe RNNCell modules are designed to fit in with the scan function in JAX::

  _, initial_params = LSTMCell.init(rng_1, time_series[0])
  model = nn.Model(LSTMCell, initial_params)
  carry = LSTMCell.initialize_carry(rng_2, (batch_size,), memory_size)
  carry, y = jax.lax.scan(model, carry, time_series)

"""

import abc

from . import activation
from . import base
from . import initializers
from . import linear

from jax import numpy as jnp
from jax import random
from jax import lax
import numpy as np


class RNNCellBase(base.Module):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  RNN cell base class."""

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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  LSTM cell."""

  def apply(self, carry, inputs,
            gate_fn=activation.sigmoid, activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""A long short-term memory (LSTM) cell.

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
      carry: the hidden state of the LSTM cell,
        initialized using `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = linear.Dense.partial(
        inputs=h, features=hidden_features, bias=True,
        kernel_init=recurrent_kernel_init, bias_init=bias_init)
    dense_i = linear.Dense.partial(
        inputs=inputs, features=hidden_features, bias=False,
        kernel_init=kernel_init)
    i = gate_fn(dense_i(name='ii') + dense_h(name='hi'))
    f = gate_fn(dense_i(name='if') + dense_h(name='hf'))
    g = activation_fn(dense_i(name='ig') + dense_h(name='hg'))
    o = gate_fn(dense_i(name='io') + dense_h(name='ho'))
    new_c = f * c + i * g
    new_h = o * activation_fn(new_c)
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


class OptimizedLSTMCell(RNNCellBase):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  More efficient LSTM Cell that concatenates state components before matmul.

  Parameters are compatible with `flax.nn.LSTMCell`.
  """

  class DummyDense(base.Module):
    """Dummy module for creating parameters matching `flax.nn.Dense`."""

    def apply(self,
              inputs,
              features,
              kernel_init,
              bias_init,
              bias=True):
      k = self.param('kernel', (inputs.shape[-1], features), kernel_init)
      b = (self.param('bias', (features,), bias_init)
           if bias else jnp.zeros((features,)))
      return k, b

  def apply(self,
            carry,
            inputs,
            gate_fn=activation.sigmoid,
            activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""A long short-term memory (LSTM) cell.

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
      carry: the hidden state of the LSTM cell, initialized using
        `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step. All
        dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)

    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]

    def _concat_dense(inputs, params, use_bias=True):
      kernels, biases = zip(*params.values())
      kernel = jnp.asarray(jnp.concatenate(kernels, axis=-1), jnp.float32)

      y = jnp.dot(inputs, kernel)
      if use_bias:
        bias = jnp.asarray(jnp.concatenate(biases, axis=-1), jnp.float32)
        y = y + bias

      # Split the result back into individual (i, f, g, o) outputs.
      split_indices = np.cumsum([b.shape[0] for b in biases[:-1]])
      ys = jnp.split(y, split_indices, axis=-1)
      return dict(zip(params.keys(), ys))

    # Create the params in the same order as LSTMCell for initialization
    # compatibility.
    dense_params_h = {}
    dense_params_i = {}
    for component in ['i', 'f', 'g', 'o']:
      dense_params_i[component] = OptimizedLSTMCell.DummyDense(
          inputs=inputs, features=hidden_features, bias=False,
          kernel_init=kernel_init, bias_init=bias_init,
          name=f'i{component}')
      dense_params_h[component] = OptimizedLSTMCell.DummyDense(
          inputs=h, features=hidden_features, bias=True,
          kernel_init=recurrent_kernel_init, bias_init=bias_init,
          name=f'h{component}')
    dense_h = _concat_dense(h, dense_params_h, use_bias=True)
    dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

    i = gate_fn(dense_h['i'] + dense_i['i'])
    f = gate_fn(dense_h['f'] + dense_i['f'])
    g = activation_fn(dense_h['g'] + dense_i['g'])
    o = gate_fn(dense_h['o'] + dense_i['o'])

    new_c = f * c + i * g
    new_h = o * activation_fn(new_c)
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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  GRU cell."""

  def apply(self, carry, inputs,
            gate_fn=activation.sigmoid, activation_fn=activation.tanh,
            kernel_init=linear.default_kernel_init,
            recurrent_kernel_init=initializers.orthogonal(),
            bias_init=initializers.zeros):
    r"""Gated recurrent unit (GRU) cell.

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
      carry: the hidden state of the LSTM cell,
        initialized using `GRUCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)
    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = linear.Dense.partial(
        inputs=h, features=hidden_features, bias=False,
        kernel_init=recurrent_kernel_init, bias_init=bias_init)
    dense_i = linear.Dense.partial(
        inputs=inputs, features=hidden_features, bias=True,
        kernel_init=kernel_init, bias_init=bias_init)
    r = gate_fn(dense_i(name='ir') + dense_h(name='hr'))
    z = gate_fn(dense_i(name='iz') + dense_h(name='hz'))
    # add bias because the linear transformations aren't directly summed.
    n = activation_fn(dense_i(name='in') + r * dense_h(name='hn', bias=True))
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
  r"""DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A convolutional LSTM cell.

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
  """

  def apply(self, 
            carry, 
            inputs, 
            features, 
            kernel_size, 
            strides=None,
            padding='SAME',
            bias=True, 
            dtype=jnp.float32):
    """Constructs a convolutional LSTM.

    Args:
      carry: the hidden state of the Conv2DLSTM cell,
        initialized using `Conv2DLSTM.initialize_carry`.
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    input_to_hidden = linear.Conv.partial(
        features=4*features, 
        kernel_size=kernel_size, 
        strides=strides,
        padding=padding,
        bias=bias, 
        dtype=dtype,
        name="ih")

    hidden_to_hidden = linear.Conv.partial(
        features=4*features, 
        kernel_size=kernel_size, 
        strides=strides,
        padding=padding,
        bias=bias, 
        dtype=dtype,
        name="hh")

    gates = input_to_hidden(inputs) + hidden_to_hidden(h)
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

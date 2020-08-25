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

from . import activation
from . import base
from . import initializers
from . import linear

from jax import random


class RNNCellBase(base.Module):
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
  """LSTM cell."""

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


class GRUCell(RNNCellBase):
  """GRU cell."""

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


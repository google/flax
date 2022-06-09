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
see: https://flax.readthedocs.io/en/latest/design_notes/lift.html.
"""

from functools import partial   # pylint: disable=g-importing-member
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import jax
from flax.linen import transforms
from flax.linen.activation import sigmoid
from flax.linen.activation import tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import orthogonal
from flax.linen.initializers import zeros
from flax.linen.linear import Conv
from flax.linen.linear import default_kernel_init
from flax.linen.linear import Dense
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import Module
from flax.core import lift
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from jax import random
import numpy as np

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


class RNNCellBase(Module):
  """RNN cell base class."""

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
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
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros)
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
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
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
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
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

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
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros).
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
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
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
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
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros)
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

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
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
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
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
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

class RNNBase(Module):
  """Base class for RNN layers, in contains an `rnn_forward` method
  that runs scan with a cell over the inputs. Subclasses can parameterize
  `rnn_forward` to define their `__call__` method.

  Here is an example of a barebone LSTM layer implementation::

    class LSTM(RNNBase):
      units: int

      @compact
      def __call__(self, inputs):
        return self.rnn_forward(
          LSTMCell(), inputs,
          init_key=jax.random.PRNGKey(0), time_axis=-2,
          carry_size=self.units, stateful=False,
          return_state=False, reverse=False, 
          initial_state=None, variable_axes={},
          variable_broadcast='params', variable_carry=False, 
          split_rngs={'params': False},
        )
  """
  def rnn_forward(
    self,
    cell: RNNCellBase,
    inputs: jnp.ndarray,
    *,
    init_key: jnp.ndarray,
    time_axis: int,
    carry_size: Union[int, Tuple[int, ...]],
    stateful: bool,
    return_state: bool,
    reset_state: bool,
    initial_state: Optional[Any],
    mask: Optional[jnp.ndarray],
    zero_output_for_mask: bool,
    # scan args
    reverse: bool,
    unroll: int,
    variable_axes: Mapping[lift.CollectionFilter,lift.InOutScanAxis],
    variable_broadcast: lift.CollectionFilter,
    variable_carry: lift.CollectionFilter,
    split_rngs: Mapping[lift.PRNGSequenceFilter, bool],
  ):
    """
    Takes an RNNCellBase `cell` and an input sequence `x` and `scan`s the 
    cell over the sequence.

    **Collections**
  
    * `memory`: if `stateful` is True, the RNN cell's state is stored in a variable
      named `carry` under the `memory` collection.
    * The `cell` might add its own collections (like `params`) as needed.
    
    **Input shape**
    
    `x` should be an ndarray of shape `(*batch, time, *feature)`where `*batch` are the batch
    dimensions, `time` is the time dimension, and `*feature` are the feature dimensions. The 
    time dimension can be in any position, internally the input will be processed as 
    `(*batch, *feature)` at each time step. The feature dimensions should be last and are 
    specified by the number of dimensions in `carry_size`. The batch dimensions will simply
    be the remaining ones that are not the time axis or the features.

    **Output shape**

    If `return_state` is True, the output will be a `(carry, output)` tuple, otherwise
    only the output will be returned. The `output` will be of shape `(*batch, time, *feature_out)`.

    Args:
      cell: an `RNNCellBase` instance.
      x: the input sequence.
      init_key: a random key to initialize the cell's carry.
      time_axis: the axis corresponding to the time dimension.
      carry_size: the `size` argument passed to `cell.initialize_carry`.
      stateful: if True, the RNN cell's state is store in a variable named `carry` 
        in the `memory` collection.
      return_state: if True, a (carry, output) tuple is returned, otherwise only
        the output is returned.
      reset_state: if True, the RNN cell's state is reset to the initial state before
        processing the sequence. 
      reverse: if True, the input sequence is processed backwards, this argument is
        passed to `scan`.
      unrool: how many scan iterations to unroll within a single
        iteration of a loop, this argument is passed to `scan`.
      initial_state: an optional initial `carry` for the RNN cell. If stateful is True,
        this will override the current value of the `carry`.
      mask: an optional mask to apply to the input sequence. If given, it should be
        an binary array of shape `(*batch, time)` indicating whether a given timestep
        should be masked. True indicates that the timestep is used, False indicates
        that the timestep is ignored.
      zero_output_for_mask: when `mask` is given, if True, the output of masked-out
        timesteps is set to zeros, else the output is set to the value of the
        previous available timestep, if there are no previous available timesteps the
        output is set to zeros.
      variable_axes: the variable collections that are scanned over, this argument will 
        be passed to `nn.scan`.
      variable_broadcast: specifies the broadcasted variable collections, this argument
        will be passed to `nn.scan`.
      variable_carry: specifies the variable collections that are carried through
        the loop, this argument will be passed to `nn.scan`.
      split_rngs: a mapping from PRNGSequenceFilter to bool specifying whether to
        a PRNG collection should be split such that its values are different at each
        step, or replicated such that its values remain the same at each step. This
        argument will be passed to `nn.scan`.
    """

    num_feature_dims = 1 if isinstance(carry_size, int) else len(carry_size)
    time_axis = time_axis if time_axis >= 0 else time_axis + inputs.ndim
    mask_axis = 0 if mask is None else mask.ndim - 1

    if num_feature_dims < 1:
      raise ValueError(
        f'\'carry_size\' must have at least 1 element, got: \'{carry_size}\''
      )
    elif num_feature_dims >= len(inputs.shape):
      raise ValueError(
        f'The length of \'carry_size\' must be less than the length of the input '
        f'shape, got: \'{carry_size}\' and \'{inputs.shape}\''
      )
    
    num_batch_dims = inputs.ndim - (num_feature_dims + 1) # features dims + time dim

    # WARNING: this definition of initialization state
    # is known to be problematic: https://github.com/google/flax/issues/652#issuecomment-1124216543
    # Epecifically, this module will not update its cache if 
    # 'params' are mutable, which can arise if users use `mutable=True`.
    is_initializing = self.is_mutable_collection("params")

    # validate mask and calculate is_padding_mask
    if mask is not None:
      if mask.ndim == num_batch_dims:
        if mask.dtype != jnp.int32:
          raise ValueError(
            f'\'mask\' must be an int32 array, got: \'{mask.dtype}\''
          )
        is_padding_mask = True
      elif mask.ndim == num_batch_dims + 1:
        if mask.dtype != jnp.bool_:
          raise ValueError(
            f'\'mask\' must be a bool array, got: \'{mask.dtype}\''
          )
        is_padding_mask = False
      else:
        raise ValueError(
          f'\'mask\' must have {num_batch_dims} or {num_batch_dims + 1} '
          f'dimensions, got: \'{mask.ndim}\''
        )
    else:
      is_padding_mask = False

    # get carry
    if initial_state is not None:
      initial_carry = initial_state
    elif not reset_state and stateful and self.has_variable('memory', 'carry'):
      initial_carry = self.get_variable('memory', 'carry')
    else:
      shape_without_time = inputs.shape[:time_axis] + inputs.shape[time_axis + 1:]
      initial_carry = cell.initialize_carry(
        init_key,
        batch_dims=shape_without_time[:num_batch_dims],
        size=carry_size,
      )

    def scan_fn(
      cell: RNNCellBase, state: Tuple[Any, Array], x: Array, 
      mask: Optional[jnp.ndarray]
    ) -> Tuple[Tuple[Any, Array], Tuple[Any, Array]]:
      
      carry_in, prev_non_masked_output = state
      carry_next, y_next = cell(carry_in, x)

      if mask is None or is_padding_mask:
        carry_out, y_out = carry_next, y_next
      else:
        def apply_mask(value, masked_value):
          # reshape mask to have same number of dimensions as value
          mask_ = mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))
          return jnp.where(mask_, value, masked_value)
        
        carry_out = jax.tree_map(apply_mask, carry_next, carry_in)

        if zero_output_for_mask:
          y_out = jax.tree_map(apply_mask, y_next, jnp.zeros_like(y_next))
        else:
          y_out = jax.tree_map(apply_mask, y_next, prev_non_masked_output)
          prev_non_masked_output = y_out

      return (carry_out, prev_non_masked_output), (carry_out, y_out)

    scan = transforms.scan(
      scan_fn,
      in_axes=(time_axis, mask_axis), 
      out_axes=(0, time_axis),
      reverse=reverse,
      unroll=unroll,
      variable_axes=variable_axes,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      split_rngs=split_rngs,
    )

    if mask is not None and not zero_output_for_mask and not is_padding_mask:
      first_input = jax.lax.index_in_dim(inputs, 0, time_axis, keepdims=False)
      initial_non_masked_output = jnp.zeros_like(cell(initial_carry, first_input)[1])
    else:
      initial_non_masked_output = None

    (carry, _), (carry_t, outputs) = scan(cell, (initial_carry, initial_non_masked_output), inputs, mask)

    if is_padding_mask:
      carry = jnp.take_along_axis(carry_t, mask, axis=0)

    # if the module is initializing keep the initial carry
    if is_initializing:
      carry = initial_carry

    if stateful:
      self.put_variable('memory', 'carry', carry)

    if return_state:
      return carry, outputs
    else:
      return outputs

class RNN(RNNBase):
  """
  A `RNN` layer that takes an arbitrary `RNNCellBase` and performs a `scan` over
  the input's time dimension.

  Example::
    >>> import jax.numpy as jnp
    >>> import jax
    >>> import flax.linen as nn
    >>> x = jnp.ones((10, 50, 32))
    >>> lstm = nn.RNN(nn.LSTMCell(), 64)
    >>> variables = lstm.init(jax.random.PRNGKey(0), x)
    >>> y = lstm.apply(variables, x)
    >>> y.shape
    (10, 50, 64)

  **Collections**

    * `memory`: if `stateful` is True, the RNN cell's state is stored in a variable
      named `carry` under the `memory` collection.
    * The `cell` might add its own collections (like `params`) as needed.

  **Input shape**
    
  `x` should be an ndarray of shape `(*batch, time, *feature)`where `*batch` are the batch
  dimension, `time` is the time dimension, and `*feature` are the feature dimensions. The time
  dimension can be in any position, internally the input will be processed as 
  `(*batch, *feature)` at each time step. The feature dimensions should be last and are 
  specified by the number of dimensions in `carry_size`. The batch dimensions will simply
  be the remaining ones that are not the time axis or the features.

  **Output shape**

  If `return_state` is True, the output will be a `(carry, output)` tuple, otherwise
  only the output will be returned. The `output` will be of shape `(*batch, time, *feature)`.

  Attributes:
    cell: the `RNNCellBase` instance.
    carry_size: the `size` argument passed to `cell.initialize_carry`.
    time_axis: the axis from the input corresponding to the time dimension.
    stateful: if True, the RNN cell's state is stored in a variable named `carry`
      in the `memory` collection.
    return_state: if True, a (carry, output) tuple is returned, otherwise only
      the output is returned. Defaults to False.
    reverse: if True, the input sequence is processed backwards. Defaults to False.
    unroll: how many scan iterations to unroll within a single iteration of a loop,
      defaults to 1.
    zero_output_for_mask: when `mask` is given, if True, the output of masked-out
      timesteps is set to zeros, else the output is set to the value of the
      previous available timestep, if there are no previous available timesteps the
      output is set to zeros.
    variable_axes: the variable collections that are scanned over, this argument will
      be passed to `nn.scan`.
    variable_broadcast: specifies the broadcasted variable collections, this argument
      will be passed to `nn.scan`.
    variable_carry: specifies the variable collections that are carried through
      the loop, this argument will be passed to `nn.scan`.
    split_rngs: a mapping from PRNGSequenceFilter to bool specifying whether to
      a PRNG collection should be split such that its values are different at each
      step, or replicated such that its values remain the same at each step. This
      argument will be passed to `nn.scan`.
  """
  cell: RNNCellBase
  carry_size: Union[int, Tuple[int, ...]]
  time_axis: Optional[int] = None
  stateful: bool = False
  return_state: bool = False
  reverse: bool = False
  unroll: int = 1
  zero_output_for_mask: bool = True
  # scan args
  variable_axes: Mapping[lift.CollectionFilter,lift.InOutScanAxis] = FrozenDict()
  variable_broadcast: lift.CollectionFilter = ('params',)
  variable_carry: lift.CollectionFilter = False
  split_rngs: Mapping[lift.PRNGSequenceFilter, bool] = FrozenDict({'params': False})
  
  def __call__(self, 
    inputs: jnp.ndarray, init_key: Optional[jnp.ndarray] = None, 
    initial_state: Optional[Any] = None, reset_state: bool = False,
    mask: Optional[jnp.ndarray] = None,
    # overrides
    return_state: Optional[bool] = None, reverse: Optional[bool] = None,
    zero_output_for_mask: Optional[bool] = None,
  ):
    """
    Arguments:
      inputs: input tensor.
      init_key: PRNG key.
      initial_state: initial state of the RNN cell.
      reset_state: if True, the RNN cell's state is reset to the initial state before
        processing the sequence.
      return_state: if True, a (carry, output) tuple is returned, otherwise only the
        output is returned. If given, this argument overrides the `return_state` attribute
        defined in the constructor.
      reverse: if True, the input sequence is processed backwards. If given, this argument
        overrides the `reverse` attribute defined in the constructor.
      zero_output_for_mask: when `mask` is given, if True, the output of masked-out
        timesteps is set to zeros, else the output is set to the value of the
        previous available timestep, if there are no previous available timesteps the
        output is set to zeros. If given, this argument overrides the `zero_output_for_mask`
        attribute defined in the constructor.
    """
    num_feature_dims = 1 if isinstance(self.carry_size, int) else len(self.carry_size)

    return self.rnn_forward(
      self.cell, inputs,
      init_key=init_key if init_key is not None else jax.random.PRNGKey(0),
      time_axis=self.time_axis 
      if self.time_axis is not None 
      else -(num_feature_dims + 1), # 1 before features dims
      carry_size=self.carry_size,
      stateful=self.stateful,
      return_state=return_state if return_state is not None else self.return_state,
      reset_state=reset_state,
      initial_state=initial_state,
      mask=mask,
      zero_output_for_mask=zero_output_for_mask 
      if zero_output_for_mask is not None 
      else self.zero_output_for_mask,
      # scan args
      reverse=reverse if reverse is not None else self.reverse,
      unroll=self.unroll,
      variable_axes=self.variable_axes,
      variable_broadcast=self.variable_broadcast,
      variable_carry=self.variable_carry,
      split_rngs=self.split_rngs,
    )


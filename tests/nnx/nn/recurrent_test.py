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

import jax, jax.numpy as jnp
from jax import random

from flax import linen
from flax import nnx
from flax.nnx.nn import initializers

import numpy as np

from absl.testing import absltest


class TestLSTMCell(absltest.TestCase):
  def test_basic(self):
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3))
    carry = module.initialize_carry(x.shape, module.rngs)
    new_carry, y = module(carry, x)
    self.assertEqual(y.shape, (2, 4))

  def test_lstm_sequence(self):
    """Test LSTMCell over a sequence of inputs."""
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(0),
    )
    x = random.normal(random.PRNGKey(1), (5, 2, 3))  # seq_len, batch, feature
    carry = module.initialize_carry(x.shape[1:], module.rngs)
    outputs = []
    for t in range(x.shape[0]):
      carry, y = module(carry, x[t])
      outputs.append(y)
    outputs = jnp.stack(outputs)
    self.assertEqual(outputs.shape, (5, 2, 4))

  def test_lstm_with_different_dtypes(self):
    """Test LSTMCell with different data types."""
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      dtype=jnp.bfloat16,
      param_dtype=jnp.bfloat16,
      rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3), dtype=jnp.bfloat16)
    carry = module.initialize_carry(x.shape, module.rngs)
    new_carry, y = module(carry, x)
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertEqual(y.shape, (2, 4))

  def test_lstm_with_custom_activations(self):
    """Test LSTMCell with custom activation functions."""
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      gate_fn=jax.nn.relu,
      activation_fn=jax.nn.elu,
      rngs=nnx.Rngs(0),
    )
    x = jnp.ones((1, 3))
    carry = module.initialize_carry(x.shape, module.rngs)
    new_carry, y = module(carry, x)
    self.assertEqual(y.shape, (1, 4))

  def test_lstm_initialize_carry(self):
    """Test the initialize_carry method."""
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      carry_init=initializers.ones,
      rngs=nnx.Rngs(0),
    )
    x_shape = (1, 3)
    carry = module.initialize_carry(x_shape, module.rngs)
    c, h = carry
    self.assertTrue(jnp.all(c == 1.0))
    self.assertTrue(jnp.all(h == 1.0))
    self.assertEqual(c.shape, (1, 4))
    self.assertEqual(h.shape, (1, 4))

  def test_lstm_with_variable_sequence_length(self):
    """Test LSTMCell with variable sequence lengths."""
    module = nnx.LSTMCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(0))

    # Simulate a batch with variable sequence lengths
    x = jnp.array(
      [
        [[1, 2, 3], [4, 5, 6], [0, 0, 0]],  # Sequence length 2
        [[7, 8, 9], [10, 11, 12], [13, 14, 15]],  # Sequence length 3
      ]
    )  # Shape: (batch_size=2, max_seq_length=3, features=3)

    seq_lengths = jnp.array([2, 3])  # Actual lengths for each sequence
    batch_size = x.shape[0]
    max_seq_length = x.shape[1]
    carry = module.initialize_carry((batch_size, 3), module.rngs)
    outputs = []
    for t in range(max_seq_length):
      input_t = x[:, t, :]
      carry, y = module(carry, input_t)
      outputs.append(y)
    outputs = jnp.stack(
      outputs, axis=1
    )  # Shape: (batch_size, max_seq_length, hidden_features)

    # Zero out outputs beyond the actual sequence lengths
    mask = jnp.arange(max_seq_length)[None, :] < seq_lengths[:, None]
    outputs = outputs * mask[:, :, None]
    self.assertEqual(outputs.shape, (2, 3, 4))

  def test_lstm_stateful(self):
    """Test that LSTMCell maintains state across calls."""
    module = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(0),
    )
    x1 = jnp.ones((1, 3))
    x2 = jnp.ones((1, 3)) * 2
    carry = module.initialize_carry(x1.shape)
    carry, y1 = module(carry, x1)
    carry, y2 = module(carry, x2)
    self.assertEqual(y1.shape, (1, 4))
    self.assertEqual(y2.shape, (1, 4))

  def test_lstm_equivalence_with_flax_linen(self):
    """Test that nnx.LSTMCell produces the same outputs as flax.linen.LSTMCell."""
    in_features = 3
    hidden_features = 4
    key = random.PRNGKey(42)
    x = random.normal(key, (1, in_features))

    # Initialize nnx.LSTMCell
    rngs_nnx = nnx.Rngs(0)
    module_nnx = nnx.LSTMCell(
      in_features=in_features,
      hidden_features=hidden_features,
      rngs=rngs_nnx,
    )
    carry_nnx = module_nnx.initialize_carry(x.shape, rngs_nnx)
    # Initialize flax.linen.LSTMCell
    module_linen = linen.LSTMCell(
      features=hidden_features,
    )
    carry_linen = module_linen.initialize_carry(random.PRNGKey(0), x.shape)
    variables_linen = module_linen.init(random.PRNGKey(1), carry_linen, x)

    # Copy parameters from flax.linen.LSTMCell to nnx.LSTMCell
    params_linen = variables_linen['params']
    # Map the parameters from linen to nnx
    # Assuming the parameter names and shapes are compatible
    # For a precise mapping, you might need to adjust parameter names
    # Get the parameters from nnx module
    nnx_params = module_nnx.__dict__

    # Map parameters from linen to nnx
    for gate in ['i', 'f', 'g', 'o']:
      # Input kernels (input to gate)
      if gate == 'f':
        nnx_layer = getattr(module_nnx, f'if_')
      else:
        nnx_layer = getattr(module_nnx, f'i{gate}')
      linen_params = params_linen[f'i{gate}']
      nnx_layer.kernel.value = linen_params['kernel']
      if nnx_layer.use_bias:
        nnx_layer.bias.value = linen_params['bias']
      # Hidden kernels (hidden state to gate)
      nnx_layer = getattr(module_nnx, f'h{gate}')
      linen_params = params_linen[f'h{gate}']
      nnx_layer.kernel.value = linen_params['kernel']
      if nnx_layer.use_bias:
        nnx_layer.bias.value = linen_params['bias']

    # Run both modules
    new_carry_nnx, y_nnx = module_nnx(carry_nnx, x)
    new_carry_linen, y_linen = module_linen.apply(
      variables_linen, carry_linen, x
    )

    # Compare outputs
    np.testing.assert_allclose(y_nnx, y_linen, atol=1e-5)
    # Compare carries
    for c_nnx, c_linen in zip(new_carry_nnx, new_carry_linen):
      np.testing.assert_allclose(c_nnx, c_linen, atol=1e-5)


class TestRNN(absltest.TestCase):
  def test_rnn_with_lstm_cell(self):
    """Test RNN module using LSTMCell."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(0),
    )

    # Initialize the RNN module with the LSTMCell
    rnn = nnx.RNN(cell)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.ones((2, 5, 3))

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(
      outputs.shape, (2, 5, 4)
    )  # Output features should match hidden_features

  def test_rnn_with_gru_cell(self):
    """Test RNN module using GRUCell."""
    # Initialize the GRUCell
    cell = nnx.GRUCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(1),
    )

    # Initialize the RNN module with the GRUCell
    rnn = nnx.RNN(cell)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.ones((2, 5, 3))

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(
      outputs.shape, (2, 5, 4)
    )  # Output features should match hidden_features

  def test_rnn_time_major(self):
    """Test RNN module with time_major=True."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(2),
    )

    # Initialize the RNN module with time_major=True
    rnn = nnx.RNN(cell, time_major=True)

    # Create input data (seq_length=5, batch_size=2, features=3)
    x = jnp.ones((5, 2, 3))

    # Initialize the carry
    carry = cell.initialize_carry(x.shape[1:2] + x.shape[2:], cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(
      outputs.shape, (5, 2, 4)
    )  # Output features should match hidden_features

  def test_rnn_reverse(self):
    """Test RNN module with reverse=True."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(3),
    )

    # Initialize the RNN module with reverse=True
    rnn = nnx.RNN(cell, reverse=True)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.tile(jnp.arange(5), (2, 1)).reshape(
      2, 5, 1
    )  # Distinct values to check reversal
    x = jnp.concatenate([x, x, x], axis=-1)  # Shape: (2, 5, 3)

    # Run the RNN module
    outputs = rnn(x)

    # Check if the outputs are in reverse order
    outputs_reversed = outputs[:, ::-1, :]
    # Since we used distinct input values, we can compare outputs to check reversal
    # For simplicity, just check the shapes here
    self.assertEqual(outputs.shape, (2, 5, 4))
    self.assertEqual(outputs_reversed.shape, (2, 5, 4))

  def test_rnn_with_seq_lengths(self):
    """Test RNN module with variable sequence lengths."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(4),
    )

    # Initialize the RNN module
    rnn = nnx.RNN(cell, return_carry=True)

    # Create input data with padding (batch_size=2, seq_length=5, features=3)
    x = jnp.array(
      [
        [
          [1, 1, 1],
          [2, 2, 2],
          [3, 3, 3],
          [0, 0, 0],
          [0, 0, 0],
        ],  # Sequence length 3
        [
          [4, 4, 4],
          [5, 5, 5],
          [6, 6, 6],
          [7, 7, 7],
          [8, 8, 8],
        ],  # Sequence length 5
      ]
    )  # Shape: (2, 5, 3)

    seq_lengths = jnp.array([3, 5])  # Actual lengths for each sequence

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    final_carry, outputs = rnn(x, initial_carry=carry, seq_lengths=seq_lengths)

    self.assertEqual(outputs.shape, (2, 5, 4))

    self.assertEqual(
      final_carry[0].shape, (2, 4)
    )  # c: (batch_size, hidden_features)
    self.assertEqual(
      final_carry[1].shape, (2, 4)
    )  # h: (batch_size, hidden_features)

    # Todo: a better test by matching the outputs with the expected values

  def test_rnn_with_keep_order(self):
    """Test RNN module with reverse=True and keep_order=True."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(5),
    )

    # Initialize the RNN module with reverse=True and keep_order=True
    rnn = nnx.RNN(cell, reverse=True, keep_order=True)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.tile(jnp.arange(5), (2, 1)).reshape(
      2, 5, 1
    )  # Distinct values to check reversal
    x = jnp.concatenate([x, x, x], axis=-1)  # Shape: (2, 5, 3)

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    # Check if the outputs are in the original order despite processing in reverse
    self.assertEqual(outputs.shape, (2, 5, 4))

  def test_rnn_equivalence_with_flax_linen(self):
    """Test that nnx.RNN produces the same outputs as flax.linen.RNN."""
    in_features = 3
    hidden_features = 4
    seq_length = 5
    batch_size = 2
    key = random.PRNGKey(42)

    # Create input data
    x = random.normal(key, (batch_size, seq_length, in_features))

    # Initialize nnx.LSTMCell and RNN
    rngs_nnx = nnx.Rngs(0)
    cell_nnx = nnx.LSTMCell(
      in_features=in_features,
      hidden_features=hidden_features,
      rngs=rngs_nnx,
    )
    rnn_nnx = nnx.RNN(cell_nnx)

    # Initialize flax.linen.LSTMCell and RNN
    cell_linen = linen.LSTMCell(features=hidden_features)
    rnn_linen = linen.RNN(cell_linen)
    carry_linen = cell_linen.initialize_carry(random.PRNGKey(0), x[:, 0].shape)
    variables_linen = rnn_linen.init(random.PRNGKey(1), x)

    # Copy parameters from flax.linen to nnx
    params_linen = variables_linen['params']['cell']
    # Copy cell parameters
    for gate in ['i', 'f', 'g', 'o']:
      # Input kernels
      if gate == 'f':
        nnx_layer = getattr(cell_nnx, f'if_')
      else:
        nnx_layer = getattr(cell_nnx, f'i{gate}')
      linen_params = params_linen[f'i{gate}']
      nnx_layer.kernel.value = linen_params['kernel']
      if nnx_layer.use_bias:
        nnx_layer.bias.value = linen_params['bias']
      # Hidden kernels
      nnx_layer = getattr(cell_nnx, f'h{gate}')
      linen_params = params_linen[f'h{gate}']
      nnx_layer.kernel.value = linen_params['kernel']
      if nnx_layer.use_bias:
        nnx_layer.bias.value = linen_params['bias']

    # Initialize carries
    carry_nnx = cell_nnx.initialize_carry((batch_size, in_features), rngs_nnx)

    # Run nnx.RNN
    outputs_nnx = rnn_nnx(x, initial_carry=carry_nnx)

    # Run flax.linen.RNN
    outputs_linen = rnn_linen.apply(
      variables_linen, x, initial_carry=carry_linen
    )

    # Compare outputs
    np.testing.assert_allclose(outputs_nnx, outputs_linen, atol=1e-5)

  def test_rnn_with_unroll(self):
    """Test RNN module with unroll parameter."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(6))

    # Initialize the RNN module with unroll=2
    rnn = nnx.RNN(cell, unroll=2)

    # Create input data (batch_size=2, seq_length=6, features=3)
    x = jnp.ones((2, 6, 3))

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(
      outputs.shape, (2, 6, 4)
    )  # Output features should match hidden_features

  def test_rnn_with_custom_cell(self):
    """Test RNN module with a custom RNN cell."""

    class CustomRNNCell(nnx.Module):
      """A simple custom RNN cell."""

      in_features: int
      hidden_features: int

      def __init__(self, in_features, hidden_features, rngs):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.rngs = rngs
        self.dense = nnx.Linear(
          in_features=in_features + hidden_features,
          out_features=hidden_features,
          rngs=rngs,
        )

      def __call__(self, carry, inputs):
        h = carry
        x = jnp.concatenate([inputs, h], axis=-1)
        new_h = jax.nn.tanh(self.dense(x))
        return new_h, new_h

      def initialize_carry(self, input_shape, rngs):
        batch_size = input_shape[0]
        h = jnp.zeros((batch_size, self.hidden_features))
        return h

      @property
      def num_feature_axes(self) -> int:
        return 1

    # Initialize the custom RNN cell
    cell = CustomRNNCell(in_features=3, hidden_features=4, rngs=nnx.Rngs(7))

    # Initialize the RNN module
    rnn = nnx.RNN(cell)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.ones((2, 5, 3))

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(
      outputs.shape, (2, 5, 4)
    )  # Output features should match hidden_features

  def test_rnn_with_different_dtypes(self):
    """Test RNN module with different data types."""
    # Initialize the LSTMCell with float16
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      dtype=jnp.float16,
      param_dtype=jnp.float16,
      rngs=nnx.Rngs(8),
    )

    # Initialize the RNN module
    rnn = nnx.RNN(cell)

    # Create input data (batch_size=2, seq_length=5, features=3)
    x = jnp.ones((2, 5, 3), dtype=jnp.float16)

    # Initialize the carry
    carry = cell.initialize_carry((2, 3), cell.rngs)

    # Run the RNN module
    outputs = rnn(x, initial_carry=carry)

    self.assertEqual(outputs.dtype, jnp.float16)
    self.assertEqual(outputs.shape, (2, 5, 4))

  def test_rnn_with_variable_batch_size(self):
    """Test RNN module with variable batch sizes."""
    # Initialize the LSTMCell
    cell = nnx.LSTMCell(
      in_features=3,
      hidden_features=4,
      rngs=nnx.Rngs(9),
    )

    # Initialize the RNN module
    rnn = nnx.RNN(cell)

    for batch_size in [1, 2, 5]:
      # Create input data (batch_size, seq_length=5, features=3)
      x = jnp.ones((batch_size, 5, 3))

      # Initialize the carry
      carry = cell.initialize_carry((batch_size, 3), cell.rngs)

      # Run the RNN module
      outputs = rnn(x, initial_carry=carry)

      self.assertEqual(outputs.shape, (batch_size, 5, 4))

  def test_recurrent_dropout(self):
    class LSTMWithRecurrentDropout(nnx.OptimizedLSTMCell):
      def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        hidden_features: int,
        dropout_rate: float,
        **kwargs,
      ):
        super().__init__(
          in_features=in_features,
          hidden_features=hidden_features,
          rngs=rngs,
          **kwargs,
        )
        self.recurrent_dropout = nnx.Dropout(
          rate=dropout_rate, rng_collection='recurrent_dropout', rngs=rngs
        )

      def __call__(self, carry, x):
        h, c = carry
        new_h, new_c = super().__call__((h, c), x)
        new_h = jax.tree.map(self.recurrent_dropout, new_h)
        return new_h, new_c

    class RNNWithRecurrentDropout(nnx.Module):
      def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        hidden_features: int = 32,
        dropout_rate: float = 0.5,
        recurrent_dropout_rate: float = 0.25,
      ):
        cell = LSTMWithRecurrentDropout(
          in_features=in_features,
          hidden_features=hidden_features,
          rngs=rngs,
          dropout_rate=recurrent_dropout_rate,
        )
        self.lstm = nnx.RNN(cell, broadcast_rngs='recurrent_dropout')
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dense = nnx.Linear(
          in_features=hidden_features, out_features=1, rngs=rngs
        )

      def __call__(self, x):
        x = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]  # Use only the final hidden state
        return self.dense(x)

    model = RNNWithRecurrentDropout(
      in_features=32,
      hidden_features=64,
      dropout_rate=0.2,
      recurrent_dropout_rate=0.1,
      rngs=nnx.Rngs(0, recurrent_dropout=1),
    )

    x = jnp.ones((8, 10, 32))
    self.assertEqual(model.lstm.cell.rngs.recurrent_dropout.count.value, 0)
    y = model(x)

    self.assertEqual(y.shape, (8, 1))
    self.assertEqual(model.lstm.cell.rngs.recurrent_dropout.count.value, 1)


if __name__ == '__main__':
  absltest.main()

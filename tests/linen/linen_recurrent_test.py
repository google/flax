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

"""Recurrent tests."""


from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from flax import errors
from flax import linen as nn

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class RNNTest(absltest.TestCase):
  def test_rnn_basic_forward(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_multiple_batch_dims(self):
    batch_dims = (10, 11)
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((*batch_dims, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (*batch_dims, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_unroll(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, unroll=10)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)


  def test_rnn_carry_size(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  
  def test_rnn_time_axis(self):
    seq_len = 40
    batch_size = 10
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, time_axis=0, stateful=True)
    
    xs = jnp.ones((seq_len, batch_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    ys: jnp.ndarray 
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (seq_len, batch_size, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  
  def test_rnn_stateful(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    ys: jnp.ndarray
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_stateful_reset(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    ys: jnp.ndarray
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

    # change batch size
    batch_size = 20
    xs = jnp.ones((batch_size, seq_len, channels_in))

    with self.assertRaisesRegex(TypeError, 'got incompatible shapes'):
      rnn.apply(variables, xs, mutable=['memory'])

    # reset state
    ys, updates = rnn.apply(variables, xs, mutable=['memory'], reset_state=True)
    variables = variables.copy(updates)
    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # run normally
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)
    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))
  
  def test_rnn_with_spatial_dimensions(self):
    batch_size = 10
    seq_len = 40
    kernel_size = [3, 3]
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(
      nn.ConvLSTM(channels_out, kernel_size), 
      stateful=True,
      carry_size=(*kernel_size, channels_out),
    )
    
    xs = jnp.ones((batch_size, seq_len, *kernel_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape[:-1], (batch_size, *kernel_size))
      self.assertIn(carry.shape[-1], [channels_in, channels_out])
      

    ys: jnp.ndarray
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape[:-1], (batch_size, *kernel_size))
      self.assertIn(carry.shape[-1], [channels_in, channels_out])

    self.assertEqual(ys.shape, (batch_size, seq_len, *kernel_size, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out * 4,))
      self.assertIn(layer_params['kernel'].shape[2], [channels_in, channels_out, channels_out * 4])
      self.assertEqual(layer_params['kernel'].shape[3], channels_out * 4)
      

  def test_go_backwards(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True, reverse=True)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    ys: jnp.ndarray
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_return_state(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True, return_state=True)
    
    xs = jnp.ones((batch_size, seq_len, channels_out))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # carry state should be zeros on init
    for carry in jax.tree_leaves(variables['memory']['carry']):
      assert np.allclose(carry, jnp.zeros_like(carry))
      self.assertEqual(carry.shape, (batch_size, channels_out))

    ys: jnp.ndarray
    (output_state, ys), updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)

    # carry state should not be zeros after apply
    carry_leaves = jax.tree_leaves(variables['memory']['carry'])
    for carry, carry_out in zip(carry_leaves, output_state):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      assert carry is carry_out
      self.assertEqual(carry.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
    
  def test_initial_state(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    xs = jnp.ones((batch_size, seq_len, channels_in))

    # stateful RNN
    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True)
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    ys: jnp.ndarray
    # step 1
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)
    # step 2
    ys, updates = rnn.apply(variables, xs, mutable=['memory'])
    variables = variables.copy(updates)
    stateful_carry = variables['memory']['carry']

    # non-stateful RNN
    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_state=True)
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    # step 1
    carry, ys = rnn.apply(variables, xs)
    # step 2
    carry, ys = rnn.apply(variables, xs, initial_state=carry)

    # compare carry states
    for carry_stateful, carry_functional in zip(stateful_carry, carry):
      assert np.allclose(carry_stateful, carry_functional)
      self.assertEqual(carry_stateful.shape, (batch_size, channels_out))

  def test_rnn_argument_overrides(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, stateful=True)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    # override return_state
    (output_state, ys), updates = rnn.apply(variables, xs, return_state=True, 
                                            mutable=['memory'])
    variables = variables.copy(updates)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # carry state should not be zeros after apply
    carry_leaves = jax.tree_leaves(variables['memory']['carry'])
    for carry, carry_out in zip(carry_leaves, output_state):
      assert not np.allclose(carry, jnp.zeros_like(carry))
      assert carry is carry_out
      self.assertEqual(carry.shape, (batch_size, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_mask(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, shape=(batch_size, seq_len))

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)
      for seq_idx in range(seq_len):
        if mask[batch_idx, seq_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs are zeros
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], jnp.zeros_like(ys[batch_idx, seq_idx]), rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_int_mask(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=seq_len+1)

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)
      for seq_idx in range(seq_len):
        if seq_idx < mask[batch_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs are zeros
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], jnp.zeros_like(ys[batch_idx, seq_idx]), rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_mask_no_zero_outputs(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, shape=(batch_size, seq_len))

    rnn = nn.RNN(nn.LSTMCell(), channels_out, zero_output_for_mask=False)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)
      prev_non_masked_output = jnp.zeros_like(ys[batch_idx, 0])

      for seq_idx in range(seq_len):
        if mask[batch_idx, seq_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)
          prev_non_masked_output = cell_output[0]

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs the previous non-masked output
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], prev_non_masked_output, rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_int_mask_no_zero_outputs(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=seq_len+1)

    rnn = nn.RNN(nn.LSTMCell(), channels_out, zero_output_for_mask=False)
    
    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)
      prev_non_masked_output = jnp.zeros_like(ys[batch_idx, 0])

      for seq_idx in range(seq_len):
        if seq_idx < mask[batch_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)
          prev_non_masked_output = cell_output[0]

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs the previous non-masked output
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], prev_non_masked_output, rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_mask_with_spatial(self):
    batch_size = 3
    seq_len = 4
    kernel_size = [3, 3]
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, shape=(batch_size, seq_len))

    rnn = nn.RNN(
      nn.ConvLSTM(channels_out, kernel_size), 
      carry_size=(*kernel_size, channels_out),
    )
    
    xs = jnp.ones((batch_size, seq_len, *kernel_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, *kernel_size, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), (*kernel_size, channels_out))
      for seq_idx in range(seq_len):
        if mask[batch_idx, seq_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs are zeros
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], jnp.zeros_like(ys[batch_idx, seq_idx]), rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)
  
  def test_rnn_int_mask_with_spatial(self):
    batch_size = 3
    seq_len = 4
    kernel_size = [3, 3]
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=seq_len+1)

    rnn = nn.RNN(
      nn.ConvLSTM(channels_out, kernel_size), 
      carry_size=(*kernel_size, channels_out),
    )
    
    xs = jnp.ones((batch_size, seq_len, *kernel_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (batch_size, seq_len, *kernel_size, channels_out))

    # manually compute the carry and output from cell
    cell_variables = {"params": variables['params']['cell']}
    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), (*kernel_size, channels_out))
      for seq_idx in range(seq_len):
        if seq_idx < mask[batch_idx]:
          cell_input = xs[batch_idx:batch_idx+1, seq_idx]
          cell_carry, cell_output = rnn.cell.apply(cell_variables, cell_carry, cell_input)

          # check non-masked outputs are the same
          np.testing.assert_allclose(cell_output[0], ys[batch_idx, seq_idx], rtol=1e-4)
        else:
          # check masked outputs are zeros
          np.testing.assert_allclose(
            ys[batch_idx, seq_idx], jnp.zeros_like(ys[batch_idx, seq_idx]), rtol=1e-4)

      # check final carry state is the same
      for cell_carry_i, carry_i in zip(cell_carry, carry):
        np.testing.assert_allclose(cell_carry_i[0], carry_i[batch_idx], rtol=1e-4)

  def test_rnn_mask_multiple_batch_dims(self):
    batch_dims = (3, 7)
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, shape=(*batch_dims, seq_len))

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((*batch_dims, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (*batch_dims, seq_len, channels_out))

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)
  
  def test_rnn_int_mask_multiple_batch_dims(self):
    batch_dims = (3, 7)
    seq_len = 4
    channels_in = 5
    channels_out = 15

    key = jax.random.PRNGKey(0)
    mask = jax.random.randint(key, shape=batch_dims, minval=0, maxval=seq_len+1)

    rnn = nn.RNN(nn.LSTMCell(), channels_out)
    
    xs = jnp.ones((*batch_dims, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, mask=mask, return_state=True)

    self.assertEqual(ys.shape, (*batch_dims, seq_len, channels_out))

    # check params
    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

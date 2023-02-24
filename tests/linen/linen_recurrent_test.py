
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
import pytest
import einops
from flax.linen.recurrent import _select_last

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class RNNTest(absltest.TestCase):
  def test_rnn_basic_forward(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

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

    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_carry=True)

    xs = jnp.ones((*batch_dims, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

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

    rnn = nn.RNN(nn.LSTMCell(), channels_out, unroll=10, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_time_major(self):
    seq_len = 40
    batch_size = 10
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, time_major=True, return_carry=True)

    xs = jnp.ones((seq_len, batch_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

    # carry state should not be zeros after apply
    for leaf in jax.tree_util.tree_leaves(carry):
      assert not np.allclose(leaf, jnp.zeros_like(leaf))
      self.assertEqual(leaf.shape, (batch_size, channels_out))

    self.assertEqual(ys.shape, (seq_len, batch_size, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(layer_params['kernel'].shape[0], [channels_in, channels_out])
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_with_spatial_dimensions(self):
    batch_size = 10
    seq_len = 40
    kernel_size = (3, 3)
    image_size = (32, 32)
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(
      nn.ConvLSTMCell(channels_out, kernel_size),
      cell_size=(*image_size, channels_out),
    )

    xs = jnp.ones((batch_size, seq_len, *image_size, channels_in))
    variables = rnn.init(jax.random.PRNGKey(0), xs)

    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs, return_carry=True)

    # carry state should not be zeros after apply
    for leaf in jax.tree_util.tree_leaves(carry):
      assert not np.allclose(leaf, jnp.zeros_like(leaf))
      self.assertEqual(leaf.shape[:-1], (batch_size, *image_size))
      self.assertIn(leaf.shape[-1], [channels_in, channels_out])

    self.assertEqual(ys.shape, (batch_size, seq_len, *image_size, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
          self.assertEqual(layer_params['bias'].shape, (channels_out * 4,))
      self.assertIn(layer_params['kernel'].shape[2], [channels_in, channels_out, channels_out * 4])
      self.assertEqual(layer_params['kernel'].shape[3], channels_out * 4)

  @pytest.mark.skip(reason='TODO: discuss supporting reverse instead of flip_sequences')
  def test_go_backwards(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(), channels_out, reverse=True)

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

  def test_numerical_equivalence(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.PRNGKey(0), xs)

    cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), channels_out)
    cell_params = variables['params']['cell']

    for i in range(seq_len):
      cell_carry, y = rnn.cell.apply({'params': cell_params}, cell_carry, xs[:, i, :])
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-5)

    np.testing.assert_allclose(cell_carry, carry, rtol=1e-5)

  def test_numerical_equivalence_with_mask(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    key = jax.random.PRNGKey(0)
    seq_lengths = jax.random.randint(key, (batch_size,), minval=1, maxval=seq_len + 1)
    segmentation_mask = einops.repeat(
      jnp.arange(seq_len), 'time -> batch time', batch=batch_size)
    segmentation_mask = (segmentation_mask < seq_lengths[:, None]).astype(jnp.int32)

    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.PRNGKey(0), xs, segmentation_mask=segmentation_mask)

    cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), channels_out)
    cell_params = variables['params']['cell']
    carries = []

    for i in range(seq_len):
      cell_carry, y = rnn.cell.apply({'params': cell_params}, cell_carry, xs[:, i, :])
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-5)
      carries.append(cell_carry)

    for batch_idx, length in enumerate(seq_lengths):
      t = int(length) - 1
      for carries_t_, carry_ in zip(carries[t], carry):
        np.testing.assert_allclose(carries_t_[batch_idx], carry_[batch_idx], rtol=1e-5)

  @pytest.mark.skip(reason='TODO: possible bug with scan')
  def test_numerical_equivalence_single_batch(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(nn.LSTMCell(), channels_out, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.PRNGKey(0), xs)

    cell_params = variables['params']['cell']

    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)

      for i in range(seq_len):
        cell_carry, y = rnn.cell.apply({'params': cell_params}, cell_carry, xs[batch_idx, i, :][None])
        np.testing.assert_allclose(y[0], ys[batch_idx, i, :])

      np.testing.assert_allclose(cell_carry, carry)

  def test_numerical_equivalence_single_batch_nn_scan(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    cell = nn.LSTMCell()
    rnn = nn.scan(nn.LSTMCell, in_axes=1, out_axes=1,
                   variable_broadcast='params',
                   split_rngs={'params': False})()

    xs = jnp.ones((batch_size, seq_len, channels_in))
    carry = rnn.initialize_carry(jax.random.PRNGKey(0), (batch_size,), channels_out)
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.PRNGKey(0), carry, xs)

    cell_params = variables['params']

    for batch_idx in range(batch_size):
      cell_carry = cell.initialize_carry(jax.random.PRNGKey(0), (1,), channels_out)

      for i in range(seq_len):
        cell_carry, y = cell.apply({'params': cell_params}, cell_carry, xs[batch_idx:batch_idx+1, i, :])
        np.testing.assert_allclose(y[0], ys[batch_idx, i, :], rtol=1e-5)

      carry_i = jax.tree_map(lambda x: x[batch_idx:batch_idx+1], carry)
      np.testing.assert_allclose(cell_carry, carry_i, rtol=1e-5)

  def test_numerical_equivalence_single_batch_jax_scan(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    xs = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, seq_len, channels_in))
    cell = nn.LSTMCell()
    carry = cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), channels_out)
    variables = cell.init(jax.random.PRNGKey(0), carry, xs[:, 0])
    cell_params = variables['params']

    def scan_fn(carry, x):
      return cell.apply({'params': cell_params}, carry, x)

    ys: jnp.ndarray
    carry, ys = jax.lax.scan(scan_fn, carry, xs.swapaxes(0, 1))
    ys = ys.swapaxes(0, 1)

    cell_carry = cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), channels_out)

    for i in range(seq_len):
      cell_carry, y = cell.apply({'params': cell_params}, cell_carry, xs[:, i, :])
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-4)

    np.testing.assert_allclose(cell_carry, carry, rtol=1e-4)
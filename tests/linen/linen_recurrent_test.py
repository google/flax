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

"""Recurrent tests."""


from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pytest
from flax.linen.recurrent import flip_sequences
from jax._src.test_util import sample_product

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class RNNTest(absltest.TestCase):

  def test_rnn_basic_forward(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.key(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(
          layer_params['kernel'].shape[0], [channels_in, channels_out]
      )
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_multiple_batch_dims(self):
    batch_dims = (10, 11)
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True)

    xs = jnp.ones((*batch_dims, seq_len, channels_in))
    variables = rnn.init(jax.random.key(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (*batch_dims, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(
          layer_params['kernel'].shape[0], [channels_in, channels_out]
      )
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_unroll(self):
    batch_size = 10
    seq_len = 40
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(channels_out), unroll=10, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    variables = rnn.init(jax.random.key(0), xs)
    ys: jnp.ndarray
    carry, ys = rnn.apply(variables, xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    for layer_params in variables['params']['cell'].values():
      if 'bias' in layer_params:
        self.assertEqual(layer_params['bias'].shape, (channels_out,))
      self.assertIn(
          layer_params['kernel'].shape[0], [channels_in, channels_out]
      )
      self.assertEqual(layer_params['kernel'].shape[1], channels_out)

  def test_rnn_time_major(self):
    seq_len = 40
    batch_size = 10
    channels_in = 5
    channels_out = 15

    rnn = nn.RNN(nn.LSTMCell(channels_out), time_major=True, return_carry=True)

    xs = jnp.ones((seq_len, batch_size, channels_in))
    variables = rnn.init(jax.random.key(0), xs)

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
      self.assertIn(
          layer_params['kernel'].shape[0], [channels_in, channels_out]
      )
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
    )

    xs = jnp.ones((batch_size, seq_len, *image_size, channels_in))
    variables = rnn.init(jax.random.key(0), xs)

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
      self.assertIn(
          layer_params['kernel'].shape[2],
          [channels_in, channels_out, channels_out * 4],
      )
      self.assertEqual(layer_params['kernel'].shape[3], channels_out * 4)

  def test_numerical_equivalence(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.key(0), xs)

    cell_carry = rnn.cell.initialize_carry(jax.random.key(0), xs[:, 0].shape)
    cell_params = variables['params']['cell']

    for i in range(seq_len):
      cell_carry, y = rnn.cell.apply(
          {'params': cell_params}, cell_carry, xs[:, i, :]
      )
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-5)

    np.testing.assert_allclose(cell_carry, carry, rtol=1e-5)

  def test_numerical_equivalence_with_mask(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    key = jax.random.key(0)
    seq_lengths = jax.random.randint(
        key, (batch_size,), minval=1, maxval=seq_len + 1
    )

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(
        jax.random.key(0), xs, seq_lengths=seq_lengths
    )

    cell_carry = rnn.cell.initialize_carry(jax.random.key(0), xs[:, 0].shape)
    cell_params = variables['params']['cell']
    carries = []

    for i in range(seq_len):
      cell_carry, y = rnn.cell.apply(
          {'params': cell_params}, cell_carry, xs[:, i, :]
      )
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-5)
      carries.append(cell_carry)

    for batch_idx, length in enumerate(seq_lengths):
      t = int(length) - 1
      for carries_t_, carry_ in zip(carries[t], carry):
        np.testing.assert_allclose(
            carries_t_[batch_idx], carry_[batch_idx], rtol=1e-5
        )

  def test_numerical_equivalence_single_batch(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.key(0), xs)

    cell_params = variables['params']['cell']

    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.key(0), xs[:1, 0].shape)

      for i in range(seq_len):
        cell_carry, y = rnn.cell.apply(
            {'params': cell_params}, cell_carry, xs[batch_idx, i, :][None]
        )
        np.testing.assert_allclose(y[0], ys[batch_idx, i, :], rtol=1e-6)

      carry_i = jax.tree_map(lambda x: x[batch_idx : batch_idx + 1], carry)
      np.testing.assert_allclose(cell_carry, carry_i, rtol=1e-6)

  def test_numerical_equivalence_single_batch_nn_scan(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    cell: nn.LSTMCell = nn.LSTMCell(channels_out)
    rnn: nn.LSTMCell = nn.scan(
        nn.LSTMCell,
        in_axes=1,
        out_axes=1,
        variable_broadcast='params',
        split_rngs={'params': False},
    )(channels_out)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    carry = rnn.initialize_carry(jax.random.key(0), xs[:, 0].shape)
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.key(0), carry, xs)

    cell_params = variables['params']

    for batch_idx in range(batch_size):
      cell_carry = cell.initialize_carry(jax.random.key(0), xs[:1, 0].shape)

      for i in range(seq_len):
        cell_carry, y = cell.apply(
            {'params': cell_params},
            cell_carry,
            xs[batch_idx : batch_idx + 1, i, :],
        )
        np.testing.assert_allclose(y[0], ys[batch_idx, i, :], rtol=1e-5)

      carry_i = jax.tree_map(lambda x: x[batch_idx : batch_idx + 1], carry)
      np.testing.assert_allclose(cell_carry, carry_i, rtol=1e-5)

  def test_numerical_equivalence_single_batch_jax_scan(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    xs = jax.random.uniform(
        jax.random.key(0), (batch_size, seq_len, channels_in)
    )
    cell: nn.LSTMCell = nn.LSTMCell(channels_out)
    carry = cell.initialize_carry(jax.random.key(0), xs[:, 0].shape)
    variables = cell.init(jax.random.key(0), carry, xs[:, 0])
    cell_params = variables['params']

    def scan_fn(carry, x):
      return cell.apply({'params': cell_params}, carry, x)

    ys: jnp.ndarray
    carry, ys = jax.lax.scan(scan_fn, carry, xs.swapaxes(0, 1))
    ys = ys.swapaxes(0, 1)

    cell_carry = cell.initialize_carry(jax.random.key(0), xs[:, 0].shape)

    for i in range(seq_len):
      cell_carry, y = cell.apply(
          {'params': cell_params}, cell_carry, xs[:, i, :]
      )
      np.testing.assert_allclose(y, ys[:, i, :], rtol=1e-4)

    np.testing.assert_allclose(cell_carry, carry, rtol=1e-4)

  def test_reverse(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(nn.LSTMCell(channels_out), return_carry=True, reverse=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.key(0), xs)

    cell_params = variables['params']['cell']

    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.key(0), xs[:1, 0].shape)

      for i in range(seq_len):
        cell_carry, y = rnn.cell.apply(
            {'params': cell_params},
            cell_carry,
            xs[batch_idx, seq_len - i - 1, :][None],
        )
        np.testing.assert_allclose(y[0], ys[batch_idx, i, :], rtol=1e-5)

      np.testing.assert_allclose(
          cell_carry,
          jax.tree_map(lambda x: x[batch_idx : batch_idx + 1], carry),
          rtol=1e-5,
      )

  def test_reverse_but_keep_order(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    rnn = nn.RNN(
        nn.LSTMCell(channels_out),
        return_carry=True,
        reverse=True,
        keep_order=True,
    )

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = rnn.init_with_output(jax.random.key(0), xs)

    cell_params = variables['params']['cell']

    for batch_idx in range(batch_size):
      cell_carry = rnn.cell.initialize_carry(jax.random.key(0), xs[:1, 0].shape)

      for i in range(seq_len):
        cell_carry, y = rnn.cell.apply(
            {'params': cell_params},
            cell_carry,
            xs[batch_idx, seq_len - i - 1, :][None],
        )
        np.testing.assert_allclose(
            y[0], ys[batch_idx, seq_len - i - 1, :], rtol=1e-5
        )

      np.testing.assert_allclose(
          cell_carry,
          jax.tree_map(lambda x: x[batch_idx : batch_idx + 1], carry),
          rtol=1e-5,
      )

  def test_flip_sequence(self):
    x = jnp.arange(2 * 5).reshape((2, 5))
    seq_lengths = jnp.array([4, 2])

    flipped = flip_sequences(x, seq_lengths, num_batch_dims=1, time_major=False)

    self.assertEqual(flipped.shape, (2, 5))
    np.testing.assert_allclose(flipped[0, :4], [3, 2, 1, 0])
    np.testing.assert_allclose(flipped[1, :2], [6, 5])

  def test_flip_sequence_more_feature_dims(self):
    x = jnp.arange(2 * 5 * 3).reshape((2, 5, 3))
    seq_lengths = jnp.array([4, 2])

    flipped = flip_sequences(x, seq_lengths, num_batch_dims=1, time_major=False)

    self.assertEqual(flipped.shape, (2, 5, 3))
    np.testing.assert_allclose(flipped[0, :4], x[0, :4][::-1])
    np.testing.assert_allclose(flipped[1, :2], x[1, :2][::-1])

  def test_flip_sequence_time_major(self):
    x = jnp.arange(2 * 5).reshape((5, 2))
    seq_lengths = jnp.array([4, 2])

    flipped = flip_sequences(x, seq_lengths, num_batch_dims=1, time_major=True)

    self.assertEqual(flipped.shape, (5, 2))
    np.testing.assert_allclose(flipped[:4, 0], x[:4, 0][::-1])
    np.testing.assert_allclose(flipped[:2, 1], x[:2, 1][::-1])

  def test_flip_sequence_time_major_more_feature_dims(self):
    x = jnp.arange(2 * 5 * 3).reshape((5, 2, 3))
    seq_lengths = jnp.array([4, 2])

    flipped = flip_sequences(x, seq_lengths, num_batch_dims=1, time_major=True)

    self.assertEqual(flipped.shape, (5, 2, 3))
    np.testing.assert_allclose(flipped[:4, 0], x[:4, 0][::-1])
    np.testing.assert_allclose(flipped[:2, 1], x[:2, 1][::-1])

  def test_basic_seq_lengths(self):
    x = jnp.ones((2, 10, 6))
    lstm = nn.RNN(nn.LSTMCell(265))
    variables = lstm.init(jax.random.key(0), x)
    y = lstm.apply(variables, x, seq_lengths=jnp.array([5, 5]))


class BidirectionalTest(absltest.TestCase):

  def test_bidirectional(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    bdirectional = nn.Bidirectional(
        nn.RNN(nn.LSTMCell(channels_out)), nn.RNN(nn.LSTMCell(channels_out))
    )

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    ys, variables = bdirectional.init_with_output(jax.random.key(0), xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out * 2))

  def test_shared_cell(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    cell = nn.LSTMCell(channels_out)
    bdirectional = nn.Bidirectional(nn.RNN(cell), nn.RNN(cell))

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    ys, variables = bdirectional.init_with_output(jax.random.key(0), xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out * 2))

  def test_custom_merge_fn(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    bdirectional = nn.Bidirectional(
        nn.RNN(nn.LSTMCell(channels_out)),
        nn.RNN(nn.LSTMCell(channels_out)),
        merge_fn=lambda x, y: x + y,
    )

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    ys, variables = bdirectional.init_with_output(jax.random.key(0), xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

  def test_return_carry(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    bdirectional = nn.Bidirectional(
        nn.RNN(nn.LSTMCell(channels_out)),
        nn.RNN(nn.LSTMCell(channels_out)),
        return_carry=True,
    )

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = bdirectional.init_with_output(
        jax.random.key(0), xs
    )
    carry_forward, carry_backward = carry

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out * 2))
    self.assertEqual(
        jax.tree_map(jnp.shape, carry_forward),
        ((batch_size, channels_out), (batch_size, channels_out)),
    )
    self.assertEqual(
        jax.tree_map(jnp.shape, carry_backward),
        ((batch_size, channels_out), (batch_size, channels_out)),
    )


class TestRecurrentDeprecation(parameterized.TestCase):

  @parameterized.product(
      cell_type=[nn.LSTMCell, nn.GRUCell, nn.OptimizedLSTMCell]
  )
  def test_constructor(self, cell_type):
    with self.assertRaisesRegex(TypeError, 'The RNNCellBase API has changed'):
      cell_type()

  @parameterized.product(
      cell_type=[nn.LSTMCell, nn.GRUCell, nn.OptimizedLSTMCell]
  )
  def test_initialize_carry(self, cell_type):
    key = jax.random.key(0)
    with self.assertRaisesRegex(TypeError, 'The RNNCellBase API has changed'):
      cell_type.initialize_carry(key, (2,), 3)

  def test_rnn(self):
    cell = nn.LSTMCell(3)
    with self.assertRaisesRegex(TypeError, 'The RNNCellBase API has changed'):
      nn.RNN(cell, cell_size=8)


class CudnnLSTMTest(parameterized.TestCase):

  def test_cuddn_lstm(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    module = nn.CudnnLSTM(features=channels_out)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    ys, variables = module.init_with_output(jax.random.PRNGKey(0), xs)

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

  def test_cuddn_lstm_return_carry(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    module = nn.CudnnLSTM(features=channels_out, return_carry=True)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    (carry, ys), variables = module.init_with_output(jax.random.PRNGKey(0), xs)
    c, h = carry

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))
    self.assertEqual(c.shape, (1, batch_size, channels_out))
    self.assertEqual(h.shape, (1, batch_size, channels_out))

  def test_seq_lengths(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    module = nn.CudnnLSTM(features=channels_out)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    seq_lengths = jnp.array([3, 2, 1], dtype=jnp.int32)
    ys: jnp.ndarray
    ys, variables = module.init_with_output(
        jax.random.PRNGKey(0), xs, seq_lengths=seq_lengths
    )

    self.assertEqual(ys.shape, (batch_size, seq_len, channels_out))

    np.testing.assert_allclose(ys[0, 3], 0)
    np.testing.assert_allclose(ys[1, 2:], 0)
    np.testing.assert_allclose(ys[2, 1:], 0)

  @sample_product(time_major=[True, False])
  def test_cuddn_multibatch(self, time_major: bool):
    batch_size = (2, 3)
    seq_len = 4
    channels_in = 5
    channels_out = 6

    module = nn.CudnnLSTM(features=channels_out, time_major=time_major)

    if time_major:
      xs = jnp.ones((seq_len, *batch_size, channels_in))
    else:
      xs = jnp.ones((*batch_size, seq_len, channels_in))
    ys: jnp.ndarray
    ys, variables = module.init_with_output(jax.random.PRNGKey(0), xs)

    if time_major:
      self.assertEqual(ys.shape, (seq_len, *batch_size, channels_out))
    else:
      self.assertEqual(ys.shape, (*batch_size, seq_len, channels_out))

  @sample_product(time_major=[True, False])
  def test_cuddn_no_batch(self, time_major: bool):
    seq_len = 4
    channels_in = 5
    channels_out = 6

    module = nn.CudnnLSTM(features=channels_out, time_major=time_major)

    xs = jnp.ones((seq_len, channels_in))

    ys: jnp.ndarray
    ys, variables = module.init_with_output(jax.random.PRNGKey(0), xs)

    self.assertEqual(ys.shape, (seq_len, channels_out))

  # skip tests
  @pytest.mark.skip(
      reason=(
          "LSTMCell doesn't use bias term for the Dense layers applied to the"
          ' inputs'
      )
  )
  def test_compare_cudnn_with_rnn(self):
    batch_size = 3
    seq_len = 4
    channels_in = 5
    channels_out = 6

    cudnn_module = nn.CudnnLSTM(features=channels_out)
    rnn_module = nn.RNN(nn.OptimizedLSTMCell(), cell_size=channels_out)

    xs = jnp.ones((batch_size, seq_len, channels_in))
    ys_cudnn: jnp.ndarray
    ys_cudnn, variables_cudnn = cudnn_module.init_with_output(
        jax.random.PRNGKey(0), xs
    )

    variables_rnn = rnn_module.init(jax.random.PRNGKey(0), xs)

    W_ih, W_hh, b_ih, b_hh = cudnn_module.unpack_weights(
        variables_cudnn['params']['weights'], channels_in
    )
    W_ii, W_if, W_ig, W_io = jnp.split(W_ih[0], 4, axis=0)
    W_hi, W_hf, W_hg, W_ho = jnp.split(W_hh[0], 4, axis=0)
    b_ii, b_if, b_ig, b_io = jnp.split(b_ih[0], 4, axis=0)
    b_hi, b_hf, b_hg, b_ho = jnp.split(b_hh[0], 4, axis=0)

    variables_rnn_dict = variables_rnn.unfreeze()
    variables_rnn_dict['params']['cell']['hf']['kernel'] = W_hf.T
    variables_rnn_dict['params']['cell']['hf']['bias'] = b_hf
    variables_rnn_dict['params']['cell']['hg']['kernel'] = W_hg.T
    variables_rnn_dict['params']['cell']['hg']['bias'] = b_hg
    variables_rnn_dict['params']['cell']['hi']['kernel'] = W_hi.T
    variables_rnn_dict['params']['cell']['hi']['bias'] = b_hi
    variables_rnn_dict['params']['cell']['ho']['kernel'] = W_ho.T
    variables_rnn_dict['params']['cell']['ho']['bias'] = b_ho
    variables_rnn_dict['params']['cell']['if']['kernel'] = W_if.T
    variables_rnn_dict['params']['cell']['ig']['kernel'] = W_ig.T
    variables_rnn_dict['params']['cell']['ii']['kernel'] = W_ii.T
    variables_rnn_dict['params']['cell']['io']['kernel'] = W_io.T

    # b_ii, b_if, b_ig, and b_io are not used in the LSTMCell
    # hard to compare the results. Try zeroing them out in the `weights` array
    # in the future.
    ys_rnn: jnp.ndarray
    ys_rnn = rnn_module.apply(variables_rnn_dict, xs)

    np.testing.assert_allclose(ys_cudnn, ys_rnn, atol=1e-5)


if __name__ == '__main__':
  absltest.main()

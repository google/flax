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

"""Tests for sst2.models."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import jax.test_util
import numpy as np

import models


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class ModelTest(parameterized.TestCase):

  def test_embedder_returns_correct_output_shape(self):
    """Tests if the embedder returns the correct shape."""
    vocab_size = 5
    embedding_size = 3
    model = models.Embedder(
        vocab_size=vocab_size,
        embedding_size=embedding_size)
    rng = jax.random.PRNGKey(0)
    token_ids = np.array([[2, 4, 3], [2, 6, 3]], dtype=np.int32)
    output, _ = model.init_with_output(rng, token_ids, deterministic=True)
    self.assertEqual((token_ids.shape) + (embedding_size,), output.shape)

  def test_lstm_returns_correct_output_shape(self):
    """Tests if the simple LSTM returns the correct shape."""
    batch_size = 2
    seq_len = 3
    embedding_size = 4
    hidden_size = 5
    model = models.SimpleLSTM()
    rng = jax.random.PRNGKey(0)
    inputs = np.random.RandomState(0).normal(
        size=[batch_size, seq_len, embedding_size])
    initial_state = models.SimpleLSTM.initialize_carry((batch_size,), hidden_size)
    (_, output), _ = model.init_with_output(rng, initial_state, inputs)
    self.assertEqual((batch_size, seq_len, hidden_size), output.shape)

  def test_bilstm_returns_correct_output_shape(self):
    """Tests if the simple BiLSTM returns the correct shape."""
    batch_size = 2
    seq_len = 3
    embedding_size = 4
    hidden_size = 5
    model = models.SimpleBiLSTM(hidden_size=hidden_size)
    rng = jax.random.PRNGKey(0)
    inputs = np.random.RandomState(0).normal(
        size=[batch_size, seq_len, embedding_size])
    lengths = np.array([2, 3], dtype=np.int32)
    outputs, _ = model.init_with_output(rng, inputs, lengths)
    # We expect 2*hidden_size because we concatenate forward+backward LSTMs.
    self.assertEqual((batch_size, seq_len, 2 * hidden_size), outputs.shape)

  def test_text_classifier_returns_correct_output_shape(self):
    """Tests if a TextClassifier model returns the correct shape."""
    embedding_size = 3
    hidden_size = 7
    vocab_size = 5
    output_size = 3
    dropout_rate = 0.1
    word_dropout_rate = 0.2
    unk_idx = 1

    model = models.TextClassifier(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        output_size=output_size,
        dropout_rate=dropout_rate,
        word_dropout_rate=word_dropout_rate,
        unk_idx=unk_idx,
        deterministic=True)

    rng = jax.random.PRNGKey(0)
    token_ids = np.array([[2, 4, 3], [2, 6, 3]], dtype=np.int32)
    lengths = np.array([2, 3], dtype=np.int32)
    output, _ = model.init_with_output(rng, token_ids, lengths)
    batch_size = token_ids.shape[0]
    self.assertEqual((batch_size, output_size), output.shape)


if __name__ == '__main__':
  absltest.main()

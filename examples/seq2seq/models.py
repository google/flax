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

"""seq2seq example: Mode code."""

# See issue #620.
# pytype: disable=wrong-keyword-args

import functools
from typing import Any, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
PRNGKey = jax.random.KeyArray


class DecoderLSTMCell(nn.RNNCellBase):
  """DecoderLSTM Module wrapped in a lifted scan transform.
  Attributes:
    teacher_force: See docstring on Seq2seq module.
    vocab_size: Size of the vocabulary.
  """
  teacher_force: bool
  vocab_size: int

  @nn.compact
  def __call__(self, carry: Tuple[Array, Array], x: Array) -> Array:
    """Applies the DecoderLSTM model."""
    lstm_state, last_prediction = carry
    if not self.teacher_force:
      x = last_prediction
    lstm_state, y = nn.LSTMCell()(lstm_state, x)
    logits = nn.Dense(features=self.vocab_size)(y)
    # Sample the predicted token using a categorical distribution over the
    # logits.
    categorical_rng = self.make_rng('lstm')
    predicted_token = jax.random.categorical(categorical_rng, logits)
    # Convert to one-hot encoding.
    prediction = jax.nn.one_hot(
        predicted_token, self.vocab_size, dtype=jnp.float32)

    return (lstm_state, prediction), (logits, prediction)


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture.

  Attributes:
    teacher_force: whether to use `decoder_inputs` as input to the decoder at
      every step. If False, only the first input (i.e., the "=" token) is used,
      followed by samples taken from the previous output logits.
    hidden_size: int, the number of hidden dimensions in the encoder and decoder
      LSTMs.
    vocab_size: the size of the vocabulary.
    eos_id: EOS id.
  """
  teacher_force: bool
  hidden_size: int
  vocab_size: int
  eos_id: int = 1

  @nn.compact
  def __call__(self, encoder_inputs: Array,
               decoder_inputs: Array) -> Tuple[Array, Array]:
    """Applies the seq2seq model.

    Args:
      encoder_inputs: [batch_size, max_input_length, vocab_size].
        padded batch of input sequences to encode.
      decoder_inputs: [batch_size, max_output_length, vocab_size].
        padded batch of expected decoded sequences for teacher forcing.
        When sampling (i.e., `teacher_force = False`), only the first token is
        input into the decoder (which is the token "="), and samples are used
        for the following inputs. The second dimension of this tensor determines
        how many steps will be decoded, regardless of the value of
        `teacher_force`.

    Returns:
      Pair (logits, predictions), which are two arrays of length `batch_size`
      containing respectively decoded logits and predictions (in one hot
      encoding format).
    """
    # Encode inputs.
    encoder = nn.RNN(nn.LSTMCell(), self.hidden_size, return_carry=True, name='encoder')
    decoder = nn.RNN(DecoderLSTMCell(self.teacher_force, self.vocab_size), decoder_inputs.shape[-1],
      split_rngs={'params': False, 'lstm': True}, name='decoder')

    segmentation_mask = self.get_segmentation_mask(encoder_inputs)

    encoder_state, _ = encoder(encoder_inputs, segmentation_mask=segmentation_mask)
    logits, predictions = decoder(decoder_inputs[:, :-1], initial_carry=(encoder_state, decoder_inputs[:, 0]))

    return logits, predictions

  def get_segmentation_mask(self, inputs: Array) -> Array:
    """Get segmentation mask for inputs."""
    # undo one-hot encoding
    inputs = jnp.argmax(inputs, axis=-1)
    # calculate eos index
    eos_idx = jnp.argmax(inputs == self.eos_id, axis=-1, keepdims=True)
    # create index array
    indexes = jnp.arange(inputs.shape[1])
    indexes = jnp.broadcast_to(indexes, inputs.shape[:2])
    # return mask
    return indexes < eos_idx

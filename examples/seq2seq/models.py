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


class EncoderLSTM(nn.Module):
  eos_id: int

  @functools.partial(
      nn.scan,
      variable_broadcast='params',
      in_axes=1,
      out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    lstm_state, is_eos = carry
    new_lstm_state, y = nn.LSTMCell()(lstm_state, x)
    # Pass forward the previous state if EOS has already been reached.
    def select_carried_state(new_state, old_state):
      return jnp.where(is_eos[:, np.newaxis], old_state, new_state)
    # LSTM state is a tuple (c, h).
    carried_lstm_state = tuple(
        select_carried_state(*s) for s in zip(new_lstm_state, lstm_state))
    # Update `is_eos`.
    is_eos = jnp.logical_or(is_eos, x[:, self.eos_id])
    return (carried_lstm_state, is_eos), y


  @staticmethod
  def initialize_carry(batch_size, hidden_size):
    # Use a dummy key since the default state init fn is just zeros.
    return nn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Encoder(nn.Module):
  """LSTM encoder, returning state after finding the EOS token in the input."""
  hidden_size: int
  eos_id: int

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lstm = EncoderLSTM(name='encoder_lstm', eos_id=self.eos_id)
    init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)
    init_is_eos = jnp.zeros(batch_size, dtype=bool)
    init_carry = (init_lstm_state, init_is_eos)
    (final_state, _), _ = lstm(init_carry, inputs)
    return final_state


class DecoderLSTM(nn.Module):
  teacher_force: bool
  vocab_size: int

  @functools.partial(
      nn.scan,
      variable_broadcast='params',
      in_axes=1,
      out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    rng, lstm_state, last_prediction = carry
    carry_rng, categorical_rng = jax.random.split(rng, 2)
    if not self.teacher_force:
      x = last_prediction
    lstm_state, y = nn.LSTMCell()(lstm_state, x)
    logits = nn.Dense(features=self.vocab_size)(y)
    predicted_token = jax.random.categorical(categorical_rng, logits)
    prediction = jax.nn.one_hot(
        predicted_token, self.vocab_size, dtype=jnp.float32)

    return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool
  vocab_size: int

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (seq_length, vocab_size).
    lstm = DecoderLSTM(teacher_force=self.teacher_force,
                       vocab_size=self.vocab_size)
    init_carry = (self.make_rng('lstm'), self.init_state, inputs[:, 0])
    _, (logits, predictions) = lstm(init_carry, inputs)
    return logits, predictions


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture.

  Attributes:
    teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
    hidden_size: int, the number of hidden dimensions in the encoder and
      decoder LSTMs.
  """
  teacher_force: bool
  hidden_size: int
  vocab_size: int
  eos_id: int = 1

  @nn.compact
  def __call__(self, encoder_inputs, decoder_inputs):
    """Runs the seq2seq model.

    Args:
      encoder_inputs: padded batch of input sequences to encode, shaped
        `[batch_size, max(encoder_input_lengths), vocab_size]`.
      decoder_inputs: padded batch of expected decoded sequences for teacher
        forcing, shaped `[batch_size, max(decoder_inputs_length), vocab_size]`.
        When sampling (i.e., `teacher_force = False`), the initial time step is
        forced into the model and samples are used for the following inputs. The
        second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.

    Returns:
      Array of decoded logits.
    """
    # Encode inputs.
    init_decoder_state = Encoder(
        hidden_size=self.hidden_size, eos_id=self.eos_id)(encoder_inputs)
    # Decode outputs.
    logits, predictions = Decoder(
        init_state=init_decoder_state, 
        teacher_force=self.teacher_force,
        vocab_size=self.vocab_size)(decoder_inputs[:, :-1])

    return logits, predictions
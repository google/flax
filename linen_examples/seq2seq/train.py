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

"""seq2seq addition example."""

import random
from absl import app
from absl import flags
from absl import logging
import functools

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.core import Scope, init, apply, unfreeze, lift

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate',
    default=0.003,
    help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128, help=('Batch size for training.'))

flags.DEFINE_integer(
    'hidden_size', default=128, help=('Hidden size of the LSTM.'))

flags.DEFINE_integer(
    'num_train_steps', default=10000, help=('Number of train steps.'))

flags.DEFINE_integer(
    'decode_frequency',
    default=200,
    help=('Frequency of decoding during training, e.g. every 1000 steps.'))

flags.DEFINE_integer(
    'max_len_query_digit',
    default=3,
    help=('Maximum length of a single input digit.'))

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class CharacterTable(object):
  """Encode/decodes between strings and integer representations."""

  @property
  def pad_id(self):
    return 0

  @property
  def eos_id(self):
    return 1

  @property
  def vocab_size(self):
    return len(self._chars) + 2

  def __init__(self, chars):
    self._chars = sorted(set(chars))
    self._char_indices = dict(
        (ch, idx + 2) for idx, ch in enumerate(self._chars))
    self._indices_char = dict(
        (idx + 2, ch) for idx, ch in enumerate(self._chars))
    self._indices_char[self.pad_id] = '_'

  def encode(self, inputs):
    """Encode from string to list of integers."""
    return np.array(
        [self._char_indices[char] for char in inputs] + [self.eos_id])

  def decode(self, inputs):
    """Decode from list of integers to string."""
    chars = []
    for elem in inputs:
      if elem == self.eos_id:
        break
      chars.append(self._indices_char[elem])
    return ''.join(chars)


# We use a global CharacterTable so we don't have pass it around everywhere.
CTABLE = CharacterTable('0123456789+= ')


def get_max_input_len():
  """Returns the max length of an input sequence."""
  return FLAGS.max_len_query_digit * 2 + 2  # includes EOS


def get_max_output_len():
  """Returns the max length of an output sequence."""
  return FLAGS.max_len_query_digit + 3  # includes start token '=' and EOS.


def onehot(sequence, vocab_size):
  """One-hot encode a single sequence of integers."""
  return jnp.array(
      sequence[:, np.newaxis] == jnp.arange(vocab_size), dtype=jnp.float32)


def encode_onehot(batch_inputs, max_len):
  """One-hot encode a string input."""

  def encode_str(s):
    tokens = CTABLE.encode(s)
    if len(tokens) > max_len:
      raise ValueError(f'Sequence too long ({len(tokens)}>{max_len}): \'{s}\'')
    tokens = np.pad(tokens, [(0, max_len-len(tokens))], mode='constant')
    return onehot(tokens, CTABLE.vocab_size)

  return np.array([encode_str(inp) for inp in batch_inputs])


def decode_onehot(batch_inputs):
  """Decode a batch of one-hot encoding to strings."""
  decode_inputs = lambda inputs: CTABLE.decode(inputs.argmax(axis=-1))
  return np.array(list(map(decode_inputs, batch_inputs)))


def get_sequence_lengths(sequence_batch, eos_id=CTABLE.eos_id):
  """Returns the length of each one-hot sequence, including the EOS token."""
  # sequence_batch.shape = (batch_size, seq_length, vocab_size)
  eos_row = sequence_batch[:, :, eos_id]
  eos_idx = jnp.argmax(eos_row, axis=-1)  # returns first occurence
  # `eos_idx` is 0 if EOS is not present, so we use full length in that case.
  return jnp.where(
      eos_row[jnp.arange(eos_row.shape[0]), eos_idx],
      eos_idx + 1,
      sequence_batch.shape[1]  # if there is no EOS, use full length
  )


def mask_sequences(sequence_batch, lengths):
  """Set positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1]))


class EncoderLSTM(nn.Module):
  eos_id: int = 1

  @functools.partial(
      nn.transforms.scan,
      variable_axes={'params': nn.broadcast},
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    lstm_state, is_eos = carry
    new_lstm_state, y = nn.LSTMCell(name='lstm_cell')(lstm_state, x)
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
    # use dummy key since default state init fn is just zeros.
    return nn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""
  eos_id: int = 1
  hidden_size: int = 512

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lstm = EncoderLSTM(eos_id=self.eos_id, name='encoder_lstm')
    init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)
    init_carry = (init_lstm_state, jnp.zeros(batch_size, dtype=np.bool))
    # scan over input axis 1, we should make scan accept an axis argument again.
    inputs = jax.tree_map(lambda x: jnp.moveaxis(x, 0, 1), inputs)
    (final_state, _), _ = lstm(init_carry, inputs)
    return final_state


class DecoderLSTM(nn.Module):
  vocab_size: int
  teacher_force: bool = False

  @functools.partial(
      nn.transforms.scan,
      variable_axes={'params': nn.broadcast},
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    rng, lstm_state, last_prediction = carry
    carry_rng, categorical_rng = jax.random.split(rng, 2)
    if not self.teacher_force:
      x = last_prediction
    lstm_cell = nn.LSTMCell(name='lstm_cell')
    projection = nn.Dense(features=self.vocab_size, name='projection')
    lstm_state, y = lstm_cell(lstm_state, x)
    logits = projection(y)
    predicted_tokens = jax.random.categorical(categorical_rng, logits)
    prediction = onehot(predicted_tokens, self.vocab_size)
    return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool = False

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    lstm = DecoderLSTM(vocab_size=inputs.shape[2],
                       teacher_force=self.teacher_force)
    init_carry = (self.make_rng('lstm'), self.init_state, inputs[:, 0])
    # scan over input axis 1, we should make scan accept an axis argument again.
    inputs = jax.tree_map(lambda x: jnp.moveaxis(x, 0, 1), inputs)
    _, (logits, predictions) = lstm(init_carry, inputs)
    logits, predictions = jax.tree_map(
        lambda x: jnp.moveaxis(x, 0, 1), (logits, predictions))
    return logits, predictions


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture.

  Attributes:
    teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
    eos_id: int, the token signaling when the end of a sequence is reached.
    hidden_size: int, the number of hidden dimensions in the encoder and
      decoder LSTMs.
  """
  teacher_force: bool = True
  eos_id: int = 1
  hidden_size: int = 512

  @nn.compact
  def __call__(self, encoder_inputs, decoder_inputs):
    """Run the seq2seq model.

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
    # Encode inputs
    init_decoder_state = Encoder(
        eos_id=self.eos_id, hidden_size=self.hidden_size)(encoder_inputs)
    # Decode outputs.
    logits, predictions = Decoder(
        init_state=init_decoder_state, teacher_force=self.teacher_force)(
            decoder_inputs[:, :-1])

    return logits, predictions


def model(teacher_force=True):
  return Seq2seq(eos_id=CTABLE.eos_id, teacher_force=teacher_force,
                 hidden_size=FLAGS.hidden_size)


def get_initial_params(key):
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size
  encoder_shape = jnp.ones((1, get_max_input_len(), vocab_size), jnp.float32)
  decoder_shape = jnp.ones((1, get_max_output_len(), vocab_size), jnp.float32)
  return model().init({'params': key, 'lstm': key},
                      encoder_shape, decoder_shape)['params']


def get_examples(num_examples):
  """Returns @num_examples examples."""
  for _ in range(num_examples):
    max_digit = pow(10, FLAGS.max_len_query_digit) - 1
    key = tuple(sorted((random.randint(0, 99), random.randint(0, max_digit))))
    inputs = '{}+{}'.format(key[0], key[1])
    # Preprend output by the decoder's start token.
    outputs = '=' + str(key[0] + key[1])
    yield (inputs, outputs)


def get_batch(batch_size):
  """Returns a batch of example of size @batch_size."""
  inputs, outputs = zip(*get_examples(batch_size))
  return {
      'query': encode_onehot(inputs, max_len=get_max_input_len()),
      'answer': encode_onehot(outputs, max_len=get_max_output_len())
  }


def cross_entropy_loss(logits, labels, lengths):
  """Returns cross-entropy loss."""
  xe = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe


def compute_metrics(logits, labels):
  """Computes metrics and returns them."""
  lengths = get_sequence_lengths(labels)
  loss = cross_entropy_loss(logits, labels, lengths)
  # Computes sequence accuracy, which is the same as the accuracy during
  # inference, since teacher forcing is irrelevant when all output are correct.
  token_accuracy = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
  sequence_accuracy = (
      jnp.sum(mask_sequences(token_accuracy, lengths), axis=-1) == lengths
  )
  accuracy = jnp.mean(sequence_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(optimizer, batch, lstm_key):
  """Train one step."""
  labels = batch['answer'][:, 1:]  # remove '=' start token

  def loss_fn(params):
    """Compute cross-entropy loss."""
    logits, _ = model().apply({'params': params},
                              batch['query'],
                              batch['answer'],
                              rngs={'lstm': lstm_key})
    loss = cross_entropy_loss(logits, labels, get_sequence_lengths(labels))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, labels)
  return optimizer, metrics


def log_decode(question, inferred, golden):
  """Log the given question, inferred query, and correct query."""
  suffix = '(CORRECT)' if inferred == golden else (f'(INCORRECT) '
                                                   f'correct={golden}')
  logging.info('DECODE: %s = %s %s', question, inferred, suffix)


@jax.jit
def decode(params, inputs, key):
  """Decode inputs."""
  init_decoder_input = onehot(CTABLE.encode('=')[0:1], CTABLE.vocab_size)
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], get_max_output_len(), 1))
  _, predictions = model(teacher_force=False).apply({'params': params},
                                                    inputs,
                                                    init_decoder_inputs,
                                                    rngs={'lstm': key})
  return predictions


def decode_batch(params, batch_size, key):
  """Decode and log results for a batch."""
  batch = get_batch(batch_size)
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  inferred = decode(params, inputs, key)
  questions = decode_onehot(inputs)
  infers = decode_onehot(inferred)
  goldens = decode_onehot(outputs)
  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_model():
  """Train for a fixed number of steps and decode during training."""
  param = get_initial_params(jax.random.PRNGKey(0))
  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(param)
  key = jax.random.PRNGKey(0)
  for step in range(FLAGS.num_train_steps):
    key, lstm_key = jax.random.split(key)
    batch = get_batch(FLAGS.batch_size)
    optimizer, metrics = train_step(optimizer, batch, lstm_key)
    if step % FLAGS.decode_frequency == 0:
      key, decode_key = jax.random.split(key)
      logging.info('train step: %d, loss: %.4f, accuracy: %.2f', step,
                   metrics['loss'], metrics['accuracy'] * 100)
      decode_batch(optimizer.target, 5, decode_key)
  return optimizer.target


def main(_):
  _ = train_model()


if __name__ == '__main__':
  app.run(main)

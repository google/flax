# Copyright 2021 The Flax Authors.
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

# See issue #620.
# pytype: disable=wrong-keyword-args

import functools
import random
from typing import Any, Tuple

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', default='.', help='Where to store log output.')

flags.DEFINE_float(
    'learning_rate',
    default=0.003,
    help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128, help=('Batch size for training.'))

flags.DEFINE_integer(
    'hidden_size', default=512, help=('Hidden size of the LSTM.'))

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


def encode_onehot(batch_inputs, max_len=None):
  """One-hot encode a string input."""

  if max_len is None:
    max_len = get_max_input_len()

  def encode_str(s):
    tokens = CTABLE.encode(s)
    unpadded_len = len(tokens)
    if unpadded_len > max_len:
      raise ValueError(f'Sequence too long ({len(tokens)}>{max_len}): \'{s}\'')
    tokens = np.pad(tokens, [(0, max_len-len(tokens))], mode='constant')
    return jax.nn.one_hot(tokens, CTABLE.vocab_size, dtype=jnp.float32)

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

  @functools.partial(
      nn.transforms.scan,
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
    is_eos = jnp.logical_or(is_eos, x[:, CTABLE.eos_id])
    return (carried_lstm_state, is_eos), y


  @staticmethod
  def initialize_carry(batch_size, hidden_size):
    # use dummy key since default state init fn is just zeros.
    return nn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""
  hidden_size: int

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lstm = EncoderLSTM(name='encoder_lstm')
    init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)
    init_is_eos = jnp.zeros(batch_size, dtype=np.bool)
    init_carry = (init_lstm_state, init_is_eos)
    (final_state, _), _ = lstm(init_carry, inputs)
    return final_state


class DecoderLSTM(nn.Module):
  teacher_force: bool

  @functools.partial(
      nn.transforms.scan,
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
    logits = nn.Dense(features=CTABLE.vocab_size)(y)
    predicted_token = jax.random.categorical(categorical_rng, logits)
    prediction = jax.nn.one_hot(
        predicted_token, CTABLE.vocab_size, dtype=jnp.float32)

    return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (seq_length, vocab_size).
    lstm = DecoderLSTM(teacher_force=self.teacher_force)
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
    # Encode inputs.
    init_decoder_state = Encoder(hidden_size=self.hidden_size)(encoder_inputs)
    # Decode outputs.
    logits, predictions = Decoder(
        init_state=init_decoder_state, teacher_force=self.teacher_force)(
            decoder_inputs[:, :-1])

    return logits, predictions


def get_initial_params(model, key):
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size
  encoder_shape = jnp.ones((1, get_max_input_len(), vocab_size), jnp.float32)
  decoder_shape = jnp.ones((1, get_max_output_len(), vocab_size), jnp.float32)
  return model.init({
      'params': key,
      'lstm': key
  }, encoder_shape, decoder_shape)['params']


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
      'query': encode_onehot(inputs),
      'answer': encode_onehot(outputs),
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
def train_step(state, batch, lstm_key):
  """Train one step."""
  labels = batch['answer'][:, 1:]

  def loss_fn(params):
    logits, _ = state.apply_fn({'params': params},
                               batch['query'],
                               batch['answer'],
                               rngs={'lstm': lstm_key})
    loss = cross_entropy_loss(logits, labels, get_sequence_lengths(labels))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, labels)

  return state, metrics


def log_decode(question, inferred, golden):
  """Log the given question, inferred query, and correct query."""
  suffix = '(CORRECT)' if inferred == golden else (f'(INCORRECT) '
                                                   f'correct={golden}')
  logging.info('DECODE: %s = %s %s', question, inferred, suffix)


@jax.jit
def decode(params, inputs, key):
  """Decode inputs."""
  init_decoder_input = jax.nn.one_hot(
      CTABLE.encode('=')[0:1], CTABLE.vocab_size, dtype=jnp.float32)
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], get_max_output_len(), 1))
  model = Seq2seq(teacher_force=False, hidden_size=FLAGS.hidden_size)
  _, predictions = model.apply({'params': params},
                               inputs,
                               init_decoder_inputs,
                               rngs={'lstm': key})
  return predictions


def decode_batch(params, batch, key):
  """Decode and log results for a batch."""
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  inferred = decode(params, inputs, key)
  questions = decode_onehot(inputs)
  infers = decode_onehot(inferred)
  goldens = decode_onehot(outputs)

  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_model(workdir):
  """Train for a fixed number of steps and decode during training."""

  key = jax.random.PRNGKey(0)

  key, init_key = jax.random.split(key)
  model = Seq2seq(teacher_force=False, hidden_size=FLAGS.hidden_size)
  params = get_initial_params(model, init_key)
  tx = optax.adam(FLAGS.learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

  writer = metric_writers.create_default_writer(workdir)
  for step in range(FLAGS.num_train_steps):
    key, lstm_key = jax.random.split(key)
    batch = get_batch(FLAGS.batch_size)
    state, metrics = train_step(state, batch, lstm_key)
    if step % FLAGS.decode_frequency == 0:
      writer.write_scalars(step, metrics)
      key, lstm_key = jax.random.split(key)
      batch = get_batch(5)
      decode_batch(state.params, batch, lstm_key)

  return state


def main(_):
  _ = train_model(FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)

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
    unpadded_len = len(tokens)
    if unpadded_len > max_len:
      raise ValueError(f'Sequence too long ({len(tokens)}>{max_len}): \'{s}\'')
    tokens = np.pad(tokens, [(0, max_len-len(tokens))], mode='constant')
    return onehot(tokens, CTABLE.vocab_size), unpadded_len

  return np.array([encode_str(inp) for inp in batch_inputs])


def decode_onehot(batch_inputs):
  """Decode a batch of one-hot encoding to strings."""
  decode_inputs = lambda inputs: CTABLE.decode(inputs.argmax(axis=-1))
  return np.array(list(map(decode_inputs, batch_inputs)))


def mask_sequences(sequence_batch, lengths):
  """Set positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1]))


class EncoderLSTM(nn.Module):
  @functools.partial(
      nn.transforms.scan,
      variable_in_axes={'param': nn.broadcast},
      variable_out_axes={'param': nn.broadcast},
      split_rngs={'param': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.LSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(hidden_size):
    # use dummy key since default state init fn is just zeros.
    return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (), hidden_size)


class DecoderLSTM(nn.Module):
  teacher_force: bool

  @functools.partial(
      nn.transforms.scan,
      variable_in_axes={'param': nn.broadcast},
      variable_out_axes={'param': nn.broadcast},
      split_rngs={'param': False})
  @nn.compact
  def __call__(self, carry, x):
    rng, lstm_state, last_prediction = carry
    carry_rng, categorical_rng = jax.random.split(rng, 2)
    if not self.teacher_force:
      x = last_prediction
    lstm_state, y = nn.LSTMCell()(lstm_state, x)
    logits = nn.Dense(features=CTABLE.vocab_size)(y)
    predicted_token = jax.random.categorical(categorical_rng, logits)
    prediction = jnp.array(predicted_token == jnp.arange(CTABLE.vocab_size), 
                           dtype=jnp.float32)
    return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (seq_length, vocab_size).
    lstm = DecoderLSTM(teacher_force=self.teacher_force)
    first_token = jax.lax.slice_in_dim(inputs, 0, 1)[0]
    init_carry = (self.make_rng('lstm'), self.init_state, first_token)
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
      encoder_inputs: masked input sequences to encode, shaped
        `[len(input_sequence), vocab_size]`.
      decoder_inputs: masked expected decoded sequences for teacher
        forcing, shaped `[len(output_sequence), vocab_size]`.
        When sampling (i.e., `teacher_force = False`), the initial time step is
        forced into the model and samples are used for the following inputs. The
        first dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
    Returns:
      Array of decoded logits.
    """
    # Encoder.
    encoder = EncoderLSTM()
    init_carry = encoder.initialize_carry(self.hidden_size)
    init_decoder_state, _ = encoder(init_carry, encoder_inputs)
    # Decoder.
    decoder_inputs = jax.lax.slice_in_dim(decoder_inputs, 0, -1)
    decoder = Decoder(init_state=init_decoder_state, 
                      teacher_force=self.teacher_force)
    logits, predictions = decoder(decoder_inputs)

    return logits, predictions


def model(teacher_force=True):
  return Seq2seq(teacher_force=teacher_force, hidden_size=FLAGS.hidden_size)


def get_initial_params(key):
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size
  encoder_shape = jnp.ones((get_max_input_len(), vocab_size), jnp.float32)
  decoder_shape = jnp.ones((get_max_output_len(), vocab_size), jnp.float32)
  return model().init({'param': key, 'lstm': key},
                      encoder_shape, decoder_shape)['param']


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
  query, query_len = zip(*encode_onehot(inputs, max_len=get_max_input_len()))
  answer, ans_len = zip(*encode_onehot(outputs, max_len=get_max_output_len()))
  batch = { 'query': np.array(query), 'answer': np.array(answer)}
  masks = (np.array(query_len), np.array(ans_len))
  return batch, masks


def cross_entropy_loss(logits, labels, lengths):
  """Returns cross-entropy loss."""
  xe = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe


def compute_metrics(logits, labels, lengths):
  """Computes metrics and returns them."""
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


def loss_fn(params, batch, masks, labels, lstm_key):
  in_shapes = [{'query': '(n, _)', 'answer': '(m, _)'}]
  out_shape = f'(m +- 1, {CTABLE.vocab_size})'
  @functools.partial(jax.mask, in_shapes=in_shapes, out_shape=out_shape)
  def get_logits(example):
    logits, _ = model().apply({'param': params}, example['query'],
                              example['answer'], rngs={'lstm': lstm_key})
    return logits
  in_masks, out_masks = masks
  logits = jax.vmap(get_logits)([batch], dict(n=in_masks, m=out_masks))
  loss = cross_entropy_loss(logits, labels, out_masks)
  return loss, logits


@jax.jit
def train_step(optimizer, batch, masks, lstm_key):
  """Train one step."""
  labels = batch['answer'][:, 1:]
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target, batch, masks, labels, lstm_key)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, labels, masks[1]-1)
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
  init_decoder_inputs = jnp.tile(init_decoder_input, (get_max_output_len(), 1))
  _, predictions = model(teacher_force=False).apply({'param': params}, inputs,
                                                    init_decoder_inputs,
                                                    rngs={'lstm': key})
  return predictions


def decode_batch(params, batch_size, key):
  """Decode and log results for a batch."""
  batch, _ = get_batch(batch_size)
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  keys = jax.random.split(key, num=batch_size)
  inferred = jax.vmap(decode, in_axes=(None, 0, 0))(params, inputs, keys)
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
    batch, masks = get_batch(FLAGS.batch_size)
    optimizer, metrics = train_step(optimizer, batch, masks, lstm_key)
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

# Lint as: python3
#
# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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

from flax import jax_utils
from flax import nn
from flax import optim

import jax
import jax.numpy as jnp

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate',
    default=0.003,
    help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128, help=('Batch size for training.'))

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


class Encoder(nn.Module):
  """LSTM encoder, returning final state."""

  def apply(self, inputs, eos_id=1, hidden_size=512):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lengths = get_sequence_lengths(inputs, eos_id=eos_id)

    lstm = nn.LSTMCell.partial(name='lstm')
    init_lstm_state = nn.LSTMCell.initialize_carry(
        nn.make_rng(),
        (batch_size,),
        hidden_size)

    def encode(carry, x):
      i, state = carry
      new_state, y = lstm(state, x)
      *carried_state, = jnp.where(
          (i < lengths)[np.newaxis, :, np.newaxis], new_state, state)
      return (i+1, tuple(carried_state)), y

    (_, final_state), _ = jax_utils.scan_in_dim(
        encode,
        init=(0, init_lstm_state),
        xs=inputs,
        axis=1)
    return final_state


class Decoder(nn.Module):
  """LSTM decoder."""

  def apply(self, init_state, inputs, teacher_force=False):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    vocab_size = inputs.shape[2]
    lstm_cell = nn.LSTMCell.partial(name='lstm')
    projection = nn.Dense.partial(features=vocab_size, name='projection')

    def decode_step_fn(carry, x):
      lstm_state, last_prediction = carry
      if not teacher_force:
        x = last_prediction
      lstm_state, y = lstm_cell(lstm_state, x)
      logits = projection(y)
      predicted_tokens = jax.random.categorical(nn.make_rng(), logits)
      prediction = onehot(predicted_tokens, vocab_size)
      return (lstm_state, prediction), (logits, prediction)

    init_carry = (
        init_state,  # lstm_state
        inputs[:, 0],  # last output
    )

    _, (logits, predictions) = jax_utils.scan_in_dim(
        decode_step_fn, init_carry, xs=inputs, axis=1)
    return logits, predictions


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def _create_modules(self, eos_id, hidden_size):
    encoder = Encoder.partial(
        eos_id=eos_id, hidden_size=hidden_size).shared(name='encoder')
    decoder = Decoder.shared(name='decoder')
    return encoder, decoder

  def apply(self,
            encoder_inputs,
            decoder_inputs,
            teacher_force=True,
            eos_id=1,
            hidden_size=512):
    """Run the seq2seq model with teacher forcing.

    Args:
      encoder_inputs: padded batch of input sequences to encode, shaped
        `[batch_size, max(encoder_input_lengths), vocab_size]`.
      decoder_inputs: padded batch of expected decoded sequences for teacher
        forcing, shaped `[batch_size, max(decoder_inputs_length), vocab_size]`.
        When sampling (i.e., `teacher_force = False`), the initial time step is
        forced into the model and samples are used for the following inputs. The
        second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
      teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
      eos_id: int, the token signaling when the end of a sequence is reached.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
    Returns:
      Array of decoded logits.
    """
    encoder, decoder = self._create_modules(eos_id, hidden_size)

    # Encode inputs
    init_decoder_state = encoder(encoder_inputs)
    # Decode outputs.
    logits, predictions = decoder(
        init_decoder_state,
        decoder_inputs[:, :-1],
        teacher_force=teacher_force)

    return logits, predictions


def create_model(rng):
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size
  _, initial_params = Seq2seq.partial(eos_id=CTABLE.eos_id).init_by_shape(
      rng,
      [((1, get_max_input_len(), vocab_size), jnp.float32),
       ((1, get_max_output_len(), vocab_size), jnp.float32)])
  model = nn.Model(Seq2seq, initial_params)
  return model


def create_optimizer(model, learning_rate):
  """Creates an Adam optimizer for @model."""
  optimizer_def = optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer


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
def train_step(optimizer, batch, rng):
  """Train one step."""
  labels = batch['answer'][:, 1:]  # remove '=' start token

  def loss_fn(model):
    """Compute cross-entropy loss."""
    logits, _ = model(batch['query'], batch['answer'])
    loss = cross_entropy_loss(logits, labels, get_sequence_lengths(labels))
    return loss, logits
  with nn.stochastic(rng):
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
def decode(model, inputs):
  """Decode inputs."""
  init_decoder_input = onehot(CTABLE.encode('=')[0:1], CTABLE.vocab_size)
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], get_max_output_len(), 1))
  _, predictions = model(inputs, init_decoder_inputs, teacher_force=False)
  return predictions


def decode_batch(model, batch_size):
  """Decode and log results for a batch."""
  batch = get_batch(batch_size)
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  inferred = decode(model, inputs)
  questions = decode_onehot(inputs)
  infers = decode_onehot(inferred)
  goldens = decode_onehot(outputs)
  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_model():
  """Train for a fixed number of steps and decode during training."""
  rng = jax.random.PRNGKey(0)
  with nn.stochastic(rng):
    model = create_model(nn.make_rng())
    optimizer = create_optimizer(model, FLAGS.learning_rate)
    for step in range(FLAGS.num_train_steps):
      batch = get_batch(FLAGS.batch_size)
      optimizer, metrics = train_step(optimizer, batch, nn.make_rng())
      if step % FLAGS.decode_frequency == 0:
        logging.info('train step: %d, loss: %.4f, accuracy: %.2f', step,
                     metrics['loss'], metrics['accuracy'] * 100)
        decode_batch(optimizer.target, 5)
    return optimizer.target


def main(_):
  _ = train_model()


if __name__ == '__main__':
  app.run(main)

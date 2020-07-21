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

# Lint as: python3
"""seq2seq addition example."""

import random
from absl import app
from absl import flags
from absl import logging
from functools import partial

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.core import Scope, init, apply, unfreeze, lift

from typing import Any, Callable, Tuple

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
Dtype = Any  # this could be a real type?
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


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""
  eos_id: int = 1
  hidden_size: int = 512

  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lstm_cell = nn.LSTMCell(self, name='lstm')
    # TODO(marcvanzee): Rewrite this uinsg lift.scan to ensure rngs work
    # correctly within the loop.
    init_lstm_state = nn.LSTMCell.initialize_carry(
        self.make_rng('lstm'),
        (batch_size,),
        self.hidden_size)

    def encode_step_fn(carry, x):
      lstm_state, is_eos = carry
      new_lstm_state, y = lstm_cell(lstm_state, x)
      # Pass forward the previous state if EOS has already been reached.
      def select_carried_state(new_state, old_state):
        return jnp.where(is_eos[:, np.newaxis], old_state, new_state)
      # LSTM state is a tuple (c, h).
      carried_lstm_state = tuple(
          select_carried_state(*s) for s in zip(new_lstm_state, lstm_state))
      # Update `is_eos`.
      is_eos = jnp.logical_or(is_eos, x[:, self.eos_id])
      return (carried_lstm_state, is_eos), y

    (final_state, _), _ = jax_utils.scan_in_dim(
        encode_step_fn,
        init=(init_lstm_state, jnp.zeros(batch_size, dtype=np.bool)),
        xs=inputs,
        axis=1)
    return final_state


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool = False

  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    vocab_size = inputs.shape[2]
    lstm_cell = nn.LSTMCell(self, name='lstm')
    projection = nn.Dense(self, features=vocab_size, name='projection')

    def decode_step_fn(carry, x):
      rng, lstm_state, last_prediction = carry
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not self.teacher_force:
        x = last_prediction
      lstm_state, y = lstm_cell(lstm_state, x)
      logits = projection(y)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      prediction = onehot(predicted_tokens, vocab_size)
      return (carry_rng, lstm_state, prediction), (logits, prediction)

    _, (logits, predictions) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=(self.make_rng('lstm'), self.init_state, inputs[:, 0]),
        xs=inputs,
        axis=1)
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
        self, eos_id=self.eos_id, hidden_size=self.hidden_size)(
            encoder_inputs)
    # Decode outputs.
    logits, predictions = Decoder(
        self, init_state=init_decoder_state, teacher_force=self.teacher_force)(
            decoder_inputs[:, :-1])

    return logits, predictions


def model(teacher_force=True):
  return Seq2seq(parent=None, eos_id=CTABLE.eos_id, teacher_force=teacher_force,
                 hidden_size=FLAGS.hidden_size)


def get_param(key):
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size
  encoder_shape = jnp.ones((1, get_max_input_len(), vocab_size), jnp.float32)
  decoder_shape = jnp.ones((1, get_max_output_len(), vocab_size), jnp.float32)
  return model().initialized({
      'param': key, 'lstm': key
  }, encoder_shape, decoder_shape).variables.param


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
def train_step(optimizer, batch):
  """Train one step."""
  labels = batch['answer'][:, 1:]  # remove '=' start token

  def loss_fn(params):
    """Compute cross-entropy loss."""
    logits, _ = model().apply({'param': params},
                              batch['query'],
                              batch['answer'],
                              rngs={'lstm': jax.random.PRNGKey(0)})
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
def decode(params, inputs):
  """Decode inputs."""
  init_decoder_input = onehot(CTABLE.encode('=')[0:1], CTABLE.vocab_size)
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], get_max_output_len(), 1))
  _, predictions = model(teacher_force=False).apply({'param': params},
                                                    inputs,
                                                    init_decoder_inputs,
                                                    rngs={'lstm': jax.random.PRNGKey(0)})
  return predictions


def decode_batch(params, batch_size):
  """Decode and log results for a batch."""
  batch = get_batch(batch_size)
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  inferred = decode(params, inputs)
  questions = decode_onehot(inputs)
  infers = decode_onehot(inferred)
  goldens = decode_onehot(outputs)
  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_model():
  """Train for a fixed number of steps and decode during training."""
  param = get_param(jax.random.PRNGKey(0))
  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(param)
  for step in range(FLAGS.num_train_steps):
    batch = get_batch(FLAGS.batch_size)
    optimizer, metrics = train_step(optimizer, batch)
    if step % FLAGS.decode_frequency == 0:
      logging.info('train step: %d, loss: %.4f, accuracy: %.2f', step,
                   metrics['loss'], metrics['accuracy'] * 100)
      decode_batch(optimizer.target, 5)
  return optimizer.target


# def lstm(scope, x, ...):
#   carry_shape = ...
#   carry = scope.variable('memory', 'carry', carry_init_fn, carry_shape)
#   new_carry = lstm_logic(carry.value)
#   carry.value = new_carry
#   return ...

class LSTMCellStandard(nn.Module):
  gate_fn: Callable = nn.activation.sigmoid
  activation_fn: Callable = nn.activation.tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

  def __call__(self, carry, inputs):
    c, h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(nn.linear.Dense,
                      self,
                      features=hidden_features,
                      use_bias=True,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(nn.linear.Dense,
                      self,
                      features=hidden_features,
                      use_bias=False,
                      kernel_init=self.kernel_init)
    i = self.gate_fn(dense_i(name='ii')(inputs) + dense_h(name='hi')(h))
    f = self.gate_fn(dense_i(name='if')(inputs) + dense_h(name='hf')(h))
    g = self.activation_fn(dense_i(name='ig')(inputs) + dense_h(name='hg')(h))
    o = self.gate_fn(dense_i(name='io')(inputs) + dense_h(name='ho')(h))
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h


def initialize_standard(rng, batch_dims, size, init_fn=nn.initializers.zeros):
  key1, key2 = jax.random.split(rng)
  mem_shape = batch_dims + (size,)
  return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


def lstm_logic(carry):
  return carry * 2

def initialize_new(key, batch_dims, size, init_fn=nn.initializers.zeros):
  mem_shape = batch_dims + (size,)
  return init_fn(key, mem_shape), init_fn(key, mem_shape)


def lstm_cell_new(scope, x):
  # carry = scope.variable('memory', 'carry', initialize_new, (1,), 8)
  carry = scope.param('carry', initialize_new, (1,), 8)
  new_carry = lstm_logic(carry.value)
  carry.value = new_carry
  return x


def new(batch, batch_size, hidden_size):
  input_init = jnp.ones(batch.shape, jnp.float32)
  y, params = init(lstm_cell_new)(jax.random.PRNGKey(0), input_init)
  x = lift.scan(lstm_cell, variable_modes={'param': 'broadcast', 'state': 'carry'}, split_rngs={'param': False, 'dropout': True})
  print(x)


def standard(batch, batch_size, hidden_size):
  c, h = initialize_standard(jax.random.PRNGKey(0), (batch_size,), hidden_size)
  # shapes of both c and h are: (batch_size, hidden_size)

  carry_init = (jnp.ones(c.shape, jnp.float32), jnp.ones(h.shape, jnp.float32))
  input_init = jnp.ones(batch.shape, jnp.float32)


  lstm_cell = LSTMCellStandard(parent=None).initialized({'param': jax.random.PRNGKey(0)},
      carry_init, input_init)

  (final_state, _), _ = jax_utils.scan_in_dim(
      lstm_cell,
      init=(c, h),
      xs=batch,
      axis=1)
  print('final_state', final_state)


def train_model2():
  hidden_size = 8
  batch_size = 1

  query, _ = zip(*get_examples(batch_size))
  print('query', query)
  print('onehot', onehot)

  new(encode_onehot(query, get_max_input_len()), batch_size, hidden_size)


def main(_):
  _ = train_model2()


if __name__ == '__main__':
  app.run(main)

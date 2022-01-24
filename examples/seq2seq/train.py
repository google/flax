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

"""seq2seq addition example."""

# See issue #620.
# pytype: disable=wrong-keyword-args

import functools
from typing import Any

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from input_pipeline import mask_sequences, get_sequence_lengths
from input_pipeline import CharacterTable as CTable

import models


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


def get_model(ctable, *, teacher_force: bool = False):
  return models.Seq2seq(teacher_force=teacher_force,
                        hidden_size=FLAGS.hidden_size, eos_id=ctable.eos_id,
                        vocab_size=ctable.vocab_size)


def get_initial_params(model, rng, ctable):
  """Returns the initial parameters of a seq2seq model."""
  rng1, rng2 = jax.random.split(rng)
  variables = model.init(
      {'params': rng1, 'lstm': rng2},
      jnp.ones(ctable.encoder_input_shape, jnp.float32),
      jnp.ones(ctable.decoder_input_shape, jnp.float32)
  )
  return variables['params']


def get_train_state(rng, ctable):
  """Returns a train state."""
  model = get_model(ctable)
  params = get_initial_params(model, rng, ctable)
  tx = optax.adam(FLAGS.learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)
  return state

def mask_sequences(sequence_batch, lengths):
  """Set positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis])


class EncoderLSTM(nn.Module):
  def setup(self):
    self.lstm = nn.LSTMCell()

  @functools.partial(
      nn.scan,
      variable_broadcast='params',
      in_axes=1,
      out_axes=1,
      split_rngs={'params': False})
  def __call__(self, carry, x):
    lstm_carry, is_eos = carry
    new_lstm_carry, y = self.lstm(lstm_carry, x)
    # Pass forward the previous state if EOS has already been reached.
    def select_carried_state(new_state, old_state):
      return jnp.where(is_eos[:, np.newaxis], old_state, new_state)
    # LSTM state is a tuple (c, h).
    carried_lstm_state = nn.LSTMCarry(
      select_carried_state(new_lstm_carry.cell_state, lstm_carry.cell_state),
      select_carried_state(new_lstm_carry.hidden_state,
                           lstm_carry.hidden_state))
    # Update `is_eos`.
    is_eos = jnp.logical_or(is_eos, x[:, CTABLE.eos_id])
    new_carry = (carried_lstm_state, is_eos)
    return new_carry, y

  def initialize_carry(self, batch_dims, hidden_size, inputs):
    lstm_carry = self.lstm.initialize_carry(batch_dims, hidden_size,
                                            inputs[:, 0, :])
    is_eos = jnp.zeros(inputs.shape[:batch_dims], dtype=np.bool_)
    return lstm_carry, is_eos


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""
  hidden_size: int

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    lstm = EncoderLSTM(name='encoder_lstm')
    init_carry = lstm.initialize_carry(1, self.hidden_size, inputs)
    (final_state, _), _ = lstm(init_carry, inputs)
    return final_state


class DecoderLSTM(nn.Module):
  teacher_force: bool

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


def compute_metrics(logits, labels, eos_id):
  """Computes metrics and returns them."""
  lengths = get_sequence_lengths(labels, eos_id)
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
def train_step(state, batch, lstm_rng, eos_id):
  """Trains one step."""
  labels = batch['answer'][:, 1:]
  lstm_key = jax.random.fold_in(lstm_rng, state.step)

  def loss_fn(params):
    # The params key is not used, but LSTMCell requires it.
    logits, _ = state.apply_fn({'params': params},
                               batch['query'],
                               batch['answer'],
                               rngs={'params': jax.random.PRNGKey(0),
                                     'lstm': lstm_key})
    loss = cross_entropy_loss(
        logits, labels, get_sequence_lengths(labels, eos_id))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, labels, eos_id)

  return state, metrics


def log_decode(question, inferred, golden):
  """Logs the given question, inferred query, and correct query."""
  suffix = '(CORRECT)' if inferred == golden else (f'(INCORRECT) '
                                                   f'correct={golden}')
  logging.info('DECODE: %s = %s %s', question, inferred, suffix)


@functools.partial(jax.jit, static_argnums=3)
def decode(params, inputs, decode_rng, ctable):
  """Decodes inputs."""
  init_decoder_input = ctable.one_hot(ctable.encode('=')[0:1])
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], ctable.max_output_len, 1))
  model = get_model(ctable, teacher_force=False)
    # The params key is not used, but LSTMCell requires it.
  _, predictions = model.apply({'params': params},
                               inputs,
                               init_decoder_inputs,
                               rngs={'params': jax.random.PRNGKey(0),
                                     'lstm': key})
  return predictions


def decode_batch(state, batch, decode_rng, ctable):
  """Decodes and log results for a batch."""
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  decode_rng = jax.random.fold_in(decode_rng, state.step)
  inferred = decode(state.params, inputs, decode_rng, ctable)
  questions = ctable.decode_onehot(inputs)
  infers = ctable.decode_onehot(inferred)
  goldens = ctable.decode_onehot(outputs)

  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_and_evaluate(workdir):
  """Trains for a fixed number of steps and decode during training."""

  # TODO(marcvanzee): Integrate ctable with train_state.
  ctable = CTable('0123456789+= ', FLAGS.max_len_query_digit)
  rng = jax.random.PRNGKey(0)
  state = get_train_state(rng, ctable)

  writer = metric_writers.create_default_writer(workdir)
  for step in range(FLAGS.num_train_steps):
    batch = ctable.get_batch(FLAGS.batch_size)
    state, metrics = train_step(state, batch, rng, ctable.eos_id)
    if step and step % FLAGS.decode_frequency == 0:
      writer.write_scalars(step, metrics)
      batch = ctable.get_batch(5)
      decode_batch(state, batch, rng, ctable)

  return state


def main(_):
  _ = train_and_evaluate(FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)

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
  def eos_id(self):
    return 0

  @property
  def pad_id(self):
    return 1

  def __init__(self, chars):
    self._chars = sorted(set(chars))
    self._char_indices = dict(
        (ch, idx + 2) for idx, ch in enumerate(self._chars))
    self._indices_char = dict(
        (idx + 2, ch) for idx, ch in enumerate(self._chars))

  def encode(self, inputs):
    """Encode from string to list of integers."""
    return np.array(
        [self._char_indices[char] for char in inputs] + [self.eos_id])

  def decode(self, inputs):
    """Decode from list of integers to string."""
    return ''.join(self._indices_char[elem] for elem in inputs)

  def vocab_size(self):
    return len(self._chars) + 2


# We use a global CharacterTable so we don't have pass it around everywhere.
CTABLE = CharacterTable('0123456789+= ')


def get_max_input_len():
  """Returns the max length of an input sequence."""
  return FLAGS.max_len_query_digit * 2 + 1


def get_max_output_len():
  """Returns the max length of an output sequence."""
  return FLAGS.max_len_query_digit + 2  # includes start token '='.


@jax.jit
def encode_onehot(batch_inputs, max_len):
  """One-hot encode a string input."""

  @jax.vmap
  def encode_inputs(inputs):
    inputs = CTABLE.encode(inputs)
    if len(inputs) > max_len:
      raise ValueError(f'Sequence too long: {len(inputs)}')
    inputs = np.pad(inputs, [(0, 0), (0, max_len-len(inputs))], mode='constant')
    one_hot = np.zeros((inputs.size, CTABLE.vocab_size()))
    one_hot[np.arange(inputs.size), inputs] = 1
    return one_hot

  return encode_inputs(batch_inputs)


def decode_onehot(batch_inputs):
  """Decode a batch of one-hot encoding to strings."""
  decode_inputs = lambda inputs: CTABLE.decode(inputs.argmax(axis=-1))
  return np.array(list(map(decode_inputs, batch_inputs)))


def get_sequence_lengths(sequence_batch, eos_id=1):
  """Returns the length of each onehot sequence, including the EOS token."""
  eos_row = sequence_batch[:, :, eos_id]
  eos_idx = jnp.argmax(eos_row == eos_id, axis=-1)  # returns first occurence
  return jnp.where(
      eos_row[jnp.arange(eos_row.shape[0]), eos_idx] == eos_id,
      eos_idx + 1,
      sequence_batch.shape[1]  # if there is no EOS, use full length
  )


class Encoder(nn.Module):
  """LSTM encoder, returning final state."""

  def apply(self, inputs, eos_id=1, hidden_size=512):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    lengths = get_sequence_lengths(inputs, eos_id=eos_id)
    carry = nn.LSTMCell.initialize_carry(
        nn.make_rng(),
        (batch_size,),
        hidden_size)
    _, outputs = jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm'),
        carry,
        inputs,
        axis=1)
    return outputs[jnp.arange(batch_size), jnp.maximum(0, lengths-1), :]


class Decoder(nn.Module):
  """LSTM decoder."""

  def apply(self, carry, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    vocab_size = inputs.shape[2]
    carry, outputs = jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm'), carry, inputs, axis=1)
    x = nn.Dense(outputs, features=vocab_size, name='dense')
    return carry, x


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
            eos_id=1,
            hidden_size=512):
    """Run the seq2seq model with teacher forcing.

    Args:
      encoder_inputs: padded batch of input sequences to encode, shaped
        `[batch_size, max(encoder_input_lengths), vocab_size]`.
      decoder_inputs: padded batch of expected decoded sequences for teacher
        forcing, shaped `[batch_size, max(decoder_inputs_length), vocab_size]`.
      eos_id: int, the token signalling when the end of a sequence is reached.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
    """
    encoder, decoder = self._create_modules(eos_id, hidden_size)

    # Encode inputs
    carry = encoder(encoder_inputs)
    # Decode with teacher forcing.
    _, x = decoder(carry, decoder_inputs[:, :-1])

    return x

  @nn.module_method
  def sample(self,
             encoder_inputs,
             max_output_len,
             sample_temperature=1.0,
             eos_id=1,
             hidden_size=512):
    """Run the seq2seq model with sampling.

    Args:
      encoder_inputs: padded batch of input sequences to encode, shaped
        `[batch_size, max(encoder_input_lengths), vocab_size]`.
      max_output_len: int, maximum number of steps to decode. If None, decoding
        will continue until all sequences reach EOS.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
    """
    batch_size, _, vocab_size = encoder_inputs.shape
    encoder, decoder = self._create_modules(eos_id, hidden_size)

    # Encode inputs
    init_decoder_state = encoder(encoder_inputs)

    # Use autoregressive decoding, sampling predictions at each step.
    def decode_step_fn(carry, unused_x):
      decoder_carry, next_inputs = carry
      decoder_carry, decoder_outputs = decoder(
          decoder_carry, next_inputs[:, np.newaxis])
      decoder_outputs = decoder_outputs.squeeze()
      predicted_tokens = jax.random.categorical(
          nn.make_rng(), decoder_outputs / sample_temperature)
      # Onehot encode predictions.
      next_inputs = jnp.array(
          predicted_tokens == jnp.arange(vocab_size),
          dtype=jnp.float32)

      return (decoder_carry, next_inputs), next_inputs

    init_carry = (
        init_decoder_state,  # decoder_carry
        jnp.zeros((batch_size, vocab_size), dtype=np.float32),  # next_inputs
    )
    predictions = jax_utils.scan_in_dim(
        decode_step_fn, init_carry, xs=[None]*max_output_len, axis=1)
    return predictions


def create_model():
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size()
  _, initial_params = Seq2seq.init_by_shape(
      nn.make_rng(),
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
      'query': encode_onehot(np.array(inputs), max_len=get_max_input_len()),
      'answer': encode_onehot(np.array(outputs), max_len=get_max_output_len())
  }


def cross_entropy_loss(logits, labels):
  """Returns cross-entropy loss."""
  return -jnp.mean(jnp.sum(nn.log_softmax(logits) * labels[:, 1:], axis=-1))


def compute_metrics(logits, labels):
  """Computes metrics and returns them."""
  loss = cross_entropy_loss(logits, labels)
  # Computes sequence accuracy, which is the same as the accuracy during
  # inference, since teacher forcing is irrelevant when all output are correct.
  labels = labels[:, 1:]  # Remove start token from labels.
  accuracy = jnp.mean(
      jnp.all(jnp.argmax(logits, -1) == jnp.argmax(labels, -1), axis=1))
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(optimizer, batch, rng):
  """Train one step."""

  def loss_fn(model):
    """Compute cross-entropy loss."""
    logits = model(batch['query'], batch['answer'])
    loss = cross_entropy_loss(logits, batch['answer'])
    return loss, logits
  with nn.stochastic(rng):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['answer'])
  return optimizer, metrics


def log_decode(question, inferred, golden):
  """Log the given question, inferred query, and correct query."""
  # Remove last token from inferred string and first token from golden.
  inferred, golden = inferred[:-1], golden[1:]
  suffix = '(CORRECT)' if inferred == golden else (f'(INCORRECT) '
                                                   f'correct={golden}')
  logging.info('DECODE: %s = %s %s', question, inferred, suffix)


@jax.jit
def decode(model, inputs):
  """Decode inputs."""
  decoder_inputs = encode_onehot(
      np.array(['=']), max_len=get_max_input_len()).squeeze()
  decoder_inputs = jnp.tile(decoder_inputs, (inputs.shape[0], 1))
  return model(
      inputs, decoder_inputs, train=False, max_output_len=get_max_output_len())


def decode_batch(model, batch_size):
  """Decode and log results for a batch."""
  batch = get_batch(batch_size)
  inputs, outputs = batch['query'], batch['answer']
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
    model = create_model()
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

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
"""seq2seq addition example.

This script trains a simple LSTM on a sequence-to-sequence addition task using
an encoder-decoder architecture. The data is generated on the fly.
"""

import random
from absl import app
from absl import flags
from absl import logging

from flax import nn
from flax import optim
import jax
from jax import random as jrandom
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

  def __init__(self, chars):
    self._chars = sorted(set(chars))
    self._char_indices = dict((ch, idx) for idx, ch in enumerate(self._chars))
    self._indices_char = dict((idx, ch) for idx, ch in enumerate(self._chars))

  def encode(self, inputs):
    """Encode from string to list of integers."""
    return np.array([self._char_indices[char] for char in inputs])

  def decode(self, inputs):
    """Decode from list of integers to string."""
    return ''.join(self._indices_char[elem] for elem in inputs)

  def vocab_size(self):
    return len(self._chars)


# We use a global CharacterTable so we don't have pass it around everywhere.
CTABLE = CharacterTable('0123456789+= ')


def get_max_input_len():
  """Returns the max length of an input sequence."""
  return FLAGS.max_len_query_digit * 2 + 1


def get_max_output_len():
  """Returns the max length of an output sequence."""
  return FLAGS.max_len_query_digit + 2  # includes start token '='.


def encode_onehot(batch_inputs):
  """One-hot encode a string input."""

  def encode_inputs(inputs):
    inputs = CTABLE.encode(inputs)
    one_hot = np.zeros((inputs.size, CTABLE.vocab_size()))
    one_hot[np.arange(inputs.size), inputs] = 1
    return one_hot

  return np.array(list(map(encode_inputs, batch_inputs)))


def decode_onehot(batch_inputs):
  """Decode a batch of one-hot encoding to strings."""
  decode_inputs = lambda inputs: CTABLE.decode(inputs.argmax(axis=-1))
  return np.array(list(map(decode_inputs, batch_inputs)))


class Encoder(nn.Module):
  """LSTM encoder."""

  def apply(self, inputs, hidden_size=512):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]
    carry = nn.LSTMCell.initialize_carry(nn.make_rng(), (batch_size,),
                                         hidden_size)
    carry, _ = jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm'), carry, inputs, axis=1)
    return carry


class Decoder(nn.Module):
  """LSTM decoder."""

  def apply(self, carry, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    vocab_size = inputs.shape[2]
    carry, outputs = jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm'), carry, inputs, axis=1)
    x = nn.Dense(outputs, features=vocab_size, name='dense')
    x = nn.log_softmax(x)
    return carry, x


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def apply(self,
            encoder_inputs,
            decoder_inputs,
            train=True,
            max_output_len=None):
    """Run the seq2seq model."""
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size, _, vocab_size = encoder_inputs.shape
    carry = Encoder(encoder_inputs, name='encoder')
    decoder = Decoder.shared(name='decoder')

    # Teacher forcing.
    if train:
      _, x = decoder(carry, decoder_inputs[:, :-1])
      return x

    # No teacher forcing, feeding actual output back into the decoder.
    output = np.zeros((max_output_len, batch_size, vocab_size))
    for i in range(max_output_len):
      decoder_inputs = np.expand_dims(decoder_inputs, axis=1)
      carry, decoder_inputs = decoder(carry, decoder_inputs)
      decoder_inputs = decoder_inputs.squeeze()
      # Set next inputs to {0,1} to ensure they are the same as in training.
      decoder_inputs = np.array([
          decoder_inputs[j] == max(decoder_inputs[j]) for j in range(batch_size)
      ])
      output[:, i] = decoder_inputs
    return output


def create_model():
  """Creates a seq2seq model."""
  vocab_size = CTABLE.vocab_size()
  _, initial_params = Seq2seq.init_by_shape(
      nn.make_rng(), [((1, get_max_input_len(), vocab_size), jnp.float32),
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
    inputs = '{}+{}'.format(key[0], key[1]).ljust(get_max_input_len())
    # Preprend output by the decoder's start token.
    outputs = '=' + str(key[0] + key[1]).ljust(get_max_output_len() - 1)
    yield (inputs, outputs)


def get_batch(batch_size):
  """Returns a batch of example of size @batch_size."""
  inputs, outputs = zip(*get_examples(batch_size))
  return {
      'query': encode_onehot(np.array(inputs)),
      'answer': encode_onehot(np.array(outputs))
  }


def cross_entropy_loss(logits, labels):
  """Returns cross-entropy loss."""
  return -jnp.mean(jnp.sum(logits * labels[:, 1:], axis=-1))


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
def train_step(optimizer, batch):
  """Train one step."""

  def loss_fn(model):
    """Compute cross-entropy loss."""
    logits = model(batch['query'], batch['answer'])
    loss = cross_entropy_loss(logits, batch['answer'])
    return loss, logits
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


def decode_batch(model, batch_size):
  """Decode a batch."""
  batch = get_batch(batch_size)
  inputs, outputs = (batch['query'], batch['answer'])
  decoder_inputs = encode_onehot(np.array(['='])).squeeze()
  decoder_inputs = np.tile(decoder_inputs, (batch_size, 1))
  inferred = model(
      inputs, decoder_inputs, train=False, max_output_len=get_max_output_len())
  questions = decode_onehot(inputs)
  infers = decode_onehot(inferred)
  goldens = decode_onehot(outputs)
  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_model():
  """Train for a fixed number of steps and decode during training."""
  rng = jrandom.PRNGKey(0)

  with nn.stochastic(rng):
    model = create_model()
    optimizer = create_optimizer(model, FLAGS.learning_rate)
    for step in range(FLAGS.num_train_steps):
      batch = get_batch(FLAGS.batch_size)
      optimizer, metrics = train_step(optimizer, batch)
      if step % FLAGS.decode_frequency == 0:
        logging.info('train step: %d, loss: %.4f, accuracy: %.2f', step,
                     batch_metrics['loss'], batch_metrics['accuracy'] * 100)
        decode_batch(optimizer.target, 5)
  return optimizer.target


def main(_):
  _ = train_model()


if __name__ == '__main__':
  app.run(main)

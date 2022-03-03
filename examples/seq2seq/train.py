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
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

import models
from input_pipeline import CharacterTable as CTable
from input_pipeline import get_sequence_lengths
from input_pipeline import mask_sequences


Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any

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


def get_model(ctable: CTable, *, teacher_force: bool = False) -> models.Seq2seq:
  return models.Seq2seq(teacher_force=teacher_force,
                        hidden_size=FLAGS.hidden_size, eos_id=ctable.eos_id,
                        vocab_size=ctable.vocab_size)


def get_initial_params(model: models.Seq2seq, rng: PRNGKey,
                       ctable: CTable) -> Dict[str, Any]:
  """Returns the initial parameters of a seq2seq model."""
  rng1, rng2 = jax.random.split(rng)
  variables = model.init(
      {'params': rng1, 'lstm': rng2},
      jnp.ones(ctable.encoder_input_shape, jnp.float32),
      jnp.ones(ctable.decoder_input_shape, jnp.float32)
  )
  return variables['params']


def get_train_state(rng: PRNGKey, ctable: CTable) -> train_state.TrainState:
  """Returns a train state."""
  model = get_model(ctable)
  params = get_initial_params(model, rng, ctable)
  tx = optax.adam(FLAGS.learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)
  return state


def cross_entropy_loss(logits: Array, labels: Array, lengths: Array) -> float:
  """Returns cross-entropy loss."""
  xe = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe


def compute_metrics(logits: Array, labels: Array,
                    eos_id: int) -> Dict[str, float]:
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
def train_step(state: train_state.TrainState, batch: Array, lstm_rng: PRNGKey,
               eos_id: int) -> Tuple[train_state.TrainState, Dict[str, float]]:
  """Trains one step."""
  labels = batch['answer'][:, 1:]
  lstm_key = jax.random.fold_in(lstm_rng, state.step)

  def loss_fn(params):
    logits, _ = state.apply_fn({'params': params},
                               batch['query'],
                               batch['answer'],
                               rngs={'lstm': lstm_key})
    loss = cross_entropy_loss(
        logits, labels, get_sequence_lengths(labels, eos_id))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, labels, eos_id)

  return state, metrics


def log_decode(question: str, inferred: str, golden: str):
  """Logs the given question, inferred query, and correct query."""
  suffix = '(CORRECT)' if inferred == golden else (f'(INCORRECT) '
                                                   f'correct={golden}')
  logging.info('DECODE: %s = %s %s', question, inferred, suffix)


@functools.partial(jax.jit, static_argnums=3)
def decode(params: Dict[str, Any], inputs: Array, decode_rng: PRNGKey,
           ctable: CTable) -> Array:
  """Decodes inputs."""
  init_decoder_input = ctable.one_hot(ctable.encode('=')[0:1])
  init_decoder_inputs = jnp.tile(init_decoder_input,
                                 (inputs.shape[0], ctable.max_output_len, 1))
  model = get_model(ctable, teacher_force=False)
  _, predictions = model.apply({'params': params},
                               inputs,
                               init_decoder_inputs,
                               rngs={'lstm': decode_rng})
  return predictions


def decode_batch(state: train_state.TrainState, batch: Dict[str, Array],
                 decode_rng: PRNGKey, ctable: CTable):
  """Decodes and log results for a batch."""
  inputs, outputs = batch['query'], batch['answer'][:, 1:]
  decode_rng = jax.random.fold_in(decode_rng, state.step)
  inferred = decode(state.params, inputs, decode_rng, ctable)
  questions = ctable.decode_onehot(inputs)
  infers = ctable.decode_onehot(inferred)
  goldens = ctable.decode_onehot(outputs)

  for question, inferred, golden in zip(questions, infers, goldens):
    log_decode(question, inferred, golden)


def train_and_evaluate(workdir: str) -> train_state.TrainState:
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

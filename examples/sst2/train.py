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

"""SST-2 example.

This script trains a text classifier on the SST-2 dataset.

A sentence is encoded with an LSTM and a binary prediction is made from
the final LSTM state using an MLP.

The data is loaded using tensorflow_datasets.

For more detailed information, see README.
"""

import collections
import functools
import os
from typing import Any, Dict, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import flax
from flax import nn
import flax.examples.sst2.input_pipeline as input_pipeline
import flax.examples.sst2.model as sst2_model

import jax
import jax.numpy as jnp

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.0003,
    help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=64,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=20,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'dropout', default=0.5,
    help=('Dropout rate'))

flags.DEFINE_integer(
    'seed', default=0,
    help=('Random seed for network initialization.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data'))

flags.DEFINE_string(
    'glove_path', default=None,
    help=('Path to GloVe 840B 300D word embeddings file.'))

flags.DEFINE_integer(
    'glove_dim', default=300,
    help=('Dimensionality of the GloVe embeddings.'))


@jax.vmap
def binary_cross_entropy_loss(logit, label):
  """Numerically stable binary cross entropy loss."""
  return label * nn.softplus(-logit) + (1 - label) * nn.softplus(logit)


@jax.jit
def train_step(optimizer, inputs, labels):
  def loss_fn(model):
    logits = model(inputs)
    loss = jnp.mean(binary_cross_entropy_loss(logits, labels))
    return loss, logits
  loss, _, grad = optimizer.compute_gradient(loss_fn)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def get_predictions(logits):
  outputs = jax.nn.sigmoid(logits)
  return (outputs > 0.5).astype(jnp.int32)


def get_num_correct(logits, labels):
  return jnp.sum(get_predictions(logits) == labels)


@jax.jit
def eval_step(model: nn.Module, inputs, labels):
  logits = model(inputs, train=False)
  loss = jnp.sum(binary_cross_entropy_loss(logits, labels))
  num_correct = get_num_correct(logits, labels)
  return loss, num_correct


def evaluate(model: nn.Module, eval_ds: tf.data.Dataset):
  """Evaluates the model on a dataset."""
  count = 0
  total_loss = 0.
  total_correct = 0

  for ex in tfds.as_numpy(eval_ds):
    inputs, labels = ex['sentence'], ex['label']
    count = count + inputs.shape[0]
    loss, num_correct = eval_step(model, inputs, labels)
    total_loss += loss
    total_correct += num_correct

  loss = total_loss / count
  accuracy = 100. * total_correct / count
  metrics = dict(loss=loss.item(), acc=accuracy.item())

  return metrics


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(rng, input_shape, embeddings, model_kwargs):
  """Instantiate a new model."""
  with nn.stochastic(rng):
    emb_init = functools.partial(sst2_model.pretrained_init,
                                 embeddings=embeddings)
    model_def = sst2_model.LSTMClassifier.partial(
        emb_init=emb_init, **model_kwargs)
    _, initial_params = model_def.init_by_shape(rng, [(input_shape, jnp.int32)])
    model = nn.Model(model_def, initial_params)
    return model


def log(stats, epoch, train_metrics, valid_metrics):
  """Logs performance for an epoch."""
  train_loss = train_metrics['loss'] / train_metrics['total']
  logging.info('Epoch %02d train loss %.4f valid loss %.4f acc %.2f',
               epoch + 1, train_loss,
               valid_metrics['loss'], valid_metrics['acc'])

  # Remember the metrics for later plotting.
  stats['train_loss'].append(train_loss.item())
  for metric, value in valid_metrics.items():
    stats['valid_' + metric].append(value)


def train(
    model,
    learning_rate: float = 0.0003,
    num_epochs: int = 20,
    seed: int = 0,
    model_dir: Text = None,
    data_source: Any = None,
) -> Tuple[Dict[Text, Any], nn.Model]:
  """Training loop."""
  rng = jax.random.PRNGKey(seed)
  with nn.stochastic(rng):
    optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)
    stats = collections.defaultdict(list)
    best_score = 0.

    for epoch in range(num_epochs):
      train_metrics = collections.defaultdict(float)

      # Train for one epoch.
      for ex in tfds.as_numpy(data_source.train_batches):
        inputs, labels = ex['sentence'], ex['label']
        optimizer, loss = train_step(optimizer, inputs, labels)
        train_metrics['loss'] += loss * inputs.shape[0]
        train_metrics['total'] += inputs.shape[0]

      # Evaluate on validation data. optimizer.target is the updated model.
      valid_metrics = evaluate(optimizer.target, data_source.valid_batches)
      log(stats, epoch, train_metrics, valid_metrics)

      # Save a checkpoint if this is the best model so far.
      if valid_metrics['acc'] > best_score:
        best_score = valid_metrics['acc']
        flax.training.checkpoints.save_checkpoint(
            model_dir, optimizer, epoch + 1)

    # Done training. Restore best model.
    logging.info('Training done! Best validation accuracy: %.2f', best_score)
    optimizer = flax.training.checkpoints.restore_checkpoint(
        model_dir, optimizer)

  return stats, optimizer.target


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert FLAGS.model_dir is not None, \
      'Please provide a model_dir to save checkpoints.'
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  tf.enable_v2_behavior()

  # Prepare data.
  sst2_data_source = input_pipeline.SST2DataSource(
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      glove_path=FLAGS.glove_path,
      glove_dim=FLAGS.glove_dim,
      shuffle_seed=FLAGS.seed)

  # Create model.
  model = create_model(
      jax.random.PRNGKey(FLAGS.seed), (FLAGS.batch_size, 1),
      sst2_data_source.pretrained_embeddings,
      dict(vocab_size=sst2_data_source.vocab_size,
           dropout=FLAGS.dropout))

  # Train the model.
  train_stats, best_model = train(
      model,
      learning_rate=FLAGS.learning_rate,
      num_epochs=FLAGS.num_epochs,
      seed=FLAGS.seed,
      model_dir=FLAGS.model_dir,
      data_source=sst2_data_source)

  # Let's evaluate again to make sure we really got the best model.
  rng = jax.random.PRNGKey(FLAGS.seed)
  with nn.stochastic(rng):
    metrics = evaluate(best_model, sst2_data_source.valid_batches)
    logging.info('Best validation accuracy: %.2f', metrics['acc'])


if __name__ == '__main__':
  app.run(main)


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

"""SST-2 example.

This script trains a text classifier on the SST-2 dataset.

A sentence is encoded with an LSTM and a binary prediction is made from
the final LSTM state using an MLP.

The data is loaded using tensorflow_datasets.

For more detailed information, see README.
"""
# pylint: disable=import-error,too-many-locals,too-many-arguments

import collections
from typing import Any, Dict, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import flax
import flax.training.checkpoints
from flax import nn

import jax
import jax.numpy as jnp

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.io import gfile

import input_pipeline
import model as sst2_model


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.0005,
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

flags.DEFINE_float(
    'emb_dropout', default=0.5,
    help=('Embedding dropout rate'))

flags.DEFINE_float(
    'word_dropout_rate', default=0.1,
    help=('Word dropout rate. Replaces input words with <unk>.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data'))

flags.DEFINE_integer(
    'hidden_size', default=256,
    help=('Hidden size for the LSTM and MLP.'))

flags.DEFINE_integer(
    'embedding_size', default=256,
    help=('Size of the word embeddings.'))

flags.DEFINE_integer(
    'max_seq_len', default=55,
    help=('Maximum sequence length in the dataset.'))

flags.DEFINE_integer(
    'min_freq', default=5,
    help=('Minimum frequency for training set words to be in the vocabulary.'))

flags.DEFINE_float(
    'l2_reg', default=1e-6,
    help=('L2 regularization weight'))

flags.DEFINE_integer(
    'seed', default=0,
    help=('Random seed for network initialization.'))

flags.DEFINE_integer(
    'checkpoints_to_keep', default=1,
    help=('How many checkpoints to keep. Default: 1 (keep best model only)'))


@jax.vmap
def binary_cross_entropy_loss(logit: jnp.ndarray, label: jnp.ndarray):
  """Numerically stable binary cross entropy loss.

  This function is vmapped, so it is written for a single example, but can
  handle a batch of examples.

  Args:
    logit: The output logits.
    label: The correct labels.

  Returns:
    The binary cross entropy loss for each given logit.
  """
  return label * nn.softplus(-logit) + (1 - label) * nn.softplus(logit)


@jax.jit
def train_step(optimizer: Any, inputs: jnp.ndarray, lengths: jnp.ndarray,
               labels: jnp.ndarray, rng: Any, l2_reg: float):
  """Single optimized training step.

  Args:
    optimizer: The optimizer to use to update the weights.
    inputs: A batch of inputs. <int64>[batch_size, seq_len]
    lengths: The lengths of the sequences in the batch. <int64>[batch_size]
    labels: The labels of the sequences in the batch. <int64>[batch_size, 1]
    rng: Random number generator for dropout.
    l2_reg: L2 regularization weight.

  Returns:
    optimizer: The optimizer in its new state.
    loss: The loss for this step.
  """
  rng, new_rng = jax.random.split(rng)
  def loss_fn(model):
    with nn.stochastic(rng):
      logits = model(inputs, lengths, train=True)
    loss = jnp.mean(binary_cross_entropy_loss(logits, labels))

    # L2 regularization
    l2_params = jax.tree_leaves(model.params['lstm_classifier'])
    l2_weight = jnp.sum([jnp.sum(p ** 2) for p in l2_params])
    l2_penalty = l2_reg * l2_weight

    loss = loss + l2_penalty
    return loss, logits

  (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, new_rng


def get_predictions(logits: jnp.ndarray) -> jnp.ndarray:
  """Returns predictions given a batch of logits."""
  outputs = jax.nn.sigmoid(logits)
  return (outputs > 0.5).astype(jnp.int32)


def get_num_correct(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  """Returns the number of correct predictions."""
  return jnp.sum(get_predictions(logits) == labels)


@jax.jit
def eval_step(model: nn.Module, inputs: jnp.ndarray, lengths: jnp.ndarray,
              labels: jnp.ndarray):
  """A single evaluation step.

  Args:
    model: The model to be used for this evaluation step.
    inputs: A batch of inputs. <int64>[batch_size, seq_len]
    lengths: The lengths of the sequences in the batch. <int64>[batch_size]
    labels: The labels of the sequences in the batch. <int64>[batch_size, 1]

  Returns:
    loss: The summed loss on this batch.
    num_correct: The number of correct predictions in this batch.
  """
  logits = model(inputs, lengths, train=False)
  loss = jnp.sum(binary_cross_entropy_loss(logits, labels))
  num_correct = get_num_correct(logits, labels)
  return loss, num_correct


def evaluate(model: nn.Model, dataset: tf.data.Dataset):
  """Evaluates the model on a dataset.

  Args:
    model: A model to be evaluated.
    dataset: A dataset to be used for the evaluation. Typically valid or test.

  Returns:
    A dict with the evaluation results.
  """
  count = 0
  total_loss = 0.
  total_correct = 0

  for ex in tfds.as_numpy(dataset):
    inputs, lengths, labels = ex['sentence'], ex['length'], ex['label']
    count = count + inputs.shape[0]
    loss, num_correct = eval_step(model, inputs, lengths, labels)
    total_loss += loss.item()
    total_correct += num_correct.item()

  loss = total_loss / count
  accuracy = 100. * total_correct / count
  metrics = dict(loss=loss, acc=accuracy)

  return metrics


def log(stats, epoch, train_metrics, valid_metrics):
  """Logs performance for an epoch.

  Args:
    stats: A dictionary to be updated with the logged statistics.
    epoch: The epoch number.
    train_metrics: A dict with the training metrics for this epoch.
    valid_metrics: A dict with the validation metrics for this epoch.
  """
  train_loss = train_metrics['loss'] / train_metrics['total']
  logging.info('Epoch %02d train loss %.4f valid loss %.4f acc %.2f', epoch + 1,
               train_loss, valid_metrics['loss'], valid_metrics['acc'])

  # Remember the metrics for later plotting.
  stats['train_loss'].append(train_loss.item())
  for metric, value in valid_metrics.items():
    stats['valid_' + metric].append(value)

def train(
    model: nn.Model,
    learning_rate: float = None,
    num_epochs: int = None,
    seed: int = None,
    model_dir: Text = None,
    data_source: Any = None,
    batch_size: int = None,
    checkpoints_to_keep: int = None,
    l2_reg: float = None,
) -> Tuple[Dict[Text, Any], nn.Model]:
  """Training loop.

  Args:
    model: An initialized model to be trained.
    learning_rate: The learning rate.
    num_epochs: Train for this many epochs.
    seed: Seed for shuffling.
    model_dir: Directory to save best model.
    data_source: The data source with pre-processed data examples.
    batch_size: The batch size to use for training and validation data.
    l2_reg: L2 regularization weight.

  Returns:
    A dict with training statistics and the best model.
  """
  rng = jax.random.PRNGKey(seed)
  optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)
  stats = collections.defaultdict(list)
  best_score = 0.
  train_batches = input_pipeline.get_shuffled_batches(
      data_source.train_dataset, batch_size=batch_size, seed=seed)
  valid_batches = input_pipeline.get_batches(
      data_source.valid_dataset, batch_size=batch_size)

  for epoch in range(num_epochs):
    train_metrics = collections.defaultdict(float)

    # Train for one epoch.
    for ex in tfds.as_numpy(train_batches):
      inputs, lengths, labels = ex['sentence'], ex['length'], ex['label']
      optimizer, loss, rng = train_step(optimizer, inputs, lengths, labels, rng,
                                        l2_reg)
      train_metrics['loss'] += loss * inputs.shape[0]
      train_metrics['total'] += inputs.shape[0]

    # Evaluate on validation data. optimizer.target is the updated model.
    valid_metrics = evaluate(optimizer.target, valid_batches)
    log(stats, epoch, train_metrics, valid_metrics)

    # Save a checkpoint if this is the best model so far.
    if valid_metrics['acc'] > best_score:
      best_score = valid_metrics['acc']
      flax.training.checkpoints.save_checkpoint(
          model_dir, optimizer.target, epoch + 1, keep=checkpoints_to_keep)

  # Done training. Restore best model.
  logging.info('Training done! Best validation accuracy: %.2f', best_score)
  best_model = flax.training.checkpoints.restore_checkpoint(model_dir, model)

  return stats, best_model


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert FLAGS.model_dir is not None, 'Please provide model_dir.'
  if not gfile.exists(FLAGS.model_dir):
    gfile.makedirs(FLAGS.model_dir)

  tf.enable_v2_behavior()

  # Prepare data.
  data_source = input_pipeline.SST2DataSource(min_freq=FLAGS.min_freq)

  # Create model.
  model = sst2_model.create_model(
      FLAGS.seed,
      FLAGS.batch_size,
      FLAGS.max_seq_len,
      dict(
          vocab_size=data_source.vocab_size,
          embedding_size=FLAGS.embedding_size,
          hidden_size=FLAGS.hidden_size,
          output_size=1,
          unk_idx=data_source.unk_idx,
          dropout=FLAGS.dropout,
          emb_dropout=FLAGS.emb_dropout,
          word_dropout_rate=FLAGS.word_dropout_rate))

  # Train the model.
  train_stats, model = train(
      model,
      learning_rate=FLAGS.learning_rate,
      num_epochs=FLAGS.num_epochs,
      seed=FLAGS.seed,
      model_dir=FLAGS.model_dir,
      data_source=data_source,
      batch_size=FLAGS.batch_size,
      checkpoints_to_keep=FLAGS.checkpoints_to_keep,
      l2_reg=FLAGS.l2_reg)

  # Evaluate the best model.
  valid_batches = input_pipeline.get_batches(
      data_source.valid_dataset, batch_size=FLAGS.batch_size)
  metrics = evaluate(model, valid_batches)
  logging.info('Best validation accuracy: %.2f', metrics['acc'])


if __name__ == '__main__':
  app.run(main)

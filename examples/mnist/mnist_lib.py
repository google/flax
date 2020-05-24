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

"""Library for training an MNIST model."""

import functools

from absl import logging

import jax
from jax import numpy as jnp

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard


class CNN(nn.Module):
  """A simple CNN model."""

  # pylint: disable=arguments-differ
  def apply(self, x, num_classes):
    x = nn.Conv(x, features=32, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    x = nn.Dense(x, features=num_classes)
    x = nn.log_softmax(x)
    return x


def create_model(key, num_classes=10):
  model_def = CNN.partial(num_classes=num_classes)
  _, initial_params = model_def.init_by_shape(
      key, [((1, 28, 28, 1), jnp.float32)])
  model = nn.Model(CNN, initial_params)
  return model


def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(model)
  return optimizer


def mnist_preprocess_fn(features):
  features['image'] = tf.cast(features['image'], tf.float32) / 255.0
  return features


def get_split(split, num_examples):
  host_id = jax.host_id()
  host_count = jax.host_count()

  examples_per_host = num_examples // host_count
  num_unused_examples = num_examples - examples_per_host * host_count
  if num_unused_examples > 0:
    logging.warning('Discarding %d examples of %d examples (host count: %d).',
                    num_unused_examples, num_examples, host_count)
  return tfds.core.ReadInstruction(
      split_name=split, from_=examples_per_host * host_id,
      to=examples_per_host * (host_id + 1), unit='abs')


def preprocess_and_get_dataset(dataset_builder, split, train, batch_size):
  """Returns MNIST train dataset."""
  shuffle = False
  if train:
    shuffle = True
  ds = dataset_builder.as_dataset(split=split, shuffle_files=shuffle)
  ds = ds.repeat()
  if train:
    ds = ds.shuffle(16 * batch_size)
  ds = ds.map(
      mnist_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def get_datasets(dataset_builder, local_batch_size):
  """Load MNIST train and test datasets into memory."""
  train_split = get_split(
      'train', dataset_builder.info.splits['train'].num_examples)
  train_ds = preprocess_and_get_dataset(
      dataset_builder, split=train_split, train=True,
      batch_size=local_batch_size)

  eval_split = get_split(
      'test', dataset_builder.info.splits['test'].num_examples)
  eval_ds = preprocess_and_get_dataset(
      dataset_builder, split=eval_split, train=False,
      batch_size=local_batch_size)

  return train_ds, eval_ds


def onehot(labels, num_classes=10):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def prepare_train_iter(train_ds):
  """Prepare train iterator."""
  train_ds = iter(train_ds)
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((jax.local_device_count(), -1) + x.shape[1:])

  train_ds = map(jax.tree_map(_prepare, train_ds), train_ds)
  return jax_utils.prefetch_to_device(train_ds, size=2)


@functools.partial(jax.pmap, axis_name='batch')
def train_step(optimizer, batch):
  """Perform a single training step for the given batch."""

  def loss_fn(model):
    logits = model(batch['image'])
    loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch['label']))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)

  # Normalize across workers.
  grad = jax.lax.pmean(grad, axis_name='batch')

  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, labels=batch['label'])
  return optimizer, metrics


@functools.partial(jax.pmap, axis_name='batch')
def eval_step(model, batch):
  logits = model(batch['image'])
  metrics = compute_metrics(logits, batch['label'])
  metrics = jax.lax.pmean(metrics, axis_name='batch')
  return metrics['loss'], metrics['accuracy']


def train_and_evaluate(model_dir: str, batch_size: int, num_epochs: int,
                       learning_rate: float, momentum: float):
  """Runs a training and an evaluation loop.

  Args:
    model_dir:
    batch_size:
    num_epochs:
    learning_rate:
    momentum:
  """
  if batch_size % jax.host_count() != 0:
    raise ValueError(f"Batch size: {batch_size} not divisible by"
                     f"host count: {jax.host_count()}")

  local_batch_size = batch_size // jax.host_count()
  ds_builder = tfds.builder('mnist')

  # Create a different RNG for each host for data training.
  rng = jax.random.PRNGKey(0)
  rng, data_rng = jax.random.split(rng)

  dataset_builder = tfds.builder('mnist')
  num_classes = dataset_builder.info.features['label'].num_classes

  # Make sure every host uses a different RNG for training data.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.host_id())
  train_ds, eval_ds = get_datasets(dataset_builder, local_batch_size)

  rng, model_rng = jax.random.split(rng)
  model = create_model(model_rng, num_classes)
  optimizer = create_optimizer(
      model=model, learning_rate=learning_rate, beta=momentum)
  # Replicate parameters.
  optimizer = jax_utils.replicate(optimizer)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(model_dir)

  train_iter = prepare_train_iter(train_ds=train_ds)
  for step in range(1, num_epochs + 1):
    optimizer, metrics = train_step(optimizer, next(train_iter))
    loss, accuracy = eval_step(optimizer.target, eval_ds)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 step, loss, accuracy * 100)
    # log summaries.
    if jax.host_id() == 0:
      summary_writer.scalar('train_loss', metrics['loss'], step)
      summary_writer.scalar('train_accuracy', metrics['accuracy'], step)
      summary_writer.scalar('eval_loss', loss, step)
      summary_writer.scalar('eval_accuracy', accuracy, step)

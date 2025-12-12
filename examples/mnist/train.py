# Copyright 2024 The Flax Authors.
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

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args
from functools import partial
from typing import Any
from pathlib import Path

from absl import logging
from flax import nnx
from flax.metrics import tensorboard
import jax
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(0)  # Set the random seed for reproducibility.


class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
    self.dropout1 = nnx.Dropout(rate=0.025)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.dropout2 = nnx.Dropout(rate=0.025)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x, rngs: nnx.Rngs):
    x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
    x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
    x = self.linear2(x)
    return x



def loss_fn(model: CNN, batch, rngs):
  logits = model(batch['image'], rngs)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, rngs):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch, rngs)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.




@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch, None)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


def get_datasets(
    config: ml_collections.ConfigDict,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  """Load MNIST train and test datasets into memory."""
  batch_size = config.batch_size
  train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
  test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

  train_ds = train_ds.map(
      lambda sample: {
          'image': tf.cast(sample['image'], tf.float32) / 255,
          'label': sample['label'],
      }
  )  # normalize train set
  test_ds = test_ds.map(
      lambda sample: {
          'image': tf.cast(sample['image'], tf.float32) / 255,
          'label': sample['label'],
      }
  )  # normalize the test set.

  # Create a shuffled dataset by allocating a buffer size of 1024 to randomly
  # draw elements from.
  train_ds = train_ds.shuffle(1024)
  # Group into batches of `batch_size` and skip incomplete batches, prefetch the
  # next sample to improve latency.
  train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
  # Group into batches of `batch_size` and skip incomplete batches, prefetch the
  # next sample to improve latency.
  test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

  return train_ds, test_ds


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> None:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory path to store metrics.
  """
  train_ds, test_ds = get_datasets(config)

  # Instantiate the model.
  model = CNN(rngs=nnx.Rngs(0))

  learning_rate = config.learning_rate
  momentum = config.momentum

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  optimizer = nnx.Optimizer(
      model, optax.sgd(learning_rate, momentum), wrt=nnx.Param
  )
  metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(),
      loss=nnx.metrics.Average('loss'),
  )
  rngs = nnx.Rngs(0)

  for epoch in range(1, config.num_epochs + 1):
    # Run the optimization for one step and make a stateful update to the
    # following:
    # - The train state's model parameters
    # - The optimizer state
    # - The training loss and accuracy batch metrics
    model.train()  # Switch to train mode

    for batch in train_ds.as_numpy_iterator():
      train_step(model, optimizer, metrics, batch, rngs)
    #  Compute the training metrics.
    train_metrics = metrics.compute()
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    model.eval()  # Switch to eval mode
    for batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, batch)

    # Compute the eval metrics.
    eval_metrics = metrics.compute()
    metrics.reset()  # Reset the metrics for the next training epoch.

    logging.info(  # pylint: disable=logging-not-lazy
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
        ' test_accuracy: %.2f'
        % (
            epoch,
            train_metrics['loss'],
            train_metrics['accuracy'] * 100,
            eval_metrics['loss'],
            eval_metrics['accuracy'] * 100,
        )
    )

    # Write the metrics to TensorBoard.
    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
    summary_writer.scalar('test_loss', eval_metrics['loss'], epoch)
    summary_writer.scalar('test_accuracy', eval_metrics['accuracy'], epoch)

  summary_writer.flush()

  # Export the model to a SavedModel directory.
  from orbax.export import JaxModule, ExportManager, ServingConfig

  def exported_predict(model, y):
      return model(y, None)

  model.eval()
  jax_module = JaxModule(model, exported_predict)
  sig = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32)]
  export_mgr = ExportManager(jax_module, [
      ServingConfig('mnist_server', input_signature=sig)
  ])

  output_dir= Path(workdir) / 'mnist_export'
  export_mgr.save(str(output_dir))

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
"""PixelCNN++ example."""

from functools import partial
import datetime

from absl import app
from absl import flags
from absl import logging

import numpy as np

from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import random
from jax import lax
import jax.numpy as jnp
from jax.config import config

import tensorflow as tf

import input_pipeline
import pixelcnn

config.enable_omnistaging()

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.001, help=('The initial learning rate.'))

flags.DEFINE_float(
    'lr_decay',
    default='0.999995',
    help=('Learning rate decay, applied each optimization step.'))

flags.DEFINE_integer(
    'init_batch_size',
    default=16,
    help=('Batch size to use for data-dependent initialization.'))

flags.DEFINE_integer(
    'batch_size', default=64, help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=200, help=('Number of training epochs.'))

flags.DEFINE_float('dropout_rate', default=0.5, help=('DropOut rate.'))

flags.DEFINE_integer(
    'rng', default=0, help=('Random seed for network initialization.'))

flags.DEFINE_integer(
    'n_resnet', default=5, help=('Number of resnet layers per block.'))

flags.DEFINE_integer(
    'n_feature', default=160, help=('Number of features in each conv layer.'))

flags.DEFINE_integer(
    'n_logistic_mix',
    default=10,
    help=('Number of components in the output distribution.'))

flags.DEFINE_string(
    'model_dir',
    default='./model_data',
    help=('Directory to store model data.'))

flags.DEFINE_float(
    'polyak_decay',
    default=0.9995,
    help=('Exponential decay rate of the sum of previous model iterates '
          'during Polyak averaging.'))


def get_summary_writers():
  current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  log_dir = FLAGS.model_dir + '/log/' + current_time
  train_log_dir = log_dir + '/train'
  eval_log_dir = log_dir + '/eval'
  train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
  eval_summary_writer = tensorboard.SummaryWriter(eval_log_dir)
  return train_summary_writer, eval_summary_writer


def model(**kwargs):
  return pixelcnn.PixelCNNPP(
      depth=FLAGS.n_resnet, features=FLAGS.n_feature,
      logistic_components=FLAGS.n_logistic_mix, **kwargs)


def neg_log_likelihood_loss(nn_out, images):
  # The log-likelihood in bits per pixel-channel
  means, inv_scales, logit_weights = (
      pixelcnn.conditional_params_from_outputs(nn_out, images))
  log_likelihoods = pixelcnn.logprob_from_conditional_params(
      images, means, inv_scales, logit_weights)
  return -jnp.mean(log_likelihoods) / (jnp.log(2) * np.prod(images.shape[-3:]))


def train_step(optimizer, ema, batch, learning_rate_fn, dropout_rng):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    pcnn_out = model(dropout_p=FLAGS.dropout_rate).apply(
        {'params': params}, batch['image'], rngs={'dropout': dropout_rng})
    return neg_log_likelihood_loss(pcnn_out, batch['image'])

  lr = learning_rate_fn(optimizer.state.step)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(optimizer.target)
  grad = lax.pmean(grad, 'batch')
  optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Compute exponential moving average (aka Polyak decay)
  ema_decay = FLAGS.polyak_decay
  ema = jax.tree_multimap(lambda ema, p: ema * ema_decay + (1 - ema_decay) * p,
                          ema, optimizer.target)

  metrics = {'loss': lax.pmean(loss, 'batch'), 'learning_rate': lr}
  return optimizer, ema, metrics


def eval_step(params, batch):
  images = batch['image']
  pcnn_out = model(dropout_p=0.).apply({'params': params}, images)
  return {'loss': lax.pmean(neg_log_likelihood_loss(pcnn_out, images), 'batch')}


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def restore_checkpoint(optimizer, ema):
  return checkpoints.restore_checkpoint(FLAGS.model_dir, (optimizer, ema))


def save_checkpoint(optimizer, ema, step):
  optimizer, ema = jax_utils.unreplicate((optimizer, ema))
  checkpoints.save_checkpoint(FLAGS.model_dir, (optimizer, ema), step, keep=3)


def train():
  """Train model."""
  batch_size = FLAGS.batch_size
  n_devices = jax.device_count()
  if jax.host_count() > 1:
    raise ValueError('PixelCNN++ example should not be run on more than 1 host'
                     ' (for now)')
  if batch_size % n_devices > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  train_summary_writer, eval_summary_writer = get_summary_writers()

  # Load dataset
  data_source = input_pipeline.DataSource(
      train_batch_size=batch_size, eval_batch_size=batch_size)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds

  # Create dataset batch iterators
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # Compute steps per epoch and nb of eval steps
  steps_per_epoch = data_source.TRAIN_IMAGES // batch_size
  steps_per_eval = data_source.EVAL_IMAGES // batch_size
  steps_per_checkpoint = steps_per_epoch * 10
  num_steps = steps_per_epoch * FLAGS.num_epochs

  # Create the model using data-dependent initialization. Don't shard the init
  # batch.
  assert FLAGS.init_batch_size <= batch_size
  init_batch = next(train_iter)['image']._numpy()[:FLAGS.init_batch_size]

  rng = random.PRNGKey(FLAGS.rng)
  rng, init_rng = random.split(rng)
  rng, dropout_rng = random.split(rng)

  initial_variables = model().init({
      'params': init_rng,
      'dropout': dropout_rng
  }, init_batch)['params']
  optimizer_def = optim.Adam(
      learning_rate=FLAGS.learning_rate, beta1=0.95, beta2=0.9995)
  optimizer = optimizer_def.create(initial_variables)

  optimizer, ema = restore_checkpoint(optimizer, initial_variables)
  ema = initial_variables
  step_offset = int(optimizer.state.step)

  optimizer, ema = jax_utils.replicate((optimizer, ema))

  # Learning rate schedule
  learning_rate_fn = lambda step: FLAGS.learning_rate * FLAGS.lr_decay ** step

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      partial(train_step, learning_rate_fn=learning_rate_fn), axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Gather metrics
  train_metrics = []

  for step, batch in zip(range(step_offset, num_steps), train_iter):
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)

    # Generate a PRNG key that will be rolled into the batch.
    rng, step_rng = random.split(rng)
    sharded_rngs = common_utils.shard_prng_key(step_rng)

    # Train step
    optimizer, ema, metrics = p_train_step(optimizer, ema, batch, sharded_rngs)
    train_metrics.append(metrics)

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      # We've finished an epoch
      train_metrics = common_utils.get_metrics(train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      # Send stats to Tensorboard
      for key, vals in train_metrics.items():
        for i, val in enumerate(vals):
          train_summary_writer.scalar(key, val, step - len(vals) + i + 1)
      # Reset train metrics
      train_metrics = []

      # Evaluation
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # Load and shard the TF batch
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Step
        metrics = p_eval_step(ema, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # Get eval epoch summary for logging
      eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

      # Log epoch summary
      logging.info('Epoch %d: TRAIN loss=%.6f, EVAL loss=%.6f', epoch,
                   train_summary['loss'], eval_summary['loss'])

      eval_summary_writer.scalar('loss', eval_summary['loss'], step)
      train_summary_writer.flush()
      eval_summary_writer.flush()

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      save_checkpoint(optimizer, ema, step)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  train()


if __name__ == '__main__':
  app.run(main)

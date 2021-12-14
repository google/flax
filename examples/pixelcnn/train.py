# Copyright 2021 The Flax Authors.
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
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PixelCNN++ example."""

# See issue #620.
# pytype: disable=wrong-keyword-args

import functools
import datetime

from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

import input_pipeline
import pixelcnn


def get_summary_writers(workdir):
  current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  log_dir = workdir + '/log/' + current_time
  train_log_dir = log_dir + '/train'
  eval_log_dir = log_dir + '/eval'
  train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
  eval_summary_writer = tensorboard.SummaryWriter(eval_log_dir)
  return train_summary_writer, eval_summary_writer


def model(config: ml_collections.ConfigDict, **kwargs):
  return pixelcnn.PixelCNNPP(
      depth=config.n_resnet,
      features=config.n_feature,
      logistic_components=config.n_logistic_mix,
      **kwargs)


def neg_log_likelihood_loss(nn_out, images):
  # The log-likelihood in bits per pixel-channel
  means, inv_scales, logit_weights = (
      pixelcnn.conditional_params_from_outputs(nn_out, images))
  log_likelihoods = pixelcnn.logprob_from_conditional_params(
      images, means, inv_scales, logit_weights)
  return -jnp.mean(log_likelihoods) / (jnp.log(2) * np.prod(images.shape[-3:]))


def train_step(config: ml_collections.ConfigDict, learning_rate_fn, optimizer,
               ema, batch, dropout_rng):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    pcnn_out = model(
        config,
        dropout_p=config.dropout_rate).apply({'params': params},
                                             batch['image'],
                                             rngs={'dropout': dropout_rng},
                                             train=True)
    return neg_log_likelihood_loss(pcnn_out, batch['image'])

  lr = learning_rate_fn(optimizer.state.step)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Compute exponential moving average (aka Polyak decay)
  ema_decay = config.polyak_decay
  ema = jax.tree_multimap(lambda ema, p: ema * ema_decay + (1 - ema_decay) * p,
                          ema, optimizer.target)

  metrics = {'loss': jax.lax.pmean(loss, 'batch'), 'learning_rate': lr}
  return optimizer, ema, metrics


def eval_step(config, params, batch):
  images = batch['image']
  pcnn_out = model(config).apply({'params': params}, images, train=False)
  return {
      'loss': jax.lax.pmean(neg_log_likelihood_loss(pcnn_out, images), 'batch')
  }


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def restore_checkpoint(workdir: str, optimizer, ema):
  return checkpoints.restore_checkpoint(workdir, (optimizer, ema))


def save_checkpoint(workdir: str, optimizer, ema, step):
  optimizer, ema = jax_utils.unreplicate((optimizer, ema))
  checkpoints.save_checkpoint(workdir, (optimizer, ema), step, keep=3)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)

  batch_size = config.batch_size
  n_devices = jax.device_count()
  if jax.process_count() > 1:
    raise ValueError('PixelCNN++ example should not be run on more than 1 host'
                     ' (for now)')
  if batch_size % n_devices > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  train_summary_writer, eval_summary_writer = get_summary_writers(workdir)
  # Load dataset
  data_source = input_pipeline.DataSource(config)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds
  steps_per_epoch = data_source.ds_info.splits[
      'train'].num_examples // config.batch_size
  # Create dataset batch iterators
  train_iter = iter(train_ds)
  num_train_steps = train_ds.cardinality().numpy()
  steps_per_checkpoint = 1000

  # Create the model using data-dependent initialization. Don't shard the init
  # batch.
  assert config.init_batch_size <= batch_size
  init_batch = next(train_iter)['image']._numpy()[:config.init_batch_size]

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng, dropout_rng = jax.random.split(rng, 3)

  initial_variables = model(config).init(
      {
          'params': init_rng,
          'dropout': dropout_rng
      }, init_batch, train=False)['params']
  optimizer_def = optim.Adam(beta1=0.95, beta2=0.9995)
  optimizer = optimizer_def.create(initial_variables)

  optimizer, ema = restore_checkpoint(workdir, optimizer, initial_variables)
  ema = initial_variables
  step_offset = int(optimizer.state.step)

  optimizer, ema = jax_utils.replicate((optimizer, ema))

  # Learning rate schedule
  learning_rate_fn = lambda step: config.learning_rate * config.lr_decay**step

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(train_step, config, learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config), axis_name='batch')

  # Gather metrics
  train_metrics = []

  for step, batch in zip(range(step_offset, num_train_steps), train_iter):
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)

    # Generate a PRNG key that will be rolled into the batch.
    rng, step_rng = jax.random.split(rng)
    sharded_rngs = common_utils.shard_prng_key(step_rng)

    # Train step
    optimizer, ema, metrics = p_train_step(optimizer, ema, batch, sharded_rngs)
    train_metrics.append(metrics)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)

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
      for eval_batch in eval_ds:
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

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_train_steps:
      save_checkpoint(workdir, optimizer, ema, step)

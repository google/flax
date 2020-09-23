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

import functools
import datetime
import os

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
import flax.nn
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import random
from jax import lax
import jax.numpy as jnp

import ml_collections
from ml_collections import config_flags

import numpy as np

import tensorflow as tf

import input_pipeline
import pixelcnn


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'configs/default.py'),
    'File path to the Training hyperparameter configuration.')

flags.DEFINE_string(
    'model_dir', default='./model_data',
    help=('Directory to store model data.'))


def create_model(prng_key, example_images, module):
  with flax.nn.stochastic(jax.random.PRNGKey(0)):
    _, initial_params = module.init(prng_key, example_images)
    model = flax.nn.Model(module, initial_params)
  return model


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(
      learning_rate=learning_rate, beta1=0.95, beta2=0.9995)
  optimizer = optimizer_def.create(model)
  return optimizer


def neg_log_likelihood_loss(nn_out, images):
  # The log-likelihood in bits per pixel-channel
  means, inv_scales, logit_weights = (
      pixelcnn.conditional_params_from_outputs(nn_out, images))
  log_likelihoods = pixelcnn.logprob_from_conditional_params(
      images, means, inv_scales, logit_weights)
  return -jnp.mean(log_likelihoods) / (jnp.log(2) * np.prod(images.shape[-3:]))


def train_step(optimizer, ema, batch, prng_key, learning_rate_fn,
               dropout_rate, polyak_decay):
  """Perform a single training step."""
  def loss_fn(model):
    """loss function used for training."""
    with flax.nn.stochastic(prng_key):
      nn_out = model(batch['image'], dropout_p=dropout_rate)
    return neg_log_likelihood_loss(nn_out, batch['image'])

  lr = learning_rate_fn(optimizer.state.step)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(optimizer.target)
  grad = lax.pmean(grad, 'batch')
  optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Compute exponential moving average (aka Polyak decay)
  ema = jax.tree_multimap(
      lambda ema, p: ema * polyak_decay + (1 - polyak_decay) * p,
      ema, optimizer.target.params)

  metrics = {'loss': lax.pmean(loss, 'batch'), 'learning_rate': lr}
  return optimizer, ema, metrics


def eval_step(model, batch):
  images = batch['image']
  nn_out = model(images, dropout_p=0)
  return {'loss': lax.pmean(neg_log_likelihood_loss(nn_out, images), 'batch')}


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])
  return jax.tree_map(_prepare, xs)


def restore_checkpoint(model_dir, optimizer, ema):
  return checkpoints.restore_checkpoint(model_dir, (optimizer, ema))


def save_checkpoint(model_dir, optimizer, ema):
  # get train state from the first replica
  optimizer, ema = jax.device_get(
      jax.tree_map(lambda x: x[0], (optimizer, ema)))
  step = int(optimizer.state.step)
  checkpoints.save_checkpoint(model_dir, (optimizer, ema), step, keep=3)


def train_and_evaluate(config: ml_collections.ConfigDict, model_dir: str):
  """Executes model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    model_dir: Directory to store model data.
  """
  if jax.host_count() > 1:
    raise ValueError('PixelCNN++ example should not be run on more than 1 host'
                     ' (for now)')

  pcnn_module = pixelcnn.PixelCNNPP.partial(depth=config.n_resnet,
                                            features=config.n_feature,
                                            k=config.n_logistic_mix)

  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = model_dir + '/log/' + current_time
  train_log_dir = log_dir + '/train'
  eval_log_dir = log_dir + '/eval'
  train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
  eval_summary_writer = tensorboard.SummaryWriter(eval_log_dir)

  rng = random.PRNGKey(config.random_seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  # Load dataset
  data_source = input_pipeline.DataSource(
      train_batch_size=config.batch_size, eval_batch_size=config.batch_size)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds

  # Create dataset batch iterators
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  steps_per_epoch = (
    data_source.info.splits['train'].num_examples // config.batch_size
  )

  # Compute steps per epoch and nb of eval steps
  if config.num_train_steps == -1:
    num_steps = steps_per_epoch * config.num_epochs
  else:
    num_steps = config.num_train_steps

  if config.num_eval_steps == -1:
    steps_per_eval = (
      data_source.info.splits['test'].num_examples // config.batch_size
    )
  else:
    steps_per_eval = config.num_eval_steps

  steps_per_checkpoint = steps_per_epoch * 10

  base_learning_rate = config.learning_rate

  # Create the model using data-dependent initialization. Don't shard the init
  # batch.
  assert config.init_batch_size <= config.batch_size
  init_batch = next(train_iter)['image']._numpy()[:config.init_batch_size]
  model = create_model(rng, init_batch, pcnn_module)
  ema = model.params
  optimizer = create_optimizer(model, base_learning_rate)
  del model  # don't keep a copy of the initial model

  optimizer, ema = restore_checkpoint(model_dir, optimizer, ema)
  step_offset = int(optimizer.state.step)
  optimizer, ema = jax_utils.replicate((optimizer, ema))

  # Learning rate schedule
  learning_rate_fn = lambda step: base_learning_rate * config.lr_decay ** step

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(
        train_step, learning_rate_fn=learning_rate_fn,
        dropout_rate=config.dropout_rate, polyak_decay=config.polyak_decay),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Gather metrics
  train_metrics = []
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    # Generate a PRNG key that will be rolled into the batch
    rng, step_key = jax.random.split(rng)
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)

    # Train step
    optimizer, ema, metrics = p_train_step(optimizer, ema, batch, sharded_keys)
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
      model_ema = optimizer.target.replace(params=ema)
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # Load and shard the TF batch
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Step
        metrics = p_eval_step(model_ema, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # Get eval epoch summary for logging
      eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

      # Log epoch summary
      logging.info(
          'Epoch %d: TRAIN loss=%.6f, EVAL loss=%.6f',
          epoch, train_summary['loss'], eval_summary['loss'])

      eval_summary_writer.scalar('loss', eval_summary['loss'], step)
      train_summary_writer.flush()
      eval_summary_writer.flush()

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      save_checkpoint(model_dir, optimizer, ema)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_and_evaluate(config=FLAGS.config, model_dir=FLAGS.model_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'model_dir'])

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  app.run(main)

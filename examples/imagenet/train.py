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

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import nn
from flax import optim
from flax.examples.imagenet import input_pipeline
from flax.examples.imagenet import models
from flax.examples.utils import common_utils
from flax.metrics import tensorboard

import jax
from jax import random

import jax.nn
import jax.numpy as jnp

import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=90,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data'))

flags.DEFINE_bool(
    'use_bfloat16', default=False,
    help=('If bfloat16 should be used instead of float32.'))


def create_model(key, batch_size, image_size, model_dtype):
  input_shape = (batch_size, image_size, image_size, 3)
  model_def = models.ResNet.partial(num_classes=1000, dtype=model_dtype)
  with nn.stateful() as init_state:
    _, model = model_def.create_by_shape(key, [(input_shape, model_dtype)])
  return model, init_state


def create_optimizer(model, beta):
  optimizer_def = optim.Momentum(beta=beta,
                                 nesterov=True)
  optimizer = optimizer_def.create(model)
  optimizer = optimizer.replicate()
  return optimizer


def cross_entropy_loss(logits, labels):
  return -jnp.sum(
      common_utils.onehot(labels, num_classes=1000) * logits) / labels.size


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = common_utils.pmean(metrics)
  return metrics


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(base_learing_rate, steps_per_epoch, num_epochs):
  warmup_epochs = 5
  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = cosine_decay(base_learing_rate,
                      epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
    warmup = jnp.minimum(1., epoch / warmup_epochs)
    return lr * warmup
  return step_fn


def train_step(optimizer, state, batch, learning_rate_fn):
  """Perform a single training step."""
  def loss_fn(model):
    """loss function used for training."""
    with nn.stateful(state) as new_state:
      logits = model(batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_leaves(model.params)
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_state, logits)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  new_optimizer, _, (new_state, logits) = optimizer.optimize(
      loss_fn, learning_rate=lr)
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr
  return new_optimizer, new_state, metrics


def eval_step(model, state, batch):
  state = common_utils.pmean(state)
  with nn.stateful(state, mutable=False):
    logits = model(batch['image'], train=False)
  return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()
  tf.config.experimental.set_visible_devices([], 'GPU')

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.model_dir)

  rng = random.PRNGKey(0)

  image_size = 224

  batch_size = FLAGS.batch_size
  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // jax.host_count()
  device_batch_size = batch_size // jax.device_count()

  model_dtype = jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32
  input_dtype = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32

  train_ds = input_pipeline.load_split(local_batch_size,
                                       image_size=image_size,
                                       dtype=input_dtype,
                                       train=True)
  eval_ds = input_pipeline.load_split(local_batch_size,
                                      image_size=image_size,
                                      dtype=input_dtype,
                                      train=False)

  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  num_epochs = FLAGS.num_epochs
  steps_per_epoch = input_pipeline.TRAIN_IMAGES // batch_size
  steps_per_eval = input_pipeline.EVAL_IMAGES // batch_size
  num_steps = steps_per_epoch * num_epochs

  base_learning_rate = FLAGS.learning_rate * batch_size / 256.

  model, state = create_model(rng, device_batch_size, image_size, model_dtype)
  state = jax_utils.replicate(state)
  optimizer = create_optimizer(model, FLAGS.momentum)
  del model  # do not keep a copy of the initial model
  learning_rate_fn = create_learning_rate_fn(
      base_learning_rate, steps_per_epoch, num_epochs)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  epoch_metrics = []
  epoch = 1
  for step, batch in zip(range(num_steps), train_iter):
    batch = prepare_tf_data(batch)
    optimizer, state, metrics = p_train_step(optimizer, state, batch)
    epoch_metrics.append(metrics)
    if (step + 1) % steps_per_epoch == 0:
      epoch_metrics = common_utils.get_metrics(epoch_metrics)
      summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
      logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      if jax.host_id() == 0:
        for key, vals in epoch_metrics.items():
          tag = 'train_%s' % key
          for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

      epoch_metrics = []
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        eval_batch = prepare_tf_data(eval_batch)
        metrics = p_eval_step(optimizer.target, state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      if jax.host_id() == 0:
        for key, val in eval_metrics.items():
          tag = 'eval_%s' % key
          summary_writer.scalar(tag, val.mean(), step)
        summary_writer.flush()

      epoch += 1


if __name__ == '__main__':
  app.run(main)

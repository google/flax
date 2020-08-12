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

# Lint as: python3
"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time

from absl import app
from absl import flags
from absl import logging

import flax
from flax import jax_utils
from flax import nn
from flax import optim
import input_pipeline
import models
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import lax
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

flags.DEFINE_bool(
    'cache', default=False,
    help=('If True, cache the dataset.'))

flags.DEFINE_integer(
    'num_epochs', default=90,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))

flags.DEFINE_bool(
    'half_precision', default=False,
    help=('If bfloat16/float16 should be used instead of float32.'))


def model(**kwargs):
  # TODO: Is this slow?
  platform = jax.local_devices()[0].platform
  if FLAGS.half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return models.ResNet(num_classes=1000, dtype=model_dtype, **kwargs)

def initialized(key, image_size):
  input_shape = (1, image_size, image_size, 3)
  model_ = model()
  return model_.init({"param": key}, jnp.ones(input_shape, model_.dtype))


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
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs):
  warmup_epochs = 5
  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = cosine_decay(base_learning_rate,
                      epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
    warmup = jnp.minimum(1., epoch / warmup_epochs)
    return lr * warmup
  return step_fn


def train_step(state, batch, learning_rate_fn):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    variables = {'param': params, 'batch_stats': state.batch_stats}
    logits, new_variables = model().apply(variables, batch['image'], mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_leaves(variables['param'])
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_variables, logits)

  step = state.step
  optimizer = state.optimizer
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grad = grad_fn(optimizer.target)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(optimizer.target)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = lax.pmean(grad, axis_name='batch')
  new_variables, logits = aux[1]
  new_batch_stats = new_variables['batch_stats']
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and the old optimizer
    # state should be restored.
    new_optimizer = jax.tree_multimap(
        functools.partial(jnp.where, is_fin), new_optimizer, optimizer)
    metrics['scale'] = dynamic_scale.scale

  new_state = state.replace(
      step=step + 1, optimizer=new_optimizer, batch_stats=new_batch_stats,
      dynamic_scale=dynamic_scale)
  return new_state, metrics


def eval_step(state, batch):
  params = state.optimizer.target
  variables = {'param': params, 'batch_stats': state.batch_stats}
  logits = model(train=False).apply(variables, batch['image'], mutable=False)
  return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(batch_size, image_size, dtype, train, cache):
  ds = input_pipeline.load_split(
      batch_size, image_size=image_size, dtype=dtype, train=train, cache=cache)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  batch_stats: dict
  dynamic_scale: optim.DynamicScale


def restore_checkpoint(state):
  return checkpoints.restore_checkpoint(FLAGS.model_dir, state)


def save_checkpoint(state):
  if jax.host_id() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(FLAGS.model_dir, state, step, keep=3)


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
  return state.replace(batch_stats=avg(state.batch_stats))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
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

  platform = jax.local_devices()[0].platform

  dynamic_scale = None
  if FLAGS.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
      dynamic_scale = optim.DynamicScale()
  else:
    input_dtype = tf.float32

  train_iter = create_input_iter(
      local_batch_size, image_size, input_dtype, train=True, cache=FLAGS.cache)
  eval_iter = create_input_iter(
      local_batch_size, image_size, input_dtype, train=False, cache=FLAGS.cache)

  num_epochs = FLAGS.num_epochs
  steps_per_epoch = input_pipeline.TRAIN_IMAGES // batch_size
  steps_per_eval = input_pipeline.EVAL_IMAGES // batch_size
  steps_per_checkpoint = steps_per_epoch * 10
  num_steps = steps_per_epoch * num_epochs

  base_learning_rate = FLAGS.learning_rate * batch_size / 256.

  variables = initialized(rng, image_size)
  optimizer = optim.Momentum(beta=FLAGS.momentum, nesterov=True).create(variables['param'])
  state = TrainState(step=0, optimizer=optimizer, batch_stats=variables['batch_stats'],
                     dynamic_scale=dynamic_scale)

  state = restore_checkpoint(state)
  step_offset = int(state.step)  # step_offset > 0 if restarting from checkpoint
  state = jax_utils.replicate(state)

  learning_rate_fn = create_learning_rate_fn(
      base_learning_rate, steps_per_epoch, num_epochs)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  epoch_metrics = []
  t_loop_start = time.time()
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    epoch_metrics.append(metrics)
    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      epoch_metrics = common_utils.get_metrics(epoch_metrics)
      summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
      logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      steps_per_sec = steps_per_epoch / (time.time() - t_loop_start)
      t_loop_start = time.time()
      if jax.host_id() == 0:
        for key, vals in epoch_metrics.items():
          tag = 'train_%s' % key
          for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)
        summary_writer.scalar('steps per second', steps_per_sec, step)

      epoch_metrics = []
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
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
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

if __name__ == '__main__':
  app.run(main)

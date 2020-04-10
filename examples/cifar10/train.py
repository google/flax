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

"""CIFAR-10 example."""

import ast
import functools

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import optim
import input_pipeline
from models import pyramidnet
from models import wideresnet
from models import wideresnet_shakeshake
from flax.metrics import tensorboard
import flax.nn
from flax.training import common_utils
from flax.training import lr_schedule

import jax
from jax import random
import jax.nn
import jax.numpy as jnp

import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_string(
    'lr_schedule', default='stepped',
    help=('Learning rate schedule type; constant, stepped or cosine'))

flags.DEFINE_string(
    'lr_sched_steps', default='[[60, 0.2], [120, 0.04], [160, 0.008]]',
    help=('Learning rate schedule steps as a Python list; '
          '[[step1_epoch, step1_lr_scale], '
          '[step2_epoch, step2_lr_scale], ...]'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_float(
    'l2_reg', default=0.0005,
    help=('The amount of L2-regularization to apply.'))

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=200,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'arch', default='wrn26_10',
    help=('Network architecture'))

flags.DEFINE_float(
    'wrn_dropout_rate', default=0.3,
    help=('Wide ResNet DropOut rate'))

flags.DEFINE_integer(
    'rng', default=0,
    help=('Random seed for network initialization.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data'))


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(prng_key, batch_size, image_size, module):
  input_shape = (batch_size, image_size, image_size, 3)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = module.init_by_shape(
          prng_key, [(input_shape, jnp.float32)])
      model = flax.nn.Model(module, initial_params)
  return model, init_state


def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                 beta=beta,
                                 nesterov=True)
  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)
  return optimizer


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  error_rate = jnp.mean(jnp.argmax(logits, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
  }
  metrics = jax.lax.pmean(metrics, 'batch')
  return metrics


def train_step(optimizer, state, batch, prng_key, learning_rate_fn, l2_reg):
  """Perform a single training step."""
  def loss_fn(model):
    """loss function used for training."""
    with flax.nn.stateful(state) as new_state:
      with flax.nn.stochastic(prng_key):
        logits = model(batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    # TODO(britefury): check if applying L2 regularization to weights but
    # *not* biases improves results
    weight_penalty_params = jax.tree_leaves(model.params)
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = l2_reg * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_state, logits)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (new_state, logits)), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=lr)

  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr
  return new_optimizer, new_state, metrics


def eval_step(model, state, batch):
  state = jax.lax.pmean(state, 'batch')
  with flax.nn.stateful(state, mutable=False):
    logits = model(batch['image'], train=False)
  return compute_metrics(logits, batch['label'])


def load_and_shard_tf_batch(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def train(module, model_dir, batch_size,
          num_epochs, learning_rate, sgd_momentum,
          make_lr_fun=None, l2_reg=0.0005, run_seed=0):
  """Train model."""
  if jax.host_count() > 1:
    raise ValueError('CIFAR-10 example should not be run on '
                     'more than 1 host (for now)')

  if make_lr_fun is None:
    # No learning rate function provided
    # Default to stepped LR schedule for CIFAR-10 and Wide ResNet
    def make_lr_fun(base_lr, steps_per_epoch):  # pylint: disable=function-redefined
      return lr_schedule.create_stepped_learning_rate_schedule(
          base_lr, steps_per_epoch,
          [[60, 0.2], [120, 0.04], [160, 0.008]])

  summary_writer = tensorboard.SummaryWriter(model_dir)

  rng = random.PRNGKey(run_seed)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()

  # Load dataset
  data_source = input_pipeline.CIFAR10DataSource(
      train_batch_size=batch_size, eval_batch_size=batch_size)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds

  # Compute steps per epoch and nb of eval steps
  steps_per_epoch = data_source.TRAIN_IMAGES // batch_size
  steps_per_eval = data_source.EVAL_IMAGES // batch_size
  num_steps = steps_per_epoch * num_epochs

  base_learning_rate = learning_rate

  # Create the model
  image_size = 32
  model, state = create_model(rng, device_batch_size, image_size, module)
  state = jax_utils.replicate(state)
  optimizer = create_optimizer(model, base_learning_rate, sgd_momentum)
  del model  # don't keep a copy of the initial model

  # Learning rate schedule
  learning_rate_fn = make_lr_fun(base_learning_rate, steps_per_epoch)

  # pmap the train and eval functions
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                        l2_reg=l2_reg),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Create dataset batch iterators
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # Gather metrics
  train_metrics = []
  epoch = 1
  for step, batch in zip(range(num_steps), train_iter):
    # Generate a PRNG key that will be rolled into the batch
    rng, step_key = jax.random.split(rng)
    # Load and shard the TF batch
    batch = load_and_shard_tf_batch(batch)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)

    # Train step
    optimizer, state, metrics = p_train_step(
        optimizer, state, batch, sharded_keys)
    train_metrics.append(metrics)

    if (step + 1) % steps_per_epoch == 0:
      # We've finished an epoch
      train_metrics = common_utils.get_metrics(train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      # Send stats to Tensorboard
      for key, vals in train_metrics.items():
        tag = 'train_%s' % key
        for i, val in enumerate(vals):
          summary_writer.scalar(tag, val, step - len(vals) + i + 1)
      # Reset train metrics
      train_metrics = []

      # Evaluation
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # Load and shard the TF batch
        eval_batch = load_and_shard_tf_batch(eval_batch)
        # Step
        metrics = p_eval_step(optimizer.target, state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # Get eval epoch summary for logging
      eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)

      # Log epoch summary
      logging.info(
          'Epoch %d: TRAIN loss=%.6f, err=%.2f, EVAL loss=%.6f, err=%.2f',
          epoch, train_summary['loss'], train_summary['error_rate'] * 100.0,
          eval_summary['loss'], eval_summary['error_rate'] * 100.0)

      summary_writer.scalar('eval_loss', eval_summary['loss'], epoch)
      summary_writer.scalar('eval_error_rate', eval_summary['error_rate'],
                            epoch)
      summary_writer.flush()

      epoch += 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  if FLAGS.arch == 'wrn26_10':
    module = wideresnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=10,
        dropout_rate=FLAGS.wrn_dropout_rate)
  elif FLAGS.arch == 'wrn26_2':
    module = wideresnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=2,
        num_outputs=10,
        dropout_rate=FLAGS.wrn_dropout_rate)
  elif FLAGS.arch == 'wrn26_6_ss':
    module = wideresnet_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=4,
        channel_multiplier=6,
        num_outputs=10)
  elif FLAGS.arch == 'pyramid':
    module = pyramidnet.PyramidNetShakeDrop.partial(num_outputs=10)
  else:
    raise ValueError('Unknown architecture {}'.format(FLAGS.arch))

  if FLAGS.lr_schedule == 'constant':
    def make_lr_fun(base_lr, steps_per_epoch):
      return lr_schedule.create_constant_learning_rate_schedule(
          base_lr, steps_per_epoch)
  elif FLAGS.lr_schedule == 'stepped':
    if not FLAGS.lr_sched_steps:
      lr_sched_steps = [[60, 0.2], [120, 0.04], [160, 0.008]]
    else:
      lr_sched_steps = ast.literal_eval(FLAGS.lr_sched_steps)
    def make_lr_fun(base_lr, steps_per_epoch):
      return lr_schedule.create_stepped_learning_rate_schedule(
          base_lr, steps_per_epoch, lr_sched_steps)
  elif FLAGS.lr_schedule == 'cosine':
    def make_lr_fun(base_lr, steps_per_epoch):
      return lr_schedule.create_cosine_learning_rate_schedule(
          base_lr, steps_per_epoch, FLAGS.num_epochs)
  else:
    raise ValueError('Unknown LR schedule type {}'.format(FLAGS.lr_schedule))

  train(module, FLAGS.model_dir, FLAGS.batch_size,
        FLAGS.num_epochs, FLAGS.learning_rate,
        FLAGS.momentum, make_lr_fun, FLAGS.l2_reg, FLAGS.rng)


if __name__ == '__main__':
  app.run(main)

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

Library file containing logic for training and evaluation loops.
"""

import functools
import time

from absl import logging

import jax
from jax import lax
from jax import random

import jax.nn
import jax.numpy as jnp

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import flax
from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import input_pipeline
from resnet_v1 import ResNet


def _create_model(key, batch_size, image_size, model_dtype):
  input_shape = (batch_size, image_size, image_size, 3)
  module = ResNet.partial(num_classes=1000, dtype=model_dtype)
  with nn.stateful() as init_state:
    _, initial_params = module.init_by_shape(
        key, [(input_shape, model_dtype)])
    model = nn.Model(module, initial_params)
  return model, init_state


def _cross_entropy_loss(logits, labels):
  return -jnp.sum(
      common_utils.onehot(labels, num_classes=1000) * logits) / labels.size


def _compute_metrics(logits, labels):
  loss = _cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def _cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def _create_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs):
  warmup_epochs = 5
  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = _cosine_decay(base_learning_rate,
                       epoch - warmup_epochs,
                       num_epochs - warmup_epochs)
    warmup = jnp.minimum(1., epoch / warmup_epochs)
    return lr * warmup
  return step_fn


def _train_step(state, batch, learning_rate_fn):
  """Perform a single training step."""
  def _loss_fn(model):
    """loss function used for training."""
    with nn.stateful(state.model_state) as new_model_state:
      logits = model(batch['image'])
    loss = _cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_leaves(model.params)
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  step = state.step
  optimizer = state.optimizer
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        _loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grad = grad_fn(optimizer.target)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    aux, grad = grad_fn(optimizer.target)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = lax.pmean(grad, axis_name='batch')
  new_model_state, logits = aux[1]
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = _compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and the old optimizer
    # state should be restored.
    new_optimizer = jax.tree_multimap(
        functools.partial(jnp.where, is_fin), new_optimizer, optimizer)
    metrics['scale'] = dynamic_scale.scale

  new_state = state.replace(
      step=step + 1, optimizer=new_optimizer, model_state=new_model_state,
      dynamic_scale=dynamic_scale)
  return new_state, metrics


def _eval_step(state, batch):
  model = state.optimizer.target
  with nn.stateful(state.model_state, mutable=False):
    logits = model(batch['image'], train=False)
  return _compute_metrics(logits, batch['label'])


def _prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def _create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                       cache):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache)
  it = map(_prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class _TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: nn.Collection
  dynamic_scale: optim.DynamicScale


def _restore_checkpoint(model_dir, state):
  return checkpoints.restore_checkpoint(model_dir, state)


def _save_checkpoint(model_dir, state):
  if jax.host_id() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(model_dir, state, step, keep=3)


def _sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
  return state.replace(model_state=avg(state.model_state))


def train_and_evaluate(model_dir: str, batch_size: int, num_epochs: int,
                       learning_rate: float, momentum: float, cache: bool,
                       half_precision: bool,
                       num_train_and_eval_steps: int = -1,
                       disable_checkpointing: bool = False):
  """Runs model training and evaluation loop.

  Args:
    model_dir: Directory where the checkpoints and tensorboard summaries
      should be written to.
    batch_size: Batch size of the input.
    num_epochs: Number of epochs to cycle through before stopping.
    learning_rate: Learning rate for the momentum optimizer.
    momentum: Momentum value for the momentum optimizer.
    cache: Determines whether the dataset should be cached.
    half_precision: Determines whether bfloat16/float16 should be used
      instead of float32.
    num_train_and_eval_steps: Number of steps for training and eval.
      This is used for testing (default: -1 i.e use the entire dataset).
    disable_checkpointing: Determines whether the training and evaluation
      loop should checkpoint model weights.
  """

  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(model_dir)

  rng = random.PRNGKey(0)

  image_size = 224

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // jax.host_count()
  device_batch_size = batch_size // jax.device_count()

  platform = jax.local_devices()[0].platform

  dynamic_scale = None
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
      input_dtype = tf.bfloat16
    else:
      model_dtype = jnp.float16
      input_dtype = tf.float16
      dynamic_scale = optim.DynamicScale()
  else:
    model_dtype = jnp.float32
    input_dtype = tf.float32

  dataset_builder = tfds.builder('imagenet2012:5.*.*')
  train_iter = _create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=cache)
  eval_iter = _create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=cache)

  if num_train_and_eval_steps == -1:
    steps_per_epoch = \
        dataset_builder.info.splits['train'].num_examples // batch_size
    steps_per_eval = \
        dataset_builder.info.splits['validation'].num_examples // batch_size
  else:
    steps_per_epoch = num_train_and_eval_steps
    steps_per_eval = num_train_and_eval_steps

  steps_per_checkpoint = steps_per_epoch * 10
  num_steps = steps_per_epoch * num_epochs

  base_learning_rate = learning_rate * batch_size / 256.

  model, model_state = _create_model(
      rng, device_batch_size, image_size, model_dtype)
  optimizer = optim.Momentum(beta=momentum, nesterov=True).create(model)
  state = _TrainState(step=0, optimizer=optimizer, model_state=model_state,
                      dynamic_scale=dynamic_scale)
  del model, model_state  # do not keep a copy of the initial model

  state = _restore_checkpoint(model_dir, state)
  step_offset = int(state.step)  # step_offset > 0 if restarting from checkpoint
  state = jax_utils.replicate(state)

  learning_rate_fn = _create_learning_rate_fn(
      base_learning_rate, steps_per_epoch, num_epochs)

  p_train_step = jax.pmap(
      functools.partial(_train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(_eval_step, axis_name='batch')

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
      state = _sync_batch_stats(state)
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
    if not disable_checkpointing and ((step + 1) % steps_per_checkpoint == 0
                                      or step + 1 == num_steps):
      state = _sync_batch_stats(state)
      _save_checkpoint(model_dir, state)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

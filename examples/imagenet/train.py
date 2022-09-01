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

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any

from absl import logging
from clu import periodic_actions

import flax
from flax import jax_utils
from flax import optim
from jax.dtypes import dtype
from jax.interpreters.pxla import replicate

import input_pipeline
import models

from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import lax
from jax import random
from jax.lib import xla_client
import numpy as np

import jax.numpy as jnp

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=10, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  @functools.partial(jax.jit, backend='cpu')
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  model_state, params = variables.pop('params')
  return params, model_state


def cross_entropy_loss(logits, labels):
  return -jnp.sum(
      common_utils.onehot(labels, num_classes=10) * logits) / labels.size


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


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float,
                            steps_per_epoch: int):

  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = cosine_decay(base_learning_rate,
                      epoch - config.warmup_epochs,
                      config.num_epochs - config.warmup_epochs)
    warmup = jnp.minimum(1., epoch / config.warmup_epochs)
    return lr * warmup
  return step_fn


def train_step(apply_fn, state, batch, learning_rate_fn):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    variables = {'params': params, **state.model_state}
    logits, new_model_state = apply_fn(
        variables, batch[0], mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch[1])
    weight_penalty_params = jax.tree_leaves(variables['params'])
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
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grad = grad_fn(optimizer.target)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grad = grad_fn(optimizer.target)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = lax.pmean(grad, axis_name='batch')
  new_model_state, logits = aux[1]
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  metrics = compute_metrics(logits, batch[1])
  metrics['learning_rate'] = lr


  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and the old optimizer
    # state should be restored.
    new_optimizer = jax.tree_multimap(
        functools.partial(jnp.where, is_fin), new_optimizer, optimizer)
    # metrics['scale'] = dynamic_scale.scale

  new_state = state.replace(
      step=step + 1, optimizer=new_optimizer, model_state=new_model_state,
      dynamic_scale=dynamic_scale)
  return new_state, (metrics["accuracy"], metrics["loss"])

def eval_step(apply_fn, state, batch):
  params = state.optimizer.target
  variables = {'params': params, **state.model_state}
  logits = apply_fn(
      variables, batch[0], train=False, mutable=False)
  return compute_metrics(logits, batch[1])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access
    # Change data type from int64 to int32 as ipu-infeed does not take int64.
    if (len(x.shape) == 1):
      x = x.astype(np.int32)
    return x

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache)
  it = map(prepare_tf_data, ds)
  # it = jax_utils.prefetch_to_device(it, 2)
  return it


# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: Any
  dynamic_scale: optim.DynamicScale


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.host_id() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # An axis_name is passed to pmap which can then be used by pmean.
  # In this case each device has its own version of the batch statistics and
  # we average them.
  avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

  new_model_state = state.model_state.copy({
      'batch_stats': avg(state.model_state['batch_stats'])})
  return state.replace(model_state=new_model_state)


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = optim.DynamicScale()
  else:
    dynamic_scale = None

  params, model_state = initialized(rng, image_size, model)
  optimizer = optim.Momentum(
      beta=config.momentum, nesterov=True).create(params)
  state = TrainState(
      step=0, optimizer=optimizer, model_state=model_state,
      dynamic_scale=dynamic_scale)
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  """

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

  rng = random.PRNGKey(0)

  image_size = 224
  
  if config.batch_size % jax.device_count("ipu") > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.device_count("ipu")
  logging.info(f'batch size per IPU: {local_batch_size}')

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset)
  dataset_builder.download_and_prepare()
  train_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=config.cache)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation'].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  base_learning_rate = config.learning_rate * config.batch_size / 256.

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision)

  state = create_train_state(rng, config, model, image_size)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)

  p_train_step = functools.partial(train_step, model.apply,
                                  learning_rate_fn=learning_rate_fn)

  p_eval_step = functools.partial(eval_step, model.apply)

  to_infeed_shape = (jax.ShapedArray((local_batch_size, image_size, image_size, 3), dtype=jnp.float32),
                   jax.ShapedArray((local_batch_size, 1), dtype=jnp.int32))

  def train_loop(_, state):
    token = lax.create_token()
    batch, token = lax.infeed(
        token, shape=to_infeed_shape)
    state, metrics = p_train_step(state, batch)
    lax.outfeed(token, metrics)
    return state

  def eval_loop(_, state):
    token = lax.create_token()
    batch, token = lax.infeed(
        token, shape=to_infeed_shape)
    lax.outfeed(token, p_eval_step(state, batch))
    return state

  @functools.partial(jax.pmap, backend='ipu', donate_argnums=[2], in_axes=(None, None, None, 0), axis_name='batch')
  def train_loops(start, end, state, dummy):
    state = lax.fori_loop(start, end, train_loop, state)
    return state

  @functools.partial(jax.pmap, backend='ipu', donate_argnums=[2], in_axes=(None, None, None, 0), axis_name='batch')
  def eval_loops(start, end, state, dummy):
    state = lax.fori_loop(start, end, eval_loop, state)
    return state

  eval_metrics = []

  x = jnp.array(1, dtype=jnp.float32)
  y = jnp.array(1, dtype=jnp.float32)

  epoch_metrics = []

  logging.info('Initial compilation, this might take some minutes...')
  devices = jax.devices(backend='ipu')
  n_devices = len(devices)
  logging.info(f'IPUs: {devices}')
  dummy = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1)

  for epoch in range(config.num_epochs):
    start_time = time.time()
    logging.info('Prepare train dataset!')
    for device in devices:
      for _, batch in zip(range(steps_per_epoch), train_iter):
        batch = tuple(batch.values())
        device.transfer_to_infeed(batch)
    logging.info('Finished train dataset!')
    state = train_loops(epoch * steps_per_epoch, (epoch + 1) * steps_per_epoch, state, dummy)
    state = jax_utils.unreplicate(state)
    epoch_time = time.time() - start_time
    print("Train epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    for i, device in enumerate(devices):
      for _ in range(steps_per_epoch):
        train_acc, train_loss = device.transfer_from_outfeed(xla_client.shape_from_pyval((x, y))
                                            .with_major_to_minor_layout_if_absent())
        if i == 0:
          epoch_metrics.append({"loss":train_loss, "accuracy":train_acc})
    
    # epoch_metrics = common_utils.get_metrics(epoch_metrics)
    epoch_metrics = common_utils.stack_forest(epoch_metrics)
    summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f',
                  epoch, summary['loss'], summary['accuracy'] * 100)

    epoch_metrics = []
    eval_metrics = []

    logging.info('Prepare eval dataset!')
    for device in devices:
      for _, batch in zip(range(steps_per_eval), eval_iter):
        batch = tuple(batch.values())
        device.transfer_to_infeed(batch)
    logging.info('Finished eval dataset!')
    state = eval_loops(0, steps_per_eval, state, dummy)
    state = jax_utils.unreplicate(state)
    for i, device in enumerate(devices):
      for _ in range(steps_per_eval):
        eval_acc, eval_loss = device.transfer_from_outfeed(xla_client.shape_from_pyval((x, y))
                                            .with_major_to_minor_layout_if_absent())
        if i == 0:
          eval_metrics.append({"loss":eval_loss, "accuracy":eval_acc})
    
    # eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics = common_utils.stack_forest(eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
    eval_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec, with eval".format(epoch, eval_time))

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state

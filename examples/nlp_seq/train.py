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

"""Sequence Tagging example.

This script trains a Transformer on the Universal dependency dataset.
"""

import functools
import itertools
import os
import time
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
import input_pipeline
import models
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', default='', help=('Directory for model data.'))

flags.DEFINE_string('experiment', default='xpos', help=('Experiment name.'))

flags.DEFINE_integer(
    'batch_size', default=64, help=('Batch size for training.'))

flags.DEFINE_integer(
    'eval_frequency',
    default=100,
    help=('Frequency of eval during training, e.g. every 1000 steps.'))

flags.DEFINE_integer(
    'num_train_steps', default=75000, help=('Number of train steps.'))

flags.DEFINE_integer(
    'num_eval_steps',
    default=-1,
    help=('Number of evaluation steps. If -1 use the whole evaluation set.'))

flags.DEFINE_float('learning_rate', default=0.05, help=('Learning rate.'))

flags.DEFINE_float(
    'weight_decay',
    default=1e-1,
    help=('Decay factor for AdamW style weight decay.'))

flags.DEFINE_integer('max_length', default=256,
                     help=('Maximum length of examples.'))

flags.DEFINE_integer(
    'random_seed', default=0, help=('Integer for PRNG random seed.'))

flags.DEFINE_string('train', default='', help=('Path to training data.'))

flags.DEFINE_string('dev', default='', help=('Path to development data.'))


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model(key, input_shape, model_kwargs):
  model_def = models.Transformer.partial(train=False, **model_kwargs)
  _, initial_params = model_def.init_by_shape(key, [(input_shape, jnp.float32)])
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)
  return optimizer


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = jnp.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = np.sum(metrics, -1)
  return metrics


def train_step(optimizer, batch, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""
  train_keys = ['inputs', 'targets']
  (inputs, targets) = [batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # It's very important to handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the latter can add
  # bad stalls to the input data transfer.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr

  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, batch):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = model(inputs, train=False)
  return compute_metrics(logits, targets, weights)


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by zeros with the shape of last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  # Padding with zeros to avoid that they get counted in compute_metrics.
  return np.concatenate([x, np.tile(np.zeros_like(x[-1]), (batch_pad, 1))])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  num_eval_steps = FLAGS.num_eval_steps
  eval_freq = FLAGS.eval_frequency
  max_length = FLAGS.max_length
  random_seed = FLAGS.random_seed

  if not FLAGS.dev:
    raise app.UsageError('Please provide path to dev set.')
  if not FLAGS.train:
    raise app.UsageError('Please provide path to training set.')

  parameter_path = os.path.join(FLAGS.model_dir, FLAGS.experiment + '.params')
  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, FLAGS.experiment + '_train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, FLAGS.experiment + '_eval'))

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()

  # create the training and development dataset
  vocabs = input_pipeline.create_vocabs(FLAGS.train)
  attributes_input = [input_pipeline.CoNLLAttributes.FORM]
  attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
  train_ds = input_pipeline.sentence_dataset_dict(
      FLAGS.train,
      vocabs,
      attributes_input,
      attributes_target,
      batch_size=batch_size,
      bucket_size=max_length)

  eval_ds = input_pipeline.sentence_dataset_dict(
      FLAGS.dev,
      vocabs,
      attributes_input,
      attributes_target,
      batch_size=batch_size,
      bucket_size=max_length,
      repeat=1)
  train_iter = iter(train_ds)
  bs = device_batch_size * jax.device_count()

  rng = random.PRNGKey(random_seed)
  rng, init_rng = random.split(rng)
  input_shape = (bs, max_length)
  transformer_kwargs = {
      'vocab_size': len(vocabs['forms']),
      'output_vocab_size': len(vocabs['xpos']),
      'emb_dim': 512,
      'num_heads': 8,
      'num_layers': 6,
      'qkv_dim': 512,
      'mlp_dim': 2048,
      'max_len': max_length,
  }
  model = create_model(init_rng, tuple(input_shape), transformer_kwargs)

  optimizer = create_optimizer(model, learning_rate)
  del model  # don't keep a copy of the initial model
  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())

  metrics_all = []
  tick = time.time()
  best_dev_score = 0
  for step, batch in zip(range(num_train_steps), train_iter):
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access

    optimizer, metrics, dropout_rngs = p_train_step(
        optimizer, batch, dropout_rng=dropout_rngs)
    metrics_all.append(metrics)

    if (step + 1) % eval_freq == 0:
      metrics_all = common_utils.get_metrics(metrics_all)
      lr = metrics_all.pop('learning_rate').mean()
      metrics_sums = jax.tree_map(jnp.sum, metrics_all)
      denominator = metrics_sums.pop('denominator')
      summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
      summary['learning_rate'] = lr
      # Calculate (clipped) perplexity after averaging log-perplexities:
      summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
      logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
      if jax.host_id() == 0:
        tock = time.time()
        steps_per_sec = eval_freq / (tock - tick)
        tick = tock
        train_summary_writer.scalar('steps per second', steps_per_sec, step)
        for key, val in summary.items():
          train_summary_writer.scalar(key, val, step)
        train_summary_writer.flush()
      # reset metric accumulation for next evaluation cycle.
      metrics_all = []

      eval_metrics = []
      eval_iter = iter(eval_ds)
      if num_eval_steps == -1:
        num_iter = itertools.repeat(1)
      else:
        num_iter = range(num_eval_steps)
      for _, eval_batch in zip(num_iter, eval_iter):
        eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
        # Handle final odd-sized batch by padding instead of dropping it.
        cur_pred_batch_size = eval_batch['inputs'].shape[0]
        if cur_pred_batch_size != batch_size:
          logging.info('Uneven batch size %d.', cur_pred_batch_size)
          eval_batch = jax.tree_map(
              lambda x: pad_examples(x, batch_size), eval_batch)
        eval_batch = common_utils.shard(eval_batch)

        metrics = p_eval_step(optimizer.target, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
      eval_denominator = eval_metrics_sums.pop('denominator')
      eval_summary = jax.tree_map(
          lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
          eval_metrics_sums)

      # Calculate (clipped) perplexity after averaging log-perplexities:
      eval_summary['perplexity'] = jnp.clip(
          jnp.exp(eval_summary['loss']), a_max=1.0e4)
      logging.info('eval in step: %d, loss: %.4f, accuracy: %.4f', step,
                   eval_summary['loss'], eval_summary['accuracy'])

      if best_dev_score < eval_summary['accuracy']:
        best_dev_score = eval_summary['accuracy']
        # TODO: save model.
      eval_summary['best_dev_score'] = best_dev_score
      logging.info('best development model score %.4f', best_dev_score)
      if jax.host_id() == 0:
        for key, val in eval_summary.items():
          eval_summary_writer.scalar(key, val, step)
        eval_summary_writer.flush()


if __name__ == '__main__':
  app.run(main)

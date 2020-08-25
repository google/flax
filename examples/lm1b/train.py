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

"""Language Modeling example.

This script trains a Transformer on the lm1b dataset.
The data is loaded using tensorflow_datasets.
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
import decode
import input_pipeline
import models
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data.')

flags.DEFINE_string(
    'data_dir', default=None,
    help='Directory containing TFDS lm1b/subwords32k dataset.')

flags.DEFINE_integer(
    'batch_size', default=2048,
    help='Batch size for training.')

flags.DEFINE_integer(
    'eval_frequency', default=1000,
    help='Frequency of eval during training, e.g. every 1000 steps.')

flags.DEFINE_integer(
    'num_train_steps', default=500000,
    help='Number of training steps.')

flags.DEFINE_integer(
    'num_eval_steps', default=20,
    help='Number of evaluation steps. If -1 use the whole evaluation set.')

flags.DEFINE_float(
    'learning_rate', default=0.05,
    help='Learning rate.')

flags.DEFINE_float(
    'weight_decay', default=1e-1,
    help='Decay factor for AdamW-style weight decay.')

flags.DEFINE_integer(
    'max_target_length', default=512,
    help='Maximum length of training examples.')

flags.DEFINE_integer(
    'max_eval_target_length', default=2048,
    help='Maximum length of eval examples.')

flags.DEFINE_float(
    'sampling_temperature', default=0.6,
    help='Sampling temperature for language model inference.')

flags.DEFINE_integer(
    'sampling_top_k', default=20,
    help='Top k cutoff for logit sampling. If 0 then no top-k cutoff is used.')

flags.DEFINE_string(
    'prompt', default='I love to ',
    help='Prompt for language model sampling.')

flags.DEFINE_integer(
    'max_predict_token_length', default=50,
    help='Maximum example text inference token length.')

flags.DEFINE_bool(
    'save_checkpoints', default=True,
    help='Whether to save model checkpoints for debugging.')

flags.DEFINE_bool(
    'restore_checkpoints', default=True,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_integer(
    'checkpoint_freq', default=10000,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_integer(
    'random_seed', default=0,
    help='Integer for PRNG random seed.')


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model(key, input_shape, model_kwargs):
  module = models.TransformerLM.partial(**model_kwargs)
  with nn.attention.Cache().mutate() as cache_def:
    _, initial_params = module.init_by_shape(key,
                                         [(input_shape, jnp.float32)],
                                         cache=cache_def)
  model = nn.Model(module, initial_params)
  return model, cache_def


def create_optimizer(model, learning_rate):
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(model)
  return optimizer


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

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
    A function learning_rate(step): float -> {'learning_rate': float}, the
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
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def train_step(optimizer, inputs, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""
  weights = jnp.where(inputs > 0, 1, 0)

  # We handle PRNG splitting inside the top pmap, rather
  # than handling it outside in the training loop - doing the
  # latter can add some stalls to the devices.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Loss function used for training."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss, weight_sum = compute_weighted_cross_entropy(logits, inputs, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = compute_metrics(logits, inputs, weights)
  metrics['learning_rate'] = lr

  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, inputs):
  weights = jnp.where(inputs > 0, 1, 0)
  logits = model(inputs, train=False)
  return compute_metrics(logits, inputs, weights)


def predict_step(inputs, model, cache, prng_key):
  """Fast sampling of language model from prompt."""
  prefix_len = inputs.shape[1]
  pad_len = FLAGS.max_predict_token_length - prefix_len
  padded_inputs = jnp.pad(inputs, jnp.array([[0, 0], [0, pad_len]]))

  def tokens_ids_to_logits(ids, cache):
    """Token slice to logits from decoder model."""
    with cache.mutate() as new_cache:
      logits = model(ids, shift=False, train=False, cache=new_cache)
    # Remove singleton sequence-length dimension
    # [batch, 1, vocab] --> [batch, vocab]
    logits = logits.squeeze(axis=1)
    return logits, new_cache

  sampled_seqs = decode.temperature_sample(
      padded_inputs,
      cache,
      tokens_ids_to_logits,
      prng_key,
      temperature=FLAGS.sampling_temperature,
      topk=FLAGS.sampling_top_k,
      eos_token=2**16)  # No EOS tokens used in default lm1b dataset encoding.

  return sampled_seqs


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  num_eval_steps = FLAGS.num_eval_steps
  eval_freq = FLAGS.eval_frequency
  max_target_length = FLAGS.max_target_length
  max_eval_target_length = FLAGS.max_eval_target_length
  random_seed = FLAGS.random_seed

  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'eval'))

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  train_ds, eval_ds, info_ds = input_pipeline.get_lm1b_datasets(
      n_devices=jax.local_device_count(),
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      dynamic_batching=True,
      max_target_length=max_target_length,
      max_eval_target_length=max_eval_target_length)
  vocab_size = info_ds['text'].encoder.vocab_size
  encoder = info_ds['text'].encoder

  train_iter = iter(train_ds)
  input_shape = (batch_size, max_target_length)

  transformer_lm_kwargs = {
      'vocab_size': vocab_size,
      'emb_dim': 512,
      'num_heads': 8,
      'num_layers': 6,
      'qkv_dim': 512,
      'mlp_dim': 2048,
      'max_len': max(max_target_length, max_eval_target_length)
  }

  rng = random.PRNGKey(random_seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = random.split(rng)
  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, jax.local_device_count())

  model, cache_def = create_model(init_rng, input_shape, transformer_lm_kwargs)
  optimizer = create_optimizer(model, learning_rate)
  del model  # Don't keep a copy of the initial model.
  start_step = 0
  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    optimizer = checkpoints.restore_checkpoint(FLAGS.model_dir, optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate)
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')
  p_pred_step = jax.pmap(predict_step, axis_name='batch')

  metrics_all = []
  tick = time.time()
  for step, batch in zip(range(start_step, num_train_steps), train_iter):
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
    optimizer, metrics, dropout_rngs = p_train_step(
        optimizer, batch, dropout_rng=dropout_rngs)
    metrics_all.append(metrics)

    # Save a Checkpoint
    if ((step % FLAGS.checkpoint_freq == 0 and step > 0) or
        step == num_train_steps - 1):
      if jax.host_id() == 0 and FLAGS.save_checkpoints:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(
            FLAGS.model_dir, jax_utils.unreplicate(optimizer), step)

    # Periodic metric handling.
    if step % eval_freq == 0 and step > 0:
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
      # Reset metric accumulation for next evaluation cycle.
      metrics_all = []

      # Eval Metrics
      eval_metrics = []
      eval_iter = iter(eval_ds)
      if num_eval_steps == -1:
        num_iter = itertools.repeat(1)
      else:
        num_iter = range(num_eval_steps)
      for _, eval_batch in zip(num_iter, eval_iter):
        # pylint: disable=protected-access
        eval_batch = common_utils.shard(
            jax.tree_map(lambda x: x._numpy(), eval_batch))
        # pylint: enable=protected-access
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
      logging.info('eval in step: %d, loss: %.4f', step, eval_summary['loss'])
      if jax.host_id() == 0:
        for key, val in eval_summary.items():
          eval_summary_writer.scalar(key, val, step)
        eval_summary_writer.flush()

      # Fast inference of prompt extension using trained LM.
      rng, subrng = jax.random.split(rng)
      pred_rngs = random.split(subrng, jax.local_device_count())
      prompt = jnp.array(encoder.encode(FLAGS.prompt))
      prompt = jax_utils.replicate(prompt)
      prompt = jnp.reshape(prompt, (prompt.shape[0], 1, prompt.shape[1]))
      cache = jax_utils.replicate(
          cache_def.initialize_cache((1, FLAGS.max_predict_token_length)))
      predicted = p_pred_step(prompt, optimizer.target, cache, pred_rngs)
      predicted = tohost(predicted)
      exemplars = ''
      for n in range(predicted.shape[0]):
        exemplars += encoder.decode(predicted[n]) + '\n\n'
      if jax.host_id() == 0:
        eval_summary_writer.text('samples', exemplars, step)
        eval_summary_writer.flush()


if __name__ == '__main__':
  app.run(main)

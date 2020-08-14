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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""

import collections
import functools
import os
import time
from absl import app
from absl import flags
from absl import logging
from jax import random
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils

import bleu
import decode
import input_pipeline
import models


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help='Directory to store model data.')

flags.DEFINE_string(
    'data_dir', default=None,
    help='Tensorflow datasets directory.')

flags.DEFINE_string(
    'vocab_path', default=None,
    help='Path to load or store sentencepiece vocab file.')

flags.DEFINE_integer(
    'vocab_size', default=32000,
    help='Vocabulary size if `vocab_path` is not given.')

flags.DEFINE_string(
    'dataset_name', default='wmt17_translate/de-en',
    help='Name of TFDS translation dataset to use.')

flags.DEFINE_string(
    'eval_dataset_name', default='wmt14_translate/de-en:test',
    help='Optional name of TFDS translation dataset to use for evaluation.')

flags.DEFINE_bool(
    'reverse_translation', default=False,
    help='Reverse the direction of translation.')

flags.DEFINE_integer(
    'batch_size', default=256,
    help='Per host batch size for training.')

flags.DEFINE_integer(
    'beam_size', default=4,
    help='Beam size for inference.')

flags.DEFINE_integer(
    'eval_frequency', default=1000,
    help='Frequency of eval during training, e.g. every 1000 steps.')

flags.DEFINE_integer(
    'num_train_steps', default=500000,
    help='Number of train steps.')

flags.DEFINE_integer(
    'num_eval_steps', default=20,
    help='Number of steps to take during evaluation.')

flags.DEFINE_float(
    'learning_rate', default=0.0625,
    help='Base learning rate.')

flags.DEFINE_integer(
    'warmup_steps', default=1000,
    help='Linear learning rate warmup.')

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help='Cross entropy loss label smoothing.')

flags.DEFINE_float(
    'weight_decay', default=0.0,
    help='Decay factor for AdamW style weight decay.')

flags.DEFINE_integer(
    'max_target_length', default=256,
    help='Maximum length cutoff for training examples.')

flags.DEFINE_integer(
    'max_eval_target_length', default=256,
    help='Maximum length cutoff for eval examples.')

flags.DEFINE_integer(
    'max_predict_length', default=256,
    help='Maximum length cutoff for predicted tokens.')

flags.DEFINE_bool(
    'share_embeddings', default=True,
    help='Inputs and targets share embedding.')

flags.DEFINE_bool(
    'logits_via_embedding', default=True,
    help='Final logit transform uses embedding matrix transpose.')

flags.DEFINE_integer(
    'num_layers', default=6,
    help='Number of transformer layers.')

flags.DEFINE_integer(
    'qkv_dim', default=1024,
    help='Size of query/key/value for attention.')

flags.DEFINE_integer(
    'emb_dim', default=1024,
    help='Size of embeddings.')

flags.DEFINE_integer(
    'mlp_dim', default=4096,
    help='Size of the MLP.')

flags.DEFINE_integer(
    'num_heads', default=16,
    help='Number of attention heads.')

flags.DEFINE_float(
    'dropout_rate', default=0.1,
    help='Dropout rate.')

flags.DEFINE_float(
    'attention_dropout_rate', default=0.1,
    help='Attention dropout rate.')

flags.DEFINE_integer(
    'random_seed', default=0,
    help='Integer for PRNG random seed.')

flags.DEFINE_bool(
    'save_checkpoints', default=True,
    help='Whether to save model checkpoints.')

flags.DEFINE_bool(
    'restore_checkpoints', default=True,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_integer(
    'checkpoint_freq', default=10000,
    help='Save a checkpoint every these number of steps.')

flags.DEFINE_bool(
    'use_bfloat16', default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_string(
    'jax_backend_target', default=None,
    help=('TPU grpc target for use with cloud TPUs.'
          ' e.g. grpc://192.168.0.2:8470'))


# Training utility functions.
# -----------------------------------------------------------------------------

def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

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


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and
     off values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) + (vocab_size - 1) *
      low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = jnp.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
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


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights,
                                                    label_smoothing)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name='batch')
  return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------

def train_step(optimizer,
               batch,
               config,
               learning_rate_fn,
               label_smoothing=0.0,
               dropout_rng=None):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using 'packed examples'
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ['inputs', 'targets',
                'inputs_position', 'targets_position',
                'inputs_segmentation', 'targets_segmentation']
  (inputs, targets,
   inputs_positions, targets_positions,
   inputs_segmentation, targets_segmentation) = [
       batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # We handle PRNG splitting inside the top pmap to improve efficiency.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        {'param': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={'dropout': dropout_rng})

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing)
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


def eval_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply(
      {'param': params}, inputs, targets)

  return compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = models.Transformer(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables['cache']


def predict_step(inputs, params, cache, eos_id, max_decode_len, config,
                 beam_size=4):
  """Predict translation with fast decoding beam search on a batch."""
  batch_size = inputs.shape[0]

  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  src_padding_mask = decode.flat_batch_beam_expand(
      (inputs > 0)[..., None], beam_size)
  tgt_padding_mask = decode.flat_batch_beam_expand(
      jnp.ones((batch_size, 1, 1)), beam_size)
  encoded_inputs = decode.flat_batch_beam_expand(
      models.Transformer(config).apply(
        {'param': params}, inputs, method=models.Transformer.encode),
      beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = models.Transformer(config).apply(
        {'param': params, 'cache': flat_cache},
        encoded_inputs,
        src_padding_mask,
        flat_ids,
        tgt_padding_mask=tgt_padding_mask,
        mutable=['cache'],
        method=models.Transformer.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_id=eos_id,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]


# Utils for prediction and BLEU calculation
# -----------------------------------------------------------------------------

def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree's leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i', devices=devices)
  def pre_pmap(xs):
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)
  def post_pmap(xs):
    return jax.tree_map(lambda x: x[0], xs)
  return post_pmap(host_psum(pre_pmap(in_tree)))


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


# Main
# -----------------------------------------------------------------------------

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.jax_backend_target:
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = FLAGS.jax_backend_target

  # This seems to be necessary even when importing TF2?
  tf.enable_v2_behavior()

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.host_id() == 0:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'eval'))

  if FLAGS.batch_size % n_devices:
    raise ValueError('Batch size must be divisible by the number of devices')

  vocab_path = FLAGS.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(FLAGS.model_dir, 'sentencepiece_model')
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_wmt_datasets(
      n_devices=n_devices,
      dataset_name=FLAGS.dataset_name,
      eval_dataset_name=FLAGS.eval_dataset_name,
      shard_idx=jax.host_id(),
      shard_count=jax.host_count(),
      data_dir=FLAGS.data_dir,
      vocab_path=vocab_path,
      target_vocab_size=FLAGS.vocab_size,
      batch_size=FLAGS.batch_size,
      max_length=FLAGS.max_target_length,
      max_eval_length=FLAGS.max_eval_target_length)
  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.
  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  logging.info('Initializing model, optimizer, and step functions.')

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=FLAGS.share_embeddings,
      logits_via_embedding=FLAGS.logits_via_embedding,
      dtype=jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32,
      emb_dim=FLAGS.emb_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.qkv_dim,
      mlp_dim=FLAGS.mlp_dim,
      max_len=max(FLAGS.max_target_length, FLAGS.max_eval_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

  start_step = 0
  rng = random.PRNGKey(FLAGS.random_seed)
  rng, init_rng = random.split(rng)
  input_shape = (FLAGS.batch_size, FLAGS.max_target_length)
  target_shape = (FLAGS.batch_size, FLAGS.max_target_length)

  # call a jitted initialization function to get the initial parameter tree
  @jax.jit
  def initialize_variables(rng):
    return models.Transformer(eval_config).init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))
  initial_variables = initialize_variables(init_rng)

  # apply an optimizer to this tree
  optimizer_def = optim.Adam(
      FLAGS.learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(initial_variables['param'])

  # We access model params only from optimizer below via optimizer.target.
  del initial_variables

  if FLAGS.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    optimizer = checkpoints.restore_checkpoint(FLAGS.model_dir, optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps)

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=FLAGS.label_smoothing),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          config=eval_config,
          label_smoothing=FLAGS.label_smoothing),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=FLAGS.max_predict_length,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          config=predict_config,
          beam_size=FLAGS.beam_size),
      axis_name='batch',
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, n_devices)

  logging.info('Starting training loop.')
  metrics_all = []
  t_loop_start = time.time()
  for step, batch in zip(range(start_step, FLAGS.num_train_steps), train_iter):
    # Shard data to devices and do a training step.
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
    optimizer, metrics, dropout_rngs = p_train_step(
        optimizer, batch, dropout_rng=dropout_rngs)
    metrics_all.append(metrics)

    # Save a checkpoint on one host after every checkpoint_freq steps.
    if (FLAGS.save_checkpoints and step % FLAGS.checkpoint_freq == 0 and
        step > 0 and jax.host_id() == 0):
      checkpoints.save_checkpoint(FLAGS.model_dir,
                                  jax_utils.unreplicate(optimizer), step)

    # Periodic metric handling.
    if step % FLAGS.eval_frequency != 0 and step > 0:
      continue

    # Training Metrics
    logging.info('Gathering training metrics.')
    metrics_all = common_utils.get_metrics(metrics_all)
    lr = metrics_all.pop('learning_rate').mean()
    metrics_sums = jax.tree_map(jnp.sum, metrics_all)
    denominator = metrics_sums.pop('denominator')
    summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
    summary['learning_rate'] = lr
    steps_per_eval = FLAGS.eval_frequency if step != 0 else 1
    steps_per_sec = steps_per_eval / (time.time() - t_loop_start)
    t_loop_start = time.time()
    if jax.host_id() == 0:
      train_summary_writer.scalar('steps per second', steps_per_sec, step)
      for key, val in summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
    metrics_all = []
    logging.info('train in step: %d, loss: %.4f', step, summary['loss'])

    # Eval Metrics
    logging.info('Gathering evaluation metrics.')
    t_eval_start = time.time()
    eval_metrics = []
    eval_iter = iter(eval_ds)
    for _, eval_batch in zip(range(FLAGS.num_eval_steps), eval_iter):
      eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      eval_batch = common_utils.shard(eval_batch)
      metrics = p_eval_step(optimizer.target, eval_batch)
      eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop('denominator')
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)
    if jax.host_id() == 0:
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, step)
      eval_summary_writer.flush()
    logging.info('eval in step: %d, loss: %.4f', step, eval_summary['loss'])
    logging.info('eval time: %.4f s step %d', time.time()-t_eval_start, step)

    # Translation and BLEU Score.
    logging.info('Translating evaluation dataset.')
    t_inference_start = time.time()
    predict_iter = iter(predict_ds)
    sources, references, predictions = [], [], []
    for _, pred_batch in enumerate(predict_iter):
      pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = pred_batch['inputs'].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(
            np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        pred_batch = jax.tree_map(
            lambda x: pad_examples(x, padded_size), pred_batch)  # pylint: disable=cell-var-from-loop
      pred_batch = common_utils.shard(pred_batch)
      cache = p_init_cache(pred_batch['inputs'])
      predicted = p_pred_step(pred_batch['inputs'],
                              optimizer.target,
                              cache,
                              eos_id,
                              FLAGS.max_predict_length)
      predicted = tohost(predicted)
      inputs = tohost(pred_batch['inputs'])
      targets = tohost(pred_batch['targets'])
      # Iterate through non-padding examples of batch.
      for i, s in enumerate(predicted[:cur_pred_batch_size]):
        sources.append(decode_tokens(inputs[i]))
        references.append(decode_tokens(targets[i]))
        predictions.append(decode_tokens(s))
    logging.info('Translation: %d predictions %d references %d sources.',
                 len(predictions), len(references), len(sources))
    logging.info('Translation time: %.4f s step %d.',
                 time.time() - t_inference_start, step)

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    all_bleu_matches = per_host_sum_pmap(bleu_matches)
    bleu_score = bleu.complete_bleu(*all_bleu_matches)
    # Save translation samples for tensorboard.
    exemplars = ''
    for n in np.random.choice(np.arange(len(predictions)), 8):
      exemplars += f'{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n'
    if jax.host_id() == 0:
      eval_summary_writer.scalar('bleu', bleu_score, step)
      eval_summary_writer.text('samples', exemplars, step)
      eval_summary_writer.flush()
    logging.info('Translation BLEU Score %.4f', bleu_score)


if __name__ == '__main__':
  app.run(main)

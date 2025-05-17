# Copyright 2024 The Flax Authors.
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

This script trains a Transformer on a LM1B dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import dataclasses
import os

import input_pipeline
import jax
import jax.numpy as jnp
import models
import numpy as np
import optax
import temperature_sampler
import tensorflow as tf
import utils
from absl import logging
from clu import metric_writers, periodic_actions
from configs import default
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from utils import HasCache, TrainState

from flax import nnx
from flax.training import checkpoints, common_utils


def rsqrt_schedule(
  init_value: float,
  shift: int = 0,
):
  """Applies a reverse square-root schedule.

  The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

  Args:
    init_value: Base learning rate (before applying the rsqrt schedule).
    shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
      schedule makes it less steep in the beginning (close to 0).

  Returns:
    A schedule that applies the reverse square root.
  """

  def schedule(count):
    return init_value * (count + shift) ** -0.5 * shift**0.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules(
    [
      optax.linear_schedule(
        init_value=0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
      ),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
    ],
    boundaries=[warmup_steps],
  )


def compute_weighted_cross_entropy(
  logits, targets, weights=None, label_smoothing=0.0
):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
    confidence * jnp.log(confidence)
    + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
  )
  soft_targets = common_utils.onehot(
    targets, vocab_size, on_value=confidence, off_value=low_confidence
  )

  loss = -jnp.sum(soft_targets * nnx.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
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
    raise ValueError(
      'Incorrect shapes. Got shape %s logits and %s targets'
      % (str(logits.shape), str(targets.shape))
    )
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(
    logits, labels, weights, label_smoothing
  )
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
    'loss': loss,
    'accuracy': acc,
    'denominator': weight_sum,
  }
  return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(
  state: TrainState,
  batch,
  learning_rate_fn,
  label_smoothing=0.0,
  dropout_rng=None,
):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ['inputs', 'inputs_position', 'inputs_segmentation']
  (inputs, inputs_positions, inputs_segmentation) = (
    batch.get(k, None) for k in train_keys
  )

  weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    module = nnx.merge(state.graphdef, params)
    module.set_attributes(deterministic=False, decode=False)
    logits = module(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      rngs=nnx.Rngs(dropout=dropout_rng),
    )

    loss, weight_sum = compute_weighted_cross_entropy(
      logits, inputs, weights, label_smoothing
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, inputs, weights)
  metrics['learning_rate'] = lr

  return new_state, metrics


def eval_step(
  params: nnx.State,
  batch,
  graphdef: nnx.GraphDef[models.TransformerLM],
  label_smoothing=0.0,
):
  """Calculate evaluation metrics on a batch."""
  inputs = batch['inputs']
  weights = jnp.where(inputs > 0, 1.0, 0.0)
  module = nnx.merge(graphdef, params)
  module.set_attributes(deterministic=True, decode=False)
  logits = module(inputs)

  return compute_metrics(logits, inputs, weights, label_smoothing)


def predict_step(
  inputs,
  params: nnx.State,
  rngkey: jax.Array,
  graphdef: nnx.GraphDef[models.TransformerLM],
  eos_id: int,
  max_decode_len: int,
  config: models.TransformerConfig,
  temperature: float,
  top_k: int,
):
  """Predict language model on a batch."""
  module = nnx.merge(graphdef, params)

  # TODO(cgarciae): check how pytorch does this.
  for _path, m in module.iter_modules():
    if isinstance(m, HasCache):
      input_shape = (inputs.shape[0], max_decode_len, config.emb_dim)
      m.init_cache(input_shape, dtype=config.dtype)

  graphdef, params, cache = nnx.split(module, nnx.Param, nnx.Cache)

  def tokens_ids_to_logits(flat_ids, cache: nnx.State):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    module = nnx.merge(graphdef, params, cache)
    module.set_attributes(deterministic=True, decode=True)
    logits = module(flat_ids)
    cache = nnx.state(module, nnx.Cache)
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    logits = logits.squeeze(axis=1)
    return logits, cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  seqs = temperature_sampler.temperature_sample(
    inputs,
    cache,
    tokens_ids_to_logits,
    rngkey,
    temperature=temperature,
    topk=top_k,
    eos_token=eos_id,
  )

  return seqs


# Utils for prediction
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def evaluate(
  *,
  jit_eval_step,
  state: TrainState,
  eval_ds: tf.data.Dataset,
  num_eval_steps: int,
):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  eval_metrics = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_batch = jax.tree.map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    metrics = jit_eval_step(state.params, eval_batch, state.graphdef)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.stack_forest(eval_metrics)
  eval_metrics_sums = jax.tree.map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop('denominator')
  eval_summary = jax.tree.map(
    lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
    eval_metrics_sums,
  )
  return eval_summary


def generate_prediction(
  *,
  jit_pred_step,
  graphdef: nnx.GraphDef[models.TransformerLM],
  params: nnx.State,
  tokenized_prompts,
  eos_id,
  inference_rng,
  decode_tokens,
  config: default.Config,
  model_config: models.TransformerConfig,
):
  """Generate text from the prompt."""
  n_devices = jax.local_device_count()

  logging.info('Generating text.')
  predictions = []
  # Use batch of prompts provided by user.
  for pred_batch in jnp.array_split(
    tokenized_prompts, int(np.ceil(len(tokenized_prompts) / n_devices))
  ):
    cur_pred_batch_size = pred_batch.shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      pred_batch = jax.tree.map(
        lambda x: pad_examples(x, padded_size), pred_batch
      )  # pylint: disable=cell-var-from-loop
    pred_batch = common_utils.shard(pred_batch)
    inference_rng, sub_rng = random.split(inference_rng)
    inference_rngs = random.split(sub_rng, n_devices)

    predicted = jit_pred_step(
      pred_batch,
      params,
      inference_rngs,
      graphdef,
      eos_id,
      config.max_predict_length,
      model_config,
      config.sampling_temperature,
      config.sampling_top_k,
    )
    predicted = tohost(predicted)
    # Iterate through non-padding examples of batch.
    for s in predicted[:cur_pred_batch_size]:
      prediction = decode_tokens(s)
      logging.info('Sample: %s', str(prediction))
      predictions.append(prediction)

    # Save generated texts for tensorboard.
    exemplars = ''
    for prediction in predictions:
      exemplars += f'{prediction}\n\n'
  return exemplars


def train_and_evaluate(config: default.Config, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  workdir = os.path.abspath(workdir)
  tf.io.gfile.makedirs(workdir)

  vocab_path = config.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(workdir, 'sentencepiece_model')
    config.vocab_path = vocab_path
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  train_ds, eval_ds, _, encoder = input_pipeline.get_datasets(
    n_devices=jax.local_device_count(), config=config, vocab_path=vocab_path
  )

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = temperature_sampler.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  def encode_strings(strs, max_len):
    tokenized_batch = np.zeros((len(strs), max_len), np.int32)
    for i, s in enumerate(strs):
      toks = encoder.tokenize(s).numpy()
      # Remove EOS token in prompt.
      tokenized_batch[i, : toks.shape[0] - 1] = toks[:-1]
    return tokenized_batch

  tokenized_prompts = encode_strings(
    [config.prompts], config.max_predict_length
  )

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  model_config = models.TransformerConfig(
    vocab_size=vocab_size,
    output_vocab_size=vocab_size,
    logits_via_embedding=config.logits_via_embedding,
    dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
    emb_dim=config.emb_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    qkv_dim=config.qkv_dim,
    mlp_dim=config.mlp_dim,
    max_len=max(config.max_target_length, config.max_eval_target_length),
    dropout_rate=config.dropout_rate,
    attention_dropout_rate=config.attention_dropout_rate,
    kernel_init=nnx.initializers.xavier_uniform(),
    bias_init=nnx.initializers.normal(stddev=1e-6),
    axis_rules=config.axis_rules,
  )

  # Mesh definition
  devices_array = utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  rng, inference_rng = random.split(rng)

  def constructor(config: models.TransformerConfig, key: jax.Array):
    return models.TransformerLM(config, rngs=nnx.Rngs(params=key))

  learning_rate_fn = create_learning_rate_schedule(
    learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  optimizer = optax.adamw(
    learning_rate_fn,
    b1=0.9,
    b2=0.98,
    eps=1e-9,
    weight_decay=config.weight_decay,
  )

  state, state_sharding = utils.setup_initial_state(
    constructor, optimizer, model_config, init_rng, mesh
  )
  data_sharding = NamedSharding(mesh, P(config.data_sharding))

  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    state = checkpoints.restore_checkpoint(workdir, state)
    # Grab last step.
    start_step = int(state.step)

  writer = metric_writers.create_default_writer(
    workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dataclasses.asdict(config))

  # compile multidevice versions of train/eval/predict step fn.
  jit_train_step = jax.jit(
    train_step,
    in_shardings=(
      state_sharding,
      data_sharding,
      None,
    ),  # type: ignore
    out_shardings=(state_sharding, None),  # type: ignore
    static_argnames=("learning_rate_fn", "label_smoothing"),
    donate_argnums=0,
  )

  jit_eval_step = jax.jit(
    eval_step,
    in_shardings=(
      state_sharding.params,
      data_sharding,
    ),  # type: ignore
    out_shardings=None,  # type: ignore
    static_argnames=("graphdef", "label_smoothing"),
  )

  # Since the inputs and rngkey args for predict_step will be batched,
  # we must vmap them, otherwise the global arrays will be seen in each device
  jit_pred_step = jax.jit(
    jax.vmap(
      predict_step,
      in_axes=(
        0,
        jax.tree.map(lambda x: None, state.params),
        0,
        None,
        None,
        None,
        None,
        None,
        None,
      ),
    ),
    in_shardings=(
      data_sharding,
      state_sharding.params,
      data_sharding,
    ),  # type: ignore
    out_shardings=data_sharding,  # type: ignore
    static_argnums=tuple(range(3, 9)),
  )

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = rng

  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
    num_train_steps=config.num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
      report_progress,
      periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
    ]
  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      # Shard data to devices and do a training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = next(train_iter)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        state, metrics = jit_train_step(
          state, batch, learning_rate_fn, 0.0, dropout_rngs
        )
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          train_metrics = common_utils.stack_forest(train_metrics)
          lr = train_metrics.pop('learning_rate').mean()
          metrics_sums = jax.tree.map(jnp.sum, train_metrics)
          denominator = metrics_sums.pop('denominator')
          summary = jax.tree.map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
          summary['learning_rate'] = lr
          summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), max=1.0e4)
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed('eval'):
          eval_results = evaluate(
            jit_eval_step=jit_eval_step,
            state=state,
            eval_ds=eval_ds,
            num_eval_steps=config.num_eval_steps,
          )
          # (clipped) perplexity after averaging log-perplexitie
          eval_results['perplexity'] = jnp.clip(
            jnp.exp(eval_results['loss']), max=1.0e4
          )
          writer.write_scalars(
            step, {'eval_' + k: v for k, v in eval_results.items()}
          )

        with report_progress.timed('generate_text'):
          exemplars = generate_prediction(
            jit_pred_step=jit_pred_step,
            graphdef=state.graphdef,
            params=state.params,
            tokenized_prompts=tokenized_prompts,
            eos_id=eos_id,
            inference_rng=inference_rng,
            decode_tokens=decode_tokens,
            config=config,
            model_config=model_config,
          )
          writer.write_texts(step, {'samples': exemplars})

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
        step % config.checkpoint_every_steps == 0 or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info('Saving checkpoint step %d.', step)
        with report_progress.timed('checkpoint'):
          checkpoints.save_checkpoint_multiprocess(workdir, state, step)

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
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import nnx
import input_pipeline
import sampler as sampler_lib
import tokenizer
import transformer as transformer_lib
import utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  kv: str | None = None
  vocab: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(
        getattr(self, key) if key is not None else None for key in keys
    )


@dataclasses.dataclass(unsafe_hash=True)
class TrainConfig:
  """Configuration for training a gemma model."""

  # Path to load or store sentencepiece vocab file.
  vocab_path: str | None
  # Vocabulary size if `vocab_path` is not given.
  vocab_size: int
  # Maximum number of characters to use for training.
  max_corpus_chars: int
  # Name of TFDS translation dataset to use.
  dataset_name: str
  # Optional name of TFDS translation dataset to use for evaluation.
  eval_dataset_name: str
  # Optional name of TFDS split to use for evaluation.
  eval_split: str
  # Per device batch size for training.
  per_device_batch_size: int
  # Per device batch size for training.
  eval_per_device_batch_size: int

  # Prompt for language model sampling
  prompts: tuple[str, ...]
  # Temperature for top_p sampling.
  sampling_temperature: float
  # Top-p sampling threshold.
  sampling_top_p: float

  # Number of steps to take during training.
  num_train_steps: int
  # Number of steps to take during evaluation.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  num_eval_steps: int
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int
  # Base learning rate.
  learning_rate: float
  # Linear learning rate warmup.
  warmup_steps: int
  # Cross entropy loss label smoothing.
  label_smoothing: float
  # Decay factor for AdamW style weight decay.
  weight_decay: float
  # Maximum length cutoff for training examples.
  max_target_length: int
  # Maximum length cutoff for eval examples.
  max_eval_target_length: int

  # Gemma transformer name.
  # Possible values defined in transformer.TransformerConfig:
  # (gemma_2b, gemma_7b, gemma2_2b, gemma2_9b, gemma2_27b, gemma3_1b, gemma3_4b,
  # ...)
  transformer_name: str | None
  # or alternatively define the model using the dict of parameters
  transformer_params: dict[Any, Any] | None

  # Whether to save model checkpoints.
  save_checkpoints: bool
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool
  # Integer for PRNG random seed.
  seed: int

  # Parallelism
  mesh_axes: tuple[str, ...]
  axis_rules: MeshRules
  data_sharding: tuple[str, ...]

  # One axis for each parallelism type may hold a placeholder (-1)
  # value to auto-shard based on available slices and devices.
  # By default, product of the DCN axes should equal number of slices
  # and product of the ICI axes should equal number of devices per slice.
  # ICI (Inter-Chip Interconnection): A high-speed connection between
  # sets of TPU chips, which form the TPU network.
  # DCN (Data Center Network): A connection between the TPU networks;
  # not as fast as ICI.
  # ICI has around 100x the bandwidth of DCN, but it is not a general
  # purpose connection, which is why DCN is necessary for scaling to
  # extremely large ML models.
  dcn_data_parallelism: int = -1
  dcn_fsdp_parallelism: int = 1
  dcn_tensor_parallelism: int = 1
  ici_data_parallelism: int = 1
  ici_fsdp_parallelism: int = -1
  ici_tensor_parallelism: int = 1

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def __post_init__(self):
    if isinstance(self.axis_rules, dict):
      self.axis_rules = MeshRules(**self.axis_rules)


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
    state: utils.TrainState,
    batch,
    learning_rate_fn,
    label_smoothing=0.0,
):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ['inputs', 'inputs_position', 'inputs_segmentation', 'targets']
  (inputs, inputs_positions, inputs_segmentation, targets) = (
      batch.get(k, None) for k in train_keys
  )

  # TODO: this should be defined globally
  pad_id = 0
  weights = jnp.where(inputs > pad_id, 1, 0).astype(jnp.float32)
  input_mask = inputs > pad_id
  attention_mask = transformer_lib.make_causal_attn_mask(
      input_mask
  )  # (B, L, L)
  # inputs_segmentation: (B, L)
  mask = (
      inputs_segmentation[:, :, None] == inputs_segmentation[:, None, :]
  )  # (B, L, L)
  attention_mask = jnp.logical_and(mask, attention_mask)

  def loss_fn(params):
    """loss function used for training."""
    module = nnx.merge(state.graphdef, params)

    logits, _ = module(
        inputs,
        positions=inputs_positions,
        attention_mask=attention_mask,
        cache=None,
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr

  return new_state, metrics


def eval_step(
    params: nnx.State,
    batch,
    graphdef: nnx.GraphDef[transformer_lib.Transformer],
    label_smoothing=0.0,
):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']

  # TODO: this should be defined globally
  pad_id = 0
  weights = jnp.where(inputs > pad_id, 1, 0).astype(jnp.float32)
  input_mask = inputs > pad_id
  inputs_positions = transformer_lib.build_positions_from_mask(input_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(input_mask)

  module = nnx.merge(graphdef, params)
  logits, _ = module(
      inputs,
      positions=inputs_positions,
      attention_mask=attention_mask,
      cache=None,
  )

  return compute_metrics(logits, targets, weights, label_smoothing)


def evaluate(
    *,
    jit_eval_step,
    state: utils.TrainState,
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


def train_and_evaluate(config: TrainConfig, workdir: str):
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
  train_ds, eval_ds, encoder = input_pipeline.get_datasets(
      n_devices=jax.local_device_count(), config=config, vocab_path=vocab_path
  )

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  if config.transformer_name is not None:
    model_config = transformer_lib.TransformerConfig.from_version_name(
        config.transformer_name,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )
  else:
    assert config.transformer_params is not None
    model_config = transformer_lib.TransformerConfig.from_dict(
        **config.transformer_params,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )

  # Mesh definition
  devices_array = utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  _, inference_rng = random.split(rng)

  def constructor(config: transformer_lib.TransformerConfig, key: jax.Array):
    return transformer_lib.Transformer(config, rngs=nnx.Rngs(params=key))

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
  data_sharding = jax.NamedSharding(mesh, jax.P(config.data_sharding))

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
      ),  # type: ignore
      out_shardings=(state_sharding, None),  # type: ignore
      static_argnames=('learning_rate_fn', 'label_smoothing'),
      donate_argnums=0,
  )

  jit_eval_step = jax.jit(
      eval_step,
      in_shardings=(
          state_sharding.params,
          data_sharding,
      ),  # type: ignore
      out_shardings=None,  # type: ignore
      static_argnames=('graphdef', 'label_smoothing'),
  )

  vocab = tokenizer.load_sentencepiece_processor(vocab_path)
  sampler = sampler_lib.Sampler(
      transformer=nnx.merge(state.graphdef, state.params),
      vocab=vocab,
      cache_size=1024,
  )

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
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
        with report_progress.timed('data'):
          batch = next(train_iter)
          batch = jax.tree.map(
              lambda x: jnp.asarray(x, device=data_sharding), batch
          )

        with report_progress.timed('train_step'):
          state, metrics = jit_train_step(state, batch, learning_rate_fn, 0.0)
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      # Write batch loss and lr every step to TB
      # without overwhelming the stdout:
      if jax.process_index() == 0:
        tb_writer = writer._writers[-1]  # pylint: disable=protected-access
        lr = train_metrics[-1]['learning_rate']
        train_batch_loss = train_metrics[-1]['loss']
        denominator = train_metrics[-1]['denominator']
        tb_writer.write_scalars(
            step,
            {
                'train_learning_rate': lr,
                'train_loss': train_batch_loss / denominator,
            },
        )

      # Periodic metric handling.
      if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          train_metrics = common_utils.stack_forest(train_metrics)
          # Remove learning_rate from the summary
          _ = train_metrics.pop('learning_rate')
          metrics_sums = jax.tree.map(jnp.sum, train_metrics)
          denominator = metrics_sums.pop('denominator')
          summary = jax.tree.map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
          summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), max=1.0e4)
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed('generate_text'):
          # update sampler's transformer state:
          sampler.transformer_state = state.params
          exemplars = sampler(
              config.prompts,
              total_generation_steps=config.num_predict_steps,
              temperature=config.sampling_temperature,
              top_p=config.sampling_top_p,
              seed=inference_rng,
              echo=True,
          )
          writer.write_texts(step, {'samples': exemplars.text[0]})

        with report_progress.timed('eval'):
          eval_results = evaluate(
              jit_eval_step=jit_eval_step,
              state=state,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
          )
          # (clipped) perplexity after averaging log-perplexity
          eval_results['perplexity'] = jnp.clip(
              jnp.exp(eval_results['loss']), max=1.0e4
          )
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()}
          )

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step % config.checkpoint_every_steps == 0 or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info('Saving checkpoint step %d.', step)
        with report_progress.timed('checkpoint'):
          checkpoints.save_checkpoint_multiprocess(workdir, state, step)

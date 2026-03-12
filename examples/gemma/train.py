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
import contextlib
import dataclasses
from pathlib import Path

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import nnx
import input_pipeline
import grain
import jax
import jax.numpy as jnp
import tokenizer
import transformer as transformer_lib
import numpy as np
import optax
import orbax.checkpoint as ocp
import sampler as sampler_lib
from typing import Any, Literal
from jax.sharding import NamedSharding
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint.path import atomicity


@dataclasses.dataclass(slots=True)
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
  # Grain prefetch number of workers.
  prefetch_num_workers: int | None

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
  data_sharding: tuple[str | tuple[str], ...]

  fsdp_parallelism: int = -1
  tensor_parallelism: int = 1

  # Profiling
  with_profiler_step_trace: bool = False

  # Dataflow choice: grain or TF tensor
  input_pipeline_type: Literal["grain", "tf"] = "grain"

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def __post_init__(self):
    axis_shapes = [self.fsdp_parallelism, self.tensor_parallelism]
    assert axis_shapes.count(-1) in (0, 1), (
      f'Found unspecified values (-1) for more than one parallelism axis. '
      'At most one axis can be unspecified.'
    )

  def get_mesh_shape(self, num_devices: int) -> tuple[int, int]:
    axis_shapes = [self.fsdp_parallelism, self.tensor_parallelism]
    count = np.prod(axis_shapes)
    if count < 0:
      axis_shapes[axis_shapes.index(-1)] = int(num_devices / (-count))
    else:
      assert count == num_devices
    return tuple(axis_shapes)


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
  soft_targets = jax.nn.one_hot(targets, vocab_size) * (confidence - low_confidence) + low_confidence

  loss = -jnp.sum(soft_targets * nnx.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  # loss shape is [B, T]
  loss_per_sample = loss.sum(axis=1) * len(loss) / normalizing_factor
  return loss_per_sample


@jax.jit
def compute_perplexity(loss):
  return jnp.clip(jnp.exp(loss), max=1.0e4)


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch: dict[str, jax.Array],
    train_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> tuple[nnx.Module, nnx.Optimizer, nnx.Rngs, nnx.Metric]:
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # If such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ['inputs', 'inputs_position', 'inputs_segmentation', 'targets']
  (inputs, inputs_positions, inputs_segmentation, targets) = (
      batch.get(k, None) for k in train_keys
  )

  weights = jnp.where(inputs > pad_id, 1, 0).astype(jnp.float32)
  input_mask = inputs > pad_id
  attention_mask = transformer_lib.make_causal_attn_mask(input_mask)  # (B, L, L)
  if inputs_segmentation is not None:
    # inputs_segmentation: (B, L)
    mask = inputs_segmentation[:, :, None] == inputs_segmentation[:, None, :]  # (B, L, L)
    attention_mask = jnp.logical_and(mask, attention_mask)

  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, rngs):
    """loss function used for training."""
    module = nnx.merge(graphdef, params, nondiff)

    logits, _ = module(
        inputs,
        positions=inputs_positions,
        attention_mask=attention_mask,
        cache=None,
        rngs=rngs,
    )

    loss_per_sample = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss_per_sample.mean()
    return mean_loss, (loss_per_sample, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss_per_sample, logits)), grads = grad_fn(params, rngs.fork())
  optimizer.update(model, grads)

  # Apply pad mask on logits and targets for metrics computation
  logits = logits * weights[..., None]
  targets = targets * weights.astype(int)
  train_metrics.update(loss=loss_per_sample, logits=logits, labels=targets)
  return model, optimizer, rngs, train_metrics


def eval_step(
    model,
    batch,
    eval_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> nnx.Metric:
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']

  weights = jnp.where(inputs > pad_id, 1, 0).astype(jnp.float32)
  input_mask = inputs > pad_id
  inputs_positions = transformer_lib.build_positions_from_mask(input_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(input_mask)

  logits, _ = model(
      inputs,
      positions=inputs_positions,
      attention_mask=attention_mask,
      cache=None,
  )
  loss_per_sample = compute_weighted_cross_entropy(
    logits, targets, weights, label_smoothing
  )

  # Apply pad mask on logits and targets for metrics computation
  logits = logits * weights[..., None]
  targets = targets * weights.astype(int)
  eval_metrics.update(loss=loss_per_sample, logits=logits, labels=targets)
  return eval_metrics


def evaluate(
    *,
    jit_eval_step,
    model: nnx.Module,
    eval_ds: grain.IterDataset,
    num_eval_steps: int,
    eval_metrics: nnx.Metric,
):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_metrics = jit_eval_step(model, eval_batch, eval_metrics)

  result = eval_metrics.compute()
  if isinstance(result, dict) and "loss" in result:
    result["perplexity"] = compute_perplexity(result["loss"])

  return result


def inspect_sharding(x):
    info = x.sharding.devices_indices_map(tuple(x.shape))
    for key, value in info.items():
        logging.debug(f" - Device {key.id}: {value}")


def train_and_evaluate(config: TrainConfig, workdir: str, chpt_bucket: str | None = None):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  workdir = Path(workdir).absolute().resolve()
  workdir.mkdir(parents=True, exist_ok=True)

  if config.vocab_path is None:
    config.vocab_path = str(workdir / "sentencepiece_model")
  vocab_path = Path(config.vocab_path).absolute().resolve()
  vocab_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint_path = workdir / "checkpoints" if chpt_bucket is None else chpt_bucket

  workdir, vocab_path = str(workdir), str(vocab_path)

  # Mesh definition with default explicit axis types
  mesh = jax.make_mesh(config.get_mesh_shape(len(jax.devices())), config.mesh_axes)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  data_sharding = NamedSharding(mesh, jax.P(config.data_sharding))

  train_ds, eval_ds, encoder = input_pipeline.get_datasets(
      config=config, vocab_path=vocab_path, data_sharding=data_sharding
  )

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  # Activations dtype for mixed precision, weights dtype is float32
  dtype = jnp.bfloat16 if config.use_bfloat16 else jnp.float32
  shd_config=transformer_lib.ShardingConfig.get_default_sharding(
      fsdp_axis_name="fsdp", tensor_parallel_axis_name="tensor"
  )
  if config.transformer_name is not None:
    model_config = transformer_lib.TransformerConfig.from_version_name(
        config.transformer_name,
        num_embed=vocab_size,
        dtype=dtype,
        shd_config=shd_config,
    )
  else:
    assert config.transformer_params is not None
    model_config = transformer_lib.TransformerConfig.from_dict(
        **config.transformer_params,
        num_embed=vocab_size,
        dtype=dtype,
        shd_config=shd_config,
    )

  start_step = 0
  learning_rate_fn = create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  rngs = nnx.Rngs(params=config.seed, dropout=config.seed)

  with jax.set_mesh(mesh):
    model = transformer_lib.Transformer(model_config, rngs=rngs)
    optimizer = nnx.Optimizer(
        model,
        tx=optax.adamw(
            learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=config.weight_decay,
        ),
        wrt=nnx.Param,
    )

  checkpoint_mngr = ocp.CheckpointManager(
    checkpoint_path,
    options=ocp.CheckpointManagerOptions(
      preservation_policy=preservation_policy_lib.LatestN(1),
      temporary_path_class=atomicity.CommitFileTemporaryPath
    )
  )

  if config.restore_checkpoints and checkpoint_mngr.latest_step() is not None:
    # Restore unreplicated optimizer + model state from last checkpoint.
    target = {
      "model": nnx.state(model),
      "optimizer": nnx.state(optimizer),
      "step": 0,
    }
    checkpoint = checkpoint_mngr.restore(
      checkpoint_mngr.latest_step(),
      args=ocp.args.StandardRestore(target),
    )
    nnx.update(model, checkpoint["model"])
    nnx.update(optimizer, checkpoint["optimizer"])
    start_step = checkpoint["step"] + 1  # Add +1 to skip saving again the same step

  # check sharding:
  flat_state = nnx.to_flat_state(nnx.state(model))
  B = 1_000_000_000
  num_params = {str(key): param.size for key, param in flat_state}
  embed_num_params = sum([value for key, value in num_params.items() if "embed" in key])
  attn_num_params = sum([value for key, value in num_params.items() if "attn" in key])
  mlp_num_params = sum([value for key, value in num_params.items() if "mlp" in key])
  total_num_params = sum([value for _, value in num_params.items()])
  logging.info(
    "\nModel Number of Parameters:\n"
    f"- Total (B): {total_num_params / B}\n"
    f"- Embedding (B): {embed_num_params / B}\n"
    f"- Attentions (B): {attn_num_params / B}\n"
    f"- MLPs (B): {mlp_num_params / B}\n"
  )
  logging.debug("--- Model shardings:")
  for key, param in flat_state:
      logging.debug(f"-- {key} --")
      sharding_names = param.sharding_names if hasattr(param, 'sharding_names') else 'no sharding'
      logging.debug(f"- {param.shape} | {param.dtype}  {sharding_names}")
      inspect_sharding(param[...])

  logging.debug("--- Optimizer shardings:")
  flat_state = nnx.to_flat_state(nnx.state(optimizer))
  for key, param in flat_state:
      logging.debug(f"-- {key} --")
      sharding_names = param.sharding_names if hasattr(param, 'sharding_names') else 'no sharding'
      logging.debug(f"- {param.shape} | {param.dtype}  {sharding_names}")
      inspect_sharding(param[...])

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dataclasses.asdict(config))

  train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
  )
  eval_metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average('loss'),
      accuracy=nnx.metrics.Accuracy(),
  )

  jit_train_step = jax.jit(
      train_step,
      static_argnames=("label_smoothing", "pad_id"),
      donate_argnames=("model", "optimizer"),
  )

  jit_eval_step = jax.jit(
      eval_step,
      static_argnames=("label_smoothing", "pad_id"),
  )

  vocab = tokenizer.load_sentencepiece_processor(vocab_path)
  sampler =  sampler_lib.Sampler(vocab=vocab, cache_size=1024)

  # Main Train Loop
  # ---------------------------------------------------------------------------
  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=10),
    ]
  with metric_writers.ensure_flushes(writer), jax.set_mesh(mesh):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      maybe_profiler_step_trace = (
        jax.profiler.StepTraceAnnotation('train', step_num=step)
        if config.with_profiler_step_trace else
        contextlib.suppress()
      )

      with maybe_profiler_step_trace:
        with report_progress.timed('data'):
          batch = next(train_iter)

        with report_progress.timed('train_step'):
          model, optimizer, rngs, train_metrics = jit_train_step(
              model,
              optimizer,
              rngs,
              batch,
              train_metrics,
              0.0,  # label_smoothing
              encoder.pad_id(),  # pad_id
          )

      # Quick indication that training is happening.
      if step < 20:
        logging.info(
          "Finished training step %d. Batch size: %d, Loss: %.5f, LR: %.5f",
          step,
          len(batch['inputs']),
          train_metrics.compute()["loss"],
          learning_rate_fn(step + 1),
        )
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          summary = train_metrics.compute()
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)

        with report_progress.timed('generate_text'):
          model.eval()
          # update sampler's transformer state:
          exemplars = sampler(
              config.prompts,
              total_generation_steps=config.num_predict_steps,
              temperature=config.sampling_temperature,
              top_p=config.sampling_top_p,
              seed=config.seed,
              echo=True,
              dtype=dtype,
              transformer=model,
              data_sharding=data_sharding,
          )
          writer.write_texts(step, {'samples': exemplars.text})

        with report_progress.timed('eval'):
          eval_results = evaluate(
              jit_eval_step=jit_eval_step,
              model=model,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
              eval_metrics=eval_metrics,
          )
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()}
          )
          model.train()

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          (step > 0 and step % config.checkpoint_every_steps == 0) or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info('Saving checkpoint step %d.', step)
        with report_progress.timed('checkpoint'):
          checkpoint = {
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
            "step": step,
          }
          checkpoint_mngr.save(step, args=ocp.args.StandardSave(checkpoint))

  checkpoint_mngr.wait_until_finished()
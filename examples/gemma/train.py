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

This script trains a Gemma transformer on the LM1B dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error
import contextlib
import dataclasses
import time
from functools import partial
from pathlib import Path
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import nnx
import input_pipeline
import sampler as sampler_lib
import sow_lib
import tokenizer
import transformer as transformer_lib
from train_cfg import TrainConfig
from transformer_cfg import ShardingConfig
import grain
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint.path import atomicity


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
  soft_targets = (
      jax.nn.one_hot(targets, vocab_size) * (confidence - low_confidence)
      + low_confidence
  )

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
def jax_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch: dict[str, jax.Array],
    train_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
    with_capture: bool = False,
) -> tuple[nnx.State, nnx.State | None]:
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # If such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ["inputs", "inputs_position", "inputs_segmentation", "targets"]
  inputs, inputs_positions, inputs_segmentation, targets = (
      batch.get(k, None) for k in train_keys
  )

  input_mask = inputs > pad_id
  weights = input_mask.astype(jnp.float32)
  attention_mask = transformer_lib.make_causal_attn_mask(
      input_mask
  )  # (B, L, L)
  if inputs_segmentation is not None:
    # inputs_segmentation: (B, L)
    mask = (
        inputs_segmentation[:, :, None] == inputs_segmentation[:, None, :]
    )  # (B, L, L)
    attention_mask = jnp.logical_and(mask, attention_mask)

  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, rngs):
    """loss function used for training."""
    module = nnx.merge(graphdef, params, nondiff)
    if with_capture:
      forward = nnx.capture(module, nnx.Intermediate)
    else:
      forward = module
    output = forward(
        inputs,
        positions=inputs_positions,
        attention_mask=attention_mask,
        cache=None,
        rngs=rngs,
    )
    # output is (preds, cache) if with_capture=False
    # and is ((preds, cache), intermediates) if with_capture=True
    if with_capture:
      (logits, _), intermediates = output
    else:
      logits, _ = output
      intermediates = None

    loss_per_sample = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss_per_sample.mean()
    return mean_loss, (loss_per_sample, logits, intermediates)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss_per_sample, logits, intermediates)), grads = grad_fn(params, rngs.fork())
  optimizer.update(model, grads)

  # Apply pad mask on logits and targets for metrics computation
  train_metrics.update(
      loss=loss_per_sample,
      logits=logits,
      labels=targets,
      mask={"accuracy": input_mask},
  )
  return nnx.state((model, optimizer, rngs, train_metrics)), intermediates


def nnx_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch: dict[str, jax.Array],
    train_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
    with_capture: bool = False,
) -> nnx.State | None:
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

  input_mask = inputs > pad_id
  weights = input_mask.astype(jnp.float32)
  attention_mask = transformer_lib.make_causal_attn_mask(
      input_mask
  )  # (B, L, L)
  if inputs_segmentation is not None:
    # inputs_segmentation: (B, L)
    mask = (
        inputs_segmentation[:, :, None] == inputs_segmentation[:, None, :]
    )  # (B, L, L)
    attention_mask = jnp.logical_and(mask, attention_mask)

  def loss_fn(model, rngs):
    """loss function used for training."""
    if with_capture:
      forward = nnx.capture(model, nnx.Intermediate)
    else:
      forward = model
    output = forward(
        inputs,
        positions=inputs_positions,
        attention_mask=attention_mask,
        cache=None,
        rngs=rngs,
    )
    # output is (preds, cache) if with_capture=False
    # and is ((preds, cache), intermediates) if with_capture=True
    if with_capture:
      (logits, _), intermediates = output
    else:
      logits, _ = output
      intermediates = None

    loss_per_sample = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss_per_sample.mean()
    return mean_loss, (loss_per_sample, logits, intermediates)

  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (_, (loss_per_sample, logits, intermediates)), grads = grad_fn(model, rngs.fork())
  optimizer.update(model, grads)

  # Apply pad mask on logits and targets for metrics computation
  train_metrics.update(
      loss=loss_per_sample,
      logits=logits,
      labels=targets,
      mask={"accuracy": input_mask},
  )
  return intermediates


def eval_step(
    model,
    batch,
    eval_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> nnx.Metric:
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']

  input_mask = inputs > pad_id
  weights = input_mask.astype(jnp.float32)
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
  eval_metrics.update(
      loss=loss_per_sample,
      logits=logits,
      labels=targets,
      mask={"accuracy": input_mask},
  )
  return eval_metrics


def jax_eval_step(
    model,
    batch,
    eval_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> nnx.Metric:
  return eval_step(model, batch, eval_metrics, label_smoothing, pad_id)


def nnx_eval_step(
    model,
    batch,
    eval_metrics: nnx.Metric,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> None:
  eval_step(model, batch, eval_metrics, label_smoothing, pad_id)


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
    output = jit_eval_step(model, eval_batch, eval_metrics)
    if output is not None:
      eval_metrics = output

  result = eval_metrics.compute()
  if isinstance(result, dict) and "loss" in result:
    result["perplexity"] = compute_perplexity(result["loss"])

  return result


def inspect_sharding(x):
  info = x.sharding.devices_indices_map(tuple(x.shape))
  for key, value in info.items():
    logging.debug(f" - Device {key.id}: {value}")


def _filter_intermediates(x):
  # assume single host so we skip allgather from all hosts
  # by converting to np we remove all shardings
  if jnp.isdtype(x, "real floating"):
    x = x.astype(jnp.float32)
  x = np.asarray(x)
  x = x[0, 0, ...].reshape(-1)
  if len(x) < 10:
    return x
  if np.isdtype(x.dtype, "integral"):
    return x.min().item(), x.max().item()
  # Mask -inf for attn masks
  x = x[x > -1e10]
  if len(x) > 0:
    return x.min().item(), x.mean().item(), x.max().item()
  return x


def display_intermediates(intermediates, placeholder=None):
  if placeholder is not None:
    placeholder = jax.tree.map(lambda p: _filter_intermediates(p), placeholder)
    placeholder = nnx.to_flat_state(placeholder)
    for k, v in placeholder:
      logging.info(f"{k}, stats: {v.get_value()[0]}")

  return intermediates


def train_and_evaluate(
    config: TrainConfig, workdir: str, chpt_bucket: str | None = None
):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  if config.use_nnx_tree_mode:
    logging.info("Set NNX tree-mode")
    flax.config.update("nnx_graph_mode", False)

  if config.use_nnx_transforms in ("all", "jit-only", "grad-only"):
    logging.info(f"Use NNX transforms: {config.use_nnx_transforms}")

  workdir = Path(workdir).absolute().resolve()
  workdir.mkdir(parents=True, exist_ok=True)

  if config.vocab_path is None:
    config.vocab_path = str(workdir / "sentencepiece_model")
  vocab_path = Path(config.vocab_path).absolute().resolve()
  vocab_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint_path = (
      workdir / "checkpoints" if chpt_bucket is None else chpt_bucket
  )

  workdir, vocab_path = str(workdir), str(vocab_path)

  # Mesh definition with default explicit axis types
  mesh = jax.make_mesh(
      config.get_mesh_shape(len(jax.devices())), config.mesh_axes
  )
  logging.info(f"Using mesh: {mesh}")

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
  shd_config = ShardingConfig.fsdp_tp_sharding(
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
    model = transformer_lib.Transformer(
        model_config,
        rngs=rngs,
        sow_config=sow_lib.SowConfig(**config.sow_config),
    )
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
    eval_model = nnx.view(model, deterministic=True)

  checkpoint_mngr = ocp.CheckpointManager(
      checkpoint_path,
      options=ocp.CheckpointManagerOptions(
          preservation_policy=preservation_policy_lib.LatestN(1),
          temporary_path_class=atomicity.CommitFileTemporaryPath,
      ),
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
    start_step = (
        checkpoint["step"] + 1
    )  # Add +1 to skip saving again the same step

  # check sharding:
  flat_state = nnx.to_flat_state(nnx.state(model))
  B = 1_000_000_000
  num_params = {str(key): param.size for key, param in flat_state}
  embed_num_params = sum(
      [value for key, value in num_params.items() if "embed" in key]
  )
  attn_num_params = sum(
      [value for key, value in num_params.items() if "attn" in key]
  )
  mlp_num_params = sum(
      [value for key, value in num_params.items() if "mlp" in key]
  )
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
    out_sharding = (
        param.out_sharding if hasattr(param, "out_sharding") else "no sharding"
    )
    logging.debug(f"- {param.shape} | {param.dtype}  {out_sharding}")
    inspect_sharding(param[...])

  logging.debug("--- Optimizer shardings:")
  flat_state = nnx.to_flat_state(nnx.state(optimizer))
  for key, param in flat_state:
    logging.debug(f"-- {key} --")
    out_sharding = (
        param.out_sharding if hasattr(param, "out_sharding") else "no sharding"
    )
    logging.debug(f"- {param.shape} | {param.dtype}  {out_sharding}")
    inspect_sharding(param[...])

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dataclasses.asdict(config))

  train_metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average("loss"),
      accuracy=nnx.metrics.Accuracy(),
  )
  eval_metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average("loss"),
      accuracy=nnx.metrics.Accuracy(),
  )

  nnx_train_step_ = partial(nnx_train_step, with_capture=config.sow_config is not None)
  jax_train_step_ = partial(jax_train_step, with_capture=config.sow_config is not None)

  jit_fn = nnx.jit if config.use_nnx_transforms in ("all", "jit-only") else jax.jit
  jit_train_step = jit_fn(
      nnx_train_step_ if config.use_nnx_transforms in ("all", "grad-only") else jax_train_step_,
      static_argnames=("label_smoothing", "pad_id"),
      donate_argnames=("model", "optimizer"),
  )
  jit_eval_step = jit_fn(
      nnx_eval_step
      if config.use_nnx_transforms in ("all", "grad-only")
      else jax_eval_step,
      static_argnames=("label_smoothing", "pad_id"),
  )

  vocab = tokenizer.load_sentencepiece_processor(vocab_path)
  sampler = sampler_lib.Sampler(vocab=vocab, cache_size=1024)

  # Main Train Loop
  # ---------------------------------------------------------------------------
  logging.info('Starting training loop.')
  prev_intermediates = None   # for async display of intermediates
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
    started = time.perf_counter()
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      maybe_profiler_step_trace = (
          jax.profiler.StepTraceAnnotation("train", step_num=step)
          if config.with_profiler_step_trace
          else contextlib.suppress()
      )

      with maybe_profiler_step_trace:
        with report_progress.timed('data'):
          batch = next(train_iter)

        with report_progress.timed('train_step'):
          output = jit_train_step(
              model,
              optimizer,
              rngs,
              batch,
              train_metrics,
              0.0,  # label_smoothing
              encoder.pad_id(),  # pad_id
          )

          if isinstance(output, tuple):
            updates, intermediates = output
            nnx.update((model, optimizer, rngs, train_metrics), updates)
          else:
            intermediates = output

      # Quick indication that training is happening.
      if step < 20:
        logging.info(
            "Finished training step %d. Batch size: %d, Loss: %.5f, LR: %.5f",
            step,
            len(batch["inputs"]),
            train_metrics.compute()["loss"],
            learning_rate_fn(step + 1),
        )
      for h in hooks:
        h(step)

      # Display intermediates every 100 iters
      if step % 100 == 0:
        prev_intermediates = display_intermediates(intermediates, prev_intermediates)

      # Periodic metric handling.
      if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          summary = train_metrics.compute()
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)

        with report_progress.timed('generate_text'):
          # update sampler's transformer state:
          exemplars = sampler(
              config.prompts,
              total_generation_steps=config.num_predict_steps,
              temperature=config.sampling_temperature,
              top_p=config.sampling_top_p,
              seed=jnp.array(config.seed),
              echo=True,
              dtype=dtype,
              transformer=eval_model,
              data_sharding=data_sharding,
          )
          writer.write_texts(step, {"samples": exemplars.text})  # pytype: disable=wrong-arg-types

        with report_progress.timed('eval'):
          eval_results = evaluate(
              jit_eval_step=jit_eval_step,
              model=eval_model,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
              eval_metrics=eval_metrics,
          )
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()}
          )

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step > 0 and step % config.checkpoint_every_steps == 0
      ) or is_last_step
      if config.save_checkpoints and save_checkpoint:
        logging.info('Saving checkpoint step %d.', step)
        with report_progress.timed('checkpoint'):
          checkpoint = {
              "model": nnx.state(model),
              "optimizer": nnx.state(optimizer),
              "step": step,
          }
          checkpoint_mngr.save(step, args=ocp.args.StandardSave(checkpoint))

    elapsed = time.perf_counter() - started
    logging.info(f"Total training loop time: {elapsed / 60} minutes")
  checkpoint_mngr.wait_until_finished()

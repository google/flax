# Copyright 2022 The Flax Authors.
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

import collections
import functools
import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

import input_pipeline
import models
import temperature_sampler


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
    return init_value * (count + shift)**-.5 * shift**.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0, end_value=learning_rate, transition_steps=warmup_steps),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
  ],
                              boundaries=[warmup_steps])


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.0):
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
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
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
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
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
      "loss": loss,
      "accuracy": acc,
      "denominator": weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name="batch")
  return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(state,
               batch,
               config,
               learning_rate_fn,
               label_smoothing=0.0,
               dropout_rng=None):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = ["inputs", "inputs_position", "inputs_segmentation"]
  (inputs, inputs_positions, inputs_segmentation
   ) = (batch.get(k, None) for k in train_keys)

  weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.TransformerLM(config).apply(
        {"params": params},
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        rngs={"dropout": dropout_rng})

    loss, weight_sum = compute_weighted_cross_entropy(logits, inputs, weights,
                                                      label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, inputs, weights)
  metrics["learning_rate"] = lr

  return new_state, metrics


def eval_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs = batch["inputs"]
  weights = jnp.where(inputs > 0, 1.0, 0.0)
  logits = models.TransformerLM(config).apply({"params": params}, inputs)

  return compute_metrics(logits, inputs, weights, label_smoothing)


def predict_step(inputs,
                 params,
                 rngkey,
                 eos_id,
                 max_decode_len,
                 config,
                 temperature,
                 top_k):
  """Predict language model on a batch."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = models.TransformerLM(config).init(
      jax.random.PRNGKey(0),
      jnp.ones(target_shape, config.dtype))
  cache = initial_variables["cache"]

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = models.TransformerLM(config).apply(
        {
            "params": params,
            "cache": flat_cache
        },
        flat_ids,
        mutable=["cache"])
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  seqs = temperature_sampler.temperature_sample(
      inputs,
      cache,
      tokens_ids_to_logits,
      rngkey,
      temperature=temperature,
      topk=top_k,
      eos_token=eos_id)

  return seqs


# Utils for prediction
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree"s leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.process_index].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

  def pre_pmap(xs):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)

  def post_pmap(xs):
    return jax.tree_util.tree_map(lambda x: x[0], xs)

  return post_pmap(host_psum(pre_pmap(in_tree)))


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def evaluate(*, p_eval_step, params, eval_ds: tf.data.Dataset,
             num_eval_steps: int):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info("Gathering evaluation metrics.")
  eval_metrics = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_batch = jax.tree_util.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(params, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop("denominator")
  eval_summary = jax.tree_util.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  return eval_summary


def generate_prediction(*, p_pred_step, params,
                        tokenized_prompts,
                        eos_id,
                        inference_rng,
                        decode_tokens,
                        max_predict_length: int):
  """Generate text from the prompt."""
  n_devices = jax.local_device_count()

  logging.info("Generating text.")
  predictions = []
  # Use batch of prompts provided by user.
  for pred_batch in jnp.array_split(
      tokenized_prompts, int(np.ceil(len(tokenized_prompts) / n_devices))):
    cur_pred_batch_size = pred_batch.shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(
          np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      pred_batch = jax.tree_util.tree_map(
          lambda x: pad_examples(x, padded_size), pred_batch)  # pylint: disable=cell-var-from-loop
    pred_batch = common_utils.shard(pred_batch)
    inference_rng, sub_rng = random.split(inference_rng)
    inference_rngs = random.split(sub_rng, n_devices)

    predicted = p_pred_step(pred_batch, params, inference_rngs,
                            eos_id, max_predict_length)
    predicted = tohost(predicted)
    # Iterate through non-padding examples of batch.
    for s in predicted[:cur_pred_batch_size]:
      prediction = decode_tokens(s)
      logging.info("Sample: %s", str(prediction))
      predictions.append(prediction)

    # Save generated texts for tensorboard.
    exemplars = ""
    for prediction in predictions:
      exemplars += f"{prediction}\n\n"
  return exemplars


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)

  vocab_path = config.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(workdir, "sentencepiece_model")
    config.vocab_path = vocab_path
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info("Initializing dataset.")
  train_ds, eval_ds, _, encoder = input_pipeline.get_datasets(
      n_devices=jax.local_device_count(),
      config=config,
      vocab_path=vocab_path)

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = temperature_sampler.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode("utf-8")

  def encode_strings(strs, max_len):
    tokenized_batch = np.zeros((len(strs), max_len), np.int32)
    for i, s in enumerate(strs):
      toks = encoder.tokenize(s).numpy()
      # Remove EOS token in prompt.
      tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
    return tokenized_batch

  tokenized_prompts = encode_strings(
      [config.prompts], config.max_predict_length)

  logging.info("Initializing model, optimizer, and step functions.")
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
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
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  rng, inference_rng = random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_target_length)

  m = models.TransformerLM(eval_config)
  initial_variables = jax.jit(m.init)(init_rng,
                                      jnp.ones(input_shape, jnp.float32))

  learning_rate_fn = create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps)

  optimizer = optax.adamw(
      learning_rate_fn, b1=0.9, b2=0.98, eps=1e-9,
      weight_decay=config.weight_decay
      )
  state = train_state.TrainState.create(
      apply_fn=m.apply,
      params=initial_variables["params"],
      tx=optimizer
      )
  # We access model params only from optimizer below.
  del initial_variables

  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    state = checkpoints.restore_checkpoint(workdir, state)
    # Grab last step.
    start_step = int(state.step)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if start_step == 0:
    writer.write_hparams(dict(config))

  # Replicate optimizer.
  state = jax_utils.replicate(state)

  # compile multidevice versions of train/eval/predict step fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn),
      axis_name="batch",
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, config=eval_config),
      axis_name="batch")

  p_pred_step = jax.pmap(
      functools.partial(
          predict_step, config=predict_config,
          temperature=config.sampling_temperature,
          top_k=config.sampling_top_k),
      axis_name="batch",
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  logging.info("Starting training loop.")
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=5)
    ]
  train_metrics = []
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      # Shard data to devices and do a training step.
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(train_iter)))
        state, metrics = p_train_step(
            state, batch, dropout_rng=dropout_rngs)
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed("training_metrics"):
          logging.info("Gathering training metrics.")
          train_metrics = common_utils.get_metrics(train_metrics)
          lr = train_metrics.pop("learning_rate").mean()
          metrics_sums = jax.tree_util.tree_map(jnp.sum, train_metrics)
          denominator = metrics_sums.pop("denominator")
          summary = jax.tree_util.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
          summary["learning_rate"] = lr
          summary["perplexity"] = jnp.clip(
              jnp.exp(summary["loss"]), a_max=1.0e4)
          summary = {"train_" + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed("eval"):
          eval_results = evaluate(
              p_eval_step=p_eval_step,
              params=state.params,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps)
          # (clipped) perplexity after averaging log-perplexitie
          eval_results["perplexity"] = jnp.clip(
              jnp.exp(eval_results["loss"]), a_max=1.0e4)
          writer.write_scalars(
              step, {"eval_" + k: v for k, v in eval_results.items()})

        with report_progress.timed("generate_text"):
          exemplars = generate_prediction(
              p_pred_step=p_pred_step,
              params=state.params,
              tokenized_prompts=tokenized_prompts,
              eos_id=eos_id,
              inference_rng=inference_rng,
              decode_tokens=decode_tokens,
              max_predict_length=config.max_predict_length)
          writer.write_texts(step, {"samples": exemplars})

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (step % config.checkpoint_every_steps == 0 or
                         is_last_step)
      if config.save_checkpoints and save_checkpoint and jax.process_index() == 0:
        with report_progress.timed("checkpoint"):
          checkpoints.save_checkpoint(workdir, jax_utils.unreplicate(state),
                                      step)

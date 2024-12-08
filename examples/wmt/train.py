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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
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
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

import bleu
import decode
import input_pipeline
import models


class TrainState(train_state.TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale

class Fp8TrainState(train_state.Fp8TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale

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
    A schedule `count -> learning_rate`.
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
        "Incorrect shapes. Got shape %s logits and %s targets"
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
    raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s targets"
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
      "loss": loss,
      "accuracy": acc,
      "denominator": weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name="batch")
  return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------


def train_step(
    state,
    batch,
    config,
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
  train_keys = [
      "inputs",
      "targets",
      "inputs_position",
      "targets_position",
      "inputs_segmentation",
      "targets_segmentation",
  ]
  (
      inputs,
      targets,
      inputs_positions,
      targets_positions,
      inputs_segmentation,
      targets_segmentation,
  ) = (batch.get(k, None) for k in train_keys)

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        params if config.use_fp8 else {"params": params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={"dropout": dropout_rng},
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step

  if state.dynamic_scale:
    # dynamic scale takes care of averaging gradients across replicas
    grad_fn = state.dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name="batch"
    )
    dynamic_scale, is_fin, (_, logits), grads = grad_fn(state.params)
    state = state.replace(dynamic_scale=dynamic_scale)
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, targets, weights)
  metrics["learning_rate"] = learning_rate_fn(step)

  if state.dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    select_fn = functools.partial(jnp.where, is_fin)
    new_state = new_state.replace(
        opt_state=jax.tree_util.tree_map(
            select_fn, new_state.opt_state, state.opt_state
        ),
        params=jax.tree_util.tree_map(
            select_fn, new_state.params, state.params
        ),
    )
    metrics["loss_scale"] = dynamic_scale.scale * metrics["denominator"]

  return new_state, metrics


def eval_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch["inputs"], batch["targets"]
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply(
      params if config.use_fp8 else {"params": params}, inputs, targets)

  return compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = models.Transformer(config).init(
      jax.random.key(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype),
  )
  return initial_variables["cache"]


def predict_step(
    inputs, params, cache, eos_id, max_decode_len, config, beam_size=4
):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item"s data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = decode.flat_batch_beam_expand(
      models.Transformer(config).apply(
          params if config.use_fp8 else {"params": params},
          inputs, method=models.Transformer.encode
      ),
      beam_size,
  )
  raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    params_predict = params if config.use_fp8 else {"params": params}
    params_predict["cache"] = flat_cache
    flat_logits, new_vars = models.Transformer(config).apply(
        params_predict,
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=["cache"],
        method=models.Transformer.decode,
    )
    new_flat_cache = new_vars["cache"]
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
      max_decode_len=max_decode_len,
  )

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
  """Execute psum on in_tree"s leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.process_index].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

  def pre_pmap(xs):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs
    )

  def post_pmap(xs):
    return jax.tree_util.tree_map(lambda x: x[0], xs)

  return post_pmap(host_psum(pre_pmap(in_tree)))


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def evaluate(
    *, p_eval_step, params, eval_ds: tf.data.Dataset, num_eval_steps: int
):
  """Evaluate the params an return a dictionary with the metrics."""
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
      eval_metrics_sums,
  )
  return eval_summary


def translate_and_calculate_bleu(
    *,
    p_pred_step,
    p_init_cache,
    params,
    predict_ds: tf.data.Dataset,
    decode_tokens,
    max_predict_length: int,
):
  """Translates the `predict_ds` and calculates the BLEU score."""
  n_devices = jax.local_device_count()
  logging.info("Translating evaluation dataset.")
  sources, references, predictions = [], [], []
  for pred_batch in predict_ds:
    pred_batch = jax.tree_util.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
    # Handle final odd-sized batch by padding instead of dropping it.
    cur_pred_batch_size = pred_batch["inputs"].shape[0]
    if cur_pred_batch_size % n_devices:
      padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
      pred_batch = jax.tree_util.tree_map(
          lambda x: pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
          pred_batch,
      )
    pred_batch = common_utils.shard(pred_batch)
    cache = p_init_cache(pred_batch["inputs"])
    predicted = p_pred_step(
        pred_batch["inputs"], params, cache, decode.EOS_ID, max_predict_length
    )
    predicted = tohost(predicted)
    inputs = tohost(pred_batch["inputs"])
    targets = tohost(pred_batch["targets"])
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:cur_pred_batch_size]):
      sources.append(decode_tokens(inputs[i]))
      references.append(decode_tokens(targets[i]))
      predictions.append(decode_tokens(s))
  logging.info(
      "Translation: %d predictions %d references %d sources.",
      len(predictions),
      len(references),
      len(sources),
  )

  # Calculate BLEU score for translated eval corpus against reference.
  bleu_matches = bleu.bleu_partial(references, predictions)
  all_bleu_matches = per_host_sum_pmap(bleu_matches)
  bleu_score = bleu.complete_bleu(*all_bleu_matches)
  # Save translation samples for tensorboard.
  exemplars = ""
  for n in np.random.choice(np.arange(len(predictions)), 8):
    exemplars += f"{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n"
  return exemplars, bleu_score


def preferred_dtype(config):
  platform = jax.local_devices()[0].platform
  if config.use_mixed_precision:
    if platform == "tpu":
      return jnp.bfloat16
    elif platform == "gpu":
      return jnp.bfloat16 if config.use_fp8 else jnp.float16
  return jnp.float32


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
  train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_wmt_datasets(
      n_devices=jax.local_device_count(),
      config=config,
      reverse_translation=config.reverse_translation,
      vocab_path=vocab_path,
  )

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode("utf-8")

  if config.num_predict_steps > 0:
    predict_ds = predict_ds.take(config.num_predict_steps)

  logging.info("Initializing model, optimizer, and step functions.")

  dtype = preferred_dtype(config)

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=dtype,
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
      bias_init=nn.initializers.normal(stddev=1e-6),
      use_fp8=config.use_fp8
  )
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

  start_step = 0
  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_target_length)
  target_shape = (config.per_device_batch_size, config.max_target_length)

  m = models.Transformer(eval_config)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32),
  )

  # Create train state with Adam optimizer and weight decay.
  learning_rate_fn = create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )
  dynamic_scale = None
  if dtype == jnp.float16:
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  TrainStateProxy = Fp8TrainState if config.use_fp8 else TrainState
  state = TrainStateProxy.create(
      apply_fn=m.apply,
      params=initial_variables if config.use_fp8 else initial_variables["params"],
      tx=optax.adamw(
          learning_rate=learning_rate_fn,
          b1=0.9,
          b2=0.98,
          eps=1e-9,
          weight_decay=config.weight_decay,
      ),
      dynamic_scale=dynamic_scale,
  )

  # We access model params only via state.params
  del initial_variables

  if config.restore_checkpoints:
    # Restore unreplicated optimizer + model state from last checkpoint.
    state = checkpoints.restore_checkpoint(workdir, state)
    # Grab last step.
    start_step = int(state.step)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dict(config))

  # Replicate state.
  state = jax_utils.replicate(state)

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=config.label_smoothing,
      ),
      axis_name="batch",
      donate_argnums=(0,),
  )  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(eval_step, config=eval_config), axis_name="batch"
  )
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=config.max_predict_length,
          config=predict_config,
      ),
      axis_name="batch",
  )
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step, config=predict_config, beam_size=config.beam_size
      ),
      axis_name="batch",
      static_broadcasted_argnums=(3, 4),
  )  # eos token, max_length are constant

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  logging.info("Starting training loop.")
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
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, next(train_iter))
        )
        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
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
          summary = jax.tree_util.tree_map(
              lambda x: x / denominator, metrics_sums
          )  # pylint: disable=cell-var-from-loop
          summary["learning_rate"] = lr
          summary = {"train_" + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed("eval"):
          eval_results = evaluate(
              p_eval_step=p_eval_step,
              params=state.params,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
          )
          writer.write_scalars(
              step, {"eval_" + k: v for k, v in eval_results.items()}
          )
        if not config.use_fp8:
          with report_progress.timed("translate_and_bleu"):
            exemplars, bleu_score = translate_and_calculate_bleu(
                p_pred_step=p_pred_step,
                p_init_cache=p_init_cache,
                params=state.params,
                predict_ds=predict_ds,
                decode_tokens=decode_tokens,
                max_predict_length=config.max_predict_length,
            )
            writer.write_scalars(step, {"bleu": bleu_score})
            writer.write_texts(step, {"samples": exemplars})

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step % config.checkpoint_every_steps == 0 or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info("Saving checkpoint step %d.", step)
        with report_progress.timed("checkpoint"):
          checkpoints.save_checkpoint_multiprocess(
              workdir, jax_utils.unreplicate(state), step
          )

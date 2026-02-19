# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WMT helper functions for benchmarking."""

import functools
from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.wmt import models
from flax.examples.wmt.configs import default as wmt_config
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


class TrainState(train_state.TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale


def rsqrt_schedule(init_value: float, shift: int = 0):
  def schedule(count):
    return init_value * (count + shift) ** -0.5 * shift**0.5
  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
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


def preferred_dtype(config):
  platform = jax.local_devices()[0].platform
  if config.use_mixed_precision:
    if platform == 'tpu':
      return jnp.bfloat16
    elif platform == 'gpu':
      return jnp.float16
  return jnp.float32


def compute_weighted_cross_entropy(
    logits, targets, weights=None, label_smoothing=0.0
):
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

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
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


def wmt_train_step(
    state,
    batch,
    config,
    learning_rate_fn,
    label_smoothing=0.0,
    dropout_rng=None,
):
  train_keys = [
      'inputs',
      'targets',
      'inputs_position',
      'targets_position',
      'inputs_segmentation',
      'targets_segmentation',
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
    logits = models.Transformer(config).apply(
        {'params': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={'dropout': dropout_rng},
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step

  if state.dynamic_scale:
    grad_fn = state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
    dynamic_scale, is_fin, (_, logits), grads = grad_fn(state.params)
    state = state.replace(dynamic_scale=dynamic_scale)
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = learning_rate_fn(step)

  if state.dynamic_scale:
    select_fn = functools.partial(jnp.where, is_fin)
    new_state = new_state.replace(
        opt_state=jax.tree_util.tree_map(
            select_fn, new_state.opt_state, state.opt_state
        ),
        params=jax.tree_util.tree_map(
            select_fn, new_state.params, state.params
        ),
    )
    metrics['loss_scale'] = dynamic_scale.scale * metrics['denominator']

  return new_state, metrics


def get_fake_batch(config: ml_collections.ConfigDict) -> dict[str, Any]:
  rng = jax.random.key(0)
  batch_size = config.per_device_batch_size
  max_len = config.max_target_length

  inputs = jax.random.randint(
      rng, (batch_size, max_len), 0, config.vocab_size, jnp.int32
  )
  targets = jax.random.randint(
      rng, (batch_size, max_len), 0, config.vocab_size, jnp.int32
  )

  return {
      'inputs': inputs,
      'targets': targets,
      'inputs_position': None,
      'targets_position': None,
      'inputs_segmentation': None,
      'targets_segmentation': None,
  }


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  dtype = preferred_dtype(config)

  train_config = models.TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.vocab_size,
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
      kernel_init=jax.nn.initializers.xavier_uniform(),
      bias_init=jax.nn.initializers.normal(stddev=1e-6),
  )

  model = models.Transformer(train_config)

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

  rng = jax.random.key(0)
  init_rng, dropout_rng = jax.random.split(rng)

  batch = get_fake_batch(config)
  inputs = batch['inputs']
  targets = batch['targets']

  initial_variables = model.init(init_rng, inputs, targets)

  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.use_mixed_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()

  state = TrainState.create(
      apply_fn=model.apply,
      params=initial_variables['params'],
      tx=optimizer,
      dynamic_scale=dynamic_scale,
  )

  jit_train_step = jax.jit(
      wmt_train_step,
      static_argnums=(2, 3, 4),
  )

  return (
      jit_train_step,
      (state, batch, train_config, learning_rate_fn, 0.0, dropout_rng),
      {},
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, wmt_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, wmt_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

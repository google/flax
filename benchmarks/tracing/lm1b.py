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
"""LM1B helper functions for benchmarking."""

import functools
from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.lm1b import models
from flax.examples.lm1b.configs import default as lm1b_config
from flax.training import common_utils
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


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


def get_fake_batch(config: ml_collections.ConfigDict) -> dict[str, jnp.ndarray]:
  batch_size = config.per_device_batch_size
  max_len = config.max_target_length

  inputs = jax.random.randint(
      jax.random.key(0),
      (batch_size, max_len),
      minval=0,
      maxval=config.vocab_size,
      dtype=jnp.int32,
  )
  inputs_position = jnp.tile(
      jnp.arange(max_len, dtype=jnp.int32), (batch_size, 1)
  )
  inputs_segmentation = jnp.ones((batch_size, max_len), dtype=jnp.int32)

  return {
      'inputs': inputs,
      'inputs_position': inputs_position,
      'inputs_segmentation': inputs_segmentation,
  }


@functools.partial(jax.jit, static_argnums=(2, 3))
def bench_train_step(state, batch, config, learning_rate_fn):

  def compute_metrics(logits, labels, weights):
    loss, weight_sum = compute_weighted_cross_entropy(
        logits, labels, weights, 0.0
    )
    acc, _ = compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    return metrics

  inputs = batch['inputs']
  inputs_positions = batch['inputs_position']
  inputs_segmentation = batch['inputs_segmentation']

  weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)
  dropout_rng = jax.random.fold_in(jax.random.key(0), state.step)

  def loss_fn(params):
    logits = models.TransformerLM(config).apply(
        {'params': params},
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        rngs={'dropout': dropout_rng},
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, inputs, weights, 0.0
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


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  train_config = models.TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.vocab_size,
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
      kernel_init=jax.nn.initializers.xavier_uniform(),
      bias_init=jax.nn.initializers.normal(stddev=1e-6),
  )

  model = models.TransformerLM(train_config)

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
  init_rng, _ = jax.random.split(rng)

  initial_variables = model.init(
      init_rng,
      jnp.ones(
          (config.per_device_batch_size, config.max_target_length), jnp.int32
      ),
      jnp.ones(
          (config.per_device_batch_size, config.max_target_length), jnp.int32
      ),
      jnp.ones(
          (config.per_device_batch_size, config.max_target_length), jnp.int32
      ),
  )

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=initial_variables['params'],
      tx=optimizer,
  )

  batch = get_fake_batch(config)

  return (
      bench_train_step,
      (state, batch, train_config, learning_rate_fn),
      {},
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_lm1b_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, lm1b_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_lm1b_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, lm1b_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

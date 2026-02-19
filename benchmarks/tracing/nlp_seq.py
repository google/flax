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
"""NLP Sequence Tagging helper functions for benchmarking."""

import functools
from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.nlp_seq import models
from flax.examples.nlp_seq.configs import default as nlp_seq_config
from flax.training import common_utils
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
):
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
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
        ret *= decay_factor ** (step // steps_per_decay)
      elif name == 'cosine_decay':
        progress = jnp.maximum(
            0.0, (step - warmup_steps) / float(steps_per_cycle)
        )
        ret *= jnp.maximum(
            0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0)))
        )
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )
  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
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
  batch_size = config.batch_size
  max_len = config.max_length
  vocab_size = config.vocab_size

  inputs = jax.random.randint(
      jax.random.key(0),
      (batch_size, max_len),
      minval=0,
      maxval=vocab_size,
      dtype=jnp.int32,
  )
  targets = jax.random.randint(
      jax.random.key(1),
      (batch_size, max_len),
      minval=0,
      maxval=config.output_vocab_size,
      dtype=jnp.int32,
  )

  return {
      'inputs': inputs,
      'targets': targets,
  }


@functools.partial(jax.jit, static_argnums=(2, 3))
def bench_train_step(state, batch, config, learning_rate_fn):

  def local_compute_metrics(logits, labels, weights):
    loss, weight_sum = compute_weighted_cross_entropy(
        logits, labels, weights
    )
    acc, _ = compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    return metrics

  inputs = batch['inputs']
  targets = batch['targets']

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
  dropout_rng = jax.random.fold_in(jax.random.key(0), state.step)

  def loss_fn(params):
    model = models.Transformer(config)
    logits = model.apply(
        {'params': params},
        inputs=inputs,
        train=True,
        rngs={'dropout': dropout_rng},
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights
    )
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)

  metrics = local_compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = lr

  return new_state, metrics


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  if not hasattr(config, 'vocab_size'):
    config.vocab_size = 30000
  if not hasattr(config, 'output_vocab_size'):
    config.output_vocab_size = 50

  model_config = models.TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.output_vocab_size,
      max_len=config.max_length,
  )

  model = models.Transformer(model_config)

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=config.learning_rate
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

  init_batch = jnp.ones((config.batch_size, config.max_length), jnp.int32)
  initial_variables = model.init(init_rng, inputs=init_batch, train=False)

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=initial_variables['params'],
      tx=optimizer,
  )

  batch = get_fake_batch(config)

  return (
      bench_train_step,
      (state, batch, model_config, learning_rate_fn),
      {},
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_nlp_seq_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, nlp_seq_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_nlp_seq_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, nlp_seq_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

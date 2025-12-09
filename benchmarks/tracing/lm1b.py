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

from flax.examples.lm1b import models
from flax.examples.lm1b import train
import jax
import jax.numpy as jnp
import ml_collections
import optax


def get_fake_batch(config: ml_collections.ConfigDict) -> dict[str, jnp.ndarray]:
  """Generate a batch of fake LM1B data.

  Args:
    config: The training configuration.

  Returns:
    A dictionary with 'inputs', 'inputs_position', and 'inputs_segmentation'.
  """
  batch_size = config.per_device_batch_size
  max_len = config.max_target_length

  # Inputs: integers [0, vocab_size)
  inputs = jax.random.randint(
      jax.random.key(0),
      (batch_size, max_len),
      minval=0,
      maxval=config.vocab_size,
      dtype=jnp.int32,
  )

  # Inputs position: integers [0, max_len)
  inputs_position = jnp.tile(
      jnp.arange(max_len, dtype=jnp.int32), (batch_size, 1)
  )

  # Inputs segmentation: all ones (single segment)
  inputs_segmentation = jnp.ones((batch_size, max_len), dtype=jnp.int32)

  return {
      'inputs': inputs,
      'inputs_position': inputs_position,
      'inputs_segmentation': inputs_segmentation,
  }


@functools.partial(jax.jit, static_argnums=(2, 3))
def bench_train_step(state, batch, config, learning_rate_fn):
  """Perform a single training step (JIT-compiled)."""

  def compute_metrics(logits, labels, weights):
    loss, weight_sum = train.compute_weighted_cross_entropy(
        logits, labels, weights, 0.0
    )
    acc, _ = train.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    return metrics

  # Unpack batch
  inputs = batch['inputs']
  inputs_positions = batch['inputs_position']
  inputs_segmentation = batch['inputs_segmentation']

  weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)
  dropout_rng = jax.random.fold_in(jax.random.key(0), state.step)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.TransformerLM(config).apply(
        {'params': params},
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        rngs={'dropout': dropout_rng},
    )

    loss, weight_sum = train.compute_weighted_cross_entropy(
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
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, and kwargs.
  """
  # Create model configuration
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

  # Create model
  model = models.TransformerLM(train_config)

  # Create learning rate function
  learning_rate_fn = train.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  # Create optimizer
  optimizer = optax.adamw(
      learning_rate_fn,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=config.weight_decay,
  )

  # Create initial state
  rng = jax.random.key(0)
  init_rng, _ = jax.random.split(rng)

  # We need to mock the mesh for setup_initial_state or just create state manually
  # setup_initial_state uses mesh for sharding, which we might want to avoid for simple tracing
  # Let's create state manually to be safe and simple

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

  from flax.training import train_state

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=initial_variables['params'],
      tx=optimizer,
  )

  # Generate fake batch
  batch = get_fake_batch(config)

  # Return bench_train_step and its arguments
  return (
      bench_train_step,
      (state, batch, train_config, learning_rate_fn),
      {},
  )

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

from flax.examples.nlp_seq import models
import jax
import jax.numpy as jnp
import ml_collections
import optax


def get_fake_batch(config: ml_collections.ConfigDict) -> dict[str, jnp.ndarray]:
  """Generate a batch of fake NLP sequence data.

  Args:
    config: The training configuration.

  Returns:
    A dictionary with 'inputs' and 'targets'.
  """
  batch_size = config.batch_size
  max_len = config.max_length
  vocab_size = config.vocab_size

  # Inputs: integers [0, vocab_size)
  inputs = jax.random.randint(
      jax.random.key(0),
      (batch_size, max_len),
      minval=0,
      maxval=vocab_size,
      dtype=jnp.int32,
  )

  # Targets: integers [0, output_vocab_size)
  # We use a small output vocab size for targets (e.g. POS tags)
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
  """Perform a single training step (JIT-compiled)."""
  from flax.examples.nlp_seq import train  # pylint: disable=g-import-not-at-top

  def compute_metrics(logits, labels, weights):
    loss, weight_sum = train.compute_weighted_cross_entropy(
        logits, labels, weights
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
  targets = batch['targets']

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
  dropout_rng = jax.random.fold_in(jax.random.key(0), state.step)

  def loss_fn(params):
    """loss function used for training."""
    # Re-create model with config to ensure it's bound
    model = models.Transformer(config)
    logits = model.apply(
        {'params': params},
        inputs=inputs,
        train=True,
        rngs={'dropout': dropout_rng},
    )

    loss, weight_sum = train.compute_weighted_cross_entropy(
        logits, targets, weights
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


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, and kwargs.
  """
  from flax.examples.nlp_seq import train  # pylint: disable=g-import-not-at-top

  # Set default vocab sizes if not present (they are not in default config)
  if not hasattr(config, 'vocab_size'):
    config.vocab_size = 30000
  if not hasattr(config, 'output_vocab_size'):
    config.output_vocab_size = 50

  # Create model configuration
  model_config = models.TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.output_vocab_size,
      max_len=config.max_length,
  )

  # Create model
  model = models.Transformer(model_config)

  # Create learning rate function
  learning_rate_fn = train.create_learning_rate_scheduler(
      base_learning_rate=config.learning_rate
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

  init_batch = jnp.ones((config.max_length, 1), jnp.float32)
  # The model expects inputs of shape (batch, len) but init uses (len, 1)?
  # Wait, let's check train.py initialize_variables:
  # init_batch = jnp.ones((config.max_len, 1), jnp.float32)
  # init_variables = model.init(init_rng, inputs=init_batch, train=False)
  # This looks like (len, batch=1) or (batch=len, len=1)?
  # models.Transformer.__call__ asserts inputs.ndim == 2.
  # If init_batch is (max_len, 1), then batch=max_len, len=1?
  # Let's check models.py:
  # x = inputs.astype('int32')
  # x = nn.Embed(...)(x)
  # x = AddPositionEmbs(config)(x)
  # AddPositionEmbs expects (batch, seq_len, emb_dim).
  # If inputs is (max_len, 1), embed output is (max_len, 1, emb_dim).
  # AddPositionEmbs: length = inputs.shape[1] = 1.
  # This seems to verify it works for any length up to max_len.
  # But for benchmarking we should probably use the actual batch shape to be safe.

  init_batch = jnp.ones((config.batch_size, config.max_length), jnp.int32)
  initial_variables = model.init(init_rng, inputs=init_batch, train=False)

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
      (state, batch, model_config, learning_rate_fn),
      {},
  )

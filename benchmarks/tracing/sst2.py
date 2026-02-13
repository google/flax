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
"""SST2 helper functions."""

from typing import Any

from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.sst2 import models
from flax.examples.sst2.configs import default as sst2_config
from flax.training import train_state as train_state_lib
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import optax

Array = jnp.ndarray
TrainState = train_state_lib.TrainState


@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: Array, logits: Array) -> Array:
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = logits >= zeros
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))


def model_from_config(config: ml_collections.ConfigDict):
  model = models.TextClassifier(
      embedding_size=config.embedding_size,
      hidden_size=config.hidden_size,
      vocab_size=config.vocab_size,
      output_size=config.output_size,
      dropout_rate=config.dropout_rate,
      word_dropout_rate=config.word_dropout_rate,
      unk_idx=config.unk_idx,
  )
  return model


def get_initial_params(rng, model):
  token_ids = jnp.ones((2, 3), jnp.int32)
  lengths = jnp.ones((2,), dtype=jnp.int32)
  variables = model.init(rng, token_ids, lengths, deterministic=True)
  return variables['params']


def create_train_state(rng, config: ml_collections.ConfigDict, model):
  params = get_initial_params(rng, model)
  tx = optax.chain(
      optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum),
      optax.add_decayed_weights(weight_decay=config.weight_decay),
  )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return state


def compute_metrics(*, labels: Array, logits: Array):
  if labels.ndim == 1:
    labels = jnp.expand_dims(labels, axis=1)
  loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  binary_predictions = logits >= 0.0
  binary_accuracy = jnp.equal(binary_predictions, labels)
  return {
      'loss': jnp.sum(loss),
      'accuracy': jnp.sum(binary_accuracy),
      'count': logits.shape[0],
  }


def train_step(
    state: TrainState,
    batch: dict[str, Array],
    rngs: dict[str, Any],
) -> tuple[TrainState, Any]:
  step = state.step
  rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

  def loss_fn(params):
    variables = {'params': params}
    logits = state.apply_fn(
        variables,
        batch['token_ids'],
        batch['length'],
        deterministic=False,
        rngs=rngs,
    )

    labels = batch['label']
    if labels.ndim == 1:
      labels = jnp.expand_dims(labels, 1)
    loss = jnp.mean(
        sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  value, grads = grad_fn(state.params)
  (_, logits) = value

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return new_state, metrics


def get_fake_batch(batch_size: int) -> dict[str, Any]:
  rng = jax.random.key(0)
  max_length = 60
  token_ids = jax.random.randint(
      rng, (batch_size, max_length), 0, 1000, jnp.int32
  )
  lengths = jnp.full((batch_size,), max_length, jnp.int32)
  labels = jax.random.uniform(rng, (batch_size,), jnp.float32)
  return {
      'token_ids': token_ids,
      'length': lengths,
      'label': labels,
  }


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  rng = jax.random.key(0)
  config = config.copy_and_resolve_references()
  if config.vocab_size is None:
    config.vocab_size = 1000
  model = model_from_config(config)
  state = create_train_state(rng, config, model)
  batch = get_fake_batch(config.batch_size)
  _, dropout_rng = jax.random.split(rng)
  rngs = {'dropout': dropout_rng}
  train_step_jit = jax.jit(train_step)
  return train_step_jit, (state, batch, rngs), {}


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_sst2_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, sst2_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_sst2_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, sst2_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

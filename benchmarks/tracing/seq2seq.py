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
"""Seq2Seq helper functions."""

from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.seq2seq import models
from flax.examples.seq2seq.configs import default as seq2seq_config
from flax.examples.seq2seq.input_pipeline import CharacterTable
from flax.examples.seq2seq.input_pipeline import get_sequence_lengths
from flax.examples.seq2seq.input_pipeline import mask_sequences
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import optax


def cross_entropy_loss(logits, labels, lengths):
  xe = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe


def compute_metrics(logits, labels, eos_id):
  lengths = get_sequence_lengths(labels, eos_id)
  loss = cross_entropy_loss(logits, labels, lengths)
  token_accuracy = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
  sequence_accuracy = (
      jnp.sum(mask_sequences(token_accuracy, lengths), axis=-1) == lengths
  )
  accuracy = jnp.mean(sequence_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def seq2seq_train_step(state, batch, lstm_rng, eos_id):
  labels = batch['answer'][:, 1:]
  lstm_key = jax.random.fold_in(lstm_rng, state.step)

  def loss_fn(params):
    logits, _ = state.apply_fn(
        {'params': params},
        batch['query'],
        batch['answer'],
        rngs={'lstm': lstm_key},
    )
    loss = cross_entropy_loss(
        logits, labels, get_sequence_lengths(labels, eos_id)
    )
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, labels, eos_id)

  return state, metrics


def get_fake_batch(batch_size: int, ctable: CharacterTable) -> dict[str, Any]:
  return ctable.get_batch(batch_size)


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  rng = jax.random.key(0)
  ctable = CharacterTable("0123456789+= ", config.max_len_query_digit)

  model = models.Seq2seq(
      teacher_force=False,
      hidden_size=config.hidden_size,
      eos_id=ctable.eos_id,
      vocab_size=ctable.vocab_size,
  )

  rng1, rng2 = jax.random.split(rng)
  params = model.init(
      {'params': rng1, 'lstm': rng2},
      jnp.ones(ctable.encoder_input_shape, jnp.float32),
      jnp.ones(ctable.decoder_input_shape, jnp.float32),
  )['params']

  tx = optax.adam(config.learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )

  batch = get_fake_batch(config.batch_size, ctable)
  return seq2seq_train_step, (state, batch, rng, ctable.eos_id), {}


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_seq2seq_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, seq2seq_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_seq2seq_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, seq2seq_config.get_config, state
  )


if __name__ == "__main__":
  tracing_benchmark.run_benchmarks()

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
"""Gemma helper functions."""

from typing import Any

from flax import nnx
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.gemma import transformer as transformer_lib
from flax.examples.gemma import utils
from flax.examples.gemma.configs import default as gemma_config
from flax.training import common_utils
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

  loss = -jnp.sum(soft_targets * nnx.log_softmax(logits), axis=-1)
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


def train_step(
    state: utils.TrainState,
    batch,
    learning_rate_fn,
    label_smoothing=0.0,
):
  train_keys = ['inputs', 'inputs_position', 'inputs_segmentation', 'targets']
  (inputs, inputs_positions, inputs_segmentation, targets) = (
      batch.get(k, None) for k in train_keys
  )

  pad_id = 0
  weights = jnp.where(inputs > pad_id, 1, 0).astype(jnp.float32)
  input_mask = inputs > pad_id
  attention_mask = transformer_lib.make_causal_attn_mask(input_mask)
  mask = (
      inputs_segmentation[:, :, None] == inputs_segmentation[:, None, :]
  )
  attention_mask = jnp.logical_and(mask, attention_mask)

  def loss_fn(params):
    module = nnx.merge(state.graphdef, params)

    logits, _ = module(
        inputs,
        positions=inputs_positions,
        attention_mask=attention_mask,
        cache=None,
    )

    loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing
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


def get_fake_batch(batch_size: int) -> Any:
  rng = jax.random.PRNGKey(0)
  batch = {}
  for k in (
      'inputs',
      'inputs_position',
      'inputs_segmentation',
      'targets',
      'targets_position',
      'targets_segmentation',
  ):
    batch[k] = jax.random.randint(rng, (batch_size, 128), 0, 9999999, jnp.int32)
  return batch


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
    vocab_size: int | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  if vocab_size is None:
    vocab_size = config.vocab_size

  if config.transformer_name is not None:
    model_config = transformer_lib.TransformerConfig.from_version_name(
        config.transformer_name,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )
  else:
    assert config.transformer_params is not None
    model_config = transformer_lib.TransformerConfig.from_dict(
        **config.transformer_params,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )

  devices_array = utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)

  def constructor(config: transformer_lib.TransformerConfig, key: jax.Array):
    return transformer_lib.Transformer(config, rngs=nnx.Rngs(params=key))

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

  state, state_sharding = utils.setup_initial_state(
      constructor, optimizer, model_config, init_rng, mesh
  )
  data_sharding = jax.NamedSharding(mesh, jax.P(config.data_sharding))
  jit_train_step = jax.jit(
      train_step,
      in_shardings=(
          state_sharding,
          data_sharding,
      ),  # type: ignore
      out_shardings=(state_sharding, None),  # type: ignore
      static_argnames=('learning_rate_fn', 'label_smoothing'),
      donate_argnums=0,
  )

  batch = get_fake_batch(config.per_device_batch_size)
  batch = jax.tree.map(lambda x: jnp.asarray(x, device=data_sharding), batch)

  return (
      jit_train_step,
      (state, batch, learning_rate_fn, 0.0),
      dict(),
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, gemma_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, gemma_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

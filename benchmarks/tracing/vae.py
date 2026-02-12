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
"""VAE helper functions."""

from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.vae import models
from flax.examples.vae.configs import default as vae_config
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import optax


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def train_step(state, batch, z_rng, latents):
  def loss_fn(params):
    recon_x, mean, logvar = models.model(latents).apply(
        {'params': params}, batch, z_rng
    )
    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def get_fake_batch(batch_size: int) -> Any:
  return jnp.ones((batch_size, 784), jnp.float32)


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  rng = jax.random.key(0)
  rng, key = jax.random.split(rng)
  batch = get_fake_batch(config.batch_size)
  params = models.model(config.latents).init(key, batch, rng)['params']
  state = train_state.TrainState.create(
      apply_fn=models.model(config.latents).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )
  train_step_jit = jax.jit(train_step, static_argnames=('latents',))
  return (
      train_step_jit,
      (state, batch, rng, config.latents),
      dict(),
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_vae_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, vae_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_vae_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, vae_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()

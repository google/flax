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

from flax.examples.vae import models
from flax.examples.vae import train
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax


def get_fake_batch(batch_size: int) -> Any:
  """Returns fake data for the given batch size.

  Args:
    batch_size: The global batch size to generate.

  Returns:
    A properly sharded global batch of data.
  """
  return jnp.ones((batch_size, 784), jnp.float32)


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args and kwargs for the apply function, and
    any metadata the training loop needs.
  """
  rng = jax.random.key(0)
  rng, key = jax.random.split(rng)
  batch = get_fake_batch(config.batch_size)
  params = models.model(config.latents).init(key, batch, rng)['params']
  state = train_state.TrainState.create(
      apply_fn=models.model(config.latents).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )
  # Wrap with jit, making latents static since it's a python int
  train_step_jit = jax.jit(train.train_step, static_argnames=('latents',))
  return (
      train_step_jit,
      (state, batch, rng, config.latents),
      dict(),
  )

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

from flax.examples.sst2 import train
import jax
import jax.numpy as jnp
import ml_collections


def get_fake_batch(batch_size: int) -> dict[str, Any]:
  """Returns fake data for the given batch size.

  Args:
    batch_size: The global batch size to generate.

  Returns:
    A fake batch dictionary with token_ids, length, and label.
  """
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
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, kwargs, and any metadata.
  """
  rng = jax.random.key(0)
  config = config.copy_and_resolve_references()
  if config.vocab_size is None:
    config.vocab_size = 1000
  model = train.model_from_config(config)
  state = train.create_train_state(rng, config, model)
  batch = get_fake_batch(config.batch_size)
  _, dropout_rng = jax.random.split(rng)
  rngs = {'dropout': dropout_rng}
  train_step_jit = jax.jit(train.train_step)
  return train_step_jit, (state, batch, rngs), {}

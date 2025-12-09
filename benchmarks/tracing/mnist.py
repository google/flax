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
"""MNIST helper functions."""

from typing import Any

from flax.examples.mnist import train
import jax
import jax.numpy as jnp
import ml_collections


def get_fake_batch(batch_size: int) -> tuple[Any, Any]:
  """Returns fake data for the given batch size.

  Args:
    batch_size: The global batch size to generate.

  Returns:
    A tuple of (images, labels) with fake data.
  """
  rng = jax.random.PRNGKey(0)
  images = jax.random.normal(rng, (batch_size, 28, 28, 1), jnp.float32)
  labels = jax.random.randint(rng, (batch_size,), 0, 10, jnp.int32)
  return images, labels


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, kwargs, and any metadata.
  """
  rng = jax.random.PRNGKey(0)
  state = train.create_train_state(rng, config)
  images, labels = get_fake_batch(config.batch_size)
  return train.apply_model, (state, images, labels), {}

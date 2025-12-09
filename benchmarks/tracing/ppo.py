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
"""PPO helper functions for RL benchmarking."""

from typing import Any

from flax.examples.ppo import models
from flax.examples.ppo import ppo_lib
import jax
import jax.numpy as jnp
import ml_collections


def get_fake_batch(batch_size: int = 256) -> tuple[jnp.ndarray, ...]:
  """Generate a batch of fake Atari observations and trajectories.

  Args:
    batch_size: Size of the minibatch.

  Returns:
    A tuple of (states, actions, old_log_probs, returns, advantages).
  """
  # Atari observations: (batch_size, height, width, stacked_frames)
  states = jax.random.randint(
      jax.random.key(0),
      (batch_size, 84, 84, 4),
      minval=0,
      maxval=256,
      dtype=jnp.int32,
  )

  # Actions: discrete action space (e.g., 6 actions for Pong)
  actions = jax.random.randint(
      jax.random.key(1), (batch_size,), minval=0, maxval=6, dtype=jnp.int32
  )

  # Old log probabilities from behavior policy
  old_log_probs = jax.random.normal(
      jax.random.key(2), (batch_size,), dtype=jnp.float32
  )

  # Returns (discounted cumulative rewards)
  returns = jax.random.normal(
      jax.random.key(3), (batch_size,), dtype=jnp.float32
  )

  # Advantages (GAE advantages)
  advantages = jax.random.normal(
      jax.random.key(4), (batch_size,), dtype=jnp.float32
  )

  return (states, actions, old_log_probs, returns, advantages)


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, and kwargs.
  """
  # Create model (6 actions for Pong)
  num_outputs = 6
  model = models.ActorCritic(num_outputs=num_outputs)

  # Initialize model parameters
  rng = jax.random.key(0)
  init_shape = jnp.ones((1, 84, 84, 4), jnp.float32)
  initial_params = model.init(rng, init_shape)['params']

  # Create train state
  # For benchmarking, we don't need actual training loops, just one step
  train_steps = 1000  # Dummy value for state creation
  state = ppo_lib.create_train_state(initial_params, model, config, train_steps)

  # Generate fake trajectories
  trajectories = get_fake_batch(config.batch_size)

  # PPO hyperparameters
  clip_param = config.clip_param
  vf_coeff = config.vf_coeff
  entropy_coeff = config.entropy_coeff

  # ppo_lib.train_step is already JIT-compiled with static batch_size
  return (
      ppo_lib.train_step,
      (state, trajectories, config.batch_size),
      {
          'clip_param': clip_param,
          'vf_coeff': vf_coeff,
          'entropy_coeff': entropy_coeff,
      },
  )

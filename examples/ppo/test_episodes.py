# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test policy by playing a full Atari game."""

import itertools
from typing import Any, Callable

import flax
import numpy as np

import agent
import env_utils


def policy_test(
    n_episodes: int,
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    game: str):
  """Perform a test of the policy in Atari environment.

  Args:
    n_episodes: number of full Atari episodes to test on
    apply_fn: the actor-critic apply function
    params: actor-critic model parameters, they define the policy being tested
    game: defines the Atari game to test on

  Returns:
    total_reward: obtained score
  """
  test_env = env_utils.create_env(game, clip_rewards=False)
  for _ in range(n_episodes):
    obs = test_env.reset()
    state = obs[None, ...]  # add batch dimension
    total_reward = 0.0
    for t in itertools.count():
      log_probs, _ = agent.policy_action(apply_fn, params, state)
      probs = np.exp(np.array(log_probs, dtype=np.float32))
      probabilities = probs[0] / probs[0].sum()
      action = np.random.choice(probs.shape[1], p=probabilities)
      obs, reward, done, _ = test_env.step(action)
      total_reward += reward
      next_state = obs[None, ...] if not done else None
      state = next_state
      if done:
        break
  return total_reward

# Copyright 2021 The Flax Authors.
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

"""Agent utilities, incl. choosing the move and running in separate process."""

import multiprocessing
import collections
import jax
import numpy as onp

import env_utils

@jax.jit
def policy_action(model, state):
  """Forward pass of the network."""
  out = model(state)
  return out


ExpTuple = collections.namedtuple(
    'ExpTuple', ['state', 'action', 'reward', 'value', 'log_prob', 'done'])


class RemoteSimulator:
  """Wrap functionality for an agent emulating Atari in a separate process.

  An object of this class is created for every agent.
  """

  def __init__(self, game: str):
    """Start the remote process and create Pipe() to communicate with it."""
    parent_conn, child_conn = multiprocessing.Pipe()
    self.proc = multiprocessing.Process(
        target=rcv_action_send_exp, args=(child_conn, game))
    self.proc.daemon = True
    self.conn = parent_conn
    self.proc.start()


def rcv_action_send_exp(conn, game: str):
  """Run the remote agents.

  Receive action from the main learner, perform one step of simulation and
  send back collected experience.
  """
  env = env_utils.create_env(game, clip_rewards=True)
  while True:
    obs = env.reset()
    done = False
    # Observations fetched from Atari env need additional batch dimension.
    state = obs[None, ...]
    while not done:
      conn.send(state)
      action = conn.recv()
      obs, reward, done, _ = env.step(action)
      next_state = obs[None, ...] if not done else None
      experience = (state, action, reward, done)
      conn.send(experience)
      if done:
        break
      state = next_state

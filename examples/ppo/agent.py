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

  def __init__(self, game):
    parent_conn, child_conn = multiprocessing.Pipe()
    self.proc = multiprocessing.Process(
        target=rcv_action_send_exp, args=(child_conn, game))
    self.conn = parent_conn
    self.proc.start()


def rcv_action_send_exp(conn, game):
  """Run the remote agents.

  Receive action from the main learner, perform one step of simulation and
  send back collected experience.
  """
  env = env_utils.create_env(game, clip_rewards=True)
  while True:
    obs = env.reset()
    done = False
    state = get_state(obs)
    while not done:
      conn.send(state)
      action, value, log_prob = conn.recv()
      obs, reward, done, _ = env.step(action)
      next_state = get_state(obs) if not done else None
      experience = ExpTuple(state, action, reward, value, log_prob, done)
      conn.send(experience)
      if done:
        break
      state = next_state

def get_state(observation):
  """Convert Atari env observation into a NumPy array, add batch dimension."""
  state = onp.array(observation)
  return state[None, ...]

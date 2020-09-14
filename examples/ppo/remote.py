import multiprocessing
import numpy as onp
from env import create_env


class RemoteSimulator:
  """Class that wraps basic functionality needed for an agent
  emulating Atari in a separate process.
  An object of this class is created for every agent.
  """
  def __init__(self):
    parent_conn, child_conn = multiprocessing.Pipe()
    self.proc = multiprocessing.Process(
      target=rcv_action_send_exp, args=(child_conn,))
    self.conn = parent_conn
    self.proc.start()


def rcv_action_send_exp(conn):
  """Function running on remote agents. Receives action from
  the main learner, performs one step of simulation and
  sends back collected experience.
  """
  env = create_env()
  while True:
    obs = env.reset()
    done = False
    state = get_state(obs)
    while not done:
      conn.send(state)
      action, value, log_prob = conn.recv()
      obs, reward, done, _ = env.step(action)
      next_state = get_state(obs) if not done else None
      # maybe a dictionary instead of a tuple would be better?
      experience = (state, action, reward, value, log_prob, done)
      conn.send(experience)
      if done:
        break
      state = next_state


def get_state(observation):
  state = onp.array(observation)
  return state[None, ...]

import time
import itertools
import gym
import flax
import numpy as onp

from env import create_env
from remote import get_state
from agent import policy_action

def test(n_episodes: int, model: flax.nn.base.Model, render: bool = False):
  test_env = create_env()
  if render:
    test_env = gym.wrappers.Monitor(
      test_env, "./rendered/" + "ddqn_pong_recording", force=True)
  for e in range(n_episodes):
    obs = test_env.reset()
    state = get_state(obs)
    total_reward = 0.0
    for t in itertools.count():
      probs, _ = policy_action(model, state)
      probs = onp.array(probs, dtype=onp.float64)
      probabilities = probs[0] / probs[0].sum()
      action = onp.random.choice(probs.shape[1], p=probabilities)
      obs, reward, done, _ = test_env.step(action)
      total_reward += reward
      if render:
        test_env.render()
        time.sleep(0.01)
      if not done:
        next_state = get_state(obs)
      else:
        next_state = None
      state = next_state
      if done:
        print(f"------> TEST FINISHED: finished Episode {e} with reward {total_reward}")
        break
  del test_env

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
  all_probabilities = []
  for e in range(n_episodes):
    obs = test_env.reset()
    state = get_state(obs)
    total_reward = 0.0
    for t in itertools.count():
      log_probs, _ = policy_action(model, state)
      probs = onp.exp(onp.array(log_probs, dtype=onp.float32))
      probabilities = probs[0] / probs[0].sum()
      all_probabilities.append(probabilities)
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
        all_probabilities = onp.stack(all_probabilities, axis=0)
        print(f"all_probabilities shape {all_probabilities.shape}")
        vars = onp.var(all_probabilities, axis=0)
        print(f"------> TEST FINISHED: reward {total_reward} in {t} steps")
        print(f"Variance of probabilities across encuntered states {vars}")
        break
  del test_env

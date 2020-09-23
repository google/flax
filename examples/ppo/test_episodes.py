"""Test policy by playing a full Atari game."""

import itertools
import flax
import numpy as onp

import env_utils
import remote
import agent

def policy_test(
  n_episodes: int,
  model: flax.nn.base.Model,
  game: str):
  """Perform a test of the policy in Atari environment.

  Args:
    n_episodes: number of full Atari episodes to test on
    model: the actor-critic model being tested
    game: defines the Atari game to test on

  Returns:
    None
  """
  test_env = env_utils.create_env(game, clip_rewards=False)
  all_probabilities = []
  for _ in range(n_episodes):
    obs = test_env.reset()
    state = remote.get_state(obs)
    total_reward = 0.0
    for t in itertools.count():
      log_probs, _ = agent.policy_action(model, state)
      probs = onp.exp(onp.array(log_probs, dtype=onp.float32))
      probabilities = probs[0] / probs[0].sum()
      all_probabilities.append(probabilities)
      action = onp.random.choice(probs.shape[1], p=probabilities)
      obs, reward, done, _ = test_env.step(action)
      total_reward += reward
      if not done:
        next_state = remote.get_state(obs)
      else:
        next_state = None
      state = next_state
      if done:
        all_probabilities = onp.stack(all_probabilities, axis=0)
        vars = onp.var(all_probabilities, axis=0)
        print(f"------> TEST FINISHED: reward {total_reward} in {t} steps")
        print(f"Variance of probabilities across encuntered states {vars}")
        break
  del test_env

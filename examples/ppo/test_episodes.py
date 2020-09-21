"""Test policy by playing a full Atari game."""

import time
import itertools
import gym
import flax
import numpy as onp

import env_utils
import remote
import agent

def test(n_episodes : int,
  model : flax.nn.base.Model,
  game : str,
  render : bool = False):
  """Perform a test of the policy in Atari environment.

  Args:
    n_episodes: number of full Atari episodes to test on
    model: the actor-critic model being tested
    game: defines the Atari game to test on
    render: whether to render the test environment

  Returns:
    None
  """
  test_env = env_utils.create_env(game, clip_rewards=False)
  if render:
    test_env = gym.wrappers.Monitor(
      test_env, "./rendered/" + "ddqn_pong_recording", force=True)
  all_probabilities = []
  for e in range(n_episodes):
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
      if render:
        test_env.render()
        time.sleep(0.01)
      if not done:
        next_state = remote.get_state(obs)
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

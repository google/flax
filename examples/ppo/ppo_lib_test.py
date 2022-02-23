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

"""Unit tests for the PPO example."""

from absl.testing import absltest
from flax.training import train_state
import jax
import ml_collections
import numpy as np
import numpy.testing as np_testing

import agent
import env_utils
import models
import ppo_lib


# test GAE
class TestGAE(absltest.TestCase):
  def test_gae_shape_on_random(self):
    # create random data, simulating 4 parallel envs and 20 time_steps
    envs, steps = 10, 100
    rewards = np.random.choice([-1., 0., 1.], size=(steps, envs),
                                p=[0.01, 0.98, 0.01])
    terminal_masks = np.ones(shape=(steps, envs), dtype=np.float64)
    values = np.random.random(size=(steps + 1, envs))
    discount = 0.99
    gae_param = 0.95
    adv = ppo_lib.gae_advantages(rewards, terminal_masks, values, discount,
                                 gae_param)
    self.assertEqual(adv.shape, (steps, envs))

  def test_gae_hardcoded(self):
    #test on small example that can be verified by hand
    rewards = np.array([[1., 0.], [0., 0.], [-1., 1.]])
    #one of the two episodes terminated in the middle
    terminal_masks = np.array([[1., 1.], [0., 1.], [1., 1.]])
    values = np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    discount = 0.5
    gae_param = 0.25
    correct_gae = np.array([[0.375, -0.5546875], [-1., -0.4375], [-1.5, 0.5]])
    actual_gae = ppo_lib.gae_advantages(rewards, terminal_masks, values,
                                        discount, gae_param)
    np_testing.assert_allclose(actual_gae, correct_gae)
# test environment and preprocessing
class TestEnvironmentPreprocessing(absltest.TestCase):
  def choose_random_game(self):
    games = ['BeamRider', 'Breakout', 'Pong',
             'Qbert', 'Seaquest', 'SpaceInvaders']
    ind = np.random.choice(len(games))
    return games[ind] + "NoFrameskip-v4"

  def test_creation(self):
    frame_shape = (84, 84, 4)
    game = self.choose_random_game()
    env = env_utils.create_env(game, clip_rewards=True)
    obs = env.reset()
    self.assertEqual(obs.shape, frame_shape)

  def test_step(self):
    frame_shape = (84, 84, 4)
    game = self.choose_random_game()
    env = env_utils.create_env(game, clip_rewards=True)
    obs = env.reset()
    actions = [1, 2, 3, 0]
    for a in actions:
      obs, reward, done, info = env.step(a)
      self.assertEqual(obs.shape, frame_shape)
      self.assertTrue(reward <= 1. and reward >= -1.)
      self.assertTrue(isinstance(done, bool))
      self.assertTrue(isinstance(info, dict))

# test the model (creation and forward pass)
class TestModel(absltest.TestCase):
  def choose_random_outputs(self):
    return np.random.choice([4, 5, 6, 7, 8, 9])

  def test_model(self):
    outputs = self.choose_random_outputs()
    module = models.ActorCritic(num_outputs=outputs)
    params = ppo_lib.get_initial_params(jax.random.PRNGKey(0), module)
    test_batch_size, obs_shape = 10, (84, 84, 4)
    random_input = np.random.random(size=(test_batch_size,) + obs_shape)
    log_probs, values = agent.policy_action(
        module.apply, params, random_input)
    self.assertEqual(values.shape, (test_batch_size, 1))
    sum_probs = np.sum(np.exp(log_probs), axis=1)
    self.assertEqual(sum_probs.shape, (test_batch_size, ))
    np_testing.assert_allclose(sum_probs, np.ones((test_batch_size, )),
                                atol=1e-6)

# test one optimization step
class TestOptimizationStep(absltest.TestCase):
  def generate_random_data(self, num_actions):
    data_len = 256 # equal to one default-sized batch
    state_shape = (84, 84, 4)
    states = np.random.randint(0, 255, size=((data_len, ) + state_shape))
    actions = np.random.choice(num_actions, size=data_len)
    old_log_probs = np.random.random(size=data_len)
    returns = np.random.random(size=data_len)
    advantages = np.random.random(size=data_len)
    return states, actions, old_log_probs, returns, advantages

  def test_optimization_step(self):
    num_outputs = 4
    trn_data = self.generate_random_data(num_actions=num_outputs)
    clip_param = 0.1
    vf_coeff = 0.5
    entropy_coeff = 0.01
    batch_size = 256
    module = models.ActorCritic(num_outputs)
    initial_params = ppo_lib.get_initial_params(jax.random.PRNGKey(0), module)
    config = ml_collections.ConfigDict({
      'learning_rate': 2.5e-4,
      'decaying_lr_and_clip_param': True,
    })
    state = ppo_lib.create_train_state(initial_params, module, config, 1000)
    state, _ = ppo_lib.train_step(
        state, trn_data, batch_size,
        clip_param=clip_param,
        vf_coeff=vf_coeff,
        entropy_coeff=entropy_coeff)
    self.assertIsInstance(state, train_state.TrainState)

if __name__ == '__main__':
  absltest.main()

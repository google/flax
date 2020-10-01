# Copyright 2020 The Flax Authors.
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

import jax
import flax
import numpy as onp
import numpy.testing as onp_testing
from absl.testing import absltest

import ppo_lib
import env_utils
import models
import agent

# test GAE
class TestGAE(absltest.TestCase):
  def test_gae_shape_on_random(self):
    # create random data, simulating 4 parallel envs and 20 time_steps
    envs, steps = 10, 100
    rewards = onp.random.choice([-1., 0., 1.], size=(steps, envs),
                                p=[0.01, 0.98, 0.01])
    terminal_masks = onp.ones(shape=(steps, envs), dtype=onp.float64)
    values = onp.random.random(size=(steps + 1, envs))
    discount = 0.99
    gae_param = 0.95
    adv = ppo_lib.gae_advantages(rewards, terminal_masks, values, discount,
                                 gae_param)
    self.assertEqual(adv.shape, (steps, envs))
    
  def test_gae_hardcoded(self):
    #test on small example that can be verified by hand
    rewards = onp.array([[1., 0.], [0., 0.], [-1., 1.]])
    #one of the two episodes terminated in the middle
    terminal_masks = onp.array([[1., 1.], [0., 1.], [1., 1.]])
    values = onp.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    discount = 0.5
    gae_param = 0.25
    correct_gae = onp.array([[0.375, -0.5546875], [-1., -0.4375], [-1.5, 0.5]])
    actual_gae = ppo_lib.gae_advantages(rewards, terminal_masks, values,
                                        discount, gae_param)
    onp_testing.assert_allclose(actual_gae, correct_gae)
# test environment and preprocessing
class TestEnvironmentPreprocessing(absltest.TestCase):
  def choose_random_game(self):
    games = ['BeamRider', 'Breakout', 'Pong',
             'Qbert', 'Seaquest', 'SpaceInvaders']
    ind = onp.random.choice(len(games))
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
    return onp.random.choice([4, 5, 6, 7, 8, 9])

  def test_model(self):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outputs = self.choose_random_outputs()
    initial_params = models.get_initial_params(subkey, num_outputs=outputs)
    lr = 2.5e-4
    optimizer = models.create_optimizer(initial_params, lr)
    self.assertTrue(isinstance(optimizer, flax.optim.base.Optimizer))
    test_batch_size, obs_shape = 10, (84, 84, 4)
    random_input = onp.random.random(size=(test_batch_size,) + obs_shape)
    log_probs, values = agent.policy_action(optimizer.target, random_input)
    self.assertEqual(values.shape, (test_batch_size, 1))
    sum_probs = onp.sum(onp.exp(log_probs), axis=1)
    self.assertEqual(sum_probs.shape, (test_batch_size, ))
    onp_testing.assert_allclose(sum_probs, onp.ones((test_batch_size, )),
                                atol=1e-6)

# test one optimization step
class TestOptimizationStep(absltest.TestCase):
  def generate_random_data(self, num_actions):
    data_len = 256 # equal to one default-sized batch
    state_shape = (84, 84, 4)
    states = onp.random.randint(0, 255, size=((data_len, ) + state_shape))
    actions = onp.random.choice(num_actions, size=data_len)
    old_log_probs = onp.random.random(size=data_len)
    returns = onp.random.random(size=data_len)
    advantages = onp.random.random(size=data_len)
    return states, actions, old_log_probs, returns, advantages

  def test_optimization_step(self):
    num_outputs = 4
    trn_data = self.generate_random_data(num_actions=num_outputs)
    clip_param = 0.1
    vf_coeff = 0.5
    entropy_coeff = 0.01
    lr = 2.5e-4
    batch_size = 256
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    initial_params = models.get_initial_params(subkey, num_outputs=num_outputs)
    lr = 2.5e-4
    optimizer = models.create_optimizer(initial_params, lr)
    optimizer, _ = ppo_lib.train_step(
        optimizer, trn_data, clip_param, vf_coeff, entropy_coeff, lr,
        batch_size)
    self.assertTrue(isinstance(optimizer, flax.optim.base.Optimizer))

if __name__ == '__main__':
  absltest.main()

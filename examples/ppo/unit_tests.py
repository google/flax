import jax
import flax
from flax import nn
import numpy as onp
import numpy.testing as onp_testing
from absl.testing import absltest

import main
import env_utils
import models

#test GAE
class TestGAE(absltest.TestCase):
  def test_gae_random(self):
    # create random data, simulating 4 parallel envs and 20 time_steps
    envs, steps = 10, 100
    rewards = onp.random.choice([-1., 0., 1.], size=(steps, envs),
                                p=[0.01, 0.98, 0.01])
    terminal_masks = onp.ones(shape=(steps, envs), dtype=onp.float64)
    values = onp.random.random(size=(steps + 1, envs))
    discount = 0.99
    gae_param = 0.95
    adv = main.gae_advantages(rewards, terminal_masks, values, discount, 
                              gae_param)
    self.assertEqual(adv.shape, (steps, envs))
    # test the property A_{t} = \delta_t + \gamma*\lambda*A_{t+1}
    # for each agent separately
    for e in range(envs):
      for t in range(steps-1):
        delta = rewards[t, e] + discount * values[t+1, e] - values[t, e]
        lhs = adv[t, e]
        rhs = delta + discount * gae_param * adv[t+1, e]
        onp_testing.assert_almost_equal(lhs, rhs)

#test environment and preprocessing
class TestEnvironmentPreprocessing(absltest.TestCase):
  def choose_random_game(self):
    games = ['BeamRider', 'Breakout', 'Pong',
              'Qbert', 'Seaquest', 'SpaceInvaders']
    ind = onp.random.choice(len(games))
    return games[ind] + "NoFrameskip-v4"

  def test_creation(self):
    frame_shape = (84, 84, 4)
    game = self.choose_random_game()
    env = env_utils.create_env(game)
    obs = env.reset()
    self.assertTrue(obs.shape == frame_shape)

  def test_step(self):
    frame_shape = (84, 84, 4)
    game = self.choose_random_game()
    env = env_utils.create_env(game)
    obs = env.reset()
    actions = [1, 2, 3, 0]
    for a in actions:
      obs, reward, done, info = env.step(a)
      self.assertTrue(obs.shape == frame_shape)
      self.assertTrue(reward <= 1. and reward >= -1.)
      self.assertTrue(isinstance(done, bool))
      self.assertTrue(isinstance(info, dict))

#test the model (creation and forward pass)
class TestModel(absltest.TestCase):
  def choose_random_outputs(self):
    return onp.random.choice([4,5,6,7,8,9])

  def test_model(self):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outputs = self.choose_random_outputs()
    model = models.create_model(subkey, outputs)
    optimizer = models.create_optimizer(model, learning_rate=1e-3)
    self.assertTrue(isinstance(model, nn.base.Model))
    self.assertTrue(isinstance(optimizer, flax.optim.base.Optimizer))
    test_batch_size, obs_shape = 10, (84, 84, 4)
    random_input = onp.random.random(size=(test_batch_size,) + obs_shape)
    log_probs, values = optimizer.target(random_input)
    self.assertTrue(values.shape == (test_batch_size, 1))
    sum_probs = onp.sum(onp.exp(log_probs), axis=1)
    self.assertTrue(sum_probs.shape == (test_batch_size, ))
    onp_testing.assert_almost_equal(sum_probs, onp.ones((test_batch_size, )))


if __name__ == '__main__':
  absltest.main()

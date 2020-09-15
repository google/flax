
import jax
import flax
from flax import nn
import numpy as onp

import numpy.testing as onp_testing
from absl.testing import absltest

#test GAE
from main import gae_advantages
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
    adv = gae_advantages(rewards, terminal_masks, values, discount, gae_param)
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
from remote import RemoteSimulator, rcv_action_send_exp
from env import create_env
class TestEnvironmentPreprocessing(absltest.TestCase):
  def test_creation(self):
    frame_shape = (84, 84, 4)
    env = create_env()
    obs = env.reset()
    self.assertTrue(obs.shape == frame_shape)

  def test_step(self):
    frame_shape = (84, 84, 4)
    env = create_env()
    obs = env.reset()
    actions = [1, 2, 3, 0]
    for a in actions:
      obs, reward, done, info = env.step(a)
      self.assertTrue(obs.shape == frame_shape)
      self.assertTrue(reward <= 1. and reward >= -1.)
      self.assertTrue(isinstance(done, bool))
      self.assertTrue(isinstance(info, dict))

#test creation of the model and optimizer
from models import create_model, create_optimizer
class TestCreation(absltest.TestCase):
  def test_create(self):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    policy_model = create_model(subkey)
    policy_optimizer = create_optimizer(policy_model, learning_rate=1e-3)
    self.assertTrue(isinstance(policy_model, nn.base.Model))
    self.assertTrue(isinstance(policy_optimizer, flax.optim.base.Optimizer))

if __name__ == '__main__':
  absltest.main()
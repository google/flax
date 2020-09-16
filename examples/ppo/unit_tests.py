
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

#test the model (creation and forward pass)
from models import create_model, create_optimizer
class TestModel(absltest.TestCase):
  def test_model(self):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = create_model(subkey)
    optimizer = create_optimizer(model, learning_rate=1e-3)
    self.assertTrue(isinstance(model, nn.base.Model))
    self.assertTrue(isinstance(optimizer, flax.optim.base.Optimizer))
    test_batch_size, obs_shape = 10, (84, 84, 4)
    random_input = onp.random.random(size=(test_batch_size,) + obs_shape)
    probs, values = optimizer.target(random_input)
    self.assertTrue(values.shape == (test_batch_size, 1))
    sum_probs = onp.sum(probs, axis=1)
    self.assertTrue(sum_probs.shape == (test_batch_size, ))
    onp_testing.assert_almost_equal(sum_probs, onp.ones((test_batch_size, )))




if __name__ == '__main__':
  absltest.main()
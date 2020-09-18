import collections
import gym
import numpy as onp

from seed_rl_atari_preprocessing import AtariPreprocessing

class ClipRewardEnv(gym.RewardWrapper):
  """This class is adatpted from OpenAI baselines
  github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
  """
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """Bin reward to {+1, 0, -1} by its sign."""
    return onp.sign(reward)

class FrameStack:
  '''Class that wraps an AtariPreprocessing object and implements
  stacking of `num_frames` last frames of the game.
  '''
  def __init__(self, preproc: AtariPreprocessing, num_frames : int):
    self.preproc = preproc
    self.num_frames = num_frames
    self.frames = collections.deque(maxlen=num_frames)

  def reset(self):
    ob = self.preproc.reset()
    for _ in range(self.num_frames):
      self.frames.append(ob)
    return self._get_array()

  def step(self, action: int):
    ob, reward, done, info = self.preproc.step(action)
    self.frames.append(ob)
    return self._get_array(), reward, done, info

  def _get_array(self):
    assert len(self.frames) == self.num_frames
    return onp.concatenate(self.frames, axis=-1)

def create_env(game : str):
  '''Create a FrameStack object that serves as environment for the `game`.
  '''
  env = gym.make(game)
  env = ClipRewardEnv(env) # bin rewards to {-1., 0., 1.}
  preproc = AtariPreprocessing(env)
  stack = FrameStack(preproc, num_frames=4)
  return stack

def get_num_actions(game : str):
  """Get the number of possible actions of a given Atari game. This determines
  the number of outputs in the actor part of the actor-critic model.
  """
  env = gym.make(game)
  return env.action_space.n


import collections
import gym
import numpy as onp

from seed_rl_atari_preprocessing import AtariPreprocessing

class FrameStack:
  '''Class that wraps an AtariPreprocessing object and implements
  stacking of `num_frames` last frames of the game
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

def create_env():
  env = gym.make("PongNoFrameskip-v4")
  preproc = AtariPreprocessing(env)
  stack = FrameStack(preproc, num_frames=4)
  return stack
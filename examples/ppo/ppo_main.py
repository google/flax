# Copyright 2021 The Flax Authors.
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

# See issue #620.
# pytype: disable=wrong-keyword-args

import os
from absl import flags
from absl import app
import jax
import jax.random
import tensorflow as tf
from ml_collections import config_flags

import ppo_lib
import models
import env_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir',
    default='/tmp/ppo_training',
    help=('Directory to save checkpoints and logging info.'))

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the default configuration file.',
    lock_config=True)

flags.mark_flags_as_required(['config'])


def main(argv):
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')
  config = FLAGS.config
  game = config.game + 'NoFrameskip-v4'
  num_actions = env_utils.get_num_actions(game)
  print(f'Playing {game} with {num_actions} actions')
  module = models.ActorCritic(num_outputs=num_actions)
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  initial_params = models.get_initial_params(subkey, module)
  optimizer = models.create_optimizer(initial_params, config.learning_rate)
  optimizer = ppo_lib.train(module, optimizer, config, FLAGS.workdir)

if __name__ == '__main__':
  app.run(main)

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

import os
from absl import flags
from absl import app
import jax
import jax.random
from ml_collections import config_flags

import ppo_lib
import models
import env_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'logdir', default='/tmp/ppo_training',
    help=('Directory to save checkpoints and logging info.'))

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'default_config.py'),
    'File path to the default configuration file.')

def main(argv):
  config = FLAGS.config
  game = config.game + 'NoFrameskip-v4'
  num_actions = env_utils.get_num_actions(game)
  print(f'Playing {game} with {num_actions} actions')
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model = models.create_model(subkey, num_outputs=num_actions)
  optimizer = models.create_optimizer(model, learning_rate=config.learning_rate)
  del model
  optimizer = ppo_lib.train(optimizer, config, FLAGS.logdir)

if __name__ == '__main__':
  app.run(main)

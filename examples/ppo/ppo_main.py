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

# See issue #620.
# pytype: disable=wrong-keyword-args

from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow as tf

import env_utils
import models
import ppo_lib


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir',
    default='/tmp/ppo_training',
    help=('Directory to save checkpoints and logging info.'))

config_flags.DEFINE_config_file(
    'config',
    "configs/default.py",
    'File path to the default configuration file.',
    lock_config=True)


def main(argv):
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')
  config = FLAGS.config
  game = config.game + 'NoFrameskip-v4'
  num_actions = env_utils.get_num_actions(game)
  print(f'Playing {game} with {num_actions} actions')
  model = models.ActorCritic(num_outputs=num_actions)
  ppo_lib.train(model, config, FLAGS.workdir)

if __name__ == '__main__':
  app.run(main)

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

# Lint as: python3
"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import os

from absl import app
from absl import flags

from ml_collections import config_flags

import tensorflow as tf

# Local imports.
import train


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir', default=None,
    help=('Directory to store model data.'))

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'configs/default.py'),
    'File path to the Training hyperparameter configuration.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')
  # Require JAX omnistaging mode.
  jax.config.enable_omnistaging()

  train.train_and_evaluate(workdir=FLAGS.workdir, config=FLAGS.config)


if __name__ == '__main__':
  app.run(main)

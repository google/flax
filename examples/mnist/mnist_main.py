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

"""MNIST example.

This script trains a simple Convolutional Neural Net on the MNIST dataset.

"""

import os

from absl import app
from absl import flags
from ml_collections import config_flags

import mnist_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'configs/default.py'),
    'File path to the Training hyperparameter configuration.')

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))


def main(_):
  mnist_lib.train_and_evaluate(config=FLAGS.config, model_dir=FLAGS.model_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_dir'])
  app.run(main)

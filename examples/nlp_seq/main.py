# Copyright 2024 The Flax Authors.
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

"""Main file for running the NLP sequence tagging example.

This file is intentionally kept short to allow config-based execution.
"""

from absl import app
from absl import flags
import train
from ml_collections import config_flags


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    'configs/default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Convert config to FLAGS for train.py compatibility
  config = FLAGS.config

  # Override FLAGS with config values
  FLAGS.model_dir = config.model_dir
  FLAGS.experiment = config.experiment
  FLAGS.batch_size = config.batch_size
  FLAGS.eval_frequency = config.eval_frequency
  FLAGS.num_train_steps = config.num_train_steps
  FLAGS.learning_rate = config.learning_rate
  FLAGS.weight_decay = config.weight_decay
  FLAGS.max_length = config.max_length
  FLAGS.random_seed = config.random_seed
  FLAGS.train = config.train
  FLAGS.dev = config.dev

  # Run the training
  train.main(argv)


if __name__ == '__main__':
  app.run(main)

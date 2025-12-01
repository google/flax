# Copyright 2025 The Flax Authors.
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

"""Main script for seq2seq example."""

from absl import app
from absl import flags
from absl import logging
import train
from ml_collections import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


def main(argv):
  del argv

  config = FLAGS.config

  # Set train.FLAGS values from config
  train.FLAGS.workdir = config.workdir
  train.FLAGS.learning_rate = config.learning_rate
  train.FLAGS.batch_size = config.batch_size
  train.FLAGS.hidden_size = config.hidden_size
  train.FLAGS.num_train_steps = config.num_train_steps
  train.FLAGS.decode_frequency = config.decode_frequency
  train.FLAGS.max_len_query_digit = config.max_len_query_digit

  logging.info('Starting training with config: %s', config)
  _ = train.train_and_evaluate(train.FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)

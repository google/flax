# Copyright 2023 The Flax Authors.
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

"""Main file for running the VAE example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
import jax
import tensorflow as tf

import train


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3, help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer('batch_size', default=128, help=('Batch size for training.'))

flags.DEFINE_integer('num_epochs', default=30, help=('Number of training epochs.'))

flags.DEFINE_integer('latents', default=20, help=('Number of latent variables.'))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  train.train_and_evaluate(
      FLAGS.batch_size, FLAGS.learning_rate, FLAGS.num_epochs, FLAGS.latents
  )


if __name__ == '__main__':
  app.run(main)

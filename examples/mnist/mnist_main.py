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

from absl import app
from absl import flags

import mnist_lib

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=10,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))

flags.mark_flag_as_required('model_dir')


def main(_):
  mnist_lib.train_and_evaluate(
      model_dir=FLAGS.model_dir, num_epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
      momentum=FLAGS.momentum)


if __name__ == '__main__':
  app.run(main)

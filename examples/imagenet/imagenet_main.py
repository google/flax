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

from absl import app
from absl import flags

import imagenet_lib


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=90,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_bool(
    'cache', default=False,
    help=('If True, cache the dataset.'))

flags.DEFINE_bool(
    'half_precision', default=False,
    help=('If bfloat16/float16 should be used instead of float32.'))

flags.DEFINE_integer(
    'num_train_steps', default=-1,
    help=('Number of training steps to be executed in a single epoch.'
          'Default = -1 signifies using the entire TRAIN split.'))

flags.DEFINE_integer(
    'num_eval_steps', default=-1,
    help=('Number of evaluation steps to be executed in a single epoch.'
          'Default = -1 signifies using the entire VALIDATION split.'))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  imagenet_lib.train_and_evaluate(
      model_dir=FLAGS.model_dir, batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, learning_rate=FLAGS.learning_rate,
      momentum=FLAGS.momentum, cache=FLAGS.cache,
      half_precision=FLAGS.half_precision,
      num_train_steps=FLAGS.num_train_steps,
      num_eval_steps=FLAGS.num_eval_steps)


if __name__ == '__main__':
  app.run(main)

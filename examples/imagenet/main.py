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

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import os

from absl import app
from absl import flags
from absl import logging
from clu import platform
import train
import jax
from ml_collections import config_flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir', default=None, help='Directory to store model data.')
config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'configs/default.py'),
    'File path to the training hyperparameter configuration.')
flags.mark_flags_as_required(['workdir'])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX host: %d / %d', jax.host_id(), jax.host_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f'host_id: {jax.host_id()}, host_count: {jax.host_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train.train_and_evaluate(config=FLAGS.config, workdir=FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)

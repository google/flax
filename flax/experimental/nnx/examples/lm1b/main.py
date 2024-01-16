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

"""Main file for running the Language Modelling example with LM1B.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import jax
import tensorflow as tf
import train
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
  'config',
  'configs/default.py',
  'File path to the training hyperparameter configuration.',
  lock_config=True,
)
flags.mark_flags_as_required(['config', 'workdir'])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
    f'process_index: {jax.process_index()}, '
    f'process_count: {jax.process_count()}'
  )
  platform.work_unit().create_artifact(
    platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
  )

  train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)

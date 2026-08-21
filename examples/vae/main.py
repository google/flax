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

"""Main file for running the VAE example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging
import jax
import tensorflow as tf
import time
import train
from configs.default import get_default_config

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store logs and checkpoints.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Parse arguments and get config
  config = get_default_config()

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Simple process logging
  logging.info('Starting training process %d/%d', jax.process_index(), jax.process_count())

  start = time.perf_counter()
  train.train_and_evaluate(config)
  logging.info('Total training time: %.2f seconds', time.perf_counter() - start)

if __name__ == '__main__':
  app.run(main)

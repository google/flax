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
"""Tests for flax.examples.imagenet.imagenet_lib."""

import os
import pathlib
import tempfile

from absl.testing import absltest

import tensorflow as tf
import tensorflow_datasets as tfds

from configs import default as config_lib
import imagenet_lib


class ImageNetTest(absltest.TestCase):
  """Test cases for imagenet_lib."""

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_train_and_evaluate(self):
    """Tests training and evaluation loop using mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    # Define training configuration.
    config = config_lib.get_config()
    config.batch_size = 1
    config.num_epochs = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      imagenet_lib.train_and_evaluate(config=config, model_dir=model_dir)


if __name__ == '__main__':
  absltest.main()

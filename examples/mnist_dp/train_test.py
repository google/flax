# Copyright 2022 The Flax Authors.
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

"""Tests for flax.examples.mnist.mnist_lib."""

import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import train
from configs import dpsgd
from configs import sgd
import tensorflow as tf
import tensorflow_datasets as tfds


def get_config(config_type: str):
  all_configs = {
      'sgd': sgd.get_config,
      'dpsgd': dpsgd.get_config,
  }
  try:
    return all_configs[config_type]()
  except KeyError as err:
    raise ValueError(f'Unsupported config type: {config_type}') from err


class TrainTest(parameterized.TestCase):
  """Test cases for train."""

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  @parameterized.product(
      config_type=['sgd', 'dpsgd']
  )
  def test_train_and_evaluate(self, config_type):
    """Tests training and evaluation code by running a single step."""
    # Create a temporary directory where tensorboard metrics are written.
    workdir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    # Define training configuration.
    config = get_config(config_type)
    config.batch_size = 8

    with tfds.testing.mock_data(num_examples=40, data_dir=data_dir):
      train.train_and_evaluate(config=config, workdir=workdir)


if __name__ == '__main__':
  absltest.main()

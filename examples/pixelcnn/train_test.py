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

import pathlib
import tempfile

from absl import logging
from absl.testing import absltest
import tensorflow as tf
import tensorflow_datasets as tfds

from configs import default
import train


def get_test_config():
  config = default.get_config()
  config.init_batch_size = 8
  config.batch_size = 8
  config.num_epochs = 1
  config.n_resent = 1
  config.n_feature = 8
  return config


class TrainTest(absltest.TestCase):
  """Test cases for PixelCNN library."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_train_and_evaluate(self):
    config = get_test_config()
    workdir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))


if __name__ == '__main__':
  absltest.main()

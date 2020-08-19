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
"""Tests for flax.examples.cifar10.train."""

import pathlib
import tempfile

from absl.testing import absltest
import tensorflow_datasets as tfds

import train as cifar_train


class CifarTest(absltest.TestCase):
  """Test class for Cifar10 training and evaluation loop using mocked data."""

  def test_train_and_eval(self):
    """Tests cifar10 training and evaluation loop."""
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      cifar_train.train_and_evaluate(
          model_dir=model_dir, batch_size=1, num_epochs=1, learning_rate=0.1,
          sgd_momentum=0.9, num_train_steps=1, num_eval_steps=1)


if __name__ == '__main__':
  absltest.main()

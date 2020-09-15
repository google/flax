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

"""Tests for flax.examples.mnist.mnist_lib."""

import pathlib
import tempfile

from absl.testing import absltest

import jax
from jax import numpy as jnp

import tensorflow_datasets as tfds

from configs import default as config_lib
import mnist_lib


class MnistLibTest(absltest.TestCase):
  """Test cases for mnist_lib."""

  def test_cnn(self):
    """Tests CNN module used as the trainable model."""
    rng = jax.random.PRNGKey(0)
    output, init_params = mnist_lib.CNN.init_by_shape(
        rng, [((5, 224, 224, 3), jnp.float32)])

    self.assertEqual((5, 10), output.shape)

    # TODO(mohitreddy): Consider creating a testing module which
    # gives a parameters overview including number of parameters.
    self.assertLen(init_params, 4)

  def test_train_and_evaluate(self):
    """Tests training and evaluation code by running a single step with
       mocked data for MNIST dataset.
    """
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    # Define training configuration.
    config = config_lib.get_config()
    config.num_epochs = 1
    config.batch_size = 8

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      mnist_lib.train_and_evaluate(config=config, model_dir=model_dir)


if __name__ == '__main__':
  absltest.main()

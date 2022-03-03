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

"""Tests for flax.examples.imagenet.train."""

import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds

# Local imports.
import models
import train
from configs import default as default_lib


jax.config.update('jax_disable_most_optimizations', True)


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_create_model(self):
    """Tests creating model."""
    model = train.create_model(model_cls=models._ResNet1, half_precision=False)  # pylint: disable=protected-access
    params, batch_stats = train.initialized(random.PRNGKey(0), 224, model)
    variables = {'params': params, 'batch_stats': batch_stats}
    x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
    y = model.apply(variables, x, train=False)
    self.assertEqual(y.shape, (8, 1000))

  def test_create_model_local(self):
    """Tests creating an unshared convolution model.

    Uses smaller inputs than `test_create_model` to due to higher compute.
    """
    model = train.create_model(model_cls=models._ResNet1Local, half_precision=False)  # pylint: disable=protected-access
    params, batch_stats = train.initialized(random.PRNGKey(0), 64, model)
    variables = {'params': params, 'batch_stats': batch_stats}
    x = random.normal(random.PRNGKey(1), (1, 64, 64, 3))
    y = model.apply(variables, x, train=False)
    self.assertEqual(y.shape, (1, 1000))

  @parameterized.product(
      model=('_ResNet1', '_ResNet1Local')
  )
  def test_train_and_evaluate(self, model):
    """Tests training and evaluation loop using mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    workdir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    # Define training configuration
    config = default_lib.get_config()
    config.model = model
    config.batch_size = 1
    config.num_epochs = 1
    config.num_train_steps = 1
    config.steps_per_eval = 1

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      train.train_and_evaluate(workdir=workdir, config=config)


if __name__ == '__main__':
  absltest.main()

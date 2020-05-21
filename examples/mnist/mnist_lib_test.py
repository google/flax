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
"""Tests for flax.examples.mnist.mnist_lib."""

import os
import tempfile

from absl import logging
from absl.testing import absltest

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

import mnist_lib

class MnistLibTest(absltest.TestCase):
  """Test cases for Mnist."""

  def test_model(self):
    rng = jax.random.PRNGKey(0)
    model_def = mnist_lib.CNN.partial(num_classes=10)
    output, init_params = model_def.init_by_shape(
        rng, [((1, 28, 28, 1), jnp.float32)])

    self.assertEqual((1, 10), output.shape)

    # Assert number of layers.
    self.assertEqual(4, len(init_params))

    # Assert shape of each layer.
    self.assertEqual((32,), init_params['Conv_0']['bias'].shape)
    self.assertEqual((3, 3, 1, 32), init_params['Conv_0']['kernel'].shape)

    self.assertEqual((64,), init_params['Conv_1']['bias'].shape)
    self.assertEqual((3, 3, 32, 64), init_params['Conv_1']['kernel'].shape)

    self.assertEqual((256,), init_params['Dense_2']['bias'].shape)
    self.assertEqual((3136, 256), init_params['Dense_2']['kernel'].shape)

    self.assertEqual((10,), init_params['Dense_3']['bias'].shape)
    self.assertEqual((256, 10), init_params['Dense_3']['kernel'].shape)

  def test_train_and_eval(self):
    # Get testing directory.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = parent_dir + '/testing/datasets'

    model_dir = tempfile.mkdtemp()
    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      mnist_lib.train_and_evaluate(
          model_dir=model_dir, batch_size=8, num_epochs=1,
          learning_rate=0.1, momentum=0.9)
    logging.info('workdir content: %s', tf.io.gfile.listdir(model_dir))

if __name__ == '__main__':
  absltest.main()

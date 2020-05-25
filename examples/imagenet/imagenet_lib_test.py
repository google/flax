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
import tempfile

from absl import logging
from absl.testing import absltest

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import flax.testing as testing

import imagenet_lib


class ImageNetLibTest(absltest.TestCase):

  def test_train_and_evaluate(self):
    # Get testing directory.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = parent_dir + '/testing/datasets'

    model_dir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      imagenet_lib.train_and_evaluate(
          model_dir=model_dir, batch_size=8, num_epochs=1,
          learning_rate=0.1, momentum=0.9, cache=True, half_precision=False,
          num_train_and_eval_steps=1)
    logging.info('Finished testing. Found: %s', tf.io.gfile.listdir(model_dir))

    summaries = testing.get_tensorboard_scalars(model_dir)
    logging.info('summaries: %s', summaries)


if __name__ == '__main__':
  absltest.main()

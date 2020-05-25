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
"""Tests for flax.examples.imagenet.input_pipeline."""

import os
import tempfile

from absl import logging
from absl.testing import absltest

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import input_pipeline


class InputPipelineTest(absltest.TestCase):

  def test_train_and_evaluate(self):
    # Get testing directory.
    tf.enable_v2_behavior()
    # make sure tf does not allocate gpu memory
    tf.config.experimental.set_visible_devices([], 'GPU')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = parent_dir + '/testing/datasets'

    with tfds.testing.mock_data(num_examples=32, data_dir=data_dir):
      split = 'train[{}:{}]'.format(0, 32)
      ds = tfds.load(
          'imagenet2012:5.*.*', split=split,
          decoders={'image': tfds.decode.SkipDecoding()})
      ds_iter = iter(ds)
      for example in ds_iter:
        input_pipeline.preprocess_for_train(example['image'])



if __name__ == '__main__':
  absltest.main()

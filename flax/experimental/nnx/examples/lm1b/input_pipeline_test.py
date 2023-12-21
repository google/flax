# Copyright 2023 The Flax Authors.
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

import os
import pathlib
import tempfile

import input_pipeline
import tensorflow_datasets as tfds
from absl.testing import absltest
from configs import default

# We just use different values here to verify that the input pipeline uses the
# the correct value for the 3 different datasets.
_TARGET_LENGTH = 32
_EVAL_TARGET_LENGTH = 48
_PREDICT_TARGET_LENGTH = 64


class InputPipelineTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.train_ds, self.eval_ds, self.predict_ds = self._get_datasets()

  def _get_datasets(self):
    config = default.get_config()
    config.per_device_batch_size = 1
    config.eval_per_device_batch_size = 2
    config.vocab_size = 32
    config.max_corpus_chars = 1000
    config.max_target_length = _TARGET_LENGTH
    config.max_eval_target_length = _EVAL_TARGET_LENGTH
    config.max_predict_length = _PREDICT_TARGET_LENGTH

    vocab_path = os.path.join(tempfile.mkdtemp(), 'sentencepiece_model')

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    with tfds.testing.mock_data(num_examples=128, data_dir=data_dir):
      train_ds, eval_ds, predict_ds, _ = input_pipeline.get_datasets(
        n_devices=2, config=config, vocab_path=vocab_path
      )
    return train_ds, eval_ds, predict_ds

  def test_train_ds(self):
    expected_shape = [2, _TARGET_LENGTH]  # 2 devices.
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    for batch in self.train_ds.take(3):
      self.assertEqual(
        {k: v.shape.as_list() for k, v in batch.items()},
        {
          'inputs': expected_shape,
          'inputs_position': expected_shape,
          'inputs_segmentation': expected_shape,
          'targets': expected_shape,
          'targets_position': expected_shape,
          'targets_segmentation': expected_shape,
        },
      )

  def test_eval_ds(self):
    expected_shape = [4, _EVAL_TARGET_LENGTH]  # 2 devices.
    for batch in self.eval_ds.take(3):
      self.assertEqual(
        {k: v.shape.as_list() for k, v in batch.items()},
        {
          'inputs': expected_shape,
          'targets': expected_shape,
        },
      )

  def test_predict_ds(self):
    expected_shape = [4, _PREDICT_TARGET_LENGTH]  # 2 devices.
    for batch in self.predict_ds.take(3):
      self.assertEqual(
        {k: v.shape.as_list() for k, v in batch.items()},
        {
          'inputs': expected_shape,
          'targets': expected_shape,
        },
      )


if __name__ == '__main__':
  absltest.main()

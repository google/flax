# Copyright 2024 The Flax Authors.
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
import numpy as np

from absl.testing import absltest
import tensorflow_datasets as tfds

from configs import default
import input_pipeline

# We just use different values here to verify that the input pipeline uses the
# the correct value for the 3 different datasets.
_TARGET_LENGTH = 32
_EVAL_TARGET_LENGTH = 48


class InputPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_ds, self.eval_ds = self._get_datasets()

  def _get_datasets(self):
    config = default.get_config()
    config.per_device_batch_size = 2
    config.eval_per_device_batch_size = 4
    config.vocab_size = 123
    config.max_corpus_chars = 1000
    config.max_target_length = _TARGET_LENGTH
    config.max_eval_target_length = _EVAL_TARGET_LENGTH
    config.prefetch_num_workers = 2

    vocab_path = os.path.join(tempfile.mkdtemp(), 'sentencepiece_model')

    # Go two directories up to the root of the flax directory.
    # "/path/to/flax/examples/lm1b_nnx/models_test.py" -> "/path/to/flax"
    flax_root_dir = pathlib.Path(__file__).absolute().parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    with tfds.testing.mock_data(num_examples=128, data_dir=data_dir):
      train_ds, eval_ds, _ = input_pipeline.get_datasets(config=config, vocab_path=vocab_path)
    return train_ds, eval_ds

  def test_train_ds(self):
    expected_shape = [2, _TARGET_LENGTH]
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    counter = 4
    train_iter = iter(self.train_ds)
    while counter > 0:
      counter -= 1
      batch = next(train_iter)
      self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
          'inputs': expected_shape,
          'inputs_position': expected_shape,
          'inputs_segmentation': expected_shape,
          'targets': expected_shape,
          'targets_position': expected_shape,
          'targets_segmentation': expected_shape,
        },
      )
      # batch["*_position"] and batch["*_segmentation"] are SharedMemoryArray
      # We wrap them into numpy array before calling np.testing.assert_array_equal
      # to avoid warnings like here: https://github.com/google/grain/issues/917
      np.testing.assert_array_equal(
        np.asarray(batch["inputs_position"]),
        np.asarray(batch["targets_position"]),
      )
      np.testing.assert_array_equal(
        np.asarray(batch["inputs_segmentation"][..., :-1]),
        np.asarray(batch["targets_segmentation"][..., :-1]),
      )
      np.testing.assert_array_equal(
        np.asarray(batch["inputs"][..., 1:]),
        np.asarray(batch["targets"][..., :-1]),
      )

  def test_eval_ds(self):
    expected_shape = [4, _EVAL_TARGET_LENGTH]

    counter = 4
    eval_iter = iter(self.eval_ds)
    while counter > 0:
      counter -= 1
      batch = next(eval_iter)
      self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
          'inputs': expected_shape,
          'targets': expected_shape,
        },
      )
      np.testing.assert_array_equal(
        batch["inputs"][:, 1:],
        batch["targets"][:, :-1],
      )


if __name__ == '__main__':
  absltest.main()

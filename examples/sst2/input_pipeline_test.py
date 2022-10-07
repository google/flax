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

import os
import pathlib
import tempfile

from absl.testing import absltest
import tensorflow_datasets as tfds

import input_pipeline
import vocabulary


class InputPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab_path = self._get_vocab_path()
    self.dataset = self._get_dataset(self.vocab_path)

  def _get_vocab_path(self):
    """Prepares a mock vocabulary and returns a path to it."""
    vocab_path = os.path.join(tempfile.mkdtemp(), 'vocab.txt')
    tokenized_sequences = (
        (b'this', b'is', b'a', b'test', b'sentence'),
        (b'this', b'is', b'a', b'test', b'sentence'),
        (b'this', b'is', b'a', b'test', b'sentence'),
    )
    vocab = vocabulary.Vocabulary(tokenized_sequences=tokenized_sequences)
    vocab.save(vocab_path)
    return vocab_path

  def _get_dataset(self, vocab_path):
    """Uses mock data to create the dataset."""
    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + \
        '/.tfds/metadata'  # pylint: disable=unused-variable
    with tfds.testing.mock_data(num_examples=128, data_dir=data_dir):
      return input_pipeline.TextDataset(vocab_path=vocab_path, split='train')

  def test_bucketed_dataset(self):
    """Each batch should have a length that is a multiple of bucket_size."""
    batch_size = 2
    bucket_size = 8
    for batch in self.dataset.get_bucketed_batches(
            batch_size=batch_size,
            bucket_size=bucket_size, max_input_length=60, shuffle=False).take(3):
      # Because of bucketing, sequence length must be multiple of bucket_size.
      length = batch['token_ids'].numpy().shape[-1]
      self.assertEqual(0, length % bucket_size)
      self.assertEqual((batch_size,), batch['length'].numpy().shape)
      self.assertEqual((batch_size,), batch['label'].numpy().shape)

  def test_batched_dataset(self):
    """Tests that the length of a batch matches the longest sequence."""
    batch_size = 2
    for batch in self.dataset.get_batches(
            batch_size=batch_size, shuffle=False).take(1):
      # Each batch is padded to the maximum sentence length in the batch.
      max_length_in_batch = max(batch['length'].numpy())
      length = batch['token_ids'].numpy().shape[-1]
      self.assertEqual(max_length_in_batch, length)
      self.assertEqual((batch_size,), batch['length'].numpy().shape)
      self.assertEqual((batch_size,), batch['label'].numpy().shape)

  def test_batched_dataset_fixed_length(self):
    """Tests that each batch has the fixed length."""
    batch_size = 2
    fixed_pad_length = 77
    for batch in self.dataset.get_batches(
            batch_size=batch_size, shuffle=False,
            fixed_pad_length=fixed_pad_length).take(1):
      length = batch['token_ids'].numpy().shape[-1]
      self.assertEqual(fixed_pad_length, length)


if __name__ == '__main__':
  absltest.main()

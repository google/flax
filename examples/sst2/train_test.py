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
"""Tests for flax.examples.sst2.train."""

import pathlib
import tempfile

from absl.testing import absltest

import tensorflow_datasets as tfds

import train as sst2_train


class Sst2TrainTest(absltest.TestCase):
  """Test cases for SST2 train file."""

  def test_train_and_evaluate(self):
    """Tests training and evaluation loop using TFDS mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      sst2_train.train_and_evaluate(
          seed=0, model_dir=model_dir, num_epochs=1, batch_size=8,
          embedding_size=256, hidden_size=256, min_freq=5, max_seq_len=55,
          dropout=0.5, emb_dropout=0.5, word_dropout_rate=0.1,
          learning_rate=0.0005, checkpoints_to_keep=0, l2_reg=1e-6)


if __name__ == '__main__':
  absltest.main()

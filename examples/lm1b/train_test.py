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
"""Tests for flax.examples.lm1b.train."""

import pathlib
import tempfile

from absl.testing import absltest

import tensorflow_datasets as tfds

import train as lm1b_train


class Lm1bTrainTest(absltest.TestCase):
  """Test cases for train.py."""

  def test_train_and_evaluate(self):
    """Tests training and evaluation loop using mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      lm1b_train.train_and_evaluate(
          random_seed=0, batch_size=1, learning_rate=0.05, num_train_steps=1,
          num_eval_steps=1, eval_freq=1, max_target_length=10,
          max_eval_target_length=32, weight_decay=1e-1, data_dir=None,
          model_dir=model_dir, restore_checkpoints=False,
          save_checkpoints=False, checkpoint_freq=2,
          max_predict_token_length=2, sampling_temperature=0.6,
          sampling_top_k=4, prompt_str='unittest ')


if __name__ == '__main__':
  absltest.main()
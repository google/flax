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
"""Tests for flax.examples.wmt.train."""

import pathlib
import tempfile

from absl.testing import absltest

import tensorflow_datasets as tfds

import train as wmt_train


class WmtTrainTest(absltest.TestCase):
  """Test cases for wmt train."""

  def test_train_and_evaluate(self):
    """Tests training and evaluation loop using mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      wmt_train.train_and_evaluate(
          random_seed=0, jax_backend_target=None, model_dir=model_dir,
          data_dir=None, vocab_path=None, vocab_size=16,
          dataset_name='wmt17_translate/de-en', eval_dataset_name=None,
          batch_size=1, beam_size=1, eval_frequency=1, num_train_steps=1,
          num_eval_steps=1, learning_rate=0.0625, warmup_steps=0,
          label_smoothing=0.1, weight_decay=0.0, max_target_length=10,
          max_eval_target_length=32, max_predict_length=10, emb_dim=64,
          num_heads=1, num_layers=2, qkv_dim=4, mlp_dim=8,
          share_embeddings=True, logits_via_embedding=False,
          use_bfloat16=False, restore_checkpoints=False,
          save_checkpoints=False, checkpoint_freq=2)


if __name__ == '__main__':
  absltest.main()

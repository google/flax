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

import pathlib
import tempfile

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import train
from absl import logging
from absl.testing import absltest
from configs import default

jax.config.update('jax_disable_most_optimizations', True)


class TrainTest(absltest.TestCase):
  """Test cases for LM library."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_train_and_evaluate(self):
    config = default.get_config()
    config.max_corpus_chars = 1000
    config.vocab_size = 32
    config.per_device_batch_size = 2
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_predict_steps = 1

    config.num_layers = 1
    config.qkv_dim = 128
    config.emb_dim = 128
    config.mlp_dim = 512
    config.num_heads = 2

    config.max_target_length = 32
    config.max_eval_target_length = 32
    config.max_predict_length = 32

    workdir = tempfile.mkdtemp()

    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable
    print('data_dir: ', data_dir)

    with tfds.testing.mock_data(num_examples=128, data_dir=data_dir):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))


if __name__ == '__main__':
  absltest.main()

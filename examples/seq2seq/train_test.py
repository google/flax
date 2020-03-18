# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flax.examples.seq2seq.train."""

from absl.testing import absltest

from flax import nn
import train

import jax
from jax import random


jax.config.parse_flags_with_absl()


class TrainTest(absltest.TestCase):

  def test_character_table(self):
    text = '410+19'
    enc_text = train.CTABLE.encode(text)
    dec_text = train.CTABLE.decode(enc_text)
    # The text is possibly padded with whitespace, but the trimmed output should
    # be equal to the input.
    self.assertEqual(text, dec_text.strip())

  def test_train_one_step(self):
    batch = train.get_batch(128)
    rng = random.PRNGKey(0)

    model = train.create_model(rng)
    optimizer = train.create_optimizer(model, 0.003)
    optimizer, train_metrics = train.train_step(optimizer, batch)

    self.assertLessEqual(train_metrics['loss'], 5)
    self.assertGreaterEqual(train_metrics['accuracy'], 0)


if __name__ == '__main__':
  absltest.main()

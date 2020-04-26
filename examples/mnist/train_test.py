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
"""Tests for flax.examples.mnist.train."""

from absl.testing import absltest

import train
import jax
from jax import random

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TrainTest(absltest.TestCase):

  def test_single_train_step(self):
    train_ds, _ = train.get_datasets()
    batch_size = 32
    model = train.create_model(random.PRNGKey(0))
    optimizer = train.create_optimizer(model, 0.1, 0.9)

    _, train_metrics = \
      train.train_step(optimizer=optimizer,
                       batch={k: v[:batch_size] for k, v in train_ds.items()})
    self.assertLessEqual(train_metrics['loss'], 2.302)
    self.assertGreaterEqual(train_metrics['accuracy'], 0.0625)


if __name__ == '__main__':
  absltest.main()

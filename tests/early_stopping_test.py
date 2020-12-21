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

"""Tests for flax.training.early_stopping."""

import copy
import os

from absl.testing import absltest
from flax.training import early_stopping
import jax
from jax import test_util as jtu

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class EarlyStoppingTests(absltest.TestCase):

  def test_iterator(self):
    iterator = early_stopping.EarlyStopping(steps=100)
    self.assertEqual(list(iterator), list(range(100)))

  def test_update(self):
    iterator = early_stopping.EarlyStopping(steps=100,
                                            min_delta=0, 
                                            patience=0)
    for step in iterator:
      val_metric = 1.
      iterator.update(val_metric)
    self.assertEqual(step, 1)

  def test_patience(self):
    iterator = early_stopping.EarlyStopping(steps=100,
                                            min_delta=0, 
                                            patience=0)
    iterator_patience = early_stopping.EarlyStopping(steps=100,
                                                     min_delta=0, 
                                                     patience=6)
    for step in iterator:
      val_metric = 1.
      iterator.update(val_metric)
    self.assertEqual(step, 1)

    for step in iterator_patience:
      val_metric = 1.
      iterator_patience.update(val_metric)
    self.assertEqual(step, 7)

  def test_delta(self):
    iterator = early_stopping.EarlyStopping(steps=100,
                                            min_delta=0, 
                                            patience=0)
    iterator_delta = early_stopping.EarlyStopping(steps=100,
                                                  min_delta=1e-3, 
                                                  patience=0)
    val_metric = 1.
    for step in iterator:
      val_metric -= 1e-4
      iterator.update(val_metric)
    self.assertEqual(step, 99)

    val_metric = 1.
    for step in iterator_delta:
      val_metric -= 1e-4
      iterator_delta.update(val_metric)
    self.assertEqual(step, 1)


if __name__ == '__main__':
  absltest.main()

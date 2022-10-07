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

  def test_update(self):
    es = early_stopping.EarlyStopping(min_delta=0,
                                      patience=0)

    for i in range(2):
      improve_steps = 0
      for step in range(10):
        metric = 1.
        did_improve, es = es.update(metric)
        if not did_improve:
          improve_steps += 1
        if es.should_stop:
          break

      self.assertEqual(improve_steps, 1)
      self.assertEqual(step, 1)

      es = es.reset()  # ensure object is reusable if reset.

  def test_patience(self):
    es = early_stopping.EarlyStopping(min_delta=0,
                                      patience=0)
    patient_es = early_stopping.EarlyStopping(min_delta=0,
                                              patience=6)
    for step in range(10):
      metric = 1.
      did_improve, es = es.update(metric)
      if es.should_stop:
        break

    self.assertEqual(step, 1)

    for patient_step in range(10):
      metric = 1.
      did_improve, patient_es = patient_es.update(metric)
      if patient_es.should_stop:
        break

    self.assertEqual(patient_step, 7)

  def test_delta(self):
    es = early_stopping.EarlyStopping(min_delta=0,
                                      patience=0)
    delta_es = early_stopping.EarlyStopping(min_delta=1e-3,
                                            patience=0)
    delta_patient_es = early_stopping.EarlyStopping(min_delta=1e-3,
                                                    patience=1)
    metric = 1.
    for step in range(100):
      metric -= 1e-4
      did_improve, es = es.update(metric)
      if es.should_stop:
        break

    self.assertEqual(step, 99)

    metric = 1.
    for step in range(100):
      metric -= 1e-4
      did_improve, delta_es = delta_es.update(metric)
      if delta_es.should_stop:
        break

    self.assertEqual(step, 1)

    metrics = [0.01, 0.005, 0.0033, 0.0025, 0.002,
               0.0017, 0.0014, 0.0012, 0.0011, 0.001]
    improvement_steps = 0
    for step in range(10):
      metric = metrics[step]
      did_improve, delta_patient_es = delta_patient_es.update(metric)
      if did_improve:
        improvement_steps += 1
      if delta_patient_es.should_stop:
        break

    self.assertEqual(improvement_steps, 4)  # steps 0, 1, 2, 4
    self.assertEqual(step, 6)


if __name__ == '__main__':
  absltest.main()

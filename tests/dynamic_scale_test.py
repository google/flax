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

"""Tests for flax.training.dynamic_scale."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from flax.training import dynamic_scale

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class DynamicScaleTest(absltest.TestCase):
  def test_grows_after_growth_interval_finite_steps(self):
    ds = dynamic_scale.DynamicScale(growth_interval=3, scale=4.0)
    scales = []
    for _ in range(6):
      ds, is_fin, _, _ = ds.value_and_grad(lambda p: p**2)(1.0)
      self.assertTrue(is_fin)
      scales.append(float(ds.scale))
    self.assertEqual(scales, [4.0, 4.0, 8.0, 8.0, 8.0, 16.0])

  def test_non_finite_step_backs_off_and_restarts_count(self):
    ds = dynamic_scale.DynamicScale(growth_interval=2, scale=4.0)
    scales = []
    for p in [1.0, jnp.inf, 1.0, 1.0]:
      ds, _, _, _ = ds.value_and_grad(lambda p: p**2)(p)
      scales.append(float(ds.scale))
    self.assertEqual(scales, [4.0, 2.0, 2.0, 4.0])


if __name__ == '__main__':
  absltest.main()

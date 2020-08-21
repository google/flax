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
"""Tests for flax.examples.imagenet.train."""

from absl.testing import absltest

from flax import nn
import train

import jax
from jax import random
import jax.numpy as jnp

import numpy as onp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class TrainTest(absltest.TestCase):

  def test_create_model(self):
    variables = train.initialized(random.PRNGKey(0), 224)
    x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
    y = train.model(train=False).apply(variables, x)
    self.assertEqual(y.shape, (8, 1000))


if __name__ == '__main__':
  absltest.main()

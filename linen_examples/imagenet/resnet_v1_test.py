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

"""Tests for flax.linen_examples.imagenet.resnet_v1."""

from absl.testing import absltest

import jax
from jax import numpy as jnp

import resnet_v1 as models

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class ResNetV1Test(absltest.TestCase):
  """Test cases for ResNet v1 model."""

  def test_resnet_v1_model(self):
    """Tests ResNet V1 model definition."""
    rng = jax.random.PRNGKey(0)
    model_def = models.ResNet50(num_classes=10, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.float32))

    self.assertLen(variables, 2)
    self.assertLen(variables['params'], 19)

if __name__ == '__main__':
  absltest.main()

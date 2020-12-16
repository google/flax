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

"""Tests for flax.linen_examples.imagenet.models."""

from absl.testing import absltest

import jax
from jax import numpy as jnp

import models


class ResNetV1Test(absltest.TestCase):
  """Test cases for ResNet v1 model definition."""

  def test_resnet_v1_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.ResNet50(num_classes=10, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.float32))

    self.assertLen(variables, 2)
    # Resnet50 model will create parameters for the following layers:
    #   conv + batch_norm = 2
    #   BottleneckResNetBlock in stages: [3, 4, 6, 3] = 16
    #   Followed by a Dense layer = 1
    self.assertLen(variables['params'], 19)

if __name__ == '__main__':
  absltest.main()

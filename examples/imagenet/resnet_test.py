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
"""Tests for flax.examples.imagenet.resnet."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

import resnet


class ResNetTest(absltest.TestCase):

  def test_resnet_module(self):
    rng = jax.random.PRNGKey(0)
    model_def = resnet.ResNet.partial(num_classes=10)
    output, init_params = model_def.init_by_shape(
        rng, [((8, 224, 224, 3), jnp.float32)])

    self.assertEqual((8, 10), output.shape)
    self.assertLen(init_params, 19)


if __name__ == '__main__':
  absltest.main()

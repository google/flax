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

"""Tests for models."""


from absl.testing import absltest
import models
import jax
import jax.numpy as jnp
import numpy as np

_CNN_PARAMS = 825_034


class ModelsTest(absltest.TestCase):

  def test_cnn(self):
    """Tests CNN module used as the trainable model."""
    rng = jax.random.PRNGKey(0)
    inputs = jnp.ones((1, 28, 28, 3), jnp.float32)
    output, variables = models.CNN().init_with_output(rng, inputs)

    self.assertEqual((1, 10), output.shape)
    self.assertEqual(
        _CNN_PARAMS,
        sum(np.prod(arr.shape) for arr in jax.tree_leaves(variables['params'])))


if __name__ == '__main__':
  absltest.main()

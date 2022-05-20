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

"""Tests for flax.linen.activation."""

from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn

import jax
from jax import random
import jax.numpy as jnp


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class ActivationTest(parameterized.TestCase):

  def test_prelu(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 6, 5))
    act = nn.PReLU()
    y, _ = act.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
  absltest.main()

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
"""Jax2Tf tests for flax.examples.imagenet.train."""


from absl.testing import absltest

import imagenet_lib
from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np

DEFAULT_ATOL = 1e-2


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def test_fprop(self):
    jax_model, _ = imagenet_lib.create_model(
        random.PRNGKey(0), 8, 224, jnp.float32)
    jax2tf_model = jax2tf.convert(jax_model)
    x = random.uniform(random.PRNGKey(1), (8, 224, 224, 3))
    np.testing.assert_allclose(jax_model(x), jax2tf_model(x), atol=DEFAULT_ATOL)


if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())

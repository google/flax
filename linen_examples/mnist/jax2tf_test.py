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
"""Jax2Tf tests for flax.linen_examples.mnist."""

from absl.testing import absltest

import mnist_lib
from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf

import tensorflow_datasets as tfds

BATCH_SIZE = 32
DEFAULT_ATOL = 1e-6


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def setUp(self):
    super().setUp()
    # Load mock data so that dataset is not downloaded over the network.
    with tfds.testing.mock_data(num_examples=BATCH_SIZE, data_dir=data_dir):
      self._train_ds, self._test_ds = mnist_lib.get_datasets()

  def test_single_train_step(self):
    params = mnist_lib.get_initial_params(random.PRNGKey(0))
    optimizer = mnist_lib.create_optimizer(params, 0.1, 0.9)
    batch = {k: v[:BATCH_SIZE] for k, v in self._train_ds.items()}
    jax_opt, jax_metrics = mnist_lib.train_step(optimizer, batch)
    tf_opt, tf_metrics = jax2tf.convert(mnist_lib.train_step)(optimizer, batch)
    self.assertAllClose(jax_metrics, tf_metrics, atol=DEFAULT_ATOL)
    self.assertAllClose(jax_opt.state, tf_opt.state, atol=DEFAULT_ATOL)

  def test_eval(self):
    params = mnist_lib.get_initial_params(random.PRNGKey(0))
    jax_metrics = mnist_lib.eval_step(params, self._test_ds)
    tf_metrics = jax2tf.convert(mnist_lib.eval_step)(params, self._test_ds)
    self.assertAllClose(jax_metrics, tf_metrics, atol=DEFAULT_ATOL)


if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())

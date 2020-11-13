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
"""Jax2Tf tests for flax.linen_examples.seq2seq.train."""

from absl.testing import absltest

from flax import optim
import train
from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf

DEFAULT_ATOL = 5e-6

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


def create_test_optimizer():
  rng = random.PRNGKey(0)
  param = train.get_initial_params(rng)
  return optim.Adam(learning_rate=0.003).create(param)


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def test_fprop(self):
    key = random.PRNGKey(0)
    param = train.get_initial_params(key)
    batch, masks = train.get_batch(5)
    in_masks, out_masks = masks
    jax_logits, jax_pred = train.apply_model(batch, in_masks, out_masks, param,
                                             key)
    tf_logits, tf_pred = jax2tf.convert(train.apply_model)(batch, in_masks,
                                                           out_masks, param,
                                                           key)
    self.assertAllClose(jax_logits, tf_logits, atol=DEFAULT_ATOL)
    self.assertAllClose(jax_pred, tf_pred, atol=DEFAULT_ATOL)

  def test_train_one_step(self):
    batch, masks = train.get_batch(128)

    optimizer = create_test_optimizer()
    key = random.PRNGKey(0)
    jax_opt, jax_metrics = train.train_step(optimizer, batch, masks, key)
    tf_opt, tf_metrics = jax2tf.convert(train.train_step)(optimizer, batch,
                                                          masks, key)
    self.assertAllClose(jax_metrics, tf_metrics, atol=DEFAULT_ATOL)
    self.assertAllClose(jax_opt.state, tf_opt.state, atol=DEFAULT_ATOL)


if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())

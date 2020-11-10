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
"""Jax2Tf tests for flax.examples.seq2seq.train."""

import functools

from absl.testing import absltest

from flax import nn
import train
from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np

DEFAULT_ATOL = 5e-6


def _train_one_step(batch):
  with nn.stochastic(random.PRNGKey(0)):
    model = train.create_model()
    optimizer = train.create_optimizer(model, 0.003)
    optimizer, train_metrics = train.train_step(optimizer, batch, nn.make_rng())
  return train_metrics['loss'], train_metrics['accuracy']


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def test_fprop(self):
    batch = train.get_batch(128)
    rng = random.PRNGKey(0)

    with nn.stochastic(rng):
      jax_model = train.create_model()
      jax_logits, _ = jax_model(batch['query'], batch['answer'])
      tf_model = jax2tf.convert(jax_model)
      tf_logits, _ = tf_model(batch['query'], batch['answer'])

    np.testing.assert_allclose(jax_logits, tf_logits, atol=DEFAULT_ATOL)

  def test_decode_batch(self):
    rng = random.PRNGKey(0)
    inputs = train.get_batch(5)['query']
    init_decoder_input = train.onehot(
        train.CTABLE.encode('=')[0:1], train.CTABLE.vocab_size)
    init_decoder_inputs = jnp.tile(
        init_decoder_input, (inputs.shape[0], train.get_max_output_len(), 1))
    with nn.stochastic(rng):
      jax_model = train.create_model()
      _, jax_inferred = jax_model(
          inputs, init_decoder_inputs, teacher_force=False)

    # Reset the seed and recreate model so that same random number sequences
    # are used when calling make_rng().
    with nn.stochastic(rng):
      jax_model = train.create_model()
      tf_model = jax2tf.convert(
          functools.partial(jax_model, teacher_force=False))
      _, tf_inferred = tf_model(inputs, init_decoder_inputs)

    np.testing.assert_allclose(jax_inferred, tf_inferred, atol=DEFAULT_ATOL)

  def test_train_one_step(self):
    batch = train.get_batch(128)
    np.testing.assert_allclose(
        _train_one_step(batch),
        jax2tf.convert(_train_one_step)(batch),
        atol=DEFAULT_ATOL)


if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())

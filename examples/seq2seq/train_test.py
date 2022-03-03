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

"""Tests for flax.examples.seq2seq.train."""

import functools

from absl.testing import absltest
from flax.training import train_state
import jax
from jax import random
import numpy as np
import optax

import input_pipeline
import train
import models

jax.config.parse_flags_with_absl()


def create_ctable(chars='0123456789+= '):
  return input_pipeline.CharacterTable(chars)


def create_train_state(ctable):
  model = models.Seq2seq(teacher_force=False,
      hidden_size=train.FLAGS.hidden_size, vocab_size=ctable.vocab_size)
  params = train.get_initial_params(model, jax.random.PRNGKey(0), ctable)
  tx = optax.adam(train.FLAGS.learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)
  return state


class TrainTest(absltest.TestCase):

  def test_character_table(self):
    ctable = create_ctable()
    text = '410+19'
    enc_text = ctable.encode(text)
    dec_text = ctable.decode(enc_text)
    # The text is possibly padded with whitespace, but the trimmed output should
    # be equal to the input.
    self.assertEqual(text, dec_text.strip())

  def test_mask_sequences(self):
    np.testing.assert_equal(
        input_pipeline.mask_sequences(
            np.arange(1, 13).reshape((4, 3)),
            np.array([3, 2, 1, 0])
        ),
        np.array(
            [[1, 2, 3],
             [4, 5, 0],
             [7, 0, 0],
             [0, 0, 0]]
        )
    )

  def test_get_sequence_lengths(self):
    oh_sequence_batch = jax.vmap(
        functools.partial(jax.nn.one_hot, num_classes=4))(
            np.array([[0, 1, 0], [1, 0, 2], [1, 2, 0], [1, 2, 3]]))
    np.testing.assert_equal(
        input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=0),
        np.array([1, 2, 3, 3], np.int32)
    )
    np.testing.assert_equal(
        input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=1),
        np.array([2, 1, 1, 1], np.int32)
    )
    np.testing.assert_equal(
        input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=2),
        np.array([3, 3, 2, 2], np.int32)
    )

  def test_train_one_step(self):
    ctable = create_ctable()
    batch = ctable.get_batch(128)

    state = create_train_state(ctable)
    key = random.PRNGKey(0)
    _, train_metrics = train.train_step(state, batch, key, ctable.eos_id)

    self.assertLessEqual(train_metrics['loss'], 5)
    self.assertGreaterEqual(train_metrics['accuracy'], 0)

  def test_decode_batch(self):
    ctable = create_ctable()
    batch = ctable.get_batch(5)
    key = random.PRNGKey(0)
    state = create_train_state(ctable)
    train.decode_batch(state, batch, key, ctable)

if __name__ == '__main__':
  absltest.main()

# Copyright 2024 The Flax Authors.
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

import jax
import jax.numpy as jnp

from flax.experimental import nnx

from absl.testing import parameterized


class TestMetrics(parameterized.TestCase):
  def test_split_merge(self):
    logits = jax.random.normal(jax.random.key(0), (5, 2))
    labels = jnp.array([1, 1, 0, 1, 0])
    logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    labels2 = jnp.array([0, 1, 1, 1, 1])

    accuracy = nnx.metrics.Accuracy()
    accuracy.update(logits=logits, labels=labels)
    graphdef, state = accuracy.split()
    accuracy = nnx.merge(graphdef, state)
    self.assertEqual(accuracy.compute(), 0.6)
    accuracy.update(logits=logits2, labels=labels2)
    self.assertEqual(accuracy.compute(), 0.7)

  def test_multimetric(self):
    logits = jax.random.normal(jax.random.key(0), (5, 2))
    labels = jnp.array([1, 1, 0, 1, 0])
    logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    labels2 = jnp.array([0, 1, 1, 1, 1])
    batch_loss = jnp.array([1, 2, 3, 4])
    batch_loss2 = jnp.array([3, 2, 1, 0])

    metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average()
    )
    values = metrics.compute()
    self.assertTrue(jnp.isnan(values['accuracy']))
    self.assertTrue(jnp.isnan(values['loss']))

    metrics.update(logits=logits, labels=labels, values=batch_loss)
    values = metrics.compute()
    self.assertEqual(values['accuracy'], 0.6)
    self.assertEqual(values['loss'], 2.5)

    metrics.update(logits=logits2, labels=labels2, values=batch_loss2)
    values = metrics.compute()
    self.assertEqual(values['accuracy'], 0.7)
    self.assertEqual(values['loss'], 2.)

    metrics.reset()
    values = metrics.compute()
    self.assertTrue(jnp.isnan(values['accuracy']))
    self.assertTrue(jnp.isnan(values['loss']))
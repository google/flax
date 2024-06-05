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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp


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

  def test_welford(self):
    values = jax.random.normal(jax.random.key(0), (5, 2))

    welford = nnx.metrics.Welford()
    welford.update(values=values)
    graphdef, state = welford.split()
    welford = nnx.merge(graphdef, state)
    expected = nnx.metrics.Statistics(
        mean=values.mean(),
        standard_deviation=values.std(),
        standard_error_of_mean=values.std() / jnp.sqrt(values.size),
    )
    computed = welford.compute()
    self.assertAlmostEqual(computed.mean, expected.mean, )
    self.assertAlmostEqual(computed.standard_deviation, expected.standard_deviation)
    self.assertAlmostEqual(
        computed.standard_error_of_mean, expected.standard_error_of_mean
    )

  def test_welford_large(self):
    values = jax.random.normal(jax.random.key(0), (5, 2)) + 1e16

    welford = nnx.metrics.Welford()
    welford.update(values=values)
    graphdef, state = welford.split()
    welford = nnx.merge(graphdef, state)
    expected = nnx.metrics.Statistics(
        mean=values.mean(),
        standard_deviation=values.std(),
        standard_error_of_mean=values.std() / jnp.sqrt(values.size),
    )
    computed = welford.compute()
    self.assertAlmostEqual(computed.mean, expected.mean)
    self.assertAlmostEqual(computed.standard_deviation, expected.standard_deviation)
    self.assertAlmostEqual(
        computed.standard_error_of_mean, expected.standard_error_of_mean
    )

  def test_welford_many(self):
    values = jax.random.normal(jax.random.key(0), (50_000,))

    welford = nnx.metrics.Welford()
    welford.update(values=values)
    graphdef, state = welford.split()
    welford = nnx.merge(graphdef, state)
    computed = welford.compute()
    self.assertAlmostEqual(
        computed.mean, 0.0, delta=3 * computed.standard_error_of_mean
    )
    self.assertAlmostEqual(computed.standard_deviation, 1.0, places=2)

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
    self.assertEqual(values['loss'], 2.0)

    metrics.reset()
    values = metrics.compute()
    self.assertTrue(jnp.isnan(values['accuracy']))
    self.assertTrue(jnp.isnan(values['loss']))


if __name__ == '__main__':
  absltest.main()

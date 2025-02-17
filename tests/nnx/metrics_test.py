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
import numpy as np


class TestMetrics(parameterized.TestCase):

  def test_split_merge(self):
    logits = jnp.array(
      [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]
    )
    labels = jnp.array([1, 1, 1, 1, 1])
    logits2 = jnp.array(
      [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]
    )
    labels2 = jnp.array([1, 1, 1, 1, 0])

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
    logits = jnp.array(
      [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]
    )
    labels = jnp.array([1, 1, 0, 1, 0])
    logits2 = jnp.array(
      [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]
    )
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
    self.assertEqual(values['accuracy'], 0.5)
    self.assertEqual(values['loss'], 2.0)

    metrics.reset()
    values = metrics.compute()
    self.assertTrue(jnp.isnan(values['accuracy']))
    self.assertTrue(jnp.isnan(values['loss']))

  def test_binary_classification_accuracy(self):
    logits = jnp.array([0.4, 0.7, 0.2, 0.6])
    labels = jnp.array([0, 1, 1, 1])
    logits2 = jnp.array([0.1, 0.9, 0.8, 0.3])
    labels2 = jnp.array([0, 1, 1, 0])
    accuracy = nnx.metrics.Accuracy(threshold=0.5)
    accuracy.update(logits=logits, labels=labels)
    self.assertEqual(accuracy.compute(), 0.75)
    accuracy.update(logits=logits2, labels=labels2)
    self.assertEqual(accuracy.compute(), 0.875)

  @parameterized.parameters(
    {
      'logits': np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
      'labels': np.array([0, 0, 0, 0]),
      'threshold': None,
      'error_msg': 'For multi-class classification'
    },
    {
      'logits': np.array([0.0, 0.0, 0.0, 0.0]),
      'labels': np.array([[0, 0], [0, 0]]),
      'threshold': 0.5,
      'error_msg': 'For binary classification'
    }
  )
  def test_accuracy_dims(self, logits, labels, threshold, error_msg):
    accuracy = nnx.metrics.Accuracy(threshold=threshold)
    with self.assertRaisesRegex(ValueError, error_msg):
      accuracy.update(logits=logits, labels=labels)


if __name__ == '__main__':
  absltest.main()

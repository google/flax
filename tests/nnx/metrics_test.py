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


class TestAverage(parameterized.TestCase):

  def test_initial_compute_nan(self):
    avg = nnx.metrics.Average()
    self.assertTrue(jnp.isnan(avg.compute()))

  def test_single_batch(self):
    avg = nnx.metrics.Average()
    avg.update(values=jnp.array([1, 2, 3, 4]))
    np.testing.assert_allclose(avg.compute(), 2.5, rtol=1e-6)

  def test_multiple_batches(self):
    avg = nnx.metrics.Average()
    avg.update(values=jnp.array([1, 2, 3, 4]))
    avg.update(values=jnp.array([3, 2, 1, 0]))
    np.testing.assert_allclose(avg.compute(), 2.0, rtol=1e-6)

  def test_reset(self):
    avg = nnx.metrics.Average()
    avg.update(values=jnp.array([1, 2, 3]))
    avg.reset()
    self.assertTrue(jnp.isnan(avg.compute()))

  def test_custom_argname(self):
    avg = nnx.metrics.Average('loss')
    avg.update(loss=jnp.array([10, 20]))
    np.testing.assert_allclose(avg.compute(), 15.0, rtol=1e-6)

  def test_missing_argname(self):
    avg = nnx.metrics.Average('loss')
    with self.assertRaisesRegex(TypeError, "Expected keyword argument 'loss'"):
      avg.update(values=jnp.array([1, 2]))

  def test_scalar_float(self):
    avg = nnx.metrics.Average()
    avg.update(values=5.0)
    np.testing.assert_allclose(avg.compute(), 5.0, rtol=1e-6)

  def test_scalar_int(self):
    avg = nnx.metrics.Average()
    avg.update(values=3)
    np.testing.assert_allclose(avg.compute(), 3.0, rtol=1e-6)


class TestWelford(parameterized.TestCase):

  def test_multiple_batches(self):
    wf = nnx.metrics.Welford()
    batch1 = jnp.array([1.0, 2.0, 3.0, 4.0])
    batch2 = jnp.array([3.0, 2.0, 1.0, 0.0])
    wf.update(values=batch1)
    wf.update(values=batch2)
    all_values = jnp.concatenate([batch1, batch2])
    stats = wf.compute()
    np.testing.assert_allclose(stats.mean, all_values.mean(), rtol=1e-6)
    np.testing.assert_allclose(
      stats.standard_deviation, all_values.std(), rtol=1e-5
    )
    expected_sem = all_values.std() / jnp.sqrt(all_values.size)
    np.testing.assert_allclose(
      stats.standard_error_of_mean, expected_sem, rtol=1e-5
    )

  def test_reset(self):
    wf = nnx.metrics.Welford()
    wf.update(values=jnp.array([1.0, 2.0, 3.0]))
    wf.reset()
    stats = wf.compute()
    np.testing.assert_allclose(stats.mean, 0.0, atol=0)
    self.assertTrue(jnp.isnan(stats.standard_error_of_mean))
    self.assertTrue(jnp.isnan(stats.standard_deviation))

  def test_custom_argname(self):
    wf = nnx.metrics.Welford('loss')
    wf.update(loss=jnp.array([1.0, 2.0, 3.0]))
    stats = wf.compute()
    np.testing.assert_allclose(stats.mean, 2.0, rtol=1e-6)

  def test_missing_argname(self):
    wf = nnx.metrics.Welford('loss')
    with self.assertRaisesRegex(TypeError, "Expected keyword argument 'loss'"):
      wf.update(values=jnp.array([1.0]))


class TestAccuracy(parameterized.TestCase):

  def test_multiclass_int64_labels(self):
    logits = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([1, 0], dtype=np.int64)
    labels = jnp.asarray(labels)
    acc = nnx.metrics.Accuracy()
    acc.update(logits=logits, labels=labels)
    np.testing.assert_allclose(acc.compute(), 1.0, rtol=1e-6)

  def test_invalid_label_dtype(self):
    logits = jnp.array([[0.0, 1.0]])
    labels = jnp.array([1.0])
    acc = nnx.metrics.Accuracy()
    with self.assertRaisesRegex(ValueError, 'labels.dtype'):
      acc.update(logits=logits, labels=labels)

  def test_threshold_type_error(self):
    with self.assertRaisesRegex(TypeError, 'float'):
      nnx.metrics.Accuracy(threshold=1)


class TestMultiMetric(parameterized.TestCase):

  @parameterized.parameters('reset', 'update', 'compute', 'split')
  def test_reserved_name_error(self, name):
    with self.assertRaisesRegex(ValueError, 'reserved'):
      nnx.MultiMetric(**{name: nnx.metrics.Average()})

  @parameterized.parameters('_metric_names', '_expected_kwargs')
  def test_internal_name_error(self, name):
    with self.assertRaisesRegex(ValueError, 'reserved'):
      nnx.MultiMetric(**{name: nnx.metrics.Average()})

  def test_unmatched_kwarg_error(self):
    metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average(),
      score=nnx.metrics.Average('score'),
    )
    # Guard: validation must be active for this test.
    self.assertEqual(
      metrics._expected_kwargs, {'values', 'score'}
    )
    with self.assertRaisesRegex(
      TypeError, 'Unexpected keyword argument'
    ):
      metrics.update(
        values=jnp.array([1.0]),
        score=jnp.array([2.0]),
        typo_kwarg=jnp.array([3.0]),
      )

  def test_compute_returns_dict(self):
    metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average(),
      score=nnx.metrics.Average('score'),
    )
    metrics.update(values=jnp.array([1, 2, 3]), score=jnp.array([4, 5, 6]))
    result = metrics.compute()
    self.assertIsInstance(result, dict)
    self.assertEqual(set(result.keys()), {'loss', 'score'})
    np.testing.assert_allclose(result['loss'], 2.0, rtol=1e-6)
    np.testing.assert_allclose(result['score'], 5.0, rtol=1e-6)

  def test_split_merge(self):
    metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average(),
    )
    metrics.update(values=jnp.array([1.0, 2.0, 3.0]))
    graphdef, state = metrics.split()
    restored = nnx.merge(graphdef, state)
    self.assertEqual(restored._metric_names, ['loss'])
    self.assertEqual(
      restored._expected_kwargs, {'values'}
    )
    np.testing.assert_allclose(
      restored.compute()['loss'], 2.0, rtol=1e-6
    )

  def test_validation_disabled_with_var_keyword(self):
    metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(),
      loss=nnx.metrics.Average(),
    )
    # Accuracy.update has **_, so validation is disabled.
    self.assertIsNone(metrics._expected_kwargs)


if __name__ == '__main__':
  absltest.main()

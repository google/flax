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

"""Tests for flax.examples.vae."""

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
import optax

import models
import train


class ModelTest(absltest.TestCase):
  """Test cases for the VAE model."""

  def test_vae(self):
    """Tests the VAE module forward pass and generation."""
    model = models.model(20, rngs=nnx.Rngs(0))
    inputs = jnp.ones((2, 784), jnp.float32)
    recon_x, mean, logvar = model(inputs)
    self.assertEqual((2, 784), recon_x.shape)
    self.assertEqual((2, 20), mean.shape)
    self.assertEqual((2, 20), logvar.shape)

    z = jax.random.normal(jax.random.key(0), (4, 20))
    generated = model.generate(z)
    self.assertEqual((4, 784), generated.shape)
    self.assertTrue(bool(jnp.all((generated >= 0) & (generated <= 1))))


class TrainStepTest(absltest.TestCase):
  """Test cases for the training and evaluation functions."""

  def test_compute_metrics(self):
    """Tests the metric computation on a small batch."""
    key = jax.random.key(0)
    logits = jax.random.normal(key, (4, 784))
    x = (jax.random.uniform(key, (4, 784)) > 0.5).astype(jnp.float32)
    mean = jnp.zeros((4, 20))
    logvar = jnp.zeros((4, 20))
    metrics = train.compute_metrics(logits, x, mean, logvar)
    self.assertGreater(float(metrics['loss']), 0.0)
    self.assertAlmostEqual(
      float(metrics['loss']),
      float(metrics['bce']) + float(metrics['kld']),
      places=4,
    )

  def test_train_step_decreases_loss(self):
    """Tests that a few training steps decrease the loss."""
    model = models.model(20, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(0.001), wrt=nnx.Param)
    data_key = jax.random.key(42)

    def loss():
      batch = (jax.random.uniform(data_key, (128, 784)) > 0.5).astype(
        jnp.float32
      )
      recon_x, mean, logvar = model(batch)
      metrics = train.compute_metrics(recon_x, batch, mean, logvar)
      return float(metrics['loss'])

    initial_loss = loss()
    for _ in range(10):
      data_key, subkey = jax.random.split(data_key)
      batch = (jax.random.uniform(subkey, (128, 784)) > 0.5).astype(jnp.float32)
      train.train_step(model, optimizer, batch)
    self.assertLess(loss(), initial_loss)

  def test_eval_f(self):
    """Tests the evaluation function output shapes."""
    model = models.model(20, rngs=nnx.Rngs(0))
    key = jax.random.key(0)
    images = (jax.random.uniform(key, (16, 784)) > 0.5).astype(jnp.float32)
    z = jax.random.normal(key, (64, 20))
    metrics, comparison, sample = train.eval_f(model, images, z)
    self.assertEqual((16, 28, 28, 1), comparison.shape)
    self.assertEqual((64, 28, 28, 1), sample.shape)
    self.assertGreater(float(metrics['loss']), 0.0)


if __name__ == '__main__':
  absltest.main()

# Copyright 2023 The Flax Authors.
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
"""Training and evaluation logic."""

from absl import logging
from flax import nnx
import input_pipeline
import models
import utils as vae_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nnx.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )


def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'bce': bce_loss, 'kld': kld_loss, 'loss': bce_loss + kld_loss}

@nnx.jit
def train_step(optimizer: nnx.Optimizer, model: nnx.Module, batch, z_rng, latents):
  """Single training step for the VAE model."""
  def loss_fn(model):
    recon_x, mean, logvar = model(batch, z_rng)
    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  return loss


@nnx.jit
def eval_f(model: nnx.Module, images, z, z_rng, latents):
  """Evaluation function for the VAE model."""
  recon_images, mean, logvar = model(images, z_rng)
  comparison = jnp.concatenate([
      images[:8].reshape(-1, 28, 28, 1),
      recon_images[:8].reshape(-1, 28, 28, 1),
  ])
  generate_images = model.generate(z)
  generate_images = generate_images.reshape(-1, 28, 28, 1)
  metrics = compute_metrics(recon_images, images, mean, logvar)
  return metrics, comparison, generate_images



def train_and_evaluate(config: ml_collections.ConfigDict):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()

  logging.info('Initializing dataset.')
  train_ds = input_pipeline.build_train_set(config.batch_size, ds_builder)
  test_ds = input_pipeline.build_test_set(ds_builder)

  logging.info('Initializing model.')
  rngs = nnx.Rngs(0)
  model = models.model(784, config.latents, rngs=rngs)
  optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate))

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, config.latents))

  steps_per_epoch = (
      ds_builder.info.splits['train'].num_examples // config.batch_size
  )

  for epoch in range(config.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      loss_val = train_step(optimizer, model, batch, key, config.latents)

    metrics, comparison, sample = eval_f(
        model, test_ds, z, eval_rng, config.latents
    )
    vae_utils.save_image(
        comparison, f'results/reconstruction_{epoch}.png', nrow=8
    )
    vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print(
        'eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
            epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
        )
    )

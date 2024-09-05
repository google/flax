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
from flax import linen as nn
import input_pipeline
import models
import utils as vae_utils
from flax.training import train_state
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
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )


def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'bce': bce_loss, 'kld': kld_loss, 'loss': bce_loss + kld_loss}


def train_step(state, batch, z_rng, latents):
  def loss_fn(params):
    recon_x, mean, logvar = models.model(latents).apply(
        {'params': params}, batch, z_rng
    )

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def eval_f(params, images, z, z_rng, latents):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images, z_rng)
    comparison = jnp.concatenate([
        images[:8].reshape(-1, 28, 28, 1),
        recon_images[:8].reshape(-1, 28, 28, 1),
    ])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    return metrics, comparison, generate_images

  return nn.apply(eval_model, models.model(latents))({'params': params})


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
  init_data = jnp.ones((config.batch_size, 784), jnp.float32)
  params = models.model(config.latents).init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=models.model(config.latents).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, config.latents))

  steps_per_epoch = (
      ds_builder.info.splits['train'].num_examples // config.batch_size
  )

  for epoch in range(config.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step(state, batch, key, config.latents)

    metrics, comparison, sample = eval_f(
        state.params, test_ds, z, eval_rng, config.latents
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

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

from absl import app
from absl import flags

import numpy as np
import jax.numpy as jnp

import jax
from jax import random

from flax import nn
from flax import optim

import tensorflow as tf
import tensorflow_datasets as tfds

from utils import save_image


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'num_epochs', default=30,
    help=('Number of training epochs.')
)

flags.DEFINE_integer(
    'latents', default=20,
    help=('Number of latent variables.')
)


class Encoder(nn.Module):

  def apply(self, x, latents):
    x = nn.Dense(x, 500, name='fc1')
    x = nn.relu(x)
    mean_x = nn.Dense(x, latents, name='fc2_mean')
    logvar_x = nn.Dense(x, latents, name='fc2_logvar')
    return mean_x, logvar_x


class Decoder(nn.Module):

  def apply(self, z):
    z = nn.Dense(z, 500, name='fc1')
    z = nn.relu(z)
    z = nn.Dense(z, 784, name='fc2')
    return z


class VAE(nn.Module):

  def apply(self, x, z_rng, latents=20):
    decoder = self._create_decoder()
    mean, logvar = Encoder(x, latents, name='encoder')
    z = reparameterize(z_rng, mean, logvar)
    recon_x = decoder(z)
    return recon_x, mean, logvar

  @nn.module_method
  def generate(self, z, **unused_kwargs):
    decoder = self._create_decoder()
    return nn.sigmoid(decoder(z))

  def _create_decoder(self):
    return Decoder.shared(name='decoder')


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {
      'bce': bce_loss,
      'kld': kld_loss,
      'loss': bce_loss + kld_loss
  }


@jax.jit
def train_step(optimizer, batch, z_rng):
  def loss_fn(model):
    recon_x, mean, logvar = model(batch, z_rng)

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss, recon_x
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  _, grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer


@jax.jit
def eval(model, images, z, z_rng):
  recon_images, mean, logvar = model(images, z_rng)

  comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                recon_images[:8].reshape(-1, 28, 28, 1)])

  generate_images = model.generate(z)
  generate_images = generate_images.reshape(-1, 28, 28, 1)
  metrics = compute_metrics(recon_images, images, mean, logvar)

  return metrics, comparison, generate_images


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x


def main(argv):
  del argv
  rng = random.PRNGKey(0)
  rng, key = random.split(rng)

  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()
  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(FLAGS.batch_size)
  train_ds = tfds.as_numpy(train_ds)

  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = np.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)

  module = VAE.partial(latents=FLAGS.latents)
  _, params = module.init_by_shape(
      key, [(FLAGS.batch_size, 784)], z_rng=random.PRNGKey(0))
  vae = nn.Model(module, params)

  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(vae)
  optimizer = jax.device_put(optimizer)

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, FLAGS.latents))

  steps_per_epoch = 50000 // FLAGS.batch_size

  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      optimizer = train_step(optimizer, batch, key)

    metrics, comparison, sample = eval(optimizer.target, test_ds, z, eval_rng)
    save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)
    save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
        epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
    ))


if __name__ == '__main__':
  app.run(main)

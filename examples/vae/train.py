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

from absl import app
from absl import flags
from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import utils as vae_utils


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
  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(784, name='fc2')(z)
    return z


class VAE(nn.Module):
  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


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


def model():
  return VAE(latents=FLAGS.latents)


@jax.jit
def train_step(state, batch, z_rng):
  def loss_fn(params):
    recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


@jax.jit
def eval(params, images, z, z_rng):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images, z_rng)
    comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                  recon_images[:8].reshape(-1, 28, 28, 1)])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    return metrics, comparison, generate_images

  return nn.apply(eval_model, model())({'params': params})


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x


def main(argv):
  del argv

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

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
  train_ds = iter(tfds.as_numpy(train_ds))

  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = np.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)

  init_data = jnp.ones((FLAGS.batch_size, 784), jnp.float32)

  state = train_state.TrainState.create(
      apply_fn=model().apply,
      params=model().init(key, init_data, rng)['params'],
      tx=optax.adam(FLAGS.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, FLAGS.latents))

  steps_per_epoch = 50000 // FLAGS.batch_size

  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step(state, batch, key)

    metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
    vae_utils.save_image(
        comparison, f'results/reconstruction_{epoch}.png', nrow=8)
    vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
        epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
    ))


if __name__ == '__main__':
  app.run(main)

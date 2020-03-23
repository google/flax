from absl import app
from absl import flags

import jax.numpy as jnp
import numpy as np

import jax
from jax import random

from flax import nn
from flax import optim

import tensorflow_datasets as tfds

from utils import save_image


FLAGS = flags.FLAGS

flags.DEFINE_float(
  'learning_rate', default=1e-3,
  help=('The leanring rate for the Adam optimizer')
)

flags.DEFINE_integer(
  'batch_size', default=128,
  help=('Batch size for training')
)

flags.DEFINE_integer(
  'num_epochs', default=10,
  help=('Number of training epochs')
)


class Encoder(nn.Module):
  def apply(self, x):
    x = nn.Dense(x, 400, name='enc_fc1')
    x = nn.relu(x)
    mean_x = nn.Dense(x, 20, name='enc_fc21')
    logvar_x = nn.Dense(x, 20, name='enc_fc22')
    return mean_x, logvar_x


class Decoder(nn.Module):
  def apply(self, z):
    z = nn.Dense(z, 400, name='dec_fc1')
    z = nn.relu(z)
    z = nn.Dense(z, 784, name='dec_fc2')
    return z


class VAE(nn.Module):
  def apply(self, x):
    mean, logvar = Encoder(x, name='encoder')
    z = reparameterize(mean, logvar)
    recon_x = self._created_decoder()(z)
    return recon_x, mean, logvar

  @nn.module_method
  def generate(self, z):
    params = self.get_param('decoder')
    return nn.sigmoid(Decoder.call(params, z))

  @nn.module_method
  def generate_one_liner(self, z):
    return nn.sigmoid(Decoder(z, name='decoder'))

  @nn.module_method
  def generate_shared(self, z):
    return nn.sigmoid(self._created_decoder()(z))

  def _created_decoder(self):
    return Decoder.shared(name='decoder')


def reparameterize(mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = np.random.normal(size=logvar.shape)
  return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
  return - 0.5 * jnp.sum(1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  return - jnp.sum(labels * nn.log_sigmoid(logits) + (1 - labels) * (nn.log_sigmoid(logits) - logits))


def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x)
  kld_loss = kl_divergence(mean, logvar)
  return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(bce_loss + kld_loss)}


@jax.jit
def train_step(optimizer, batch):
  def loss_fn(model):
    x = batch['image']
    recon_x, mean, logvar = model(x)

    bce_loss = binary_cross_entropy_with_logits(recon_x, x)
    kld_loss = kl_divergence(mean, logvar)
    loss = jnp.mean(bce_loss + kld_loss)
    return loss, recon_x
  optimizer, _, _ = optimizer.optimize(loss_fn)
  return optimizer


@jax.jit
def eval(model, eval_ds, z):
  xs = eval_ds['image'] / 255.0
  xs = xs.reshape(-1, 784)
  recon_xs, mean, logvar = model(xs)

  comparison = jnp.concatenate([xs[:8].reshape(-1, 28, 28, 1),
                                recon_xs[:8].reshape(-1, 28, 28, 1)])

  generate_xs = model.generate(z)
  generate_xs = generate_xs.reshape(-1, 28, 28, 1)

  return compute_metrics(recon_xs, xs, mean, logvar), comparison, generate_xs


def main(argv):
  key = random.PRNGKey(0)
  train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
  train_ds = train_ds.cache().shuffle(1000).batch(FLAGS.batch_size)
  test_ds = tfds.as_numpy(tfds.load('mnist', split=tfds.Split.TEST, batch_size=-1))

  _, params = VAE.init_by_shape(key, [((1, 784), jnp.float32)])
  vae = nn.Model(VAE, params)

  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(vae)

  for epoch in range(FLAGS.num_epochs):
    for batch in tfds.as_numpy(train_ds):
      batch['image'] = batch['image'].reshape(-1, 784) / 255.0
      optimizer = train_step(optimizer, batch)

    z = np.random.normal(size=(64, 20))
    metrics, comparison, sample = eval(optimizer.target, test_ds, z)
    save_image(comparison, 'results/reconstruction_' + str(epoch) + '.png', nrow=8)
    save_image(sample, 'results/sample_' + str(epoch) + '.png', nrow=8)

    print("eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}".format(
      epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
    ))


if __name__ == '__main__':
    app.run(main)

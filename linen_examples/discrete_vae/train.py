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

# See issue #620.
# pytype: disable=attribute-error
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args

from typing import Any

from absl import app
from absl import flags
from flax import linen as nn
from flax import optim
from flax.linen_examples.vae import utils as vae_utils
import jax
from jax import lax
from jax import random
from jax.config import config
from jax.experimental.kmeans import _kmeans
from jax.experimental.optimizers import clip_grads
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

config.enable_omnistaging()


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
    'clusters', default=10, help=('Number of clusters in kmeans.'))

flags.DEFINE_integer(
    'slices',
    default=20,
    help=(
        'Number of slices per observation. The concept of the slize is explained in https://arxiv.org/pdf/1803.03382.pdf'
    ))

flags.DEFINE_integer(
    'grad_clip', default=0.5, help=('Clip gradients at the value'))

flags.DEFINE_integer(
    'momentum',
    default=0.99,
    help=('The default momentum for updates of the codebook'))

flags.DEFINE_integer(
    'beta',
    default=0.001,
    help=("""The coefficient of the commitment loss, see
    https://arxiv.org/pdf/1711.00937.pdf for a broader
    discussion of the losses."""))


class DiscreteEncoder(nn.Module):
  """Discrete encder - a barebone MLP."""
  clusters: int
  slices: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    x = nn.Dense(500, name='fc2')(x)
    x = nn.relu(x)
    x = nn.Dense(self.clusters * self.slices, name='fc3_clusters')(x)
    return x


class Decoder(nn.Module):
  """Discrete deccder - a barebone MLP."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(784, name='fc2')(z)
    return z


class DiscreteVAE(nn.Module):
  """DiscreteVAE Module.

  Attributes:
    clusters: number of clusters
    update_codebook: if true, the codebook stored in batch_stats will be
      updated. Typically set to False when training and to True when evaluating.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
  """
  clusters: int = 20
  slices: int = 10
  update_codebook: bool = False
  momentum: float = 0.99

  def setup(self):
    self.encoder = DiscreteEncoder(self.clusters, self.slices)
    self.decoder = Decoder()

  @nn.module.compact
  def __call__(self, x, init_rng):
    # we detect if we're in initialization via empty variable tree.
    obs = self.encoder(x)
    codebook_input_shape = (obs.shape[0] * self.slices, self.clusters)
    obs = obs.reshape(codebook_input_shape)

    initializing = not self.has_variable('batch_stats', 'codebook')
    codebook = self.variable(
        'batch_stats', 'codebook', lambda s: s[random.choice(  # pylint-disable=g-long-lambda
            init_rng, jnp.arange(s.shape[0]), shape=[
                self.clusters,
            ])], obs)

    obs_codes, new_code_book, _ = _kmeans(obs, codebook.value, False)
    old_code_book = codebook.value
    if self.update_codebook and not initializing:
      codebook.value = self.momentum * codebook.value + (
          1 - self.momentum) * new_code_book

    decoder_input_shape = (obs.shape[0] / self.slices,
                           self.clusters * self.slices)
    latent_repr = old_code_book[obs_codes, :]
    latent_repr = latent_repr.reshape(decoder_input_shape)

    # as a decoder we use a standard MLP
    recon_x = self.decoder(latent_repr)
    # The output contains the cluster identifier per every element of the batch.
    return recon_x, obs, old_code_book[obs_codes, :]

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


@jax.vmap
def l2_loss(obs_1, obs_2):
  assert obs_1.shape == obs_2.shape
  return (obs_1 - obs_2)**2


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, obs, obs_quantized):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  sq_loss = l2_loss(obs, obs_quantized).mean()
  return {
      'bce': bce_loss,
      'l2loss': sq_loss,
      'loss': bce_loss + sq_loss,
      'obs_mean': obs.mean(),
      'obs_quantized_mean': obs_quantized.mean(),
      'recon_x_mean': recon_x.mean(),
      'x_mean': x.mean(),
  }


def discrete_model():
  return DiscreteVAE(
      clusters=FLAGS.clusters, slices=FLAGS.slices,
      update_codebook=True, momentum=FLAGS.momentum)


def discrete_eval_model():
  return DiscreteVAE(
      clusters=FLAGS.clusters, slices=FLAGS.slices, update_codebook=False)


@jax.jit
def train_step(state, batch, z_rng):
  """A definition of loss function as well a single train step."""
  def loss_fn(params):
    output, mutated_variables = discrete_model().apply(
        {
            'params': params,
            **state.model_state
        },
        batch,
        z_rng,
        mutable=['batch_stats'])
    recon_x, obs, obs_quantized = output

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    sq_loss = l2_loss(obs, lax.stop_gradient(obs_quantized)).mean()
    loss = bce_loss + FLAGS.beta * sq_loss
    return loss, (mutated_variables, (recon_x, obs, obs_quantized))

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grad = grad_fn(state.optimizer.target)
  grad = clip_grads(grad, FLAGS.grad_clip)
  new_model_state, _ = aux[1]

  new_optimizer = state.optimizer.apply_gradient(grad)
  new_state = state.replace(
      optimizer=new_optimizer, model_state=new_model_state)

  return new_state


@jax.jit
def evaluate(state, images, z, z_rng):
  """A definition of eval function."""
  params = state.optimizer.target
  recon_images, obs, obs_quantized = discrete_eval_model().apply(
      {
          'params': params,
          **state.model_state
      }, images, z_rng)

  comparison = jnp.concatenate([
      images[:8].reshape(-1, 28, 28, 1),
      recon_images[:8].reshape(-1, 28, 28, 1)
  ])

  generate_images = discrete_eval_model().apply(
      {
          'params': params,
          **state.model_state
      }, z, method=DiscreteVAE.generate)
  generate_images = generate_images.reshape(-1, 28, 28, 1)
  metrics = compute_metrics(recon_images, images, obs, obs_quantized)

  return metrics, comparison, generate_images


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x


class TrainState:
  optimizer: Any
  model_state: Any
  step: Any


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
  init_variables = discrete_model().init(key, init_data, rng)
  model_state, params = init_variables.pop('params')

  optimizer = optim.Adam(learning_rate=FLAGS.learning_rate).create(params)
  optimizer = jax.device_put(optimizer)

  state = TrainState(optimizer=optimizer, model_state=model_state, step=0)

  rng, z_key, eval_rng = random.split(rng, 3)

  steps_per_epoch = 50000 // FLAGS.batch_size

  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step(state, batch, key)
    # The step information allows adding a learning schedule in train_step.
    state = state.replace(step=state.step + 1)
    code_book = state.model_state['batch_stats']['codebook']

    # We sample 64 images, each consisting of DiscreteVAE.slices
    z = code_book[random.choice(
        z_key, jnp.arange(code_book.shape[0]), shape=[
            64 * FLAGS.slices,
        ])]
    decoder_input_shape = (64, FLAGS.clusters * FLAGS.slices)
    z = z.reshape(decoder_input_shape)

    metrics, comparison, sample = evaluate(state, test_ds, z, eval_rng)
    vae_utils.save_image(
        comparison, f'results/reconstruction_{epoch}.png', nrow=8)
    vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
        epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
    ))


if __name__ == '__main__':
  app.run(main)

# Copyright 2021 The Flax Authors.
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

# Lint as: python3
"""Sampling from PixelCNN++ using fixed-point iteration.
"""

import functools

from flax import optim
import pixelcnn
import train
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as onp
from PIL import Image


def generate_sample(config: ml_collections.ConfigDict, workdir: str):
  """Loads the latest model in `workdir` and samples a batch of images."""
  batch_size = config.sample_batch_size
  rng = random.PRNGKey(config.sample_rng_seed)
  rng, model_rng = random.split(rng)
  rng, dropout_rng = random.split(rng)

  # Create a model with dummy parameters and a dummy optimizer.
  init_batch = jnp.zeros((1, 32, 32, 3))

  params = train.model(config).init(
      {
          'params': model_rng,
          'dropout': dropout_rng
      }, init_batch)['params']
  optimizer_def = optim.Adam(
      learning_rate=config.learning_rate, beta1=0.95, beta2=0.9995)
  optimizer = optimizer_def.create(params)

  _, params = train.restore_checkpoint(workdir, optimizer, params)

  # Initialize batch of images
  device_count = jax.local_device_count()
  assert not batch_size % device_count, (
      'Sampling batch size must be a multiple of the device count, got '
      'sample_batch_size={}, device_count={}.'.format(batch_size,
                                                      device_count))
  sample_prev = jnp.zeros((device_count, batch_size // device_count, 32, 32, 3))

  # and batch of rng keys
  sample_rng = random.split(rng, device_count)

  # Generate sample using fixed-point iteration
  sample = sample_iteration(config, sample_rng, params, sample_prev)
  while jnp.any(sample != sample_prev):
    sample_prev, sample = sample, sample_iteration(config, sample_rng, params,
                                                   sample)
  return jnp.reshape(sample, (batch_size, 32, 32, 3))


def _categorical_onehot(rng, logit_probs):
  """Sample from a categorical distribution and one-hot encode the sample."""
  nr_mix = logit_probs.shape[-3]
  idxs = random.categorical(rng, logit_probs, axis=-3)
  return jnp.moveaxis(idxs[..., jnp.newaxis] == jnp.arange(nr_mix), -1, -3)


def conditional_params_to_sample(rng, conditional_params):
  means, inv_scales, logit_probs = conditional_params
  rng_mix, rng_logistic = random.split(rng)
  # Add channel dimension to one-hot mixture indicator
  mix_indicator = _categorical_onehot(rng_mix, logit_probs)[..., jnp.newaxis]
  # Use the mixture indicator to select the mean and inverse scale
  mean      = jnp.sum(means      * mix_indicator, -4)
  inv_scale = jnp.sum(inv_scales * mix_indicator, -4)
  sample = mean + random.logistic(rng_logistic, mean.shape) / inv_scale
  return snap_to_grid(sample)


@functools.partial(jax.pmap, static_broadcasted_argnums=2)
def sample_iteration(config, rng, params, sample):
  """PixelCNN++ sampling expressed as a fixed-point iteration."""
  rng, dropout_rng = random.split(rng)
  out = train.model(config).apply({'params': params},
                                  sample,
                                  rngs={'dropout': dropout_rng})
  c_params = pixelcnn.conditional_params_from_outputs(out, sample)
  return conditional_params_to_sample(rng, c_params)


def snap_to_grid(sample):
  return jnp.clip(jnp.round((sample + 1) * 127.5) / 127.5 - 1, -1., 1.)


def save_images(batch, fname):
  n_rows = batch.shape[0] // 16
  batch = onp.uint8(jnp.round((batch + 1) * 127.5))
  out = onp.full((1 + 33 * n_rows, 1 + 33 * 16, 3), 255, 'uint8')
  for i, im in enumerate(batch):
    top  = 1 + 33 * (i // 16)
    left = 1 + 33 * (i %  16)
    out[top:top + 32, left:left + 32] = im
  Image.fromarray(out).save(fname)

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

"""VAE model definitions."""

from flax import nnx
from jax import random
import jax.numpy as jnp


class Encoder(nnx.Module):
  """VAE Encoder."""

  def __init__(self, input_features: int, latents: int, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(input_features, 500, rngs=rngs)
    self.fc2_mean = nnx.Linear(500, latents, rngs=rngs)
    self.fc2_logvar = nnx.Linear(500, latents, rngs=rngs)

  def __call__(self, x):
    x = self.fc1(x)
    x = nnx.relu(x)
    mean_x = self.fc2_mean(x)
    logvar_x = self.fc2_logvar(x)
    return mean_x, logvar_x


class Decoder(nnx.Module):
  """VAE Decoder."""

  def __init__(self, latents: int, output_features: int, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(latents, 500, rngs=rngs)
    self.fc2 = nnx.Linear(500, output_features, rngs=rngs)

  def __call__(self, z):
    z = self.fc1(z)
    z = nnx.relu(z)
    z = self.fc2(z)
    return z


class VAE(nnx.Module):
  """Full VAE model."""

  def __init__(self, input_features: int, latents: int, rngs: nnx.Rngs):
    self.encoder = Encoder(input_features=input_features, latents=latents, rngs=rngs)
    self.decoder = Decoder(latents=latents, output_features=input_features, rngs=rngs)

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nnx.sigmoid(self.decoder(z))


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def model(input_features: int, latents: int, rngs: nnx.Rngs):
  return VAE(input_features=input_features, latents=latents, rngs=rngs)

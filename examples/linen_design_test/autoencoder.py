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

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

import jax
from jax import numpy as jnp, random, lax
import numpy as np

from flax import linen as nn
from flax.linen import Module, Dense, compact



# A concise MLP defined via lazy submodule initialization
class MLP(Module):
  widths: Iterable

  @compact
  def __call__(self, x):
    for width in self.widths[:-1]:
      x = nn.relu(Dense(width)(x))
    return Dense(self.widths[-1])(x)


# An autoencoder exposes multiple methods, so we define all
# submodules in setup().
class AutoEncoder(Module):
  encoder_widths: Iterable
  decoder_widths: Iterable
  input_shape: Tuple = None

  def setup(self):
    # Submodules attached in `setup` get names via attribute assignment
    self.encoder = MLP(self.encoder_widths)
    self.decoder = MLP(self.decoder_widths + (jnp.prod(self.input_shape), ))

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    assert x.shape[-len(self.input_shape):] == self.input_shape
    return self.encoder(jnp.reshape(x, (x.shape[0], -1)))

  def decode(self, z):
    z = self.decoder(z)
    x = nn.sigmoid(z)
    x = jnp.reshape(x, (x.shape[0],) + self.input_shape)
    return x


# `ae` is a detached module, which has no variables.
ae = AutoEncoder(
    encoder_widths=(32, 32, 32),
    decoder_widths=(32, 32, 32),
    input_shape=(28, 28, 1))


# `ae.initialized` returnes a materialized copy of `ae` by
# running through an input to create submodules defined lazily.
params = ae.init(
    {'params': random.PRNGKey(42)},
    jnp.ones((1, 28, 28, 1)))


# Now you can use `ae` as a normal object, calling any methods defined on AutoEncoder
print("reconstruct", jnp.shape(ae.apply(params, jnp.ones((1, 28, 28, 1)))))
print("encoder", jnp.shape(ae.apply(params, jnp.ones((1, 28, 28, 1)), method=ae.encode)))


# `ae.variables` is a frozen dict that looks like
# {'params': {"decoder": {"Dense_0": {"bias": ..., "kernel": ...}, ...}}
print("var shapes", jax.tree_util.tree_map(jnp.shape, params))


# TODO(avital, levskaya): resurrect this example once interactive api is restored.


# You can access submodules defined in setup(), they are just references on
# the autoencoder instance
# encoder = ae.encoder
# print("encoder var shapes", jax.tree_util.tree_map(jnp.shape, encoder.variables))


# # You can also access submodules that were defined in-line.
# # (We may add syntactic sugar here, e.g. to allow `ae.encoder.Dense_0`)
# encoder_dense0 = ae.encoder.children['Dense_0']
# print("encoder dense0 var shapes", jax.tree_util.tree_map(jnp.shape, encoder_dense0.variables))

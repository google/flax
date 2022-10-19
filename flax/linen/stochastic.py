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

"""Stochastic modules."""

from typing import Optional, Sequence

from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from jax import lax
from jax import random
import jax.numpy as jnp


class Dropout(Module):
  """Create a dropout layer.

    Note: When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
    to include an RNG seed named `'dropout'`. For example::

      model.apply({'params': params}, inputs=inputs, train=True, rngs={'dropout': dropout_rng})`

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng_collection: the rng collection name to use when requesting an rng key.
  """
  rate: float
  broadcast_dims: Sequence[int] = ()
  deterministic: Optional[bool] = None
  rng_collection: str = 'dropout'

  @compact
  def __call__(self, inputs, deterministic: Optional[bool] = None):
    """Applies a random dropout mask to the input.

    Args:
      inputs: the inputs that should be randomly masked.
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    deterministic = merge_param(
        'deterministic', self.deterministic, deterministic)

    if (self.rate == 0.) or deterministic:
      return inputs

    # Prevent gradient NaNs in 1.0 edge-case.
    if self.rate == 1.0:
      return jnp.zeros_like(inputs)

    keep_prob = 1. - self.rate
    rng = self.make_rng(self.rng_collection)
    broadcast_shape = list(inputs.shape)
    for dim in self.broadcast_dims:
      broadcast_shape[dim] = 1
    mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
    mask = jnp.broadcast_to(mask, inputs.shape)
    return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

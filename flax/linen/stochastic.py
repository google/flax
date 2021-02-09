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

"""Stochastic modules.
"""

from typing import Iterable, Optional

from jax import lax
from jax import random
import jax.numpy as jnp

from flax.linen.module import Module, compact, merge_param


class Dropout(Module):
  """Create a dropout layer.

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
  """
  rate: float
  broadcast_dims: Iterable[int] = ()
  deterministic: Optional[bool] = None

  @compact
  def __call__(self, inputs, deterministic: Optional[bool] = None, rng=None):
    """Applies a random dropout mask to the input.

    Args:
      inputs: the inputs that should be randomly masked.
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng: an optional `jax.random.PRNGKey`. By default `nn.make_rng()` will
        be used.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    deterministic = merge_param(
        'deterministic', self.deterministic, deterministic)
    if self.rate == 0.:
      return inputs
    keep_prob = 1. - self.rate
    if deterministic:
      return inputs
    else:
      if rng is None:
        rng = self.make_rng('dropout')
      broadcast_shape = list(inputs.shape)
      for dim in self.broadcast_dims:
        broadcast_shape[dim] = 1
      mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
      mask = jnp.broadcast_to(mask, inputs.shape)
      return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

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

import contextlib

from . import utils
from jax import lax
from jax import random
import jax.numpy as jnp


_prng_stack = utils.CallStack()


class _PRNGFrame:
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Random Number generator scope responsible for generation prngs in a stochastic context."""

  def __init__(self, rng):
    self.base_rng = rng
    self.counter = 0
    self.level = utils._trace_level(utils._current_trace())

  def make_rng(self):
    # when calling make_rng within a jax transformations
    # the rng could be implicitly reused (eg. in jit, vmap, scan, ...).
    # We raise an error to avoid silent errors.
    level = utils._trace_level(utils._current_trace())
    if level > self.level:
      raise ValueError('stochastic operations are not allowed when the'
                       ' stochastic context is created outside of the'
                       ' current Jax transformation')
    self.counter += 1
    return random.fold_in(self.base_rng, self.counter)


@contextlib.contextmanager
def stochastic(rng):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  A context manager for stochastic computations.

  Args:
    rng: the random number generator used as a seed for the stochastic context.
  Yields:
    A scope in which unique rngs can be created using `nn.make_rng()`.
  """
  with _prng_stack.frame(_PRNGFrame(rng)):
    yield


def is_stochastic():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns true if a stochastic scope is currently active."""
  return bool(_prng_stack)


def make_rng():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Create a new unique random number generator in a stochastic scope.

  In combination with `nn.stochastic()` this function is used to generate random
  keys without manually passing around and splitting a random number generator::

    with nn.stochastic(rng):
      x = random.normal(nn.make_rng(), shape)
      x_drop = nn.dropout(x, 0.5)


  Returns:
    A unique jax.random.PRNGKey.
  """
  if not _prng_stack:
    raise ValueError('Use the `nn.stochastic()` context manager to enable'
                     ' stochastic computations.')
  rng_frame = _prng_stack[-1]
  return rng_frame.make_rng()


def dropout(inputs, rate, deterministic=False, rng=None):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Applies a random dropout mask to the input.

  Args:
    inputs: the inputs that should be randomly masked.
    rate: the probablity of masking out a value.
    deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
      masked, whereas if true, no mask is applied and the inputs are returned as
      is.
    rng: an optional `jax.random.PRNGKey`. By default `nn.make_rng()` will
      be used.
  Returns:
    The masked inputs.
  """
  if rate == 0.:
    return inputs
  keep_prob = 1. - rate

  if deterministic:
    return inputs
  else:
    if rng is None:
      rng = make_rng()
    mask = random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
    return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

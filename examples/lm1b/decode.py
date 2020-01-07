# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode from a trained model."""

import jax
import jax.numpy as jnp
import numpy as np


def multinomial(rng, logits):
  """Draws samples from a multinomial distribution.

  Args:
    rng: A JAX PRNGKey.
    logits: Unnormalized log-probabilities.

  Returns:
    sampled categories.
  """
  probs = jax.nn.softmax(logits)
  probs = jnp.cumsum(probs, axis=-1)
  a = jax.random.uniform(rng, logits.shape[:-1] + (1,))
  out = jnp.argmin(a > probs, axis=-1)
  return out[..., None]


@jax.jit
def predict(model, inputs):
  """Given the model and an input, predicts the output.

  Args:
    model: a model (e.g. a trained model: optimizer.target).
    inputs: input to the model (encoded text).

  Returns:
    output of the model.

  """
  return model(inputs, train=False)


def sample_sequence(model, length, context, num_samples=1, temperature=1.0):
  """Samples from the given model.

  Args:
    model: a model (e.g. a trained model: optimizer.target).
    length: desirable length of the samples.
    context: context (prefix to be fed to the model).
    num_samples: number of samples to draw.
    temperature: sampling temperature.

  Returns:
    Generated samples.

  """
  # TODO(dehghani): add repetition penalty from CTRL
  # (https://arxiv.org/abs/1909.05858)
  # TODO(dehghani): use caching
  # TODO(dehghani): implement beam search

  context = np.repeat(context[None, ...], num_samples, axis=0)
  generated = context
  for _ in range(length):
    outputs = predict(model, generated)
    next_token_logits = outputs[:, -1, :] / (
        temperature if temperature > 0 else 1.)
    if temperature == 0:  # greedy sampling:
      next_token = jnp.argmax(next_token_logits, axis=-1)[..., None]
    else:
      next_token = multinomial(jax.random.PRNGKey(0), next_token_logits)

    generated = jnp.concatenate((generated, next_token), axis=1)
  return generated

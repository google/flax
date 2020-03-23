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

"""LSTM classifier model for SST-2."""

import functools
from typing import Any, Callable, Text

import flax
from flax import nn
import jax
import jax.numpy as jnp
from jax import lax

import numpy as np

# pylint: disable=arguments-differ


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model(rng, batch_size, model_kwargs):
  """Instantiate a new model."""
  with nn.stochastic(rng):
    model_def = LSTMClassifier.partial(**model_kwargs)
    _, initial_params = model_def.init_by_shape(
        rng, [((batch_size, 55), jnp.int32), ((batch_size,), jnp.int32)])
    model = nn.Model(model_def, initial_params)
    return model


def load_model(model_dir: Text, vocab_size: int, batch_size: int):
  """Load a model from checkpoint."""
  model = create_model(jax.random.PRNGKey(0), batch_size,
                       dict(vocab_size=vocab_size))
  with nn.stochastic(jax.random.PRNGKey(0)):
    optimizer = flax.optim.Adam(learning_rate=0.001).create(model)
    optimizer = flax.training.checkpoints.restore_checkpoint(
        model_dir, optimizer)
  model = optimizer.target
  return model


def pretrained_init(_, shape, embed: np.ndarray = None, dtype=np.float32):
  assert embed.shape == shape, \
      'The pretrained embeddings do not have the expected shape.'
  return embed.astype(dtype)


class Embedding(nn.Module):
  """Embedding Module."""

  def apply(self,
            inputs: jnp.ndarray,
            num_embeddings: int,
            features: int,
            emb_init: Callable[..., np.ndarray] = nn.initializers.normal(
                stddev=0.1),
            frozen: bool = True):
    # inputs.shape = <int64>[batch_size, seq_length]
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    embed = jnp.take(embedding, inputs, axis=0)
    if frozen:  # Keep the embeddings fixed at initial (pretrained) values.
      embed = lax.stop_gradient(embed)
    return embed


class LSTMEncoder(nn.Module):
  """LSTM encoder. Turns a sequence of vectors into a vector."""

  def apply(self,
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            hidden_size: int = 256):
    # inputs.shape = <float32>[batch_size, seq_length, emb_size].
    batch_size = inputs.shape[0]
    carry = nn.LSTMCell.initialize_carry(nn.make_rng(), (batch_size,),
                                         hidden_size)
    _, outputs = nn.attention.scan_in_dim(
        nn.LSTMCell.partial(name='lstm'), carry, inputs, axis=1)
    return outputs[jnp.arange(batch_size), jnp.maximum(0, lengths-1), :]


class MLP(nn.Module):
  """A 2-layer MLP."""

  def apply(self,
            inputs: np.ndarray,
            hidden_size: int = 256,
            output_size: int = 1,
            output_bias: bool = False,
            dropout: float = 0.25,
            train: bool = True):
    # inputs.shape = <float32>[batch_size, seq_length, hidden_size]
    hidden = nn.Dense(inputs, hidden_size, name='hidden')
    hidden = nn.tanh(hidden)
    if train:
      hidden = nn.dropout(hidden, rate=dropout)
    output = nn.Dense(hidden, output_size, bias=output_bias, name='output')
    return output


class LSTMClassifier(nn.Module):
  """LSTM classifier."""

  def apply(self,
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            vocab_size: int = 0,
            embedding_size: int = 300,
            hidden_size: int = 256,
            output_size: int = 1,
            dropout: float = 0.5,
            emb_init: Callable[..., Any] = nn.initializers.normal(stddev=0.1),
            train: bool = True):
    """Encodes the input sequence and makes a prediction using an MLP."""
    # inputs.shape = <int64>[batch_size, seq_length]
    # Embed the inputs. Shape: <float32>[batch_size, seq_length, emb_size]
    embed = Embedding(inputs, vocab_size, embedding_size, emb_init=emb_init,
                      name='embed')
    if train:
      embed = nn.dropout(embed, rate=dropout)

    # pylint: disable=unpacking-non-sequence
    # Encode the sequence of embedding using an LSTM.
    hidden = LSTMEncoder(embed, lengths, hidden_size=hidden_size,
                         name='encoder')
    if train:
      hidden = nn.dropout(hidden, rate=dropout)

    # Predict the class using an MLP.
    logits = MLP(hidden, hidden_size=hidden_size, output_size=output_size,
                 output_bias=False, dropout=dropout, name='mlp', train=train)
    return logits


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
from typing import Any, Callable, Dict, Text

import flax
from flax import nn
import jax
import jax.numpy as jnp
from jax import lax

import numpy as np

# pylint: disable=arguments-differ,too-many-arguments


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_model(seed: int, batch_size: int, max_len: int,
                 model_kwargs: Dict[Text, Any]):
  """Instantiates a new model."""
  module = TextClassifier.partial(train=False, **model_kwargs)
  _, initial_params = module.init_by_shape(
      jax.random.PRNGKey(seed),
      [((batch_size, max_len), jnp.int32),
       ((batch_size,), jnp.int32)])
  model = nn.Model(module, initial_params)
  return model


def word_dropout(inputs: jnp.ndarray, rate: float, unk_idx: int, 
        deterministic: bool = False):
  """Replaces a fraction (rate) of inputs with <unk>."""
  if deterministic or rate == 0.:
    return inputs

  mask = jax.random.bernoulli(nn.make_rng(), p=rate, shape=inputs.shape)
  return jnp.where(mask, jnp.array([unk_idx]), inputs)


class Embedding(nn.Module):
  """Embedding Module."""

  def apply(self,
            inputs: jnp.ndarray,
            num_embeddings: int,
            features: int,
            emb_init: Callable[...,
                               np.ndarray] = nn.initializers.normal(stddev=0.1),
            frozen: bool = False):
    # inputs.shape = <int64>[batch_size, seq_length]
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    embed = jnp.take(embedding, inputs, axis=0)
    if frozen:  # Keep the embeddings fixed at initial (pretrained) values.
      embed = lax.stop_gradient(embed)
    return embed


class LSTM(nn.Module):
  """LSTM encoder. Turns a sequence of vectors into a vector."""

  def apply(self,
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            hidden_size: int = None):
    # inputs.shape = <float32>[batch_size, seq_length, emb_size].
    # lengths.shape = <int64>[batch_size,]
    batch_size = inputs.shape[0]
    carry = nn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size,), hidden_size)
    _, outputs = flax.jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm_cell'), carry, inputs, axis=1)
    return outputs[jnp.arange(batch_size), jnp.maximum(0, lengths - 1), :]


class MLP(nn.Module):
  """A 2-layer MLP."""

  def apply(self,
            inputs: jnp.ndarray,
            hidden_size: int = None,
            output_size: int = None,
            output_bias: bool = False,
            dropout: float = None,
            train: bool = None):
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
            embed: jnp.ndarray,
            lengths: jnp.ndarray,
            hidden_size: int = None,
            output_size: int = None,
            dropout: float = None,
            emb_dropout: float = None,
            train: bool = None):
    """Encodes the input sequence and makes a prediction using an MLP."""
    # embed <float32>[batch_size, seq_length, embedding_size]
    # lengths <int64>[batch_size]
    if train:
      embed = nn.dropout(embed, rate=emb_dropout)

    # Encode the sequence of embedding using an LSTM.
    hidden = LSTM(embed, lengths, hidden_size=hidden_size, name='lstm')
    if train:
      hidden = nn.dropout(hidden, rate=dropout)

    # Predict the class using an MLP.
    logits = MLP(
        hidden,
        hidden_size=hidden_size,
        output_size=output_size,
        output_bias=False,
        dropout=dropout,
        name='mlp',
        train=train)
    return logits


class TextClassifier(nn.Module):
  """Full classification model."""

  def apply(self,
            inputs: jnp.ndarray,
            lengths: jnp.ndarray,
            unk_idx: int = 1,
            vocab_size: int = None,
            embedding_size: int = None,
            word_dropout_rate: float = None,
            freeze_embeddings: bool = None,
            train: bool = False,
            emb_init: Callable[..., Any] = nn.initializers.normal(stddev=0.1),
            **kwargs):
    # Apply word dropout.
    if train:
      inputs = word_dropout(inputs, rate=word_dropout_rate, unk_idx=unk_idx)

    # Embed the inputs.
    embed = Embedding(
        inputs,
        vocab_size,
        embedding_size,
        emb_init=emb_init,
        frozen=freeze_embeddings,
        name='embed')

    # Encode with LSTM and classify.
    logits = LSTMClassifier(
        embed, lengths, train=train, name='lstm_classifier', **kwargs)
    return logits

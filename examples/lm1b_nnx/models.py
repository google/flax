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

"""Transformer-based language model.

Reusing decoder only model from examples/wmt.
"""

# pylint: disable=attribute-defined-outside-init
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from flax import nnx
from configs import default

Shape = tuple[int, ...]
Dtype = Any


@dataclasses.dataclass(unsafe_hash=True)
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  vocab_size: int
  output_vocab_size: int
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  kernel_init: nnx.Initializer = nnx.initializers.xavier_uniform()
  bias_init: nnx.Initializer = nnx.initializers.normal(stddev=1e-6)
  posemb_init: nnx.Initializer | None = None
  axis_rules: default.MeshRules = dataclasses.field(
    default_factory=default.MeshRules
  )

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def shift_right(x: jax.Array, axis: int = 1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths: list[tuple[int, int]] = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
    x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x: jax.Array, segment_ids=None, axis: int = 1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= segment_ids == shift_right(segment_ids, axis=axis)
  return shifted


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nnx.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(
    self,
    config: TransformerConfig,
    *,
    decode: bool = False,
    rngs: nnx.Rngs,
  ):
    self.config = config
    self.decode = decode
    self.pos_emb_shape = (1, config.max_len, config.emb_dim)

    if config.posemb_init is not None:
      self.pos_embedding = nnx.Param(
        config.posemb_init(rngs(), self.pos_emb_shape)
      )
    else:
      self.pos_embedding = None

  def __call__(self, inputs: jax.Array, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, (
      'Number of dimensions should be 3, but it is: %d' % inputs.ndim
    )
    length = inputs.shape[1]

    if self.pos_embedding is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(
        None, self.pos_emb_shape
      )
    else:
      pos_embedding = self.pos_embedding.value

    # We use a cache position index for tracking decoding position.
    if self.decode:
      _, _, df = pos_embedding.shape
      # equivalent to pos_embedding[:, i:i+1] but traceable
      pos_embedding = lax.dynamic_slice(
        pos_embedding, jnp.array((0, self.cache_index.value, 0)), (1, 1, df)
      )
      self.cache_index.value += 1
    else:
      pos_embedding = pos_embedding[:, :length, :]

    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pos_embedding
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pos_embedding[0], inputs_positions, axis=0)

  def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.uint32))


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
    self.config = config

    self.linear1 = nnx.Linear(
      config.emb_dim,
      config.mlp_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init,
        config.axis_rules('embed', 'mlp'),
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init,
        config.axis_rules('mlp'),
      ),
      rngs=rngs,
    )
    self.linear2 = nnx.Linear(
      config.mlp_dim,
      config.emb_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init,
        config.axis_rules('mlp', 'embed'),
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init,
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=config.dropout_rate)

  def __call__(self, inputs: jax.Array, *, rngs: nnx.Rngs | None = None):
    """Applies Transformer MlpBlock module."""
    x = self.linear1(inputs)
    x = nnx.relu(x)
    x = self.dropout(x, rngs=rngs)
    output = self.linear2(x)
    output = self.dropout(output, rngs=rngs)
    return output


class EncoderDecoder1DBlock(nnx.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
    self.config = config

    self.ln1 = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        config.axis_rules('embed'),
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(),
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.ln2 = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        config.axis_rules('embed'),
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(),
        config.axis_rules('embed'),
      ),
      rngs=rngs,
    )
    self.attention = nnx.MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.emb_dim,
      qkv_features=config.qkv_dim,
      dtype=config.dtype,
      kernel_init=nnx.with_partitioning(
        config.kernel_init, config.axis_rules('embed', 'kv')
      ),
      bias_init=nnx.with_partitioning(
        config.bias_init, config.axis_rules('embed')
      ),
      use_bias=False,
      broadcast_dropout=False,
      dropout_rate=config.attention_dropout_rate,
      rngs=rngs,
      keep_rngs=False,
    )
    self.mlp = MlpBlock(config=config, rngs=rngs)
    self.dropout = nnx.Dropout(rate=config.dropout_rate)

  def __call__(
    self,
    inputs: jax.Array,
    *,
    decoder_mask: jax.Array | None = None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    # Decoder block.
    assert inputs.ndim == 3
    x = self.ln1(inputs)
    x = self.attention(x, mask=decoder_mask, rngs=rngs)
    x = self.dropout(x, rngs=rngs)
    x = x + inputs
    # MLP block.
    z = self.ln2(x)
    z = self.mlp(z, rngs=rngs)
    return x + z


class Decoder(nnx.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  def __init__(
    self,
    config: TransformerConfig,
    shared_embedding: nnx.Embed | None = None,
    *,
    decode: bool = False,
    rngs: nnx.Rngs,
  ):
    self.config = config
    self.decode = decode
    self.shared_embedding = shared_embedding

    # Target Embedding
    if self.shared_embedding is None:
      self.output_embed = nnx.Embed(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        embedding_init=nnx.with_partitioning(
          nnx.initializers.normal(stddev=1.0),
          config.axis_rules('vocab', 'embed'),
        ),
        rngs=rngs,
      )
    else:
      self.output_embed = self.shared_embedding

    self.posembed_output = AddPositionEmbs(config=config, rngs=rngs)
    self.dropout = nnx.Dropout(rate=config.dropout_rate)
    for idx in range(config.num_layers):
      layer = EncoderDecoder1DBlock(config=config, rngs=rngs)
      setattr(self, f'encoderdecoderblock_{idx}', layer)

    self.encoderdecoder_norm = nnx.LayerNorm(
      num_features=config.emb_dim,
      dtype=config.dtype,
      bias_init=nnx.with_partitioning(
        nnx.initializers.zeros_init(), config.axis_rules('embed')
      ),
      scale_init=nnx.with_partitioning(
        nnx.initializers.ones_init(), config.axis_rules('embed')
      ),
      rngs=rngs,
    )
    if not config.logits_via_embedding:
      self.logitdense = nnx.Linear(
        in_features=config.emb_dim,
        out_features=config.output_vocab_size,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
          config.kernel_init, config.axis_rules('embed', 'vocab')
        ),
        bias_init=nnx.with_partitioning(
          config.bias_init, config.axis_rules('vocab')
        ),
        rngs=rngs,
      )
    else:
      self.logitdense = None

  def __call__(
    self,
    inputs,
    *,
    inputs_positions=None,
    inputs_segmentation=None,
    decoder_mask=None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config
    assert inputs.ndim == 2  # (batch, len)

    y = inputs.astype('int32')
    if not self.decode:
      y = shift_inputs(y, segment_ids=inputs_segmentation)
    y = self.output_embed(y)
    y = self.posembed_output(y, inputs_positions=inputs_positions)
    y = self.dropout(y, rngs=rngs)

    y = y.astype(config.dtype)

    # Target-Input Decoder
    for idx in range(config.num_layers):
      # TODO(cgarciae): use a list of layers instead of getattr
      layer: EncoderDecoder1DBlock = getattr(self, f'encoderdecoderblock_{idx}')
      y = layer(
        y,
        decoder_mask=decoder_mask,
        rngs=rngs,
      )
    y = self.encoderdecoder_norm(y)

    # Decoded Logits
    if self.logitdense:
      logits = self.logitdense(y)
    else:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    return logits


class TransformerLM(nnx.Module):
  """Transformer pure decoder stack for language modelling.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  def __init__(
    self, config: TransformerConfig, *, decode: bool = False, rngs: nnx.Rngs
  ):
    self.config = config
    self.decode = decode
    self.decoder = Decoder(config=config, shared_embedding=None, rngs=rngs)

  def __call__(
    self,
    inputs,
    *,
    inputs_positions=None,
    inputs_segmentation=None,
    rngs: nnx.Rngs | None = None,
  ):
    """Applies TransformerLM on the inputs.

    Args:
      inputs: target data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    config = self.config

    # Make padding attention masks.
    if self.decode:
      # for fast autoregressive decoding we use no decoder mask
      decoder_mask = None
    else:
      decoder_mask = nnx.combine_masks(
        nnx.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
        nnx.make_causal_mask(inputs, dtype=config.dtype),
      )

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nnx.combine_masks(
        decoder_mask,
        nnx.make_attention_mask(
          inputs_segmentation,
          inputs_segmentation,
          jnp.equal,
          dtype=config.dtype,
        ),
      )

    logits = self.decoder(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      decoder_mask=decoder_mask,
      rngs=rngs,
    )
    return logits.astype(self.config.dtype)

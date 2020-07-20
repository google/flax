# Copyright 2020 The Flax Authors.
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

"""Transformer-based machine translation model."""

# pylint: disable=attribute-defined-outside-init

from typing import Callable, Any, Optional

from jax import lax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax import struct


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: Optional[int] = None
  output_vocab_size: Optional[int] = None
  share_embeddings: bool = False
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
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
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
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    posemb_init: positional embedding initializer, if None, then use a
      fixed (non-learned) sinusoidal embedding table.
  """
  config: TransformerConfig
  posemb_init: Callable = None  # TODO(levskaya) move to config?!

  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if cfg.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(self,
                 cfg.mlp_dim,
                 dtype=cfg.dtype,
                 kernel_init=cfg.kernel_init,
                 bias_init=cfg.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(self, rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    output = nn.Dense(self,
                      actual_out_dim,
                      dtype=cfg.dtype,
                      kernel_init=cfg.kernel_init,
                      bias_init=cfg.bias_init)(x)
    output = nn.Dropout(self, rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def __call__(self,
               inputs,
               inputs_segmentation=None, # REFACTOR
               padding_mask=None): # REFACTOR
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(self, dtype=cfg.dtype)(inputs)
    x = nn.SelfAttention(
        self,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(
            x,
            segmentation=inputs_segmentation, # REFACTOR
            padding_mask=padding_mask) # REFACTOR

    x = nn.Dropout(self, rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(self, dtype=cfg.dtype)(x)
    y = MlpBlock(self, config=cfg)(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def __call__(self,
               targets,
               encoded,
               inputs_segmentation=None,  # REFACTOR
               targets_segmentation=None,  # REFACTOR
               padding_mask=None,  # REFACTOR
               key_padding_mask=None):  # REFACTOR
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      key_padding_mask: bool, mask padding tokens

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(self, dtype=cfg.dtype)(targets)
    x = nn.SelfAttention(
        self,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode)(x,
                           padding_mask=padding_mask,
                           segmentation=targets_segmentation)
    x = nn.Dropout(self, rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(self, dtype=cfg.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        self,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(
            y,
            encoded,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=targets_segmentation,
            key_segmentation=inputs_segmentation)

    y = nn.Dropout(self, rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(self, dtype=cfg.dtype)(y)
    z = MlpBlock(self, config=cfg)(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config

    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          self,
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(self, config=cfg, name='posembed_input')(
        x, inputs_positions=inputs_positions)
    x = nn.Dropout(self, rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)

    x = x.astype(cfg.dtype)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(self, config=cfg, name=f'encoderblock_{lyr}')(
          x,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation)

    encoded = nn.LayerNorm(self, dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  def __call__(self,
               encoded,
               src_padding_mask,
               targets,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None,
               tgt_padding_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[..., None]

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          self,
          num_embeddings=cfg.output_vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if not cfg.decode:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(self, config=cfg, name='posembed_output')(
        y, inputs_positions=targets_positions)
    y = nn.Dropout(self, rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)

    y = y.astype(cfg.dtype)

    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          self, config=cfg, name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              padding_mask=tgt_padding_mask,
              key_padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation,
              targets_segmentation=targets_segmentation)
    y = nn.LayerNorm(self, dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          self,
          cfg.output_vocab_size,
          dtype=cfg.dtype,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          name='logitdense')(y)
    return logits


class Transformer(nn.MultiModule):
  """Transformer Model for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def setup(self):
    cfg = self.config

    if cfg.share_embeddings:
      if cfg.output_vocab_size is not None:
        assert cfg.output_vocab_size == cfg.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          self,
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      self.shared_embedding = None

    self.encoder = Encoder(self,
                           config=cfg,
                           shared_embedding=self.shared_embedding)
    self.decoder = Decoder(self,
                           config=cfg,
                           shared_embedding=self.shared_embedding)

  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None,
               tgt_padding_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.

    Returns:
      output of a transformer decoder.
    """
    src_padding_mask = (inputs > 0)[..., None]

    encoded = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    logits = self.decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask)

    return logits.astype(self.config.dtype)

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.
  def encode(self,
             inputs,
             inputs_positions=None,
             inputs_segmentation=None):
    return self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None):
    logits = self.decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask)
    return logits.astype(self.config.dtype)

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

from flax.core import Scope, init, apply, nn

from typing import Callable, Any, Optional

from flax import struct

from jax import lax
import jax.numpy as jnp
import numpy as np


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


def add_position_embs(
    scope,
    inputs,
    inputs_positions=None,
    max_len=512,
    posemb_init=None,
    decode=False):
  """Applies AddPositionEmbs module.
  By default this layer uses a fixed sinusoidal embedding table. If a
  learned position embedding is desired, pass an initializer to
  posemb_init.
  Args:
    inputs: input data.
    inputs_positions: input position indices for packed sequences.
    max_len: maximum possible length for the input.
    posemb_init: positional embedding initializer, if None, then use a
      fixed (non-learned) sinusoidal embedding table.
    cache: flax attention cache for fast decoding.
  Returns:
    output: `(bs, timesteps, in_dim)`
  """
  # inputs.shape is (batch_size, seq_len, emb_dim)
  assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                            ' but it is: %d' % inputs.ndim)
  length = inputs.shape[1]
  pos_emb_shape = (1, max_len, inputs.shape[-1])
  if posemb_init is None:
    # Use a fixed (non-learned) sinusoidal position embedding.
    pos_embedding = sinusoidal_init(
        max_len=max_len)(None, pos_emb_shape, None)
  else:
    pos_embedding = scope.param('pos_embedding', posemb_init, pos_emb_shape)
  pe = pos_embedding[:, :length, :]

  if decode:
    if not scope.has_variable('cache', 'idx'):
      cache_idx = jnp.zeros((), jnp.uint32)
    else:
      cache_idx = scope.get_variable('cache', 'idx')
      cache_idx = cache_idx + 1
      _, _, df = pos_embedding.shape
      pe = lax.dynamic_slice(pos_embedding,
                              jnp.array((0, cache_idx, 0)),
                              jnp.array((1, 1, df)))
    scope.put_variable('cache', 'idx', cache_idx)

  if inputs_positions is None:
    # normal unpacked case:
    return inputs + pe
  else:
    # for packed data we need to use known position indices:
    return inputs + jnp.take(pe[0], inputs_positions, axis=0)


@struct.dataclass
class TransformerConfig:
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

  inputs_segmentation: Any = None
  padding_mask: Any = None
  targets_segmentation: Any = None

  kernel_init: Any = nn.initializers.xavier_uniform()
  bias_init: Any = nn.initializers.normal(stddev=1e-6)


def attention(scope, x, config, inputs_kv, causal_mask,
              segmentation=None, key_segmentation=None, key_padding_mask=None):
  return nn.multi_head_dot_product_attention(
    scope,
    x,
    num_heads=config.num_heads,
    dtype=config.dtype,
    inputs_kv=x,
    qkv_features=config.qkv_dim,
    attention_axis=(1,),
    causal_mask=causal_mask,
    padding_mask=config.padding_mask,
    segmentation=segmentation,
    key_segmentation=key_segmentation,
    key_padding_mask=key_padding_mask,
    kernel_init=config.kernel_init,
    bias_init=config.bias_init,
    bias=False,
    broadcast_dropout=False,
    dropout_rate=config.attention_dropout_rate,
    deterministic=config.deterministic,
    cache=config.decode
  )


def mlp_block(scope, inputs, config: TransformerConfig)
  """Applies Transformer MlpBlock module."""
  dense = functools.partial(nn.dense,
      dtype=config.dtype,
      kernel_init=config.kernel_init, bias_init=config.bias_init)
  dropout = functools.partial(nn.dropout,
      rate=config.dropout_rate,
      deterministic=config.deterministic)
  x = scope.child(dense)(inputs, config.mlp_dim)
  x = nn.relu(x)
  x = scope.child(dropout)(x)
  output = scope.child(dense)(x, inputs.shape[-1])
  output = scope.child(dropout)(output)
  return output


def encoder_1d_block(scope, inputs, config: TransformerConfig):
  """Applies Encoder1DBlock module.
  Args:
    inputs: input data.
  Returns:
    output after transformer encoder block.
  """
  norm = functools.partial(nn.layer_norm, dtype=dtype)

  # Attention block.
  assert inputs.ndim == 3
  x = scope.child(norm)(inputs)
  x = scope.child(attention)(
      x, config,
      inputs_kv=x,
      causal_mask=False,
      segmention=config.inputs_segmentation,

  )
  x = scope.child(nn.dropout)(x,
      rate=config.dropout_rate,
      deterministic=config.deterministic)
  x = x + inputs

  # MLP block.
  y = x = scope.child(norm)(x)
  y = scope.child(mlp_block)(
      y,
      config=config)

  return x + y

def encoder_decoder_1d_block(scope, targets, encoded, config: TransformerConfig):
  """Applies EncoderDecoder1DBlock module.
  Args:
    targets: input data for decoder
    encoded: input data from encoder
  Returns:
    output after transformer encoder-decoder block.
  """

  # Decoder block.
  assert targets.ndim == 3

  dropout = functools.partial(nn.dropout,
      rate=config.dropout_rate, deterministic=config.deterministic)
  norm = functools.partial(nn.layer_norm, dtype=dtype)

  x = scope.child(norm)(targets)
  x = scope.child(attention)(x, config,
      inputs_kv=x,
      causal_mask=True,
      segmentation=config.targets_segmentation)
  x = scope.child(dropout)(x)
  x = x + targets

  # Encoder-Decoder block.
  y = scope.child(nn.layer_norm)(x, dtype=dtype)
  y = scope.child(attention)(
      y, config,
      inputs_kv=encoded,
      segmentation=config.targets_segmentation,
      key_padding_mask=config.key_padding_mask,
      key_segmentation=config.inputs_segmentation,
  )
  y = scope.child(dropout)(y)
  y = y + x

  # MLP block.
  z = scope.child(norm)(y)
  z = scope.child(mlp_block)(z, config=config)

  return y + z


def encoder(
    scope,
    inputs,
    config: TransformerConfig,
    shared_embedding = None):
  """Applies Transformer model on the inputs.
  Args:
    inputs: input data
  Returns:
    output of a transformer encoder.
  """
  assert inputs.ndim == 2  # (batch, len)

  # Padding Masks
  src_padding_mask = (inputs > 0)[..., None]

  # Input Embedding
  if shared_embedding is None:
    input_embed = scope.child(nn.embedding)(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    input_embed = shared_embedding
  x = inputs.astype(jnp.int32)
  x = input_embed.lookup(x)
  x = scope.child(add_position_embs, 'posembed_input')(
      x,
      inputs_positions=config.inputs_positions,
      max_len=config.max_len)
  x = scope.child(nn.dropout)(x, rate=config.dropout_rate, deterministic=config.deterministic)

  x = x.astype(dtype)

  # Input Encoder
  for lyr in range(num_layers):
    x = scope.child(encoder_1d_block, f'encoderblock_{lyr}')(
        x, config=config)
  encoded = scope.child(nn.layer_norm, 'encoder_norm')(x, dtype=config.dtype)

  return encoded


def decoder(scope, encoded, targets, config):
  """Applies Transformer model on the inputs.
  Args:
    encoded: encoded input data from encoder.
    targets: target inputs.
    src_padding_mask: padding mask for inputs.
    tgt_padding_mask: target tokens padding mask.
    shared_embedding: a shared embedding layer to use.
  Returns:
    output of a transformer decoder.
  """
  assert encoded.ndim == 3  # (batch, len, depth)
  assert targets.ndim == 2  # (batch, len)

  # Target Embedding
  if config.shared_embedding is None:
    output_embed = scope.child(nn.embedding)(
        num_embeddings=config.output_vocab_size,
        features=config.emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    output_embed = shared_embedding

  y = targets.astype('int32')
  if shift:
    y = shift_right(y)
  y = output_embed.lookup(y)
  y = scope.child(add_position_embs, 'posembed_output')(
      y,
      inputs_positions=config.targets_positions,
      max_len=config.max_len,
      decode=config.decode)
  y = scope.child(nn.dropout)(y, rate=config.dropout_rate, deterministic=config.deterministic)

  y = y.astype(config.dtype)

  # Target-Input Decoder
  for lyr in range(num_layers):
    y = scope.child(encoder_decoder_1d_block, f'encoderdecoderblock_{lyr}')(
        y,
        encoded,
        config=config)
  y = scope.child(nn.layer_norm, 'encoderdecoder_norm')(y, dtype=config.dtype)

  # Decoded Logits
  if config.logits_via_embedding:
    # Use the transpose of embedding matrix for logit transform.
    logits = output_embed.attend(y.astype(jnp.float32))
    # Correctly normalize pre-softmax logits for this shared case.
    logits = logits / jnp.sqrt(y.shape[-1])
  else:
    logits = scope.child(nn.dense, 'logitdense')(
        y,
        output_vocab_size,
        dtype=dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)
  return logits


def transformer(scope, inputs, targets, config):
  """Applies Transformer model on the inputs.
  Args:
    inputs: input data.
    targets: target data.
  Returns:
    output of a transformer decoder.
  """
  if output_vocab_size is None:
    output_vocab_size = vocab_size

  if share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  src_padding_mask = (inputs > 0)[..., None]

  encoded = scope.child(encoder, 'encoder')(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      train=train,
      vocab_size=vocab_size,
      shared_embedding=shared_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  logits = scope.child(decoder, 'decoder')(
      encoded,
      src_padding_mask,
      targets,
      targets_positions=targets_positions,
      inputs_segmentation=inputs_segmentation,
      targets_segmentation=targets_segmentation,
      tgt_padding_mask=tgt_padding_mask,
      train=train,
      shift=shift,
      cache=cache,
      output_vocab_size=output_vocab_size,
      shared_embedding=shared_embedding,
      logits_via_embedding=logits_via_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  return logits.astype(jnp.float32) if use_bfloat16 else logits


def transformer_encode(scope, inputs, config):
  """Applies Transformer model on the inputs.
  Args:
    inputs: input data.
    targets: target data.
  Returns:
    output of a transformer decoder.
  """
  # if config.output_vocab_size is None:
  #   output_vocab_size = vocab_size

  if config.share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  encoded = scope.child(encoder, 'encoder')(
      inputs,
      config=confg)

  return encoded


def transformer_decode(scope, encoded, targets, ):
  """Applies Transformer model on the inputs.
  Args:
    inputs: input data.
    targets: target data.
  Returns:
    output of a transformer decoder.
  """
  if output_vocab_size is None:
    output_vocab_size = vocab_size

  if share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  logits = scope.child(decoder, 'decoder')(encoded, src_padding_mask, targets, config)

  return logits.astype(jnp.float32) if use_bfloat16 else logits


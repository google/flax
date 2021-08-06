"""
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
from jax._src.lax.lax import DType
import jax.numpy as jnp
import numpy as np
from flax.linen.attention import MultiHeadDotProductAttention
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.linen.normalization import LayerNorm
from flax.linen.stochastic import Dropout

Dtype = Any


def sequential(*layers) -> Callable[[np.ndarray], np.ndarray]:

  def _lambda(x):
    for layer in layers:
      x = layer(x)
    return x

  return _lambda


@dataclass
class TransformerEncoderLayer(Module):
  """
  """

  num_heads: int
  qkv_features: int
  out_features: Optional[int] = None
  dropout_rate: float = 0.0
  activation: Callable[[np.ndarray], np.ndarray] = jax.nn.relu
  dtype: DType = jnp.float32

  @compact
  def __call__(self,
               src: np.ndarray,
               mask: Optional[np.ndarray] = None,
               deterministic: Optional[bool] = None) -> np.ndarray:

    src2 = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        out_features=self.out_features,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
    )(src, src, mask=mask)

    out_features = src2.shape[-1]

    src = src + Dropout(rate=self.dropout_rate)(src2)
    src = LayerNorm()(src)

    # MLP
    src2 = sequential(
        Dense(features=out_features),
        self.activation,
        Dropout(rate=self.dropout_rate),
        Dense(features=out_features),
    )(
        src)

    src = src + Dropout(rate=self.dropout_rate)(src2)
    src = LayerNorm()(src)
    return src


@dataclass
class TransformerEncoder(Module):

  encoder_layer: Callable[[], Module]
  num_layers: int
  norm: Optional[Callable[[], Module]] = None

  @compact
  def __call__(
      self,
      src: np.ndarray,
      mask: Optional[np.ndarray] = None,
      # src_key_padding_mask: Optional[np.ndarray] = None,
  ) -> np.ndarray:

    output = src

    for _ in range(self.num_layers):
      output = self.encoder_layer()(output, mask=mask)

    if self.norm is not None:
      output = self.norm()(output)

    return output


@dataclass
class TransformerDecoderLayer(Module):

  num_heads: int
  qkv_features: int
  out_features: Optional[int] = None
  dropout_rate: float = 0.1
  activation: Callable[[np.ndarray], np.ndarray] = jax.nn.relu
  dtype: DType = jnp.float32

  @compact
  def __call__(
      self,
      tgt: np.ndarray,
      memory: np.ndarray,
      tgt_mask: Optional[np.ndarray] = None,
      memory_mask: Optional[np.ndarray] = None,
      deterministic: Optional[bool] = None
      # tgt_key_padding_mask: Optional[np.ndarray] = None,
      # memory_key_padding_mask: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """
    """
    tgt2 = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        out_features=self.out_features,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
    )(tgt, tgt, mask=tgt_mask)

    tgt = tgt + Dropout(rate=self.dropout_rate)(tgt2)
    tgt = LayerNorm()(tgt)
    tgt2 = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        out_features=self.out_features,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
    )(
        tgt,
        memory,
        mask=memory_mask,
    )
    tgt = tgt + Dropout(rate=self.dropout_rate)(tgt2)
    tgt = LayerNorm()(tgt)
    tgt2 = sequential(
        Dense(features=self.out_features),
        self.activation,
        Dropout(rate=self.dropout_rate),
        Dense(features=self.out_features),
    )(
        tgt)
    tgt = tgt + Dropout(rate=self.dropout_rate)(tgt2)
    tgt = LayerNorm()(tgt)
    return tgt


@dataclass
class TransformerDecoder(Module):

  decoder_layer: Callable[[], Module]
  num_layers: int
  norm: Optional[Callable[[], Module]] = None

  @compact
  def __call__(
      self,
      tgt: np.ndarray,
      memory: np.ndarray,
      tgt_mask: Optional[np.ndarray] = None,
      memory_mask: Optional[np.ndarray] = None,
      # tgt_key_padding_mask: Optional[np.ndarray] = None,
      # memory_key_padding_mask: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """
    """
    output = tgt

    for _ in range(self.num_layers):
      output = self.decoder_layer()(
          output,
          memory,
          tgt_mask=tgt_mask,
          memory_mask=memory_mask,
          # tgt_key_padding_mask=tgt_key_padding_mask,
          # memory_key_padding_mask=memory_key_padding_mask,
      )

    if self.norm is not None:
      output = self.norm()(output)

    return output


@dataclass
class Transformer(Module):

  head_size: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  output_size: Optional[int] = None
  dropout: float = 0.1
  activation: Callable[[np.ndarray], np.ndarray] = jax.nn.relu
  custom_encoder: Optional[Callable] = None
  custom_decoder: Optional[Callable] = None

  @compact
  def __call__(
      self,
      src: np.ndarray,
      tgt: np.ndarray,
      src_mask: Optional[np.ndarray] = None,
      tgt_mask: Optional[np.ndarray] = None,
      memory_mask: Optional[np.ndarray] = None,
      # src_key_padding_mask: Optional[np.ndarray] = None,
      # tgt_key_padding_mask: Optional[np.ndarray] = None,
      # memory_key_padding_mask: Optional[np.ndarray] = None,
  ) -> np.ndarray:

    if src.shape[0] != tgt.shape[0]:
      raise RuntimeError("the batch number of src and tgt must be equal")

    if self.custom_encoder is not None:
      encoder = self.custom_encoder()
    else:
      encoder = TransformerEncoder(
          lambda: TransformerEncoderLayer(
              num_heads=self.head_size,
              qkv_features=self.num_heads,
              out_features=self.output_size,
              dropout_rate=self.dropout,
              activation=self.activation,
              dtype=self.dtype,
          ),
          num_layers=self.num_encoder_layers,
          norm=lambda: LayerNorm(),
      )

    if self.custom_decoder is not None:
      decoder = self.custom_decoder()
    else:
      decoder = TransformerDecoder(
          lambda: TransformerDecoderLayer(
              self.head_size,
              self.num_heads,
              self.output_size,
              self.dropout,
              self.activation,
          ),
          num_layers=self.num_decoder_layers,
          norm=lambda: LayerNorm(),
      )

    memory = encoder(
        src,
        mask=src_mask,
        # src_key_padding_mask=src_key_padding_mask
    )
    output = decoder(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        # tgt_key_padding_mask=tgt_key_padding_mask,
        # memory_key_padding_mask=memory_key_padding_mask,
    )

    return output

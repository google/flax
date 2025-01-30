# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Gemma transformer."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
from typing import Any

from flax import nnx
import helpers
import layers
import modules
import params as params_lib
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-importing-member,g-multiple-import

Cache = dict[str, modules.LayerCache]


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[modules.AttentionType]
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None

  @classmethod
  def from_path(cls, path: str) -> TransformerConfig:
    """Creates a TransformerConfig from loaded parameters."""
    metadata = params_lib.load_metadata(path)
    params = params_lib.load_params(path)

    try:
      model = metadata['somewhere in orbax checkpoint']

      if model in ('gemma-2-27-pt', 'gemma-2-27-it'):
        return cls.gemma_27b()
      elif model in ('gemma-2-9-pt', 'gemma-2-9-it'):
        return cls.gemma_9b()
    except KeyError:
      # V1 model that does not include model metadata.
      # Fall back to previous method
      return cls.from_params(params)

    raise ValueError('Verify checkpoint path is a Gemma checkpoint')

  @classmethod
  def from_params(cls, params: params_lib.Params) -> TransformerConfig:
    """Creates a TransformerConfig from loaded parameters.

    Use for V1 models only.

    Args:
      params: Model parameters

    Returns:
      TransformerConfig.
    """
    use_qkv_einsum = 'qkv_einsum' in params['transformer']['layer_0']['attn']
    if use_qkv_einsum:
      return cls.gemma_7b()
    elif not use_qkv_einsum:  # And something else
      return cls.gemma_2b()
    else:
      raise ValueError(
          'Params are not a Gemma 2b, or 7b variant. These may be a different'
          ' Gemma Architecture. Use from_path function to load params.'
      )

  @classmethod
  def gemma_2b(cls):
    num_layers = 18
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_7b(cls):
    num_layers = 28
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * 28,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_27b(cls):
    num_layers = 46
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=4608,
        hidden_dim=72728,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        final_logit_softcap=30.0,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma_9b(cls):
    num_layers = 42
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=28672,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  new_key = []
  for k in key:
    if k.startswith('layer_'):
      prefix, suffix = k.split('layer_')
      assert not prefix, prefix
      new_key.append('layers')
      new_key.append(int(suffix))
    elif k == 'gating_einsum':
      new_key.append('gate_proj')
      new_key.append('kernel')
    elif k == 'linear':
      new_key.append('down_proj')
      new_key.append('kernel')
    else:
      new_key.append(k)

  return tuple(new_key)


def _assign_linen_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
) -> dict[tuple[str, ...], Any]:
  if 'gate_proj' in mapped_path:
    state[mapped_path].value = val[0]
    state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]
  else:
    state[mapped_path].value = val
  return state


class Transformer(nnx.Module):
  """Gemma transformer."""

  @classmethod
  def from_params(
      cls, params: params_lib.Params, config: None | TransformerConfig = None
  ) -> Transformer:
    if config is None:
      config = TransformerConfig.from_params(params)
    return helpers.module_from_linen_variables(
        module_factory=lambda: cls(config, rngs=nnx.Rngs(params=0)),
        variables=params['transformer'],
        map_key_fn=_map_linen_var_names,
        assign_val_fn=_assign_linen_params_to_nnx_state,
    )

  def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs):
    self.embedder = modules.Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        rngs=rngs,
    )
    self.layers = [
        modules.Block(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            embed_dim=config.embed_dim,
            head_dim=config.head_dim,
            hidden_dim=config.hidden_dim,
            sliding_window_size=config.sliding_window_size,
            use_post_attn_norm=config.use_post_attn_norm,
            use_post_ffw_norm=config.use_post_ffw_norm,
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            attn_type=attn_type,
            rngs=rngs,
        )
        for _, attn_type in zip(
            range(config.num_layers), config.attention_types
        )
    ]
    self.final_norm = layers.RMSNorm(config.embed_dim, rngs=rngs)
    self.final_logits_softcap = config.final_logit_softcap

  def __call__(
      self,
      last_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: Array,  # [B, L, L']
  ) -> tuple[Array, Cache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(last_tokens)
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    if self.final_logits_softcap is not None:
      logits /= self.final_logits_softcap
      logits = jnp.tanh(logits) * self.final_logits_softcap

    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def embed_dim(self) -> int:
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    return len(self.layers)

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> Cache:
    """Initializes a new Transformer cache."""
    return {
        f'layer_{i}': self.layers[i].init_cache(
            cache_size=cache_size,
            batch_size=batch_size,
            dtype=dtype,
        )
        for i in range(self.num_layers)
    }


def make_causal_attn_mask(
    input_mask: Array,
) -> Array:
  """Attention mask in batch mode.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def build_positions_from_mask(input_mask: Array) -> Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)

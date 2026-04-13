# Copyright 2026 The Flax Authors.
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

from collections.abc import Iterable
import dataclasses
import enum
from typing import Any

import params as params_lib
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_270m = 18
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62

DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def make_attention_layers_types(
    pattern: tuple[AttentionType, ...],
    num_layers: int,
) -> tuple[AttentionType, ...]:
  """Returns the list of attention types for every layers."""

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for gemma transformer."""

  emb_vd: P | None = None
  q_weight_ndh: P | None = None
  kv_weight_cndh: P | None = None
  qkv_weight_cndh: P | None = None
  o_weight_nhd: P | None = None
  ffw_weight_df: P | None = None
  ffw_weight_fd: P | None = None
  rms_norm_weight: P | None = None
  act_btd: P | None = None
  act_btf: P | None = None
  act_btnh: P | None = None
  act_sbtnh: P | None = None

  fsdp_axis_name: str = "fsdp"
  tensor_parallel_axis_name: str = "tensor"

  @staticmethod
  def no_sharding():
    return ShardingConfig()

  @staticmethod
  def fsdp_tp_sharding(
      is_sampling: bool = False,
      fsdp_axis_name: str = "fsdp",
      tensor_parallel_axis_name: str = "tp",
  ):
    fsdp = fsdp_axis_name if not is_sampling else None
    tp = tensor_parallel_axis_name

    return ShardingConfig(
        emb_vd=P(tp, fsdp),
        q_weight_ndh=P(tp, fsdp, None),
        kv_weight_cndh=P(None, tp, fsdp, None),
        qkv_weight_cndh=P(None, tp, fsdp, None),
        o_weight_nhd=P(tp, None, fsdp),
        ffw_weight_df=P(fsdp, tp),
        ffw_weight_fd=P(tp, fsdp),
        rms_norm_weight=P(
            tp,
        ),
        act_btd=P(fsdp, None, None if is_sampling else tp),
        act_btf=P(fsdp, None, tp),
        act_btnh=P(fsdp, None, tp, None),
        act_sbtnh=P(None, fsdp, None, tp, None),
        tensor_parallel_axis_name=tensor_parallel_axis_name,
        fsdp_axis_name=fsdp_axis_name,
    )


@dataclasses.dataclass(slots=True, frozen=True)
class TransformerConfig:
  """Configuration for the text-only gemma transformer."""

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
  attention_types: Iterable[AttentionType]
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  transpose_gating_einsum: bool = False
  local_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  use_qk_norm: bool = False
  sliding_window_size: int | None = None
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  dropout_rate: float = 0.0

  shd_config: ShardingConfig = ShardingConfig.no_sharding()

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads) ** -0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  @classmethod
  def from_path(cls, path: str) -> "TransformerConfig":
    """Creates a TransformerConfig from loaded parameters."""
    params = params_lib.load_params(path)

    return cls.from_params(params)

  @classmethod
  def from_params(
      cls,
      params: params_lib.Params,
      shd_config: ShardingConfig = ShardingConfig.no_sharding(),
  ) -> "TransformerConfig":
    """Creates a TransformerConfig from loaded parameters.

    Args:
      params: Model parameters

    Returns:
      TransformerConfig.
    """

    # Post Attn Norm is only used starting from Gemma 2.
    use_post_attn_norm = (
        "post_attention_norm" in params["transformer"]["layer_0"]
    )

    # QK Norm is only used starting from Gemma 3.
    use_qk_norm = "_query_norm" in params["transformer"]["layer_0"]["attn"]

    # Num layers will give use the model size.
    layer_names = [
        name for name in params["transformer"].keys() if "layer" in name
    ]
    layer_names = [name.replace("layer_", "") for name in layer_names]
    num_layers = max([int(layer) for layer in layer_names]) + 1

    # set dtype and weight_dtype according to params
    flat_params, _ = jax.tree.flatten(params)
    wdtypes = {p.dtype for p in flat_params}
    assert len(wdtypes) == 1, wdtypes
    wdtype = next(iter(wdtypes))
    assert wdtype in (jnp.float32, jnp.bfloat16), wdtypes
    dtype = weight_dtype = wdtype

    if not use_post_attn_norm:  # Gemma 1.
      if num_layers == _NUM_LAYERS_GEMMA_2B:
        return cls.gemma_2b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA_7B:
        return cls.gemma_7b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      raise ValueError(
          "Guessing Gemma 1 model, but could not determine size from params."
      )
    elif not use_qk_norm:  # Gemma 2.
      if num_layers == _NUM_LAYERS_GEMMA2_2B:
        return cls.gemma2_2b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA2_9B:
        return cls.gemma2_9b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA2_27B:
        return cls.gemma2_27b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      raise ValueError(
          "Guessing Gemma 2 model but could not determine size from params."
      )
    else:  # Gemma 3.
      if num_layers == _NUM_LAYERS_GEMMA3_270m:
        return cls.gemma3_270m(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA3_1B:
        return cls.gemma3_1b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA3_4B:
        return cls.gemma3_4b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA3_12B:
        return cls.gemma3_12b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )
      if num_layers == _NUM_LAYERS_GEMMA3_27B:
        return cls.gemma3_27b(
            dtype=dtype, weight_dtype=weight_dtype, shd_config=shd_config
        )

    raise ValueError("Could not determine Gemma variant from params.")

  @classmethod
  def from_version_name(cls, name: str, **override) -> "TransformerConfig":
    possible_names = (
        "gemma_2b",
        "gemma_7b",
        "gemma2_2b",
        "gemma2_9b",
        "gemma2_27b",
        "gemma3_270m",
        "gemma3_1b",
        "gemma3_4b",
        "gemma3_12b",
        "gemma3_27b",
    )
    if name not in possible_names:
      raise ValueError(
          f"Unknown version name: {name}. "
          f"Please choose one of the following: {possible_names}"
      )
    if hasattr(cls, name):
      model_config = getattr(cls, name)(**override)
      return model_config
    else:
      raise RuntimeError(
          "Something wrong in TransformerConfig code. "
          f"No attribute {name} in TransformerConfig"
      )

  @classmethod
  def from_dict(cls, **config: Any) -> "TransformerConfig":
    # Deserialize query_pre_attn_norm values:
    if "query_pre_attn_norm" in config:
      config["query_pre_attn_norm"] = QueryPreAttentionNormalisation(
          config["query_pre_attn_norm"]
      )
    else:
      config["query_pre_attn_norm"] = (
          QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
      )
    return cls(**config)

  @classmethod
  def gemma_2b(cls, **override) -> "TransformerConfig":
    num_layers = _NUM_LAYERS_GEMMA_2B
    config = {
        "num_layers": num_layers,
        "num_embed": 256128,
        "embed_dim": 2048,
        "hidden_dim": 16384,
        "num_heads": 8,
        "head_dim": 256,
        "num_kv_heads": 1,
        "final_logit_softcap": None,
        "attention_types": (AttentionType.GLOBAL,) * num_layers,
        "use_post_attn_norm": False,
        "use_post_ffw_norm": False,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma_7b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA_7B
    config = {
        "num_layers": num_layers,
        "num_embed": 256128,
        "embed_dim": 3072,
        "hidden_dim": 24576,
        "num_heads": 16,
        "head_dim": 256,
        "num_kv_heads": 16,
        "final_logit_softcap": None,
        "attention_types": (AttentionType.GLOBAL,) * num_layers,
        "use_post_attn_norm": False,
        "use_post_ffw_norm": False,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma2_2b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA2_2B
    config = {
        "num_layers": num_layers,
        "num_embed": 256128,
        "embed_dim": 2304,
        "hidden_dim": 9216,
        "num_heads": 8,
        "head_dim": 256,
        "num_kv_heads": 4,
        "final_logit_softcap": 30.0,
        "attention_types": (
            (
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            )
            * int(num_layers / 2)
        ),
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
        ),
        "attn_logits_soft_cap": 50.0,
        "sliding_window_size": 4096,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma2_9b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA2_9B
    config = {
        "num_layers": num_layers,
        "num_embed": 256128,
        "embed_dim": 3584,
        "hidden_dim": 28672,
        "num_heads": 16,
        "head_dim": 256,
        "num_kv_heads": 8,
        "final_logit_softcap": 30.0,
        "attention_types": (
            (
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            )
            * int(num_layers / 2)
        ),
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "attn_logits_soft_cap": 50.0,
        "sliding_window_size": 4096,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma2_27b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA2_27B
    config = {
        "num_layers": num_layers,
        "num_embed": 256128,
        "embed_dim": 4608,
        "hidden_dim": 72728,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 16,
        "final_logit_softcap": 30.0,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "attention_types": (
            (
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            )
            * int(num_layers / 2)
        ),
        "attn_logits_soft_cap": 50.0,
        "sliding_window_size": 4096,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma3_270m(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA3_270m
    config = {
        "num_layers": num_layers,
        "final_logit_softcap": None,
        "num_embed": 262144,
        "embed_dim": 640,
        "hidden_dim": 2048,
        "num_heads": 4,
        "head_dim": 256,
        "num_kv_heads": 1,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "use_qk_norm": True,
        "attention_types": make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
        ),
        "attn_logits_soft_cap": None,
        "sliding_window_size": 512,
        "transpose_gating_einsum": True,
        "local_base_frequency": 10_000,
        "global_base_frequency": 1_000_000,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma3_1b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA3_1B
    config = {
        "num_layers": num_layers,
        "final_logit_softcap": None,
        "num_embed": 262144,
        "embed_dim": 1152,
        "hidden_dim": 6 * 1152,
        "num_heads": 4,
        "head_dim": 256,
        "num_kv_heads": 1,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "use_qk_norm": True,
        "attention_types": make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
        ),
        "attn_logits_soft_cap": None,
        "sliding_window_size": 512,
        "transpose_gating_einsum": True,
        "local_base_frequency": 10_000,
        "global_base_frequency": 1_000_000,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma3_4b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA3_4B
    config = {
        "num_layers": num_layers,
        "final_logit_softcap": None,
        "num_embed": 262_144,
        "embed_dim": 2560,
        "hidden_dim": 2560 * 8 // 2,
        "num_heads": 8,
        "head_dim": 256,
        "num_kv_heads": 4,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "use_qk_norm": True,
        "attention_types": make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
        ),
        "attn_logits_soft_cap": None,
        "sliding_window_size": 1024,
        "transpose_gating_einsum": True,
        "local_base_frequency": 10_000,
        "global_base_frequency": 1_000_000,
        "global_scale_factor": 8.0,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma3_12b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA3_12B
    config = {
        "num_layers": num_layers,
        "final_logit_softcap": None,
        "num_embed": 262144,
        "embed_dim": 30 * 128,
        "hidden_dim": 8 * 30 * 128 // 2,
        "num_heads": 16,
        "head_dim": 256,
        "num_kv_heads": 8,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "use_qk_norm": True,
        "attention_types": make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
        ),
        "attn_logits_soft_cap": None,
        "sliding_window_size": 1024,
        "transpose_gating_einsum": True,
        "local_base_frequency": 10_000,
        "global_base_frequency": 1_000_000,
        "global_scale_factor": 8.0,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  @classmethod
  def gemma3_27b(cls, **override):
    num_layers = _NUM_LAYERS_GEMMA3_27B
    config = {
        "num_layers": num_layers,
        "final_logit_softcap": None,
        "num_embed": 262144,
        "embed_dim": 5376,
        "hidden_dim": 5376 * 8 // 2,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 16,
        "use_post_attn_norm": True,
        "use_post_ffw_norm": True,
        "use_qk_norm": True,
        "attention_types": make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        "query_pre_attn_norm": (
            QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS
        ),
        "attn_logits_soft_cap": None,
        "sliding_window_size": 1024,
        "transpose_gating_einsum": True,
        "local_base_frequency": 10_000,
        "global_base_frequency": 1_000_000,
        "global_scale_factor": 8.0,
    }
    for key, value in override.items():
      config[key] = value
    return cls(**config)

  def __post_init__(self):
    if self.num_heads != self.num_kv_heads:
      if self.num_heads % self.num_kv_heads != 0:
        raise ValueError(
            f"Number of query heads ({self.num_heads}) must be divisible by "
            f"number of key/value heads ({self.num_kv_heads})."
        )

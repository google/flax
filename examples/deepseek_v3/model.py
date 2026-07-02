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
"""Minimal model definition."""

import dataclasses
import math
from dataclasses import field
from functools import partial
from jax._src.layout import Layout, DeviceLocalLayout as DLL
# from typing import tp.Any, tp.NamedTuple, tp.Protocol, tp.TypeGuard
import typing as tp
from collections.abc import Callable
from etils import epath
from warnings import warn

import jax
import jax.numpy as jnp
from jax import tree_util
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils

# from jax.experimental.pallas.ops.gpu import attention
# from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import (
  splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
  splash_attention_mask as mask_lib,
)
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import ragged_attention
from megablox import gmm as megablox_gmm
from flax import nnx


def create_mesh():
  """Always 1D because only care about FSDP."""
  devices = jax.devices()
  mesh_shape = (len(devices),)
  # Create a 1D mesh with all devices along the 'x' axis
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh(mesh_shape, devices), ('x',)
  )
  return mesh


class ShardingRules(tp.NamedTuple):
  batch: str | tuple[str, ...] | None = None
  sequence: str | tuple[str, ...] | None = None
  head_dim: str | tuple[str, ...] | None = None
  vocab_in: str | tuple[str, ...] | None = None
  vocab_out: str | tuple[str, ...] | None = None
  act_embed: str | tuple[str, ...] | None = None
  act_heads: str | tuple[str, ...] | None = None
  # attention layer
  qkv_embed: str | tuple[str, ...] | None = None
  qkv_heads: str | tuple[str, ...] | None = None
  q_lora: str | tuple[str, ...] | None = None
  kv_lora: str | tuple[str, ...] | None = None
  kv_lora_cache: str | tuple[str, ...] | None = None
  o_heads: str | tuple[str, ...] | None = None
  o_embed: str | tuple[str, ...] | None = None
  # MLP layer
  mlp_up_embed: str | tuple[str, ...] | None = None
  mlp_up_ffw: str | tuple[str, ...] | None = None
  mlp_down_ffw: str | tuple[str, ...] | None = None
  mlp_down_embed: str | tuple[str, ...] | None = None
  # MoE layer
  moe_e_experts: str | tuple[str, ...] | None = None
  moe_e_up_embed: str | tuple[str, ...] | None = None
  moe_e_up_ffw: str | tuple[str, ...] | None = None
  moe_e_down_ffw: str | tuple[str, ...] | None = None
  moe_e_down_embed: str | tuple[str, ...] | None = None
  moe_s_up_embed: str | tuple[str, ...] | None = None
  moe_s_up_ffw: str | tuple[str, ...] | None = None
  moe_s_down_ffw: str | tuple[str, ...] | None = None
  moe_s_down_embed: str | tuple[str, ...] | None = None
  moe_e_tp: str | tuple[str, ...] | None = (
    None  # moe forward function tensor parallelism
  )


BATCH_AXIS_NAME = 'x'
TENSOR_AXIS_NAME = ('y', 'z')

# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
tp_rules = ShardingRules(
  batch=BATCH_AXIS_NAME,
  sequence=None,
  head_dim=None,
  vocab_in=None,
  vocab_out=None,
  act_embed=None,
  act_heads=TENSOR_AXIS_NAME,
  # attention layer
  qkv_embed=TENSOR_AXIS_NAME,
  qkv_heads=TENSOR_AXIS_NAME,
  q_lora=None,
  kv_lora=None,
  kv_lora_cache=TENSOR_AXIS_NAME,
  o_heads=TENSOR_AXIS_NAME,
  o_embed=None,
  # MLP layer
  mlp_up_embed=None,
  mlp_up_ffw=TENSOR_AXIS_NAME,
  mlp_down_ffw=TENSOR_AXIS_NAME,
  mlp_down_embed=None,
  # MoE layer
  moe_e_experts=None,
  moe_e_up_embed=None,
  moe_e_up_ffw=TENSOR_AXIS_NAME,
  moe_e_down_ffw=TENSOR_AXIS_NAME,
  moe_e_down_embed=None,
  moe_s_up_embed=None,
  moe_s_up_ffw=TENSOR_AXIS_NAME,
  moe_s_down_ffw=TENSOR_AXIS_NAME,
  moe_s_down_embed=None,
  moe_e_tp=TENSOR_AXIS_NAME,
)


def _logical2physical(logical: P, rules: ShardingRules):
  """Converts logical to physical pspec."""

  def get_axis(rules, axis):
    if axis is None:
      return None
    if isinstance(axis, (tuple, list)):
      return tuple(getattr(rules, ax) for ax in axis)
    else:
      return getattr(rules, axis)

  spec = P(*(get_axis(rules, axis) for axis in logical))
  # check for repeated physical axes
  all_axes = jax.tree.leaves(tuple(spec))
  assert len(set(all_axes)) == len(all_axes), (
    f'Repeated physical axes found in logical spec = {logical}, spec = {spec}'
  )
  return spec


def _logical2sharding(
  logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules
):
  """Converts logical to sharding."""
  assert mesh is not None and rules is not None
  return jax.sharding.NamedSharding(mesh, _logical2physical(logical, rules))


def jax_struct(cls, meta_fields: tuple | None = None):
  """A wrapper around jax.tree_util.register_dataclass that infers data_fields"""
  if not dataclasses.is_dataclass(cls):
    cls = dataclasses.dataclass(cls)
  meta_fields = () if meta_fields is None else meta_fields
  all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
  data_fields = tuple(f for f in all_fields if f not in meta_fields)
  return tree_util.register_dataclass(
    cls, data_fields=data_fields, meta_fields=meta_fields
  )


@tree_util.register_static
@dataclasses.dataclass(unsafe_hash=True, eq=True)
class Config:
  embed: int = 7168
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  num_heads: int = 128
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  vocab_size: int = 129280
  num_layers: int = 61
  # Max seq len here can be a source of nasty bugs in incremental prefill
  # if we overflow (since dynamic slice will shunt left instead of erroring. Fix?
  max_seq_len: int = 8192
  causal: bool = True
  use_prefill_attn_kernel: bool = False
  use_decode_attn_kernel: bool = False
  weight_dtype_at_rest: 'jnp.dtype' = jnp.bfloat16
  active_weight_dtype: 'jnp.dtype' = jnp.bfloat16
  # Sharding rules
  # rules: ShardingRules = field(default_factory=lambda: ShardingRules(**fsdp_rules._asdict()))
  rules: ShardingRules | None = None
  mesh: jax.sharding.Mesh | None = None
  # Deepseek Yarn RoPE
  rope_theta: float = 1e4
  rope_scaling_factor: float = 40.0
  rope_beta_fast: float = 32
  rope_beta_slow: float = 1
  rope_mscale: float = 1
  rope_mscale_all_dim: float = 1
  rope_original_max_position_embeddings: int = 4096
  # quantization
  quant_scale_dtype: 'jnp.dtype' = jnp.float16
  quantize_mlps: bool = False
  # attention
  causal: bool = True
  # MLP
  ffw_size: int = 18432
  # MoE
  first_k_dense: int = 3
  moe_gate_dtype: 'jnp.dtype' = jnp.float32
  moe_ffw_size: int = 2048
  n_routed_experts: int = 256
  num_experts_per_tok: int = 8
  n_group: int = 8
  topk_group: int = 4
  routed_scaling_factor: float = 2.5
  n_shared_experts: int = 1
  use_megablox: bool | None = None  # None means device default
  psum_before_expert_reduce: bool = False
  quantized: bool = False

  def __post_init__(self):
    if not self.quantize_mlps and self.quantized:
      warn(
        f'{self.quantized=}, but {self.quantize_mlps=}, returning UNQUANTIZED'
      )
      self.quantized = False


@partial(
  jax_struct, meta_fields=('shape', 'logical_axes', 'initializer', 'metadata')
)
class TensorInfo:
  shape: jax.ShapeDtypeStruct
  logical_axes: tuple
  initializer: Callable | None = None
  metadata: dict = field(default_factory=dict)


# module reload friendly isinstance check
_isinstance = lambda x, cls: (type(x).__name__ == cls.__name__) and (
  type(x).__module__ == cls.__module__
)
is_param = lambda x: _isinstance(x, TensorInfo)
which_platform = lambda cfg: cfg.mesh.devices.reshape(-1)[0].platform


class _Init:
  @classmethod
  def abstract(cls, cfg: Config, *args, **kw):
    raise NotImplementedError

  @classmethod
  def shardings(cls, cfg: Config, *args, **kw):
    abstract = cls.abstract(cfg, *args, **kw)
    return jax.tree.map(
      lambda info: _logical2sharding(info.logical_axes, cfg.mesh, cfg.rules),
      abstract,
      is_leaf=is_param,
    )

  @classmethod
  def init(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kw):
    abstract = cls.abstract(cfg, *args, **kw)
    shardings = jax.tree.map(
      lambda info: _logical2sharding(info.logical_axes, cfg.mesh, cfg.rules),
      abstract,
      is_leaf=is_param,
    )

    @partial(jax.jit, out_shardings=shardings)
    def _init():
      num_leaves = len(
        jax.tree.leaves(abstract, is_leaf=is_param)
      )  # one new RNG key per tensor
      key_iter = iter(jax.random.split(key, num_leaves))
      return jax.tree.map(
        lambda info: info.initializer(
          next(key_iter), info.shape.shape, info.shape.dtype
        ),
        abstract,
        is_leaf=is_param,
      )

    return _init()


class ShardedParam(nnx.Param[jax.Array]):
  sharding: tuple[str | None, ...]

  def __init__(
    self, value: jax.Array, sharding: tuple[str | None, ...], **kwargs
  ):
    super().__init__(value, sharding=sharding, **kwargs)


def simple_quantize_info(
  x: ShardedParam,
  axis: int | tuple[int, ...],
  scale_dtype=jnp.float16,
):
  if not isinstance(axis, (list, tuple)):
    axis = (axis,)
  axis = tuple(z % len(x.value.shape) for z in axis)
  new_shape = tuple(ax for i, ax in enumerate(x.value.shape) if i not in axis)
  new_logical_axes = tuple(
    ax for i, ax in enumerate(x.sharding) if i not in axis
  )
  x.value = jnp.zeros(x.value.shape, dtype=jnp.int8)
  return ShardedParam(
    jnp.ones(new_shape, dtype=scale_dtype),
    sharding=new_logical_axes,
    quant_axis=axis,
  )


def simple_quantize(
  x: jax.Array, axis: int | tuple[int, ...], scale_dtype=jnp.float16
):
  if not isinstance(axis, (list, tuple)):
    axis = (axis,)
  axis = tuple(z % x.ndim for z in axis)
  amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
  quant = jnp.round(x / scale).astype(jnp.int8)
  scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
  return quant, scale


class MLPLayer(nnx.Module):
  def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
    self.cfg = cfg
    _init = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    dtype = cfg.weight_dtype_at_rest
    self.w_gate = ShardedParam(
      _init(rngs.params(), (cfg.embed, cfg.ffw_size), dtype),
      sharding=('mlp_up_embed', 'mlp_up_ffw'),
    )
    self.w_up = ShardedParam(
      _init(rngs.params(), (cfg.embed, cfg.ffw_size), dtype),
      sharding=('mlp_up_embed', 'mlp_up_ffw'),
    )
    self.w_down = ShardedParam(
      _init(rngs.params(), (cfg.ffw_size, cfg.embed), dtype),
      sharding=('mlp_down_ffw', 'mlp_down_embed'),
    )
    self.w_gate_scale: ShardedParam | None = None
    self.w_up_scale: ShardedParam | None = None
    self.w_down_scale: ShardedParam | None = None
    if cfg.quantized:
      scale_dtype = cfg.quant_scale_dtype
      self.w_gate_scale = simple_quantize_info(self.w_gate, 0, scale_dtype)
      self.w_up_scale = simple_quantize_info(self.w_up, 0, scale_dtype)
      self.w_down_scale = simple_quantize_info(self.w_down, 0, scale_dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    cfg = self.cfg
    dtype = cfg.active_weight_dtype
    if cfg.quantized:
      assert self.w_gate_scale is not None
      assert self.w_up_scale is not None
      assert self.w_down_scale is not None
      with jax.named_scope('gate'):
        ff_gate = jax.nn.silu(
          jnp.einsum('btd,df->btf', x, self.w_gate) * self.w_gate_scale
        ).astype(dtype)
      with jax.named_scope('up_proj'):
        ff_up = (
          jnp.einsum('btd,df->btf', x, self.w_up) * self.w_up_scale
        ).astype(dtype)
      with jax.named_scope('down_proj'):
        ff_out = (
          jnp.einsum('btf,fd->btd', ff_gate * ff_up, self.w_down)
          * self.w_down_scale
        ).astype(dtype)
    else:
      with jax.named_scope('gate'):
        ff_gate = jax.nn.silu(
          jnp.einsum('btd,df->btf', x, self.w_gate.value)
        ).astype(dtype)
      with jax.named_scope('up_proj'):
        ff_up = jnp.einsum('btd,df->btf', x, self.w_up.value).astype(dtype)
      with jax.named_scope('down_proj'):
        ff_out = jnp.einsum(
          'btf,fd->btd', ff_gate * ff_up, self.w_down.value
        ).astype(dtype)
    if cfg.mesh is not None and cfg.rules is not None:
      ff_out = jax.lax.with_sharding_constraint(
        ff_out,
        _logical2sharding(
          ('batch', 'sequence', 'act_embed'), cfg.mesh, cfg.rules
        ),
      )
    return ff_out


QuantArray = tp.Any


class MoELayer(nnx.Module):
  def __init__(self, cfg: Config, rngs: nnx.Rngs):
    self.cfg = cfg
    _einit = jax.nn.initializers.he_normal(in_axis=0, out_axis=(1, 2))
    _sinit = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    dtype = cfg.weight_dtype_at_rest
    self.w_router = ShardedParam(
      _sinit(
        rngs.params(), (cfg.embed, cfg.n_routed_experts), cfg.moe_gate_dtype
      ),
      ('moe_e_up_embed', None),
    )
    self.b_router = ShardedParam(
      jnp.zeros((cfg.n_routed_experts,), cfg.moe_gate_dtype),
      sharding=(None,),
    )

    self.we_gate: ShardedParam = ShardedParam(
      _einit(
        rngs.params(),
        (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size),
        dtype,
      ),
      sharding=('moe_e_experts', 'moe_e_up_embed', 'moe_e_up_ffw'),
    )
    self.we_up = ShardedParam(
      _einit(
        rngs.params(),
        (cfg.n_routed_experts, cfg.embed, cfg.moe_ffw_size),
        dtype,
      ),
      sharding=('moe_e_experts', 'moe_e_up_embed', 'moe_e_up_ffw'),
    )
    self.we_down = ShardedParam(
      _einit(
        rngs.params(),
        (cfg.n_routed_experts, cfg.moe_ffw_size, cfg.embed),
        dtype,
      ),
      sharding=('moe_e_experts', 'moe_e_down_ffw', 'moe_e_down_embed'),
    )
    self.shared_expert = MLPLayer(cfg, rngs=rngs)

    self.we_gate_scale: ShardedParam | None = None
    self.we_up_scale: ShardedParam | None = None
    self.we_down_scale: ShardedParam | None = None
    if cfg.quantized:
      scale_dtype = cfg.quant_scale_dtype
      self.we_gate_scale = simple_quantize_info(self.we_gate, 1, scale_dtype)
      self.we_up_scale = simple_quantize_info(self.we_up, 1, scale_dtype)
      self.we_down_scale = simple_quantize_info(self.we_down, 1, scale_dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    cfg = self.cfg
    assert x.ndim == 3
    topk_weights, topk_idx = _route_tokens_to_moe_experts(
      x, self.w_router.value, self.b_router.value, cfg
    )
    in_specs, out_specs, is_embedding_sharded, tensor_axname = self.get_specs()

    @partial(
      shard_map,
      mesh=cfg.mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_rep=False,
    )
    def _expert_fn(
      x: jax.Array,
      we_gate: jax.Array,
      we_gate_scale: jax.Array | None,
      we_up: jax.Array,
      we_up_scale: jax.Array | None,
      we_down: jax.Array,
      we_down_scale: jax.Array | None,
      topk_weights: jax.Array,
      topk_idx: jax.Array,
    ):
      # this might need to be a
      (b, s, d), e = x.shape, cfg.num_experts_per_tok
      topk_idx_ = topk_idx.reshape(-1)
      sort_idx_ = jnp.argsort(topk_idx_, axis=-1)  # [b * s * e]
      topk_idx_sort_ = topk_idx_[sort_idx_]  # [b * s * e]

      # equivalent to:
      # ```
      # x_repeat_ = jnp.repeat(x.reshape((-1, x.shape[-1])), e, axis=0)
      # x_repeat_sort_ = jnp.take_along_axis(x_repeat_, sort_idx_[:, None], axis=-2)  # [b * s, d]
      # ```
      x_repeat_sort_ = jnp.take_along_axis(
        x.reshape((-1, x.shape[-1])),
        sort_idx_[:, None] // e,
        axis=-2,  # index trick to avoid jnp.krepeat
      )  # [b * s * e, d]

      group_sizes = jnp.bincount(topk_idx_sort_, length=cfg.n_routed_experts)
      with jax.named_scope('we_gate'):
        ff_gate = jax.nn.silu(
          _moe_gmm(x_repeat_sort_, we_gate, we_gate_scale, group_sizes, cfg)
        )
      with jax.named_scope('we_up'):
        ff_up = _moe_gmm(x_repeat_sort_, we_up, we_up_scale, group_sizes, cfg)
      with jax.named_scope('we_down'):
        if ff_gate.shape[-1] < 128:
          ff_out = _moe_gmm(
            ff_gate * ff_up,
            we_down,
            we_down_scale,
            group_sizes,
            dataclasses.replace(cfg, use_megablox=False),
          )
        else:
          ff_out = _moe_gmm(
            ff_gate * ff_up, we_down, we_down_scale, group_sizes, cfg
          )

      def _collective(ff_out):
        # psum here in shard_map
        if is_embedding_sharded:  # activations are replicated
          return jax.lax.psum_scatter(
            ff_out, tensor_axname, scatter_dimension=-1, tiled=True
          )
        else:
          return jax.lax.psum(ff_out, tensor_axname)

      with jax.named_scope('moe_unpermute_and_collective'):
        if cfg.psum_before_expert_reduce:
          ff_out = _collective(ff_out)

        # unpermute tokens
        ff_out = jnp.take_along_axis(
          ff_out, jnp.argsort(sort_idx_)[..., None], axis=-2
        )

        # last embed (d) dimension is sharded now
        ff_out_expert = jnp.einsum(
          'bsed,bse->bsd', ff_out.reshape((b, s, e, -1)), topk_weights
        )

        if not cfg.psum_before_expert_reduce:
          ff_out_expert = _collective(ff_out_expert)

      return ff_out_expert

    with jax.named_scope('moe_routed_expert'):
      ff_out_expert = _expert_fn(
        x,
        self.we_gate.value,
        self.we_gate_scale.value if self.we_gate_scale is not None else None,
        self.we_up.value,
        self.we_up_scale.value if self.we_up_scale is not None else None,
        self.we_down.value,
        self.we_down_scale.value if self.we_down_scale is not None else None,
        topk_weights,
        topk_idx,
      )
    with jax.named_scope('moe_shared_expert'):
      ff_out_shared = self.shared_expert(x)

    return ff_out_expert + ff_out_shared

  def get_specs(self) -> tuple[tp.Any, tp.Any, bool, str]:
    cfg = self.cfg
    assert cfg.rules is not None

    x_sharding = _logical2physical(P('batch', 'sequence', None), cfg.rules)
    topk_weights_spec = _logical2physical(
      P('batch', 'sequence', None), cfg.rules
    )
    topk_idx_spec = _logical2physical(P('batch', 'sequence', None), cfg.rules)

    tensor_axname = _logical2physical(P('moe_e_tp'), cfg.rules)[0]

    if cfg.mesh is None:
      return P(), P(), False, tensor_axname

    we_gate_spec = P(None, None, tensor_axname)
    self.we_gate.value = jax.lax.with_sharding_constraint(
      self.we_gate.value, NamedSharding(cfg.mesh, we_gate_spec)
    )
    if self.we_gate_scale is not None:
      we_gate_scale_spec = P(None, tensor_axname)
      self.we_gate_scale.value = jax.lax.with_sharding_constraint(
        self.we_gate_scale.value, NamedSharding(cfg.mesh, we_gate_scale_spec)
      )
    else:
      we_gate_scale_spec = P()

    we_up_spec = P(None, None, tensor_axname)
    self.we_up.value = jax.lax.with_sharding_constraint(
      self.we_up.value, NamedSharding(cfg.mesh, we_up_spec)
    )
    if self.we_up_scale is not None:
      we_up_scale_spec = P(None, tensor_axname)
      self.we_up_scale.value = jax.lax.with_sharding_constraint(
        self.we_up_scale.value, NamedSharding(cfg.mesh, we_up_scale_spec)
      )
    else:
      we_up_scale_spec = P()

    we_down_spec = P(None, tensor_axname, None)
    self.we_down.value = jax.lax.with_sharding_constraint(
      self.we_down.value, NamedSharding(cfg.mesh, we_down_spec)
    )
    if self.we_down_scale is not None:
      we_down_scale_spec = P(None, None)
      self.we_down_scale.value = jax.lax.with_sharding_constraint(
        self.we_down_scale.value, NamedSharding(cfg.mesh, we_down_scale_spec)
      )
    else:
      we_down_scale_spec = P()

    in_specs = (
      x_sharding,
      we_gate_spec,
      we_gate_scale_spec,
      we_up_spec,
      we_up_scale_spec,
      we_down_spec,
      we_down_scale_spec,
      topk_weights_spec,
      topk_idx_spec,
    )

    out_specs = _logical2physical(P('batch', 'sequence', None), cfg.rules)
    is_embedding_sharded = not (
      _logical2physical(P('act_embed'), cfg.rules)[0] is None
    )
    if is_embedding_sharded:  # activations are sharded
      out_specs = P(
        *(out_specs[:-1] + (tensor_axname,))
      )  # override last axis name

    return in_specs, out_specs, is_embedding_sharded, tensor_axname


@jax_struct
class AttentionLayer(_Init):
  q_a: jax.Array | TensorInfo | QuantArray
  q_gamma: jax.Array | TensorInfo | QuantArray
  q_b: jax.Array | TensorInfo | QuantArray
  kv_a: jax.Array | TensorInfo | QuantArray
  k_pe: jax.Array | TensorInfo | QuantArray
  kv_gamma: jax.Array | TensorInfo | QuantArray
  k_b: jax.Array | TensorInfo | QuantArray
  v_b: jax.Array | TensorInfo | QuantArray
  o: jax.Array | TensorInfo | QuantArray

  @classmethod
  def abstract(cls, cfg: Config, quantized: bool = False):
    _init = lambda *out_ax: jax.nn.initializers.he_normal(
      in_axis=0, out_axis=out_ax
    )
    dtype = cfg.weight_dtype_at_rest
    _ones_init = jax.nn.initializers.constant(1.0)
    q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    SD = jax.ShapeDtypeStruct
    layer = AttentionLayer(
      q_a=TensorInfo(
        SD((cfg.embed, cfg.q_lora_rank), dtype),
        ('qkv_embed', 'q_lora'),
        _init(1),
      ),
      q_gamma=TensorInfo(
        SD((cfg.q_lora_rank,), dtype), ('q_lora',), _ones_init
      ),
      q_b=TensorInfo(
        SD((cfg.q_lora_rank, cfg.num_heads, q_head_dim), dtype),
        ('q_lora', 'qkv_heads', 'head_dim'),
        _init(1, 2),
      ),
      kv_a=TensorInfo(
        SD((cfg.embed, cfg.kv_lora_rank), dtype),
        ('qkv_embed', 'kv_lora'),
        _init(1),
      ),
      k_pe=TensorInfo(
        SD((cfg.embed, cfg.qk_rope_head_dim), dtype),
        ('qkv_embed', 'head_dim'),
        _init(1),
      ),
      kv_gamma=TensorInfo(
        SD((cfg.kv_lora_rank,), dtype), ('kv_lora',), _ones_init
      ),
      k_b=TensorInfo(
        SD((cfg.kv_lora_rank, cfg.num_heads, cfg.qk_nope_head_dim), dtype),
        ('kv_lora', 'qkv_heads', 'head_dim'),
        _init(1, 2),
      ),
      v_b=TensorInfo(
        SD((cfg.kv_lora_rank, cfg.num_heads, cfg.v_head_dim), dtype),
        ('kv_lora', 'qkv_heads', 'head_dim'),
        _init(1, 2),
      ),
      o=TensorInfo(
        SD((cfg.num_heads, cfg.v_head_dim, cfg.embed), dtype),
        ('o_heads', 'head_dim', 'o_embed'),
        _init(1, 2),
      ),
    )
    if quantized:
      scale_dtype = cfg.quant_scale_dtype
      layer.q_a = QuantArray(*simple_quantize_info(layer.q_a, 1, scale_dtype))
      layer.q_b = QuantArray(
        *simple_quantize_info(layer.q_b, (1, 2), scale_dtype)
      )
      layer.kv_a = QuantArray(*simple_quantize_info(layer.kv_a, 1, scale_dtype))
      layer.k_pe = QuantArray(*simple_quantize_info(layer.k_pe, 1, scale_dtype))
      layer.k_b = QuantArray(
        *simple_quantize_info(layer.k_b, (1, 2), scale_dtype)
      )
      layer.v_b = QuantArray(
        *simple_quantize_info(layer.v_b, (1, 2), scale_dtype)
      )
      layer.o = QuantArray(*simple_quantize_info(layer.o, (0, 1), scale_dtype))
    return layer

  @staticmethod
  def quantize(layer: 'AttentionLayer', cfg: Config):
    scale_dtype = cfg.quant_scale_dtype
    return dataclasses.replace(
      layer,
      q_a=QuantArray(*simple_quantize(layer.q_a, 1, scale_dtype)),
      q_b=QuantArray(*simple_quantize(layer.q_b, (1, 2), scale_dtype)),
      kv_a=QuantArray(*simple_quantize(layer.kv_a, 1, scale_dtype)),
      k_pe=QuantArray(*simple_quantize(layer.k_pe, 1, scale_dtype)),
      k_b=QuantArray(*simple_quantize(layer.k_b, (1, 2), scale_dtype)),
      v_b=QuantArray(*simple_quantize(layer.v_b, (1, 2), scale_dtype)),
      o=QuantArray(*simple_quantize(layer.o, (0, 1), scale_dtype)),
    )


@jax_struct
class Layer(_Init):
  mlp: MLPLayer | MoELayer
  attn: AttentionLayer
  gamma_pre_attn: jax.Array | TensorInfo
  gamma_post_attn: jax.Array | TensorInfo

  @classmethod
  def abstract(
    cls, cfg: Config, use_moe: bool = True, quantized: bool = False
  ) -> 'Layer':
    SD = jax.ShapeDtypeStruct
    _init = jax.nn.initializers.constant(1.0)
    dtype = cfg.active_weight_dtype
    return Layer(
      mlp=MoELayer.abstract(cfg, quantized=quantized)
      if use_moe
      else MLPLayer.abstract(cfg, quantized=quantized),
      attn=AttentionLayer.abstract(cfg, quantized=quantized),
      gamma_pre_attn=TensorInfo(
        SD((cfg.embed,), dtype=dtype), ('act_embed',), _init
      ),
      gamma_post_attn=TensorInfo(
        SD((cfg.embed,), dtype=dtype), ('act_embed',), _init
      ),
    )

  @staticmethod
  def quantize(layer: 'Layer', cfg: Config):
    return dataclasses.replace(
      layer,
      mlp=layer.mlp.quantize(layer.mlp, cfg),
      attn=layer.attn.quantize(layer.attn, cfg),
    )


@jax_struct
class Weights(_Init):
  layers: list[Layer]
  embedding: jax.Array | TensorInfo
  gamma_final: jax.Array | TensorInfo
  lm_head: jax.Array | TensorInfo

  @classmethod
  def abstract(cls, cfg: Config, quantized: bool = False):
    layers = [
      Layer.abstract(cfg, use_moe=i >= cfg.first_k_dense, quantized=quantized)
      for i in range(cfg.num_layers)
    ]
    return Weights(
      layers=layers,
      embedding=TensorInfo(
        jax.ShapeDtypeStruct(
          (cfg.vocab_size, cfg.embed), cfg.weight_dtype_at_rest
        ),
        ('vocab_in', 'vocab_in'),  # this is fully replicated for performance
        jax.nn.initializers.he_normal(in_axis=0, out_axis=1),
      ),
      gamma_final=TensorInfo(
        jax.ShapeDtypeStruct((cfg.embed,), cfg.weight_dtype_at_rest),
        ('act_embed',),
        jax.nn.initializers.constant(1.0),
      ),
      lm_head=TensorInfo(
        jax.ShapeDtypeStruct(
          (cfg.embed, cfg.vocab_size), cfg.weight_dtype_at_rest
        ),
        ('vocab_in', 'vocab_out'),
        jax.nn.initializers.he_normal(in_axis=1, out_axis=0),
      ),
    )

  @staticmethod
  def quantize(weights: 'Weights', cfg: Config):
    return dataclasses.replace(
      weights, layers=[layer.quantize(layer, cfg) for layer in weights.layers]
    )


@jax_struct
class KVCache(_Init):
  kv_compressed: list[jax.Array]  # [batch_size, max_seq_len, kv_lora]
  k_pe: list[jax.Array]  # [batch_size, max_seq_len, qk_rope_head_dim]
  length: (
    jax.Array
  )  # []  # sequences are right-aligned for slice udpate performance
  starts: (
    jax.Array
  )  # [batch_size]  # sequences are right-aligned, we need start indices

  @classmethod
  def abstract(
    cls,
    cfg: Config,
    batch_size: int,
    max_seq_len: int,
    dtype: int = jnp.bfloat16,
  ):
    _init = jax.nn.initializers.zeros
    kv_compressed_info = TensorInfo(
      jax.ShapeDtypeStruct((batch_size, max_seq_len, cfg.kv_lora_rank), dtype),
      ('batch', 'sequence', 'kv_lora_cache'),
      _init,
    )
    k_pe_info = TensorInfo(
      jax.ShapeDtypeStruct(
        (batch_size, max_seq_len, cfg.qk_rope_head_dim), dtype
      ),
      ('batch', 'sequence', 'head_dim'),
      _init,
    )
    cache = KVCache(
      kv_compressed=[kv_compressed_info for _ in range(cfg.num_layers)],
      k_pe=[k_pe_info for _ in range(cfg.num_layers)],
      length=TensorInfo(jax.ShapeDtypeStruct((), jnp.int32), (), _init),
      starts=TensorInfo(
        jax.ShapeDtypeStruct((batch_size,), jnp.int32), ('batch',), _init
      ),
    )
    return cache

  @property
  def time_axis(self) -> int:
    return 1


def segment_ids_to_positions(segment_ids):
  """Counts positions for segment ids."""

  def scan_fun(a, b):
    return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

  vals = (jnp.zeros_like(segment_ids), segment_ids)
  return jnp.array(
    jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype='int32'
  )


def _yarn_find_correction_dim(
  num_rotations, dim, base=10000, max_position_embeddings=2048
):
  return (
    dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
  ) / (2 * math.log(base))


def _yarn_find_correction_range(
  low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
  low = math.floor(
    _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
  )
  high = math.ceil(
    _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
  )
  return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
  if min == max:
    max += 0.001  # Prevent singularity

  linear_func = (jnp.arange(dim) - min) / (max - min)
  ramp_func = jnp.clip(linear_func, 0, 1)
  return ramp_func


def generate_pos_embeddings(positions, head_dim, cfg: Config):
  fractions = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
  freq_extra = 1.0 / (cfg.rope_theta**fractions)
  freq_inter = 1.0 / (cfg.rope_scaling_factor * cfg.rope_theta**fractions)

  low, high = _yarn_find_correction_range(
    cfg.rope_beta_fast,
    cfg.rope_beta_slow,
    head_dim,
    cfg.rope_theta,
    cfg.rope_original_max_position_embeddings,
  )
  inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, head_dim // 2).astype(
    jnp.float32
  )
  inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
  freqs = jnp.einsum(
    '...T,k->...Tk', positions, inv_freq, precision=jax.lax.Precision.HIGHEST
  )
  _yarn_get_mscale = lambda scale, mscale: jnp.where(
    scale <= 1, 1.0, 0.1 * mscale * jnp.log(scale) + 1.0
  )
  _mscale = _yarn_get_mscale(
    cfg.rope_scaling_factor, cfg.rope_mscale
  ) / _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
  sin, cos = jnp.sin(freqs) * _mscale, jnp.cos(freqs) * _mscale
  return sin, cos


def apply_rotary_embedding(x, sin, cos):
  assert x.ndim == 4
  assert sin.ndim == 3 and cos.ndim == 3
  sin, cos = (
    sin[:, None, :, :],
    cos[:, None, :, :],
  )  # [B, T, head_dim] -> [B, h, T, head_dim]

  # x1, x2 = jnp.split(x, 2, axis=-1)
  x1, x2 = x[..., ::2], x[..., 1::2]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_attention_mask(
  q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal: bool
):
  # [B, t, T]
  segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]
  # [B, t, T] -> [B, 1, t, T]
  segment_mask = segment_mask[:, None, :, :]

  if causal:
    # [b, h, t, T]
    qk = (1, 1, q_len, k_len)
    q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
    k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
    q_positions = q_iota + q_offset[:, None, None, None]
    causal_mask = q_positions >= k_iota
    combined_mask = jnp.logical_and(segment_mask, causal_mask)
    return combined_mask
  else:
    return segment_mask


def _get_attn_scale(q_head_dim: int, cfg: Config):
  scale = q_head_dim**-0.5
  if cfg.rope_scaling_factor <= 1.0:
    _yarn_mscale = 1.0
  else:
    _yarn_mscale = (
      0.1 * cfg.rope_mscale_all_dim * math.log(cfg.rope_scaling_factor) + 1.0
    )
  return scale * _yarn_mscale**2


def attention(
  q: jax.Array,
  k: jax.Array | tuple[jax.Array, jax.Array],
  v: jax.Array | tuple[jax.Array, jax.Array],
  q_segment_ids: jax.Array,
  k_segment_ids: jax.Array,
  q_offset: jax.Array,
  cfg: Config,
) -> jax.Array:
  """
  Compute attention.

  Args:
  q: Query tensor of shape (batch_size, num_heads, q_len, head_dim)
  k: Key tensor of shape (batch_size, num_heads, k_len, head_dim)
  v: Value tensor of shape (batch_size, num_heads, k_len, head_dim)
  q_segment_ids: Query segment IDs of shape (batch_size, q_len)
  k_segment_ids: Key segment IDs of shape (batch_size, k_len)
  q_offset: Query offset of shape (batch_size,)
  cfg: Configuration object

  Returns:
  Attention output of shape (batch_size, num_heads, q_len, head_dim)
  """
  scale = _get_attn_scale(q.shape[-1], cfg)

  # grouped-query attention
  b, kg, t, d = q.shape
  _, kh, T, _ = (
    k.shape if not _isinstance(k, QuantArray) else k.quant.shape
  )  # quantized
  q_ = q.reshape((b, kh, kg // kh, t, d))

  if _isinstance(k, QuantArray):  # quantized k
    qk = (
      jnp.einsum('bhgtd,bhTd->bhgtT', q_, k.quant)
      * k.scale[..., None, None, :]
      * scale
    )
  else:
    qk = jnp.einsum('bhgtd,bhTd->bhgtT', q_, k) * scale
  qk = qk.reshape((b, kg, t, T))

  mask = make_attention_mask(
    t, T, q_segment_ids, k_segment_ids, q_offset, cfg.causal
  )
  # Apply the combined mask
  qk = jnp.where(mask, qk, -1e30)
  # Jax softmax impl includes max subtraction for numerical stability, no need to do it outside.
  attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)

  # grouped-query attention
  attn_ = attn.reshape((b, kh, kg // kh, t, T))
  if _isinstance(v, QuantArray):  # quantized v
    qkv = jnp.einsum(
      'bhgtT,bhTd->bhgtd', attn_ * v.scale[..., None, None, :], v.quant
    ).astype(cfg.active_weight_dtype)
  else:
    qkv = jnp.einsum('bhgtT,bhTd->bhgtd', attn_, v).astype(
      cfg.active_weight_dtype
    )
  return qkv.reshape((b, kg, t, v.shape[-1]))


def attention_kernel(
  q, k, v, q_segment_ids, kv_segment_ids, q_offset, starts, lengths, cfg: Config
):
  """Flash attention kernel!"""

  # On TPUv3, pallas seems to only work with float32.
  # q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)

  k, k_scale = (k.quant, k.scale) if _isinstance(k, QuantArray) else (k, None)
  v, v_scale = (v.quant, v.scale) if _isinstance(v, QuantArray) else (v, None)

  # handle grouped query attention
  assert q.shape[-3] % k.shape[-3] == 0
  kv_repeats = q.shape[-3] // k.shape[-3]
  scale = _get_attn_scale(q.shape[-1], cfg)

  # shard_map
  in_specs = (
    _logical2physical(
      P('batch', 'query_heads', 'sequence', 'head_dim'), cfg.rules
    ),
    _logical2physical(
      P('batch', 'key_heads', 'sequence', 'head_dim'), cfg.rules
    ),
    _logical2physical(
      P('batch', 'key_heads', 'sequence', 'head_dim'), cfg.rules
    ),
    _logical2physical(P('batch', 'sequence'), cfg.rules),
    _logical2physical(P('batch', 'sequence'), cfg.rules),
    _logical2physical(P('batch'), cfg.rules) if starts is not None else None,
    _logical2physical(P('batch'), cfg.rules) if lengths is not None else None,
  )
  in_specs += (
    None
    if k_scale is None
    else _logical2physical(P('batch', 'key_heads', 'sequence'), cfg.rules),
  )
  in_specs += (
    None
    if v_scale is None
    else _logical2physical(P('batch', 'key_heads', 'sequence'), cfg.rules),
  )
  out_specs = _logical2physical(
    P('batch', 'query_heads', 'sequence', 'head_dim'), cfg.rules
  )

  @partial(
    shard_map,
    mesh=cfg.mesh,
    in_specs=in_specs,
    out_specs=out_specs,
    check_rep=False,
  )
  def _f(
    q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale
  ):
    q_org_shape = q.shape
    kv_repeats = q.shape[-3] // k.shape[-3]
    q = q.reshape(
      q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])
    )

    if q.shape[-2] != 1:
      mask = mask_lib.MultiHeadMask(
        [
          mask_lib.CausalMask((q.shape[-2], k.shape[-2]))
          for _ in range(q.shape[-3])
        ]
      )
      # block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 2048)
      block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
      block_sizes = splash.BlockSizes(
        block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv
      )
      attn_fn = splash.make_splash_mqa_single_device(
        mask=mask, block_sizes=block_sizes
      )
      attn_fn = jax.vmap(
        jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)
      )

      segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
      if k_scale is not None:
        k = (k * k_scale[..., None]).astype(jnp.bfloat16)
      if v_scale is not None:
        v = (v * v_scale[..., None]).astype(jnp.bfloat16)
      ret = attn_fn(q * scale, k, v, segment_ids)
    else:
      assert q.shape[-2] == 1, 'This is a decode kernel, q.shape[-2] must be 1'
      q = q[..., 0, :]
      in_axes = (1, 1, 1, None, None)
      in_axes += ((None if k_scale is None else 1),)
      in_axes += ((None if v_scale is None else 1),)
      hyperparams = dict(scale=scale, block_kv=min(k.shape[-2], 8192))
      ret = jax.vmap(
        partial(ragged_attention.ragged_decode_fwd, **hyperparams),
        in_axes=in_axes,
        out_axes=1,
      )(q, k, v, starts, lengths, k_scale, v_scale)
    return ret.reshape(q_org_shape)

  lengths = jnp.broadcast_to(lengths, starts.shape)
  return _f(
    q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale
  ).astype(jnp.bfloat16)


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
  """Apply RMS normalization."""
  rms = jnp.sqrt(
    jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
  )
  return jnp.astype(gamma * x / rms, jnp.bfloat16)


# def attention_block(
#    x: jax.Array,
#    segment_ids: jax.Array,
#    layer: Layer,
#    sin: jax.Array,
#    cos: jax.Array,
#    cfg: Config,
#    cache: KVCache | None = None,
#    idx: int | None = None,
# ):
#    x = x.astype(cfg.active_weight_dtype)
#    # Multi-head attention
#    with jax.named_scope("qkv_matmul"):
#        if all(_isinstance(z, QuantArray) for z in [layer.q, layer.k, layer.v]):
#            q = jnp.einsum("btd,dhq->bhtq", x * layer.q.scale, layer.q.quant).astype(cfg.active_weight_dtype)
#            k = jnp.einsum("btd,dhq->bhtq", x * layer.k.scale, layer.k.quant).astype(cfg.active_weight_dtype)
#            v = jnp.einsum("btd,dhq->bhtq", x * layer.v.scale, layer.v.quant).astype(cfg.active_weight_dtype)
#        else:
#            q = jnp.einsum("btd,dhq->bhtq", x, layer.q).astype(cfg.active_weight_dtype)
#            k = jnp.einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.active_weight_dtype)
#            v = jnp.einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.active_weight_dtype)
#
#    # Apply rotary embeddings
#    with jax.named_scope("rope"):
#        q = apply_rotary_embedding(q, sin, cos)
#        k = apply_rotary_embedding(k, sin, cos)
#
#    with jax.named_scope("cache_update"):
#        if cache is not None:
#            if _isinstance(cache.k[idx], QuantArray) and _isinstance(cache.v[idx], QuantArray):
#                k_quant, k_scale = simple_quantize(k, axis=-1, scale_dtype=cfg.quant_scale_dtype)
#                v_quant, v_scale = simple_quantize(v, axis=-1, scale_dtype=cfg.quant_scale_dtype)
#                k_quant = jax.lax.dynamic_update_slice_in_dim(
#                    cache.k[idx].quant, k_quant, cache.length, axis=cache.time_axis
#                )
#                v_quant = jax.lax.dynamic_update_slice_in_dim(
#                    cache.v[idx].quant, v_quant, cache.length, axis=cache.time_axis
#                )
#                k_scale = jax.lax.dynamic_update_slice_in_dim(
#                    cache.k[idx].scale, k_scale, cache.length, axis=cache.time_axis
#                )
#                v_scale = jax.lax.dynamic_update_slice_in_dim(
#                    cache.v[idx].scale, v_scale, cache.length, axis=cache.time_axis
#                )
#                k, v = QuantArray(k_quant, k_scale), QuantArray(v_quant, v_scale)
#                time_indices = jnp.arange(0, v_quant.shape[cache.time_axis])[None, :]  # [1, T]
#            else:
#                k = jax.lax.dynamic_update_slice_in_dim(
#                    cache.k[idx], k.astype(cache.k[idx].dtype), cache.length, axis=cache.time_axis
#                )
#                v = jax.lax.dynamic_update_slice_in_dim(
#                    cache.v[idx], v.astype(cache.v[idx].dtype), cache.length, axis=cache.time_axis
#                )
#                time_indices = jnp.arange(0, v.shape[cache.time_axis])[None, :]  # [1, T]
#
#            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
#            incremental_position = jnp.max(jnp.sum(segment_ids != 0, axis=-1))
#            # i.e. valid below where we've written things [B, T]
#            k_segment_ids = (
#                (time_indices >= cache.starts[:, None]) & (time_indices < (cache.length + incremental_position))
#            ).astype(jnp.int32)
#
#            # Mask our new k and v so that its very visible and easy to test kv values being entered. Tiny perf hit b/c it is unnecessary.
#            # k, v = k * k_segment_ids[:, None, :, None], v * k_segment_ids[:, None, :, None]
#
#            q_offset = cache.length[None]
#            starts, lengths = cache.starts, (cache.length + incremental_position)[None]
#        else:
#            q_segment_ids = segment_ids
#            k_segment_ids = segment_ids
#            q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
#            starts = jnp.sum(jnp.cumsum(k_segment_ids != 0, axis=-1) == 0, axis=-1)
#            lengths = k_segment_ids.shape[-1] - jnp.sum(jnp.cumsum(jnp.flip(k_segment_ids, -1) != 0, axis=-1) == 0, -1)
#
#    # Compute attention
#    with jax.named_scope("attention"):
#        if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (cfg.use_decode_attn_kernel and q.shape[-2] == 1):
#            attn_out = attention_kernel(
#                q, k, v, q_segment_ids, k_segment_ids, q_offset, starts=starts, lengths=lengths, cfg=cfg
#            )
#        else:
#            attn_out = attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg)
#
#    # Project attention output
#    with jax.named_scope("projection"):
#        if _isinstance(layer.o, QuantArray):
#            attn_out = (jnp.einsum("bhtq,hqd->btd", attn_out, layer.o.quant) * layer.o.scale).astype(
#                cfg.active_weight_dtype
#            )
#        else:
#            attn_out = jnp.einsum("bhtq,hqd->btd", attn_out, layer.o).astype(cfg.active_weight_dtype)
#
#    attn_out = jax.lax.with_sharding_constraint(
#        attn_out, _logical2sharding(("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules)
#    )
#    return attn_out, k, v


def _count_length_from_left(segment_ids):
  """Count the length of a sequence as len(seq) - len(seq's right padding)."""
  return jnp.sum(
    jnp.cumsum(jnp.flip(segment_ids != 0, axis=-1), axis=-1) > 0, axis=-1
  )


def mla_attention_block(
  x: jax.Array,
  segment_ids: jax.Array,
  attn_layer: AttentionLayer,
  sin: jax.Array,
  cos: jax.Array,
  cfg: Config,
  cache: KVCache | None = None,
  idx: int = 0,
) -> jax.Array:
  with jax.named_scope('qkv_matmul'):
    with jax.named_scope('q_proj'):
      if _isinstance(attn_layer.q_a, QuantArray):
        q_lora = jnp.einsum(
          'btd,dr->btr', x * attn_layer.q_a.scale, attn_layer.q_a.quant
        )
      else:
        q_lora = jnp.einsum('btd,dr->btr', x, attn_layer.q_a)
      q_lora = rms_norm(q_lora, attn_layer.q_gamma)
      if _isinstance(attn_layer.q_b, QuantArray):
        q = jnp.einsum(
          'btr,rhq->bhtq', q_lora * attn_layer.q_b.scale, attn_layer.q_b.quant
        )
      else:
        q = jnp.einsum('btr,rhq->bhtq', q_lora, attn_layer.q_b)
      q_embed = jnp.concatenate(
        [
          q[..., : cfg.qk_nope_head_dim],
          apply_rotary_embedding(q[..., cfg.qk_nope_head_dim :], sin, cos),
        ],
        -1,
      )

    with jax.named_scope('kv_compressed_proj'):
      if _isinstance(attn_layer.kv_a, QuantArray):
        kv_compressed = jnp.einsum(
          'btd,dr->btr', x * attn_layer.kv_a.scale, attn_layer.kv_a.quant
        )
      else:
        kv_compressed = jnp.einsum('btd,dr->btr', x, attn_layer.kv_a)
      kv_compressed = rms_norm(kv_compressed, attn_layer.kv_gamma)
    with jax.named_scope('kv_pe_proj'):
      if _isinstance(attn_layer.k_pe, QuantArray):
        k_pe = jnp.einsum(
          'btd,dq->btq', x * attn_layer.k_pe.scale, attn_layer.k_pe.quant
        )
      else:
        k_pe = jnp.einsum('btd,dq->btq', x, attn_layer.k_pe)
      k_pe = apply_rotary_embedding(k_pe[..., None, :, :], sin, cos)[
        ..., 0, :, :
      ]

  with jax.named_scope('cache_update'):
    if cache is not None:
      kv_compressed = kv_compressed.astype(cache.kv_compressed[idx].dtype)
      k_pe = k_pe.astype(cache.k_pe[idx].dtype)
      kv_compressed = jax.lax.dynamic_update_slice_in_dim(
        cache.kv_compressed[idx],
        kv_compressed,
        cache.length,
        axis=cache.time_axis,
      )
      k_pe = jax.lax.dynamic_update_slice_in_dim(
        cache.k_pe[idx], k_pe, cache.length, axis=cache.time_axis
      )

      time_indices = jnp.arange(0, k_pe.shape[cache.time_axis])[
        None, :
      ]  # [1, T]
      q_segment_ids = jnp.where(segment_ids != 0, 1, 0)

      # TODO check this
      # incremental_position = segment_ids.shape[-1]
      incremental_position = jnp.max(_count_length_from_left(segment_ids))

      # i.e. valid below where we've written things [B, T]
      k_segment_ids = (
        (time_indices >= cache.starts[:, None])
        & (time_indices < (cache.length + incremental_position))
      ).astype(jnp.int32)

      # Mask our new k and v so that its very visible and easy to test kv values being entered. Tiny perf hit b/c it is unnecessary.
      # k, v = k * k_segment_ids[:, None, :, None], v * k_segment_ids[:, None, :, None]

      q_offset = cache.length[None]
      starts, lengths = (
        cache.starts,
        (cache.length + incremental_position)[None],
      )
    else:
      q_segment_ids, k_segment_ids = segment_ids, segment_ids
      q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
      starts = jnp.sum(jnp.cumsum(k_segment_ids != 0, axis=-1) == 0, axis=-1)
      lengths = k_segment_ids.shape[-1] - jnp.sum(
        jnp.cumsum(jnp.flip(k_segment_ids, -1) != 0, axis=-1) == 0, -1
      )

  with jax.named_scope('qkv_embed'):
    with jax.named_scope('kv_nope_proj'):
      if _isinstance(attn_layer.k_b, QuantArray):
        k_nope = jnp.einsum(
          'btr,rhq->bhtq',
          kv_compressed * attn_layer.k_b.scale,
          attn_layer.k_b.quant,
        )
      else:
        k_nope = jnp.einsum('btr,rhq->bhtq', kv_compressed, attn_layer.k_b)
      if _isinstance(attn_layer.v_b, QuantArray):
        v = jnp.einsum(
          'btr,rhv->bhtv',
          kv_compressed * attn_layer.v_b.scale,
          attn_layer.v_b.quant,
        )
      else:
        v = jnp.einsum('btr,rhv->bhtv', kv_compressed, attn_layer.v_b)
      k_embed = jnp.concatenate(
        [
          k_nope,
          jnp.broadcast_to(
            k_pe[..., None, :, :], k_nope.shape[:-1] + k_pe.shape[-1:]
          ),
        ],
        axis=-1,
      )

    # q_segment_ids = jnp.ones(x.shape[-2], dtype=jnp.int32)[None, :]
    # k_segment_ids = jnp.ones(x.shape[-2], dtype=jnp.int32)[None, :]
    # q_offset = jnp.zeros((), dtype=jnp.int32)[None]

    _l2s = lambda logical_axes: _logical2sharding(
      logical_axes, cfg.mesh, cfg.rules
    )
    q_embed = jax.lax.with_sharding_constraint(
      q_embed, _l2s(('batch', 'act_heads', 'sequence', 'head_dim'))
    )
    k_embed = jax.lax.with_sharding_constraint(
      k_embed, _l2s(('batch', 'act_heads', 'sequence', 'head_dim'))
    )
    v = jax.lax.with_sharding_constraint(
      v, _l2s(('batch', 'act_heads', 'sequence', 'head_dim'))
    )

  # Compute attention
  with jax.named_scope('attention'):
    if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (
      cfg.use_decode_attn_kernel and q.shape[-2] == 1
    ):
      attn_out = attention_kernel(
        q_embed,
        k_embed,
        v,
        q_segment_ids,
        k_segment_ids,
        q_offset,
        starts=starts,
        lengths=lengths,
        cfg=cfg,
      )
    else:
      attn_out = attention(
        q_embed, k_embed, v, q_segment_ids, k_segment_ids, q_offset, cfg
      )

  # attn_out = attention(q_embed, k_embed, v, q_segment_ids, k_segment_ids, q_offset, cfg)

  with jax.named_scope('o_proj'):
    if _isinstance(attn_layer.o, QuantArray):
      attn_out = (
        jnp.einsum('bhtv,hvd->btd', attn_out, attn_layer.o.quant)
        * attn_layer.o.scale
      )
    else:
      attn_out = jnp.einsum('bhtv,hvd->btd', attn_out, attn_layer.o)
  return attn_out, kv_compressed, k_pe


def _route_tokens_to_moe_experts(
  x: jax.Array, weight: jax.Array, bias: jax.Array, cfg: Config
):
  scores = jax.nn.sigmoid(
    jnp.einsum('bsk,kj->bsj', x, weight).astype(cfg.moe_gate_dtype)
  )
  scores_with_bias = scores + bias
  group_scores = jnp.sum(
    jax.lax.top_k(
      scores_with_bias.reshape(scores.shape[:-1] + (cfg.n_group, -1)), 2
    )[0],
    axis=-1,
  )
  group_idx = jax.lax.top_k(group_scores, cfg.topk_group)[1]
  mask = jnp.any(
    jnp.arange(cfg.n_group)[:, None] == group_idx[..., None, :], axis=-1
  )
  mask = jnp.repeat(mask, scores.shape[-1] // mask.shape[-1], -1)
  masked_scores = jnp.where(mask, scores_with_bias, 0.0)
  topk_idx = jax.lax.top_k(masked_scores, cfg.num_experts_per_tok)[1]
  topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1).astype(
    cfg.moe_gate_dtype
  )
  topk_weights = (
    cfg.routed_scaling_factor
    * topk_weights
    / (jnp.sum(topk_weights, axis=-1)[..., None] + 1e-20)
  )
  return topk_weights, topk_idx


def _moe_gmm(
  lhs: jax.Array,
  rhs: jax.Array,
  rhs_scale: jax.Array | None,
  group_sizes,
  cfg: Config,
):
  if rhs_scale is None:
    assert lhs.ndim == 2 and rhs.ndim == 3, (
      f'{lhs.ndim=} != 2 and {rhs.ndim=} != 3'
    )
  else:
    assert lhs.ndim == 2 and rhs.ndim == 3, (
      f'{lhs.ndim=} != 2 and {rhs.ndim=} != 3'
    )
  use_megablox = (
    cfg.use_megablox
    if cfg.use_megablox is not None
    else (which_platform(cfg) == 'tpu')
  )
  if use_megablox:
    from aqt.jax.v2.aqt_tensor import QTensor

    interpret = which_platform(cfg) == 'cpu'

    if _isinstance(rhs, QuantArray):
      rhs_shape, rhs_quantize_dtype = rhs.shape, rhs.dtype
      scale = rhs_scale.astype(
        jnp.float32
      )  # only bfloat16 or float32 scale is supported in megablox
      rhs = QTensor(
        qvalue=rhs,
        scale=[scale[:, None, ...]],
        bias=[],
        scale_t=None,
        dequant_dtype=scale.dtype,
      )
    else:
      rhs_shape, rhs_quantize_dtype = rhs.shape, None

    _tile = lambda d: max(
      [s for s in [16, 32, 64, 128] if d % s == 0], default=d
    )
    tiling = (_tile(lhs.shape[0]), _tile(lhs.shape[-1]), _tile(rhs_shape[-2]))
    with jax.named_scope('megablox'):
      ret = megablox_gmm(
        lhs,
        rhs,
        group_sizes,
        interpret=interpret,
        tiling=tiling,
        rhs_quantize_dtype=rhs_quantize_dtype,
      )
  else:
    # with ragged_dot we need to dequantize rhs before matmul
    if rhs_scale is not None:
      rhs = (rhs * rhs_scale[:, None, ...]).astype(cfg.active_weight_dtype)
    with jax.named_scope('jax.lax.ragged_dot'):
      ret = jax.lax.ragged_dot(lhs, rhs, group_sizes)
  return ret.astype(cfg.active_weight_dtype)


def forward_layer(
  x: jax.Array,
  segment_ids: jax.Array,
  layer: Layer,
  sin: jax.Array,
  cos: jax.Array,
  idx: int,
  cfg: Config,
  cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  x = x.astype(cfg.active_weight_dtype)
  x = jax.lax.with_sharding_constraint(
    x,
    _logical2sharding(('batch', 'sequence', 'act_embed'), cfg.mesh, cfg.rules),
  )

  # Attention block
  with jax.named_scope('attn_pre_norm'):
    attn_in = rms_norm(x, layer.gamma_pre_attn)
  attn_out, kv_compressed, k_pe = mla_attention_block(
    attn_in, segment_ids, layer.attn, sin, cos, cfg, cache, idx
  )
  with jax.named_scope('residual'):
    x = x + attn_out.astype(cfg.active_weight_dtype)

  # FFN block
  with jax.named_scope('attn_post_norm'):
    ff_in = rms_norm(x, layer.gamma_post_attn)
  with jax.named_scope('ffn'):
    ff_out = (mlp_block if _isinstance(layer.mlp, MLPLayer) else moe_block)(
      ff_in, layer.mlp, cfg
    )
  with jax.named_scope('residual'):
    x = x + ff_out.astype(cfg.active_weight_dtype)

  return x, kv_compressed, k_pe


def forward(
  x: jax.Array,
  segment_ids: jax.Array,
  weights: Weights,
  cfg: Config,
  cache: KVCache | None = None,
):
  with jax.named_scope('vocab_in_proj'):
    # Embed input tokens [B, T] -> [B, T D]
    x = jax.lax.with_sharding_constraint(
      weights.embedding[x, :],
      _logical2sharding(
        ('batch', 'sequence', 'act_embed'), cfg.mesh, cfg.rules
      ),
    )
  batch = x.shape[0]
  positions = segment_ids_to_positions(segment_ids)
  # Apply rotary embeddings: [B, T, head_dim]
  if cache is not None:
    # For inference with cache, we need to index the positional embeddings
    start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
  else:
    start_indices = jnp.zeros((batch,), dtype=jnp.int32)
  # NOTE: At inference time this only works for UNPACKED sequences.
  positions = start_indices[:, None] + positions
  # [B, T, head_dim]
  sin, cos = generate_pos_embeddings(positions, cfg.qk_rope_head_dim, cfg)
  sin, cos = (
    sin.astype(cfg.active_weight_dtype),
    cos.astype(cfg.active_weight_dtype),
  )

  for idx, layer in enumerate(weights.layers):
    x, kv_compressed, k_pe = forward_layer(
      x, segment_ids, layer, sin, cos, idx, cfg, cache
    )
    if cache is not None:
      cache.kv_compressed[idx], cache.k_pe[idx] = kv_compressed, k_pe

  # Final layer norm.
  x = rms_norm(x, weights.gamma_final)
  # Project to vocabulary size
  with jax.named_scope('vocab_out_proj'):
    logits = jnp.einsum('btd,dv->btv', x, weights.lm_head)
  if cache is not None:
    # Sum where there is a valid segment id (i.e. non padding tokens) [B, T] -> [B,]
    # cache = dataclasses.replace(cache, length=cache.length + jnp.max(jnp.sum(segment_ids != 0, axis=-1)))
    cache = dataclasses.replace(
      cache, length=cache.length + jnp.max(_count_length_from_left(segment_ids))
    )
    return logits, cache
  return logits


def input_shardings(
  mesh, rules
) -> tuple[
  jax.sharding.NamedSharding,
  jax.sharding.NamedSharding,
  jax.sharding.NamedSharding,
]:
  logical_axes = {
    'x': P('batch', 'sequence'),
    'segment_ids': P('batch', 'sequence'),
    'y': P('batch', 'sequence'),
  }
  return jax.tree.map(
    partial(_logical2sharding, mesh=mesh, rules=rules), logical_axes
  )


# Checkpointing logic
def make_mngr(path='/tmp/checkpoint_manager_sharded', erase: bool = False):
  if erase:
    path = ocp.test_utils.erase_and_create_empty(path)
  options = ocp.CheckpointManagerOptions(max_to_keep=3)
  mngr = ocp.CheckpointManager(
    path, options=options, item_names=('weights', 'opt_state')
  )
  return mngr


def _save(
  mngr: ocp.CheckpointManager, weights: Weights, opt_state: tp.Any, step: int
):
  args = ocp.args.Composite(
    **{
      k: ocp.args.StandardSave(v)
      for k, v in {'weights': weights, 'opt_state': opt_state}.items()
      if v is not None
    }
  )
  mngr.save(step, args=args)
  mngr.wait_until_finished()


def _load(
  mngr: ocp.CheckpointManager,
  cfg: Config,
  step: int | None = None,
  quantized: bool = False,
):
  step = mngr.latest_step() if step is None else step
  weights_abstract = Weights.abstract(cfg, quantized=quantized)
  weights_shardings = Weights.shardings(cfg, quantized=quantized)
  weights_shapes_shardings = jax.tree.map(
    lambda x, sharding: jax.ShapeDtypeStruct(
      x.shape.shape, x.shape.dtype, sharding=sharding
    ),
    weights_abstract,
    weights_shardings,
    is_leaf=is_param,
  )
  args = {'weights': weights_shapes_shardings}
  # retrieve opt_state if it's present in the checkpoint
  # if "opt_state" in mngr.item_metadata(step):
  #    opt_shapes_shardings = init_optimizer_state(weights_shapes_shardings)
  #    args["opt_state"] = opt_shapes_shardings
  restored = mngr.restore(
    step,
    # args=ocp.args.StandardRestore({"weights": weights_shapes_shardings, "opt_state": opt_shapes_shardings}),
    args=ocp.args.Composite(
      **{k: ocp.args.StandardRestore(v) for k, v in args.items()}
    ),
  )
  return restored['weights'], restored.get('opt_state', None)


def save(data, path):
  with ocp.PyTreeCheckpointer() as ckptr:
    ckptr.save(
      epath.Path(path),
      data,
      ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 100),
    )


class _EmptyNode:
  pass


def load(path, sharding=None):
  item, transforms = sharding, None
  restore_args = jax.tree.map(
    lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding
  )
  with ocp.PyTreeCheckpointer() as ckptr:
    result = ckptr.restore(
      epath.Path(path),
      args=ocp.args.PyTreeRestore(
        item=item, transforms=transforms, restore_args=restore_args
      ),
    )
    return jax.tree.map(
      lambda x: None if isinstance(x, _EmptyNode) else x, result
    )


def _orbax_resolve_type(x):
  if isinstance(x, ocp.metadata.StringMetadata):
    return str
  elif isinstance(x, ocp.metadata.ScalarMetadata):
    return x.dtype
  elif isinstance(x, ocp.metadata.ArrayMetadata):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)
  return x


def load_pytreedef(path):
  with ocp.PyTreeCheckpointer() as ckptr:
    return jax.tree.map(_orbax_resolve_type, ckptr.metadata(epath.Path(path)))


# Inference.
@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int):
  # [bs, length] -> [bs, padded]
  if chunk.ndim == 1:
    chunk = chunk[None, :]
  chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])
  segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
  return chunk, segment_ids


def sample_next_token(logits, temperature=1.0, greedy: bool = True):
  if greedy:
    return jnp.argmax(logits, -1)
  else:
    # Apply temperature
    logits = logits / temperature
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    # Sample from the distribution
    return jax.random.categorical(jax.random.key(0), probs, axis=-1)


def sample_from_prompt(
  tokens: jax.Array,
  weights: Weights,
  cache: KVCache,
  cfg: Config,
  batch_idx: int = 0,
  num_steps: int = 20,
  greedy: bool = True,
):
  """Samples from a prompt."""

  # Calculate the next power of 2 for padding, up to cfg.max_seq.
  assert len(tokens) <= cfg.max_seq_len
  pad_to = 2 ** math.ceil(math.log2(tokens.shape[-1]))
  prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=0)
  assert prompt.ndim == 2
  cache = dataclasses.replace(cache, length=jnp.zeros_like(cache.length))
  logits, cache = jax.jit(forward)(
    prompt, prompt_segment_ids, weights, cfg, cache
  )
  next_token_logit = logits[batch_idx, cache.length - 1, :]
  eos_tokens = (128001, 128008, 128009)

  tokens = []
  for _ in range(0, num_steps):
    next_token = sample_next_token(next_token_logit, greedy=greedy)[None]
    tokens.append(next_token[0])
    if next_token[0] in eos_tokens:
      break
    prompt, prompt_segment_ids = prepare_chunk(next_token, pad_to=1, pad_id=0)
    logits, cache = jax.jit(forward)(
      prompt, prompt_segment_ids, weights, cfg, cache
    )
    next_token_logit = logits[batch_idx, 0, :]

  return tokens, cache


def prefill(
  tokens: jax.Array,
  weights: Weights,
  cache: KVCache,
  cfg: Config,
  pad_id: int = 0,
  batch_idx: int = 0,
):
  """Samples from a prompt."""

  # Calculate the next power of 2 for padding, up to cfg.max_seq.
  assert tokens.shape[-1] <= cfg.max_seq_len
  pad_to = 2 ** math.ceil(math.log2(tokens.shape[-1]))
  prompt, prompt_segment_ids = prepare_chunk(
    tokens, pad_to=pad_to, pad_id=pad_id
  )
  assert prompt.ndim == 2

  if cache is not None:
    cache_shardings = KVCache.shardings(cfg, cfg.mesh, cfg.rules)
    cache = dataclasses.replace(
      cache,
      length=jnp.zeros_like(cache.length),
      starts=jnp.sum(jnp.cumsum(tokens != pad_id, axis=-1) == 0, -1),
    )
  else:
    cache_shardings = None
  logits_shardings = jax.sharding.NamedSharding(cfg.mesh, P(None, None, None))

  logits, cache = jax.jit(
    forward,
    donate_argnums=(4,),
    out_shardings=(logits_shardings, cache_shardings),
  )(prompt, prompt_segment_ids, weights, cfg, cache)
  return logits, cache


@partial(
  jax.jit,
  out_shardings=(Layout(DLL.AUTO), Layout(DLL.AUTO)),
  donate_argnames=('cache',),
)
def decode_step(last_tokens: jax.Array, weights, cache: KVCache, cfg: Config):
  assert last_tokens.ndim == 2
  segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
  next_logits, cache = jax.jit(forward)(
    last_tokens, segment_ids, weights, cfg, cache
  )
  next_tokens = jnp.argmax(next_logits, -1)
  next_tokens = jax.lax.with_sharding_constraint(
    next_tokens, NamedSharding(cfg.mesh, P())
  )
  return next_tokens, cache

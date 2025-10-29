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
from typing import Any
import dataclasses
import gc
import json
import functools
import time
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import math
from etils import epath

import jax
from tqdm import tqdm
from jax import numpy as jnp
from jax.sharding import SingleDeviceSharding, NamedSharding, PartitionSpec as P
import numpy as np
import torch
import torch.utils.dlpack
from safetensors.torch import load_file
from transformers import (
  PreTrainedTokenizerFast,
)

from deepseek import modeling_deepseek as deepseek
from model import (
  MoELayer,
  Config,
  AttentionLayer,
  MLPLayer,
  Layer,
  Weights,
  TensorInfo,
  is_param,
  _isinstance,
)
from model import load as jax_load, save as jax_save


def load_tokenizer(
  tokenizer_path: str | os.PathLike[str] | Path,
  tokenizer_config_path: str | os.PathLike[str] | Path,
) -> PreTrainedTokenizerFast:
  config = json.loads(Path(tokenizer_config_path).read_text())
  # config["added_tokens_decoder"] = {
  #    int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
  # }
  for k in list(config.keys()):
    v = config[k]
    if 'token' in k and isinstance(v, dict):
      config[k] = v['content']
  return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)
  # return AutoTokenizer(tokenizer_file=str(tokenizer_path), **config)


def t2j(x: Any):
  return jax.tree.map(
    lambda z: jnp.copy(jax.dlpack.from_dlpack(z.contiguous())), x
  )


def j2t(x: Any):
  cpu0 = jax.devices('cpu')[0]
  x_ = jax.tree.map(lambda z: jax.device_put(z, SingleDeviceSharding(cpu0)), x)
  return jax.tree.map(lambda z: torch.utils.dlpack.from_dlpack(jnp.copy(z)), x_)


def convert_config(
  config: deepseek.DeepseekV3Config, *, quantized: bool
) -> Config:
  rope_scaling = {
    'beta_fast': 32,
    'beta_slow': 1,
    'factor': 40,
    'mscale': 1.0,
    'mscale_all_dim': 1.0,
    'original_max_position_embeddings': 4096,
    'type': 'yarn',
  }
  return Config(
    embed=config.hidden_size,
    q_lora_rank=config.q_lora_rank,
    kv_lora_rank=config.kv_lora_rank,
    num_heads=config.num_attention_heads,
    qk_nope_head_dim=config.qk_nope_head_dim,
    qk_rope_head_dim=config.qk_rope_head_dim,
    v_head_dim=config.v_head_dim,
    vocab_size=config.vocab_size,
    num_layers=config.num_hidden_layers,
    max_seq_len=8192,
    rope_theta=config.rope_theta,
    rope_scaling_factor=rope_scaling['factor'],
    rope_beta_fast=rope_scaling['beta_fast'],
    rope_beta_slow=rope_scaling['beta_slow'],
    rope_mscale=rope_scaling['mscale'],
    rope_mscale_all_dim=rope_scaling['mscale_all_dim'],
    rope_original_max_position_embeddings=rope_scaling[
      'original_max_position_embeddings'
    ],
    ffw_size=config.intermediate_size,
    first_k_dense=config.first_k_dense_replace,
    moe_gate_dtype=jnp.float32,
    moe_ffw_size=config.moe_intermediate_size,
    n_routed_experts=config.n_routed_experts,
    num_experts_per_tok=config.num_experts_per_tok,
    n_group=config.n_group,
    topk_group=config.topk_group,
    routed_scaling_factor=config.routed_scaling_factor,
    n_shared_experts=config.n_shared_experts,
    quantized=quantized,
  )

def _cast_dtype(layer, layer_abst):
  return jax.tree.map(
    lambda x, y: x.astype(y.shape.dtype), layer, layer_abst, is_leaf=is_param
  )


def convert_attn_layer(
  params_or_attn: deepseek.DeepseekV3Attention | dict, cfg: Config
):
  layer_abst = AttentionLayer.abstract(cfg)
  params = (
    params_or_attn
    if isinstance(params_or_attn, dict)
    else dict(params_or_attn.named_parameters())
  )
  layer = AttentionLayer.abstract(cfg)
  q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

  layer.q_a = t2j(params['q_a_proj.weight'].data.T)
  layer.q_gamma = t2j(params['q_a_layernorm.weight'].data)
  layer.q_b = t2j(
    params['q_b_proj.weight'].data.T.reshape(
      (cfg.q_lora_rank, cfg.num_heads, q_head_dim)
    )
  )
  layer.kv_a = t2j(
    params['kv_a_proj_with_mqa.weight'].data.T[..., : -cfg.qk_rope_head_dim]
  )
  layer.k_pe = t2j(
    params['kv_a_proj_with_mqa.weight'].data.T[..., -cfg.qk_rope_head_dim :]
  )
  layer.kv_gamma = t2j(params['kv_a_layernorm.weight'].data)
  kv_b_proj = t2j(
    params['kv_b_proj.weight'].data.T.reshape(
      (cfg.kv_lora_rank, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
    )
  )
  layer.k_b = kv_b_proj[..., : -cfg.v_head_dim]
  layer.v_b = kv_b_proj[..., -cfg.v_head_dim :]
  layer.o = t2j(
    params['o_proj.weight'].data.T.reshape(
      (cfg.num_heads, cfg.v_head_dim, cfg.embed)
    )
  )
  for field in dataclasses.fields(layer):
    name = field.name
    expected_shape = getattr(layer_abst, name).shape.shape
    actual_shape = getattr(
      getattr(layer, name).shape, 'shape', getattr(layer, name).shape
    )
    assert actual_shape == expected_shape, (
      f'For {name = } {expected_shape = } vs {actual_shape = }'
    )
  return _cast_dtype(layer, layer_abst)


def convert_mlp_layer(params_or_mlp: deepseek.DeepseekV3MLP, cfg: Config):
  layer_abst = MLPLayer.abstract(cfg)
  params = (
    params_or_mlp
    if isinstance(params_or_mlp, dict)
    else dict(params_or_mlp.named_parameters())
  )
  layer = MLPLayer.abstract(cfg)

  layer.w_gate = t2j(params['gate_proj.weight'].data.T)
  layer.w_up = t2j(params['up_proj.weight'].data.T)
  layer.w_down = t2j(params['down_proj.weight'].data.T)
  for field in dataclasses.fields(layer):
    name = field.name
    expected_shape = getattr(layer_abst, name).shape.shape
    actual_shape = getattr(
      getattr(layer, name).shape, 'shape', getattr(layer, name).shape
    )
    assert actual_shape == expected_shape, (
      f'For {name = } {expected_shape = } vs {actual_shape = }'
    )
  return _cast_dtype(layer, layer_abst)


def convert_moe_layer(
  params_or_moe: deepseek.DeepseekV3MoE | dict, cfg: Config
):
  layer_abst = MoELayer.abstract(cfg)
  params = (
    params_or_moe
    if isinstance(params_or_moe, dict)
    else dict(params_or_moe.named_parameters())
  )
  layer = MoELayer.abstract(cfg)

  layer.w_router = t2j(params['gate.weight'].data.T)
  layer.b_router = t2j(params['gate.e_score_correction_bias'].data)
  layer.ws_gate = t2j(params['shared_experts.gate_proj.weight'].data.T)
  layer.ws_up = t2j(params['shared_experts.up_proj.weight'].data.T)
  layer.ws_down = t2j(params['shared_experts.down_proj.weight'].data.T)

  layer.we_gate = t2j(
    torch.stack(
      [
        params[f'experts.{i}.gate_proj.weight'].data.T
        for i in range(cfg.n_routed_experts)
      ],
      0,
    )
  )
  layer.we_up = t2j(
    torch.stack(
      [
        params[f'experts.{i}.up_proj.weight'].data.T
        for i in range(cfg.n_routed_experts)
      ],
      0,
    )
  )
  layer.we_down = t2j(
    torch.stack(
      [
        params[f'experts.{i}.down_proj.weight'].data.T
        for i in range(cfg.n_routed_experts)
      ],
      0,
    )
  )
  for field in dataclasses.fields(layer):
    name = field.name
    expected_shape = getattr(layer_abst, name).shape.shape
    actual_shape = getattr(
      getattr(layer, name).shape, 'shape', getattr(layer, name).shape
    )
    assert actual_shape == expected_shape, (
      f'For {name = } {expected_shape = } vs {actual_shape = }'
    )
  return _cast_dtype(layer, layer_abst)


def convert_layer(
  params_or_layer: deepseek.DeepseekV3DecoderLayer | dict, cfg: Config
):
  params = (
    params_or_layer
    if isinstance(params_or_layer, dict)
    else dict(params_or_layer.named_parameters())
  )
  use_moe = len([k for k in params.keys() if 'expert' in k]) > 0

  layer_abst = Layer.abstract(cfg, use_moe=use_moe)
  layer = Layer.abstract(cfg, use_moe=use_moe)
  attn_params = {
    k[len('self_attn.') :]: v
    for (k, v) in params.items()
    if k.startswith('self_attn.')
  }
  layer.attn = convert_attn_layer(attn_params, cfg)
  mlp_params = {
    k[len('mlp.') :]: v for (k, v) in params.items() if k.startswith('mlp.')
  }
  layer.mlp = (
    convert_mlp_layer(mlp_params, cfg)
    if not use_moe
    else convert_moe_layer(mlp_params, cfg)
  )
  layer.gamma_pre_attn = t2j(params['input_layernorm.weight'].detach())
  layer.gamma_post_attn = t2j(
    params['post_attention_layernorm.weight'].detach()
  )

  for name in ['gamma_pre_attn', 'gamma_post_attn']:
    expected_shape = getattr(layer_abst, name).shape.shape
    actual_shape = getattr(
      getattr(layer, name).shape, 'shape', getattr(layer, name).shape
    )
    assert actual_shape == expected_shape, (
      f'For {name = } {expected_shape = } vs {actual_shape = }'
    )
  return _cast_dtype(layer, layer_abst)


def _extract_layer_params(params: dict[str, Any], prefix: str):
  return {
    k[len(prefix) :]: v for (k, v) in params.items() if k.startswith(prefix)
  }


def convert_model(
  params_or_model: deepseek.DeepseekV3ForCausalLM | dict, cfg: Config
):
  params = (
    params_or_model
    if isinstance(params_or_model, dict)
    else dict(params_or_model.named_parameters())
  )

  model_abst = Weights.abstract(cfg)
  model = Weights.abstract(cfg)
  layer_idxs = {
      int(re.match(r'model\.layers\.([0-9]+).*', k).group(1))
      for k in params.keys()
      if re.match(r'model\.layers\.([0-9]+).*', k) is not None
  }
  assert all(
    i in layer_idxs for i in range(len(layer_idxs))
  )  # check all layers are there
  assert len(layer_idxs) == cfg.num_layers
  model.gamma_final = t2j(params['model.norm.weight'].detach())
  model.embedding = t2j(params['model.embed_tokens.weight'].detach())
  model.lm_head = t2j(params['lm_head.weight'].T.detach())
  model.layers = [
    convert_layer(_extract_layer_params(params, f'model.layers.{i}.'), cfg)
    for i in range(cfg.num_layers)
  ]

  for name in ['embedding', 'gamma_final', 'lm_head']:
    expected_shape = getattr(model_abst, name).shape.shape
    actual_shape = getattr(
      getattr(model_abst, name).shape, 'shape', getattr(model_abst, name).shape
    )
    assert actual_shape == expected_shape, (
      f'For {name = } {expected_shape = } vs {actual_shape = }'
    )
  return _cast_dtype(model, model_abst)


########################################################################################################################


def load_param_list(
  params_map: dict[str, dict[str, Any]], root_path: Path, param_list: list[str]
) -> dict[str, torch.Tensor]:
  root_path = Path(root_path)
  files = {
    root_path / params_map[param_name]['file'] for param_name in param_list
  }
  archive = functools.reduce(
    lambda a, b: a | b, [load_file(file) for file in tqdm(files)], dict()
  )
  return {param_name: archive[param_name] for param_name in param_list}


def load_param(
  params_map: dict[str, dict[str, Any]], root_path: Path, param_name: str
) -> torch.Tensor:
  return list(load_param_list(params_map, root_path, [param_name]).values())[0]


def _dequant_params(
  params_maybe_quant: dict[str, dict[str, Any]], parallel: bool = False
) -> dict[str, torch.Tensor]:
  quant_tensors = [
    k[: -len('_scale_inv')]
    for k in params_maybe_quant.keys()
    if k.endswith('_scale_inv')
  ]
  params = params_maybe_quant.copy()

  def _dequant_tensor(args):
    t_start = time.time()
    quant, scale, name = args
    tqdm.write(f'Dequantizing {name}')
    assert quant.ndim == scale.ndim, f'{quant.shape = }, but {scale.shape = }'
    tile = [math.ceil(z / y) for z, y in zip(quant.shape, scale.shape)]
    for i, t in enumerate(tile):
      scale = torch.repeat_interleave(scale, t, i)
    if not all(
      z % y == 0 for z, y in zip(quant.shape, scale.shape)
    ):  # there's some padding in scaling
      scale = scale[[slice(0, z) for z in quant.shape]]
    assert scale.shape == quant.shape
    value = (quant.to(scale.dtype) * scale).to(torch.float32)
    tqdm.write(f'Dequantizing takes {time.time() - t_start:.4e} s')
    return value

  if parallel:
    with ThreadPoolExecutor(max_workers=16) as executor:
      values = executor.map(
        _dequant_tensor,
        [
          (params[name], params[f'{name}_scale_inv'], name)
          for name in quant_tensors
        ],
      )
  else:
    values = [
      _dequant_tensor((params[name], params[f'{name}_scale_inv'], name))
      for name in tqdm(quant_tensors)
    ]

  dequant_params = dict(zip(quant_tensors, values))

  params.update(dequant_params)
  for name in quant_tensors:
    del params[f'{name}_scale_inv']
  return params


def load_params_from_prefix(
  params_map: dict[str, dict[str, Any]], root_path: Path, layer_prefix: str
) -> dict[str, torch.Tensor]:
  all_layer_keys = [k for k in params_map.keys() if k.startswith(layer_prefix)]
  params_maybe_quant = {
    k[len(layer_prefix) :]: v
    for (k, v) in load_param_list(params_map, root_path, all_layer_keys).items()
  }
  return _dequant_params(params_maybe_quant)


def convert_hf_checkpoint(params_map, root_path, dest_path, cfg: Config):
  root_path = Path(root_path)
  dest_path = Path(dest_path)

  weights_misc = t2j(
    load_param_list(
      params_map,
      root_path,
      ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'],
    )
  )
  weights_misc = {
    'embedding': weights_misc['model.embed_tokens.weight'],
    'gamma_final': weights_misc['model.norm.weight'],
    'lm_head': jnp.copy(weights_misc['lm_head.weight'].T),
  }
  jax_save(weights_misc, dest_path / f'weights_misc')

  for idx in tqdm(range(cfg.num_layers)):
    layer = convert_layer(
      load_params_from_prefix(params_map, root_path, f'model.layers.{idx}.'),
      cfg,
    )
    layer_quant = layer.quantize(layer, cfg)
    jax_save(layer_quant, dest_path / f'layer_{idx}')
    gc.collect()


def load_model(
  root_path: Path | epath.Path, cfg: Config, quantized: bool = False
):
  root_path = epath.Path(root_path)
  weights_sharding = Weights.shardings(cfg, quantized=quantized)
  weights_abst = Weights.abstract(cfg, quantized=quantized)
  _get_shape = (
    lambda x: x.shape.shape if _isinstance(x, TensorInfo) else x.shape
  )
  _check_shapes = lambda w, other=weights_abst: jax.tree.all(
    jax.tree.map(
      lambda x, z: _get_shape(x) == _get_shape(z), w, other, is_leaf=is_param
    )
  )
  weights = Weights.abstract(cfg, quantized=quantized)

  # load embedding, gamma_final, lm_head
  misc_sharding = {
    'embedding': weights_sharding.embedding,
    'gamma_final': weights_sharding.gamma_final,
    'lm_head': weights_sharding.lm_head,
  }
  print('Loading misc weights...', flush=True)
  weights_misc = jax_load(root_path / 'weights_misc', sharding=misc_sharding)
  print('Loaded misc weights', flush=True)
  weights.embedding = weights_misc['embedding']
  weights.gamma_final = weights_misc['gamma_final']
  weights.lm_head = weights_misc['lm_head']
  assert _check_shapes(weights)

  # def _load_layer(idx):
  #    sharding = weights_sharding.layers[idx]
  #    layer = jax_load(root_path / f"layer_{idx}", sharding=sharding)
  #    return layer
  #    #weights.layers[layer_idx] = layer

  ## load layers
  # pbar = tqdm(list(range(cfg.num_layers)))
  # pbar.set_description(f"Loading layers 0 through {cfg.num_layers}")
  # with ThreadPoolExecutor(max_workers=16) as executor:
  #    #layers = list(executor.map(_load_layer, list(range(cfg.num_layers))))
  #    futs = {i: executor.submit(_load_layer, i) for i in range(cfg.num_layers)}
  #    layers = [None for _ in range(cfg.num_layers)]

  #    while len(futs) > 0:
  #        done_mask = [i for i, fut in futs.items() if fut.done()]
  #        for i in done_mask:
  #            pbar.update(1)
  #            layers[i] = futs[i].result()
  #            del futs[i]
  #        time.sleep(1e-1)

  # weights.layers = layers
  # _check_shapes(weights)

  pbar = tqdm(list(range(cfg.num_layers)))
  pbar.set_description(f'Loading layers 0 through {cfg.num_layers}')
  for layer_idx in pbar:
    sharding = weights_sharding.layers[layer_idx]
    layer = jax_load(root_path / f'layer_{layer_idx}', sharding=sharding)
    weights.layers[layer_idx] = layer
    assert _check_shapes(weights), (
      f'Shape check failed after layer idx = {layer_idx}'
    )

  return weights


def load_torch_model(
  params_map: dict[str, dict[str, Any]],
  root_path: Path,
  config: deepseek.DeepseekV3Config,
):
  model = deepseek.DeepseekV3ForCausalLM(config)
  model.eval()
  param_names = list(dict(model.named_parameters()).keys())

  extra_param_scales = [
    f'{k}_scale_inv' for k in param_names if f'{k}_scale_inv' in params_map
  ]
  print(f'Reading additional #{len(extra_param_scales)} of parameter scales')
  param_names = param_names + extra_param_scales
  params = _dequant_params(load_param_list(params_map, root_path, param_names))
  model.load_state_dict(params, strict=True)
  return model


def replicate(x):
  def _replicate(z):
    mesh = jax.make_mesh((len(z.devices()),), P('x'), devices=list(z.devices()))
    return np.array(jax.device_put(z, NamedSharding(mesh, P())))

  return jax.tree.map(_replicate, x)


def err_fn(x, y, axis=-1):
  x, y = jax.tree.map(lambda x: x.astype(jnp.float32), (x, y))
  x, y = replicate((x, y))
  # cpu0 = jax.devices("cpu")[0]
  # x, y = jax.tree.map(lambda z: jax.device_put(z, SingleDeviceSharding(cpu0)), (x, y))
  diff = np.linalg.norm(x.astype(np.float32) - y.astype(np.float32), axis=axis)
  norm = np.linalg.norm(y.astype(np.float32), axis=axis)
  return diff / (norm + 1e-9)

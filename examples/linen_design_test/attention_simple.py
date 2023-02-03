# Copyright 2022 The Flax Authors.
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

import functools
from pprint import pprint
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Union
from flax.core import Scope
from flax.core.frozen_dict import freeze, unfreeze
from flax.linen import initializers
from flax.linen import Module, compact, vmap
from flax.linen.linear import PrecisionLike
import jax
from jax import lax, numpy as jnp, random
import numpy as np



class Dense(Module):
  features: int
  use_bias: bool = True
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros_init()
  dtype: Any = jnp.float32
  precision: PrecisionLike = None

  @compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = jnp.asarray(kernel, self.dtype)
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


class SoftmaxAttn(Module):
  @compact
  def __call__(self, weights):
    norm_dims = tuple(range(weights.ndim // 2, weights.ndim))
    return jax.nn.softmax(weights, axis=norm_dims)


class Dropout(Module):
  rate: float

  @compact
  def __call__(self, x, deterministic=False, rng=None):
    if self.rate == 0.:
      return x
    keep_prob = 1. - self.rate

    if deterministic:
      return x
    else:
      if rng is None:
        rng = self.scope.make_rng('dropout')
      mask = random.bernoulli(rng, p=keep_prob, shape=x.shape)
      return lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class SoftmaxAttnWDropout(Module):
  rate: float = 0.0
  deterministic: bool = False

  @compact
  def __call__(self, x):
    x = SoftmaxAttn()(x)
    x = Dropout(self.rate)(x, deterministic=self.deterministic)
    return x


class RawDotProductAttention(Module):
  attn_module: Callable = SoftmaxAttn

  @compact
  def __call__(self, query, key, value, bias=None, dtype=jnp.float32):
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim

    n = query.ndim
    attn_weights = lax.dot_general(
        query, key,
        (((n-1,), (n - 1,)), ((), ())))
    if bias is not None:
      attn_weights += bias
    attn_weights = self.attn_module()(attn_weights)
    attn_weights = attn_weights.astype(dtype)

    contract_dims = (
        tuple(range(n - 1, attn_weights.ndim)),
        tuple(range(0, n  - 1)))
    y = lax.dot_general(
        attn_weights, value,
        (contract_dims, ((), ())))
    return y


class DotProductAttention(Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  attn_module: Callable = SoftmaxAttn

  @compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    QKVDense = functools.partial(
      Dense, features=qkv_features, use_bias=False, dtype=dtype)
    query = QKVDense(name='query')(inputs_q)
    key = QKVDense(name='key')(inputs_kv)
    value = QKVDense(name='value')(inputs_kv)

    y = RawDotProductAttention(attn_module=self.attn_module)(
      query, key, value, bias=bias, dtype=dtype)
    y = Dense(features=out_features, dtype=dtype, name='out')(y)
    return y


# Trying out a slightly more compact vmap notation:

def concise_vmap(module, in_axes, out_axes, axis_size=None, **var_specs):
  variable_axes = {k: v[0] for k, v in
                      var_specs.items() if isinstance(v, Sequence)}
  splits = {k: v[1] for k, v in var_specs.items() if isinstance(v, Sequence)}
  return vmap(module,
              in_axes=in_axes,
              out_axes=out_axes,
              variable_axes=variable_axes,
              split_rngs=splits,
              axis_size=axis_size)


class MultiHeadDotProductAttention(Module):
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  attn_module: Callable = SoftmaxAttn
  batch_axes: Sequence[int] = (0,)
  num_heads: int = 1
  broadcast_dropout: bool = False

  @compact
  def __call__(self, inputs_q, inputs_kv, bias=None, dtype=jnp.float32):
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    out_features = self.out_features or inputs_q.shape[-1]

    # Now, vmap attn.__call__ along heads and spatial dims.
    Attn = concise_vmap(DotProductAttention,
                        (None, None, None), -2,
                        param=(0, True),
                        dropout=(None, not self.broadcast_dropout),
                        axis_size=self.num_heads)
    for axis in reversed(sorted(self.batch_axes)):
      Attn = concise_vmap(Attn,
                          (axis, axis, axis), axis,
                          param=(None, False),
                          dropout=(None, not self.broadcast_dropout))

    attn = Attn(attn_module=self.attn_module,
                qkv_features=qkv_features // self.num_heads,
                out_features=out_features)

    # evaluate multi-headed-attention.
    y = attn(inputs_q, inputs_kv, bias)
    return y.mean(axis=-2)


# run it.


if __name__ == '__main__':

  inputs = jnp.ones((8, 97, 256))
  rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
  model = MultiHeadDotProductAttention(
      broadcast_dropout=False,
      qkv_features=256,
      out_features=256,
      attn_module=functools.partial(SoftmaxAttnWDropout, rate=0.1),
      num_heads=8,
      batch_axes=(0,),)

  y, params = model.init_with_output(rngs, inputs, inputs)

  print('input shape: ', inputs.shape)
  print('parameter shapes:')
  pprint(jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
  print('output shape: ', y.shape)

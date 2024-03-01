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

import typing as tp

import jax
import jax.numpy as jnp
from jax import random

from flax.experimental import nnx


class Linear(nnx.Module):
  @tp.overload
  def __init__(self, *, din: int, dout: int, rngs: nnx.Rngs):
    ...

  @tp.overload
  def __init__(self, *, dout: int):
    ...

  @tp.overload
  def __init__(
    self,
    *,
    din: tp.Optional[int] = None,
    dout: int,
    rngs: tp.Optional[nnx.Rngs] = None,
  ):
    ...

  def __init__(
    self,
    *,
    din: tp.Optional[int] = None,
    dout: int,
    rngs: tp.Optional[nnx.Rngs] = None,
  ):
    self.dout = dout
    if din is not None:
      if rngs is None:
        raise ValueError('rngs must be provided if din is provided')
      self.init_variables(din, rngs)

  def init_variables(self, din: int, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(random.uniform(key, (din, self.dout)))
    self.b = nnx.Param(jnp.zeros((self.dout,)))

  def __call__(
    self, x: jax.Array, *, rngs: tp.Optional[nnx.Rngs] = None
  ) -> jax.Array:
    if self.is_initializing and not hasattr(self, 'w'):
      if rngs is None:
        raise ValueError('rngs must be provided to initialize module')
      self.init_variables(x.shape[-1], rngs)

    return x @ self.w + self.b


class BatchNorm(nnx.Module):
  @tp.overload
  def __init__(self, *, mu: float = 0.95):
    ...

  @tp.overload
  def __init__(self, *, din: int, mu: float = 0.95, rngs: nnx.Rngs):
    ...

  @tp.overload
  def __init__(
    self,
    *,
    din: tp.Optional[int] = None,
    mu: float = 0.95,
    rngs: tp.Optional[nnx.Rngs] = None,
  ):
    ...

  def __init__(
    self,
    *,
    din: tp.Optional[int] = None,
    mu: float = 0.95,
    rngs: tp.Optional[nnx.Rngs] = None,
  ):
    self.mu = mu

    if din is not None:
      if rngs is None:
        raise ValueError('rngs must be provided if din is provided')
      self.init_variables(din, rngs)

  def init_variables(self, din: int, rngs: nnx.Rngs):
    self.scale = nnx.Param(jax.numpy.ones((din,)))
    self.bias = nnx.Param(jax.numpy.zeros((din,)))
    self.mean = nnx.BatchStat(jax.numpy.zeros((din,)))
    self.var = nnx.BatchStat(jax.numpy.ones((din,)))

  def __call__(
    self, x, *, train: bool, rngs: tp.Optional[nnx.Rngs] = None
  ) -> jax.Array:
    if self.is_initializing and not hasattr(self, 'scale'):
      if rngs is None:
        raise ValueError('rngs must be provided to initialize module')
      self.init_variables(x.shape[-1], rngs)

    if train:
      axis = tuple(range(x.ndim - 1))
      mean = jax.numpy.mean(x, axis=axis)
      var = jax.numpy.var(x, axis=axis)
      # ema update
      self.mean = self.mu * self.mean + (1 - self.mu) * mean
      self.var = self.mu * self.var + (1 - self.mu) * var
    else:
      mean, var = self.mean, self.var

    scale, bias = self.scale, self.bias
    x = (x - mean) / jax.numpy.sqrt(var + 1e-5) * scale + bias
    return x


class Dropout(nnx.Module):
  def __init__(self, rate: float):
    self.rate = rate

  def __call__(self, x: jax.Array, *, train: bool, rngs: nnx.Rngs) -> jax.Array:
    if train:
      mask = random.bernoulli(rngs.dropout(), (1 - self.rate), x.shape)
      x = x * mask / (1 - self.rate)
    return x


# ----------------------------
# test Linear
# ----------------------------
print('test Linear')

# eager
m1 = Linear(din=32, dout=10, rngs=nnx.Rngs(params=0))
y = m1(x=jnp.ones((1, 32)))
print(jax.tree_map(jnp.shape, m1.get_state()))

# lazy
m2 = Linear(dout=10)
y = m2.init(x=jnp.ones((1, 32)), rngs=nnx.Rngs(params=0))
print(jax.tree_map(jnp.shape, m2.get_state()))

# usage
y1 = m1(x=jnp.ones((1, 32)))
y2 = m2(x=jnp.ones((1, 32)))

# ----------------------------
# Test scan
# ----------------------------
print('\ntest scan')


class Block(nnx.Module):
  def __init__(
    self,
    din: tp.Optional[int] = None,
    dout: int = 10,
    rngs: tp.Optional[nnx.Rngs] = None,
  ):
    self.linear = Linear(din=din, dout=dout, rngs=rngs)
    self.bn = BatchNorm(din=dout if din is not None else None, rngs=rngs)
    self.dropout = Dropout(0.5)

  def __call__(self, x: jax.Array, _, *, train: bool, rngs: nnx.Rngs):
    x = self.linear(x, rngs=rngs)
    x = self.bn(x, train=train, rngs=rngs)
    x = self.dropout(x, train=train, rngs=rngs)
    x = jax.nn.gelu(x)
    return x, None


MLP = nnx.Scan(
  Block,
  variable_axes={nnx.Param: 0},
  variable_carry=nnx.BatchStat,
  split_rngs={'params': True, 'dropout': True},
  length=5,
)


# eager
mlp = MLP(din=10, dout=10, rngs=nnx.Rngs(params=0))
y, _ = mlp.call(jnp.ones((1, 10)), None, train=True, rngs=nnx.Rngs(dropout=1))
print(f'{y.shape=}')
print('state =', jax.tree_map(jnp.shape, mlp.get_state()))
print()

# lazy
mlp = MLP(dout=10)
mlp.init(jnp.ones((1, 10)), None, train=False, rngs=nnx.Rngs(params=0))
y, _ = mlp.call(jnp.ones((1, 10)), None, train=True, rngs=nnx.Rngs(dropout=1))
print(f'{y.shape=}')
print('state =', jax.tree_map(jnp.shape, mlp.get_state()))

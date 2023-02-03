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

from typing import Any, Optional
from dataclasses import dataclass

from absl.testing import absltest

import jax
from jax import numpy as jnp, random

from flax.core import Array, init, unfreeze, nn

from flax import struct


@dataclass
class Dense:
  features: int
  bias: bool = True
  kernel_init: Any = nn.linear.default_kernel_init
  bias_init: Any = nn.initializers.zeros_init()

  def __call__(self, scope, x):
    kernel = scope.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    y = x @ kernel
    if self.bias:
      bias = scope.param('bias', self.bias_init, (self.features,))
      y += bias.reshape((1,) * (y.ndim - 1) + (-1,))
    return y


@struct.dataclass
class ExplicitDense:
  kernel: Array
  bias: Optional[Array]

  # a fully explicit "scope free" version
  @staticmethod
  def create(rng, in_size, out_size, bias=True,
             kernel_init=nn.linear.default_kernel_init,
             bias_init=nn.initializers.zeros_init()):
    k1, k2 = random.split(rng, 2)
    kernel = kernel_init(k1, (in_size, out_size))
    if bias:
      bias = bias_init(k2, (out_size,))
    else:
      bias = None
    return ExplicitDense(kernel, bias)

  # a semi-explicit version where a scope is used to create explicit params
  @staticmethod
  def create_in_scope(scope, in_size, out_size, bias=True,
                      kernel_init=nn.linear.default_kernel_init,
                      bias_init=nn.initializers.zeros_init()):
    kernel = scope.param('kernel', kernel_init, (in_size, out_size))
    if bias:
      bias = scope.param('bias', bias_init, (out_size,))
    else:
      bias = None
    return ExplicitDense(kernel, bias)

  def __call__(self, x):
    y = x @ self.kernel
    if self.bias is not None:
      y += self.bias.reshape((1,) * (y.ndim - 1) + (-1,))
    return y

def explicit_mlp(scope, x, sizes=(3, 1)):
  for i, size in enumerate(sizes):
    dense = scope.param(f'dense_{i}', ExplicitDense.create, x.shape[-1], size)
    x = dense(x)
    if i + 1 < len(sizes):
      x = nn.relu(x)
  return x

def semi_explicit_mlp(scope, x, sizes=(3, 1)):
  for i, size in enumerate(sizes):
    dense = scope.child(ExplicitDense.create_in_scope, prefix='dense_')(x.shape[-1], size)
    x = dense(x)
    if i + 1 < len(sizes):
      x = nn.relu(x)
  return x


class DenseTest(absltest.TestCase):

  def test_dense(self):
    model = Dense(features=4)
    x = jnp.ones((1, 3))
    y, variables = init(model)(random.PRNGKey(0), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(y.shape, (1, 4))
    self.assertEqual(param_shapes, {
        'kernel': (3, 4),
        'bias': (4,),
    })

  def test_explicit_dense(self):
    x = jnp.ones((1, 3))
    y, variables = init(explicit_mlp)(random.PRNGKey(0), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(y.shape, (1, 4))
    self.assertEqual(param_shapes, {
        'kernel': (3, 4),
        'bias': (4,),
    })

  def test_explicit_dense(self):
    x = jnp.ones((1, 4))
    y, variables = init(explicit_mlp)(random.PRNGKey(0), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(y.shape, (1, 1))
    self.assertEqual(param_shapes, {
        'dense_0': ExplicitDense((4, 3), (3,)),
        'dense_1': ExplicitDense((3, 1), (1,))
    })

  def test_semi_explicit_dense(self):
    x = jnp.ones((1, 4))
    y, variables = init(semi_explicit_mlp)(random.PRNGKey(0), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(y.shape, (1, 1))
    self.assertEqual(param_shapes, {
        'dense_0': {'kernel': (4, 3), 'bias': (3,)},
        'dense_1': {'kernel': (3, 1), 'bias': (1,)}
    })


if __name__ == '__main__':
  absltest.main()

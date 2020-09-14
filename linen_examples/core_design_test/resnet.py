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


from flax.core import Scope, init, apply, unfreeze, nn

import jax
from jax import lax, random, numpy as jnp

from typing import Any
from functools import partial


Array = Any


default_norm = partial(nn.batch_norm)

def residual_block(scope: Scope, x: Array, conv, norm, act, features: int, strides=(1, 1)):
  residual = x
  x = scope.child(conv, 'conv_1')(x, features, (1, 1))
  x = scope.child(norm, 'bn_1')(x)
  x = act(x)
  x = scope.child(conv, 'conv_2')(x, 4 * features, (3, 3), strides=strides)
  x = scope.child(norm, 'bn_2')(x)
  x = act(x)
  x = scope.child(conv, 'conv_3')(x, 4 * features, (1, 1))
  x = scope.child(norm, 'bn_3')(x)

  if x.shape != residual.shape:
    residual = scope.child(conv, 'proj_conv')(residual, 4 * features, (1, 1), strides=strides)
    residual = scope.child(norm, 'proj_bn')(residual)

  return act(residual + x)


def resnet(scope: Scope, x,
           block_sizes=(3, 4, 6, 3),
           features=64, num_classes=1000,
           dtype=jnp.float32,
           norm=default_norm,
           act=nn.relu,
           ):
  conv = partial(nn.conv, bias=False, dtype=dtype)
  norm = partial(norm, dtype=dtype)

  x = scope.child(conv, 'init_conv')(x, 16, (7, 7), padding=((3, 3), (3, 3)))
  x = scope.child(norm, 'init_bn')(x)
  x = act(x)
  x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')

  for i, size in enumerate(block_sizes):
    for j in range(size):
      strides = (1, 1)
      if i > 0 and j == 0:
        strides = (2, 2)
      block_features = features * 2 ** i
      block_scope = scope.push(f'block_{i}_{j}')
      x = residual_block(block_scope, x, conv, norm, act, block_features, strides)
      # we can access parameters of the sub module by operating on the scope
      # Example:
      # block_scope.get_kind('params')['conv_1']['kernel']
  x = jnp.mean(x, (1, 2))
  x = scope.child(nn.dense, 'out')(x, num_classes)
  return x

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 224, 224, 3))
  y, params = init(resnet)(random.PRNGKey(1), x)
  print(y.shape)
  print(jax.tree_map(jnp.shape, unfreeze(params)))

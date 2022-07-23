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

from functools import partial

from absl.testing import absltest

import numpy as np

from flax.core import Scope, Array, init, apply, unfreeze, lift, nn

import jax
from jax import lax, random, numpy as jnp


default_norm = partial(nn.batch_norm)


def residual_block(scope: Scope, x: Array, conv, norm, act, features: int):
  residual = x
  x = scope.child(conv, 'conv_1')(x, features, (3, 3))
  x = scope.child(norm, 'bn_1')(x)
  x = act(x)
  x = scope.child(conv, 'conv_2')(x, features, (3, 3))
  x = scope.child(norm, 'bn_2')(x)
  return act(residual + x)

def big_resnet(scope: Scope, x, blocks=(10, 5), dtype=jnp.float32,
               norm=default_norm, act=nn.relu):
  conv = partial(nn.conv, bias=False, dtype=dtype)
  norm = partial(norm, dtype=dtype)

  # a two stage resnet where inner blocks are rematerialized to make sure
  # memory consumtion grows as O(sqrt(N)) and compute is O(N) where N is the number of blocks..
  # we use a double scan such that the compiled binary is of size O(1).
  print('total residual blocks:', np.prod(blocks))

  def body_fn(scope, x):
    return residual_block(scope, x, conv, norm, act, features=x.shape[-1])

  return lift.remat_scan(
      body_fn, lengths=blocks,
      variable_axes={'params': 0, 'batch_stats': 0},
      split_rngs={'params': True},
      policy=None)(scope, x)


class BigResnetTest(absltest.TestCase):

  def test_big_resnet(self):
    x = random.normal(random.PRNGKey(0), (1, 8, 8, 8))
    y, variables = init(big_resnet)(random.PRNGKey(1), x)
    self.assertEqual(y.shape, (1, 8, 8, 8))
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    batch_stats_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['batch_stats']))
    self.assertEqual(param_shapes, {
        'conv_1': {'kernel': (10, 5, 3, 3, 8, 8)},
        'conv_2': {'kernel': (10, 5, 3, 3, 8, 8)},
        'bn_1': {'scale': (10, 5, 8), 'bias': (10, 5, 8)},
        'bn_2': {'scale': (10, 5, 8), 'bias': (10, 5, 8)}
    })
    self.assertEqual(batch_stats_shapes, {
        'bn_1': {'var': (10, 5, 8), 'mean': (10, 5, 8)},
        'bn_2': {'var': (10, 5, 8), 'mean': (10, 5, 8)}
    })


if __name__ == '__main__':
  absltest.main()

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

from typing import Sequence, Callable

from flax.core import Scope, Array, init, unfreeze, lift, nn

from absl.testing import absltest

import jax
from jax import random, numpy as jnp


def mlp_vmap(scope: Scope, x: Array,
             sizes: Sequence[int] = (8, 1),
             act_fn: Callable[[Array], Array] = nn.relu,
             share_params: bool = False):
  if share_params:
    dense_vmap = lift.vmap(nn.dense,
                           in_axes=(0, None),
                           variable_axes={'params': None},
                           split_rngs={'params': False})
  else:
    dense_vmap = lift.vmap(nn.dense,
                           in_axes=(0, None),
                           variable_axes={'params': 0},
                           split_rngs={'params': True})

  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(dense_vmap, prefix='hidden_')(x, size)
    x = act_fn(x)

  # output layer
  return scope.child(dense_vmap, 'out')(x, sizes[-1])


class VMapTest(absltest.TestCase):

  def test_vmap_shared(self):
    x = random.normal(random.PRNGKey(0), (1, 4))
    x = jnp.concatenate([x, x], 0)

    y, variables = init(mlp_vmap)(random.PRNGKey(1), x, share_params=True)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'hidden_0' : {'kernel': (4, 8), 'bias': (8,)},
        'out': {'kernel': (8, 1), 'bias': (1,)},
    })
    self.assertEqual(y.shape, (2, 1))
    self.assertTrue(jnp.allclose(y[0], y[1]))

  def test_vmap_unshared(self):
    x = random.normal(random.PRNGKey(0), (1, 4))
    x = jnp.concatenate([x, x], 0)

    y, variables = init(mlp_vmap)(random.PRNGKey(1), x, share_params=False)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'hidden_0': {'kernel': (2, 4, 8), 'bias': (2, 8)},
        'out': {'kernel': (2, 8, 1), 'bias': (2, 1)},
    })
    self.assertEqual(y.shape, (2, 1))
    self.assertFalse(jnp.allclose(y[0], y[1]))


if __name__ == '__main__':
  absltest.main()

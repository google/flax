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


from flax.core import Scope, Array, init, unfreeze, lift, nn

from absl.testing import absltest

import jax
from jax import random, numpy as jnp


def mlp_scan(scope: Scope, xs: Array,
             share_params: bool = False):

  scope.variable('counter', 'i', jnp.zeros, ())
  def body_fn(scope, c, x):
    counter = scope.variable('counter', 'i', jnp.zeros, ())
    counter.value += 1
    x = scope.child(nn.dense)(x, 1)
    return c, x

  if share_params:
    _, ys = lift.scan(
        body_fn,
        variable_carry='counter',
        variable_broadcast='params',
        split_rngs={'params': False})(scope, (), xs)
  else:
    _, ys = lift.scan(
        body_fn,
        variable_carry='counter',
        variable_axes={'params': 0},
        split_rngs={'params': True})(scope, (), xs)

  # output layer
  return ys


class ScanTest(absltest.TestCase):

  def test_scan_unshared_params(self):
    x = random.normal(random.PRNGKey(0), (1, 4))
    x = jnp.concatenate([x, x], 0)
    y, variables = init(mlp_scan)(random.PRNGKey(1), x, share_params=False)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(variables['counter']['i'], 2)
    self.assertEqual(param_shapes, {
      'dense_0': {'kernel': (2, 4, 1), 'bias': (2, 1)},
    })

    self.assertNotEqual(y[0], y[1])
    k1, k2 = variables['params']['dense_0']['kernel']
    self.assertFalse(jnp.allclose(k1, k2))

  def test_scan_shared_params(self):
    x = random.normal(random.PRNGKey(0), (1, 4))
    x = jnp.concatenate([x, x], 0)
    y, variables = init(mlp_scan)(random.PRNGKey(1), x, share_params=True)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(variables['counter']['i'], 2)
    self.assertEqual(param_shapes, {
      'dense_0': {'kernel': (4, 1), 'bias': (1,)},
    })

    self.assertEqual(y[0], y[1])


if __name__ == '__main__':
  absltest.main()

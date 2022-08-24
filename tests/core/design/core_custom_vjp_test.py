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
from functools import partial

from absl.testing import absltest

import numpy as np

from flax.core import Scope, Array, init, apply, unfreeze, lift, nn

import jax
from jax import random, numpy as jnp


def mlp_custom_grad(scope: Scope, x: Array,
                    sizes: Sequence[int] = (8, 1),
                    act_fn: Callable[[Array], Array] = nn.relu):

  f = nn.dense

  def fwd(scope, x, features):
    y, vjp_fn = lift.vjp(partial(f, features=features), scope, x)
    return y, vjp_fn

  def bwd(features, res, y_t):
    del features
    vjp_fn = res
    params_t, *input_t = vjp_fn(y_t)
    params_t = jax.tree_util.tree_map(jnp.sign, params_t)
    return (params_t, *input_t)

  dense_custom_grad = lift.custom_vjp(
      f, forward_fn=fwd, backward_fn=bwd, nondiff_argnums=(2,))

  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(dense_custom_grad, prefix='hidden_')(x, size)
    x = act_fn(x)

  # output layer
  return scope.child(dense_custom_grad, 'out')(x, sizes[-1])


class CustomVJPTest(absltest.TestCase):

  def test_custom_vjp(self):
    x = random.normal(random.PRNGKey(0), (1, 4))
    y, variables = init(mlp_custom_grad)(random.PRNGKey(1), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    loss_fn = lambda p, x: jnp.mean(apply(mlp_custom_grad)(p, x) ** 2)
    grad = jax.grad(loss_fn)(variables, x)
    grad_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, grad['params']))
    self.assertEqual(y.shape, (1, 1))
    expected_param_shapes = {
        'hidden_0': {'kernel': (4, 8), 'bias': (8,)},
        'out': {'kernel': (8, 1), 'bias': (1,)},
    }
    self.assertEqual(param_shapes, expected_param_shapes)
    self.assertEqual(grad_shapes, expected_param_shapes)
    for g in jax.tree_util.tree_leaves(grad):
      self.assertTrue(np.all(g == np.sign(g)))


if __name__ == '__main__':
  absltest.main()

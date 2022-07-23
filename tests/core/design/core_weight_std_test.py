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
from typing import Sequence

from flax.core import Scope, Array, init, apply, unfreeze, lift, nn

from absl.testing import absltest

import jax
from jax import random, numpy as jnp


def weight_std(fn, kernel_name='kernel', eps=1e-8):
  def std(variables):
    params = variables['params']
    assert kernel_name in params
    kernel = params[kernel_name]
    redux = tuple(range(kernel.ndim - 1))
    norm = jnp.square(kernel).sum(redux, keepdims=True)
    std_kernel = kernel / jnp.sqrt(norm + eps)
    params[kernel_name] = std_kernel
    return variables

  # map_variables handles a few of nasty edge cases here...
  # the transformed kind will be immutable inside fn
  # this way we avoid lost mutations to param
  # map_variables also avoids accidental reuse of rngs
  # and it makes sure that other state is updated correctly (not twice during init!)
  return lift.map_variables(fn, "params", std, init=True)

def mlp(scope: Scope, x: Array,
        sizes: Sequence[int] = (8, 1)):
  std_dense = weight_std(partial(
      nn.dense, kernel_init=nn.initializers.normal(stddev=1e5)))
  for size in sizes[:-1]:
    x = scope.child(std_dense, prefix='hidden_')(x, size)
  return scope.child(nn.dense, 'out')(x, sizes[-1])


class WeightStdTest(absltest.TestCase):

  def test_weight_std(self):
    x = random.normal(random.PRNGKey(0), (1, 4,))
    y, variables = init(mlp)(random.PRNGKey(1), x)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'hidden_0': {'kernel': (4, 8), 'bias': (8,)},
        'out': {'kernel': (8, 1), 'bias': (1,)},
    })
    self.assertEqual(y.shape, (1, 1))
    self.assertTrue(y.ravel() < 1.)

    y2 = apply(mlp)(variables, x)
    self.assertTrue(jnp.allclose(y, y2))


if __name__ == '__main__':
  absltest.main()

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

from dataclasses import dataclass
from typing import Any, Callable, Sequence, NamedTuple, Any

from absl.testing import absltest

from flax.core import Scope, Array, init, apply, unfreeze, nn
import jax
from jax import numpy as jnp, random

from jax.scipy.linalg import expm


Initializer = Any
Flow = Any


@dataclass
class DenseFlow:
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros_init()

  def params(self, scope: Scope, features: int):
    kernel = scope.param('kernel', self.kernel_init, (features, features))
    bias = scope.param('bias', self.bias_init, (features,))
    return kernel, bias

  def forward(self, scope: Scope, x: Array):
    kernel, bias = self.params(scope, x.shape[-1])
    return jnp.dot(
      x, expm(kernel)) + bias.reshape((1,) * (x.ndim - 1) + (-1,))

  def backward(self, scope: Scope, y: Array):
    kernel, bias = self.params(scope, y.shape[-1])
    return jnp.dot(
      y - bias.reshape((1,) * (y.ndim - 1) + (-1,)), expm(-kernel))


@dataclass
class StackFlow:
  flows: Sequence[Flow]

  def forward(self, scope: Scope, x: Array):
    for i, f in enumerate(self.flows):
      x = scope.child(f.forward, name=str(i))(x)
    return x

  def backward(self, scope: Scope, x: Array):
    for i, f in reversed(tuple(enumerate(self.flows))):
      x = scope.child(f.backward, name=str(i))(x)
    return x


class FlowTest(absltest.TestCase):

  def test_flow(self):
    x = jnp.ones((1, 3))
    flow = StackFlow((DenseFlow(),) * 3)
    y, variables = init(flow.forward)(random.PRNGKey(0), x)
    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(y.shape, (1, 3))
    self.assertEqual(param_shapes, {
        '0': {'kernel': (3, 3), 'bias': (3,)},
        '1': {'kernel': (3, 3), 'bias': (3,)},
        '2': {'kernel': (3, 3), 'bias': (3,)},
    })
    x_restored = apply(flow.backward)(variables, y)
    self.assertTrue(jnp.allclose(x, x_restored))


if __name__ == '__main__':
  absltest.main()

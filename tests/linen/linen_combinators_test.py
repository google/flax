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

"""Tests for flax.linen.combinators."""

from typing import Any, Optional, Sequence

from absl.testing import absltest

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class MLP(nn.Module):
  layer_sizes: Sequence[int]
  activation: Optional[Any] = None
  activation_final: Optional[Any] = None

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for layer_size in self.layer_sizes[:-1]:
      x = nn.Dense(features=layer_size, kernel_init=nn.initializers.ones)(x)
      if self.activation is not None:
        x = self.activation(x)
    x = nn.Dense(
        features=self.layer_sizes[-1], kernel_init=nn.initializers.ones)(
            x)
    if self.activation_final is None:
      return x
    return self.activation_final(x)


class SequentialTest(absltest.TestCase):

  def test_construction(self):
    sequential = nn.Sequential([nn.Dense(4), nn.Dense(2)])
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (3, 1, 5))
    params = sequential.init(key2, x)
    output = sequential.apply(params, x)
    self.assertEqual(output.shape, (3, 1, 2))

  def test_fails_if_layers_empty(self):
    sequential = nn.Sequential([])
    with self.assertRaisesRegex(ValueError,
                                'Empty Sequential module'):
      sequential.init(random.PRNGKey(42), jnp.ones((3, 5)))

  def test_same_output_as_mlp(self):
    sequential = nn.Sequential([
        nn.Dense(4, kernel_init=nn.initializers.ones),
        nn.Dense(8, kernel_init=nn.initializers.ones),
        nn.Dense(2, kernel_init=nn.initializers.ones)
    ])
    mlp = MLP(layer_sizes=[4, 8, 2])

    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (3, 5))
    params_1 = sequential.init(key2, x)
    params_2 = mlp.init(key2, x)

    output_1 = sequential.apply(params_1, x)
    output_2 = mlp.apply(params_2, x)
    np.testing.assert_array_equal(output_1, output_2)

  def test_same_output_as_mlp_with_activation(self):
    sequential = nn.Sequential([
        nn.Dense(4, kernel_init=nn.initializers.ones), nn.relu,
        nn.Dense(8, kernel_init=nn.initializers.ones), nn.relu,
        nn.Dense(2, kernel_init=nn.initializers.ones), nn.log_softmax
    ])

    mlp = MLP(
        layer_sizes=[4, 8, 2],
        activation=nn.relu,
        activation_final=nn.log_softmax)

    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (3, 5))
    params_1 = sequential.init(key2, x)
    params_2 = mlp.init(key2, x)

    output_1 = sequential.apply(params_1, x)
    output_2 = mlp.apply(params_2, x)
    np.testing.assert_array_equal(output_1, output_2)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for flax.linen.combinators."""

from typing import Any
from collections.abc import Sequence

import jax
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp
from jax import random

from flax import linen as nn

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class MLP(nn.Module):
  layer_sizes: Sequence[int]
  activation: Any | None = None
  activation_final: Any | None = None

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for layer_size in self.layer_sizes[:-1]:
      x = nn.Dense(
        features=layer_size, kernel_init=nn.initializers.ones_init()
      )(x)
      if self.activation is not None:
        x = self.activation(x)
    x = nn.Dense(
      features=self.layer_sizes[-1], kernel_init=nn.initializers.ones_init()
    )(x)
    if self.activation_final is None:
      return x
    return self.activation_final(x)


class AttentionTuple(nn.Module):
  num_heads: int = 2
  qkv_features: int = 16

  @nn.compact
  def __call__(self, query, key_value):
    output = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads, qkv_features=self.qkv_features
    )(query, key_value)
    return output, key_value


class AttentionDict(nn.Module):
  num_heads: int = 2
  qkv_features: int = 16

  @nn.compact
  def __call__(self, query, key_value):
    output = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads, qkv_features=self.qkv_features
    )(query, key_value)
    return dict(query=output, key_value=key_value)


class SequentialTest(absltest.TestCase):
  def test_construction(self):
    sequential = nn.Sequential([nn.Dense(4), nn.Dense(2)])
    key1, key2 = random.split(random.key(0), 2)
    x = random.uniform(key1, (3, 1, 5))
    params = sequential.init(key2, x)
    output = sequential.apply(params, x)
    self.assertEqual(output.shape, (3, 1, 2))

  def test_fails_if_layers_empty(self):
    sequential = nn.Sequential([])
    with self.assertRaisesRegex(ValueError, 'Empty Sequential module'):
      sequential.init(random.key(42), jnp.ones((3, 5)))

  def test_same_output_as_mlp(self):
    sequential = nn.Sequential(
      [
        nn.Dense(4, kernel_init=nn.initializers.ones_init()),
        nn.Dense(8, kernel_init=nn.initializers.ones_init()),
        nn.Dense(2, kernel_init=nn.initializers.ones_init()),
      ]
    )
    mlp = MLP(layer_sizes=[4, 8, 2])

    key1, key2 = random.split(random.key(0), 2)
    x = random.uniform(key1, (3, 5))
    params_1 = sequential.init(key2, x)
    params_2 = mlp.init(key2, x)

    output_1 = sequential.apply(params_1, x)
    output_2 = mlp.apply(params_2, x)
    np.testing.assert_array_equal(output_1, output_2)

  def test_same_output_as_mlp_with_activation(self):
    sequential = nn.Sequential(
      [
        nn.Dense(4, kernel_init=nn.initializers.ones_init()),
        nn.relu,
        nn.Dense(8, kernel_init=nn.initializers.ones_init()),
        nn.relu,
        nn.Dense(2, kernel_init=nn.initializers.ones_init()),
        nn.log_softmax,
      ]
    )

    mlp = MLP(
      layer_sizes=[4, 8, 2],
      activation=nn.relu,
      activation_final=nn.log_softmax,
    )

    key1, key2 = random.split(random.key(0), 2)
    x = random.uniform(key1, (3, 5))
    params_1 = sequential.init(key2, x)
    params_2 = mlp.init(key2, x)

    output_1 = sequential.apply(params_1, x)
    output_2 = mlp.apply(params_2, x)
    np.testing.assert_array_equal(output_1, output_2)

  def test_tuple_output(self):
    sequential = nn.Sequential(
      [
        AttentionTuple(),
        AttentionTuple(),
      ]
    )

    key1, key2, key3 = random.split(random.key(0), 3)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key2, (9, 5))
    params_1 = sequential.init(key3, query, key_value)
    outputs = sequential.apply(params_1, query, key_value)
    np.testing.assert_equal(len(outputs), 2)
    out_query, out_key_value = outputs
    np.testing.assert_equal(out_query.shape, (3, 5))
    np.testing.assert_equal(out_key_value.shape, (9, 5))

  def test_dict_output(self):
    sequential = nn.Sequential(
      [
        AttentionDict(),
        AttentionDict(),
      ]
    )

    key1, key2, key3 = random.split(random.key(0), 3)
    query = random.uniform(key1, (3, 5))
    key_value = random.uniform(key2, (9, 5))
    params_1 = sequential.init(key3, query, key_value)
    outputs = sequential.apply(params_1, query, key_value)
    np.testing.assert_equal(len(outputs), 2)
    out_query, out_key_value = outputs['query'], outputs['key_value']
    np.testing.assert_equal(out_query.shape, (3, 5))
    np.testing.assert_equal(out_key_value.shape, (9, 5))

  def test_sequential_compact(self):

    mlp = nn.Sequential([
        lambda x: nn.Dense(x.shape[-1])(x),
        nn.relu,
        lambda x: nn.Dense(x.shape[-1])(x),
        nn.relu,
        lambda x: nn.Dense(x.shape[-1])(x),
    ])

    params = mlp.init(random.key(0), jnp.ones((3, 5)))['params']

    self.assertIn('Dense_0', params)
    self.assertIn('Dense_1', params)
    self.assertIn('Dense_2', params)


if __name__ == '__main__':
  absltest.main()

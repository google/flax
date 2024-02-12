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

import typing as tp

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx
from flax.typing import Dtype


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16], param_dtype=[jnp.float32, jnp.float16]
  )
  def test_nnx_linen_batchnorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
  ):
    class NNXModel(nnx.Module):
      def __init__(self, dtype, param_dtype, rngs):
        self.norm_layer = nnx.BatchNorm(
          3,
          use_running_average=False,
          dtype=dtype,
          param_dtype=param_dtype,
          rngs=rngs,
        )
        self.linear = nnx.Linear(
          3, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x):
        x = self.norm_layer(x)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32

      def setup(self):
        self.norm_layer = linen.BatchNorm(
          use_running_average=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x):
        x = self.norm_layer(x)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jnp.ones((1, 3))

    linen_model = LinenModel(dtype=dtype, param_dtype=param_dtype)
    variables = linen_model.init(jax.random.key(0), x)
    linen_out, batch_stats = linen_model.apply(
      variables, x, mutable=['batch_stats']
    )

    nnx_model = NNXModel(dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    nnx_model.linear.kernel = variables['params']['linear']['kernel']
    nnx_model.linear.bias = variables['params']['linear']['bias']

    nnx_out = nnx_model(x)
    assert_array_equal(linen_out, nnx_out)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16], param_dtype=[jnp.float32, jnp.float16]
  )
  def test_nnx_linen_layernorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
  ):
    class NNXModel(nnx.Module):
      def __init__(self, dtype, param_dtype, rngs):
        self.norm_layer = nnx.LayerNorm(
          3, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.linear = nnx.Linear(
          3, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x):
        x = self.norm_layer(x)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32

      def setup(self):
        self.norm_layer = linen.LayerNorm(
          dtype=self.dtype, param_dtype=self.param_dtype
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x):
        x = self.norm_layer(x)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jnp.ones((1, 3))

    linen_model = LinenModel(dtype=dtype, param_dtype=param_dtype)
    variables = linen_model.init(jax.random.key(0), x)
    linen_out = linen_model.apply(variables, x)

    nnx_model = NNXModel(dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    nnx_model.linear.kernel = variables['params']['linear']['kernel']
    nnx_model.linear.bias = variables['params']['linear']['bias']

    nnx_out = nnx_model(x)
    assert_array_equal(linen_out, nnx_out)

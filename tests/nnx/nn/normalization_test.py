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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from flax import linen
from flax import nnx
from flax.typing import Dtype


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    use_fast_variance=[True, False],
    mask=[None, np.array([True, False, True, False, True])],
  )
  def test_nnx_linen_batchnorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    use_fast_variance: bool,
    mask: tp.Optional[np.ndarray],
  ):
    class NNXModel(nnx.Module):
      __data__ = ('norm_layer', 'linear')
      def __init__(self, dtype, param_dtype, use_fast_variance, rngs):
        self.norm_layer = nnx.BatchNorm(
          5,
          use_running_average=False,
          dtype=dtype,
          param_dtype=param_dtype,
          use_fast_variance=use_fast_variance,
          rngs=rngs,
        )
        self.linear = nnx.Linear(
          5, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32
      use_fast_variance: bool = True

      def setup(self):
        self.norm_layer = linen.BatchNorm(
          use_running_average=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          use_fast_variance=use_fast_variance,
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jax.random.normal(jax.random.key(0), (10, 5))

    linen_model = LinenModel(
      dtype=dtype, param_dtype=param_dtype, use_fast_variance=use_fast_variance
    )
    variables = linen_model.init(jax.random.key(1), x)
    linen_out, batch_stats = linen_model.apply(
      variables, x, mask=mask, mutable=['batch_stats']
    )

    nnx_model = NNXModel(
      dtype=dtype,
      param_dtype=param_dtype,
      use_fast_variance=use_fast_variance,
      rngs=rngs,
    )
    nnx_model.linear.kernel.value = variables['params']['linear']['kernel']
    nnx_model.linear.bias.value = variables['params']['linear']['bias']

    nnx_out = nnx_model(x, mask=mask)
    np.testing.assert_array_equal(linen_out, nnx_out)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    use_fast_variance=[True, False],
    mask=[None, np.array([True, False, True, False, True])],
  )
  def test_nnx_linen_layernorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    use_fast_variance: bool,
    mask: tp.Optional[np.ndarray],
  ):
    class NNXModel(nnx.Module):
      __data__ = ('norm_layer', 'linear')
      def __init__(self, dtype, param_dtype, use_fast_variance, rngs):
        self.norm_layer = nnx.LayerNorm(
          5,
          dtype=dtype,
          param_dtype=param_dtype,
          use_fast_variance=use_fast_variance,
          rngs=rngs,
        )
        self.linear = nnx.Linear(
          5, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32
      use_fast_variance: bool = True

      def setup(self):
        self.norm_layer = linen.LayerNorm(
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          use_fast_variance=self.use_fast_variance,
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jax.random.normal(jax.random.key(0), (10, 5))

    linen_model = LinenModel(
      dtype=dtype, param_dtype=param_dtype, use_fast_variance=use_fast_variance
    )
    variables = linen_model.init(jax.random.key(1), x)
    linen_out = linen_model.apply(variables, x, mask=mask)
    assert isinstance(linen_out, jax.Array)

    nnx_model = NNXModel(
      dtype=dtype,
      param_dtype=param_dtype,
      use_fast_variance=use_fast_variance,
      rngs=rngs,
    )
    nnx_model.linear.kernel.value = variables['params']['linear']['kernel']
    nnx_model.linear.bias.value = variables['params']['linear']['bias']

    nnx_out = nnx_model(x, mask=mask)
    np.testing.assert_array_equal(linen_out, nnx_out)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    use_fast_variance=[True, False],
    mask=[None, np.array([True, False, True, False, True])],
  )
  def test_nnx_linen_rmsnorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    use_fast_variance: bool,
    mask: tp.Optional[np.ndarray],
  ):
    class NNXModel(nnx.Module):
      __data__ = ('norm_layer', 'linear')
      def __init__(self, dtype, param_dtype, use_fast_variance, rngs):
        self.norm_layer = nnx.RMSNorm(
          5,
          dtype=dtype,
          param_dtype=param_dtype,
          use_fast_variance=use_fast_variance,
          rngs=rngs,
        )
        self.linear = nnx.Linear(
          5, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32
      use_fast_variance: bool = True

      def setup(self):
        self.norm_layer = linen.RMSNorm(
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          use_fast_variance=self.use_fast_variance,
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jax.random.normal(jax.random.key(0), (10, 5))

    linen_model = LinenModel(
      dtype=dtype, param_dtype=param_dtype, use_fast_variance=use_fast_variance
    )
    variables = linen_model.init(jax.random.key(1), x)
    linen_out = linen_model.apply(variables, x, mask=mask)

    nnx_model = NNXModel(
      dtype=dtype,
      param_dtype=param_dtype,
      use_fast_variance=use_fast_variance,
      rngs=rngs,
    )
    nnx_model.linear.kernel.value = variables['params']['linear']['kernel']
    nnx_model.linear.bias.value = variables['params']['linear']['bias']

    nnx_out = nnx_model(x, mask=mask)
    assert isinstance(linen_out, jax.Array)
    np.testing.assert_array_equal(linen_out, nnx_out)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    use_fast_variance=[True, False],
    mask=[None, np.array([True, False, True, False, True, False])],
  )
  def test_nnx_linen_groupnorm_equivalence(
    self,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    use_fast_variance: bool,
    mask: tp.Optional[np.ndarray],
  ):
    class NNXModel(nnx.Module):
      __data__ = ('norm_layer', 'linear')
      def __init__(self, dtype, param_dtype, use_fast_variance, rngs):
        self.norm_layer = nnx.GroupNorm(
          6,
          num_groups=3,
          dtype=dtype,
          param_dtype=param_dtype,
          use_fast_variance=use_fast_variance,
          rngs=rngs,
        )
        self.linear = nnx.Linear(
          6, 4, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    class LinenModel(linen.Module):
      dtype: tp.Optional[Dtype] = None
      param_dtype: Dtype = jnp.float32
      use_fast_variance: bool = True

      def setup(self):
        self.norm_layer = linen.GroupNorm(
          num_groups=3,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          use_fast_variance=self.use_fast_variance,
        )
        self.linear = linen.Dense(
          4, dtype=self.dtype, param_dtype=self.param_dtype
        )

      def __call__(self, x, *, mask=None):
        x = self.norm_layer(x, mask=mask)
        x = self.linear(x)
        return x

    rngs = nnx.Rngs(42)
    x = jax.random.normal(jax.random.key(0), (10, 6))

    linen_model = LinenModel(
      dtype=dtype, param_dtype=param_dtype, use_fast_variance=use_fast_variance
    )
    variables = linen_model.init(jax.random.key(1), x)
    linen_out = linen_model.apply(variables, x, mask=mask)

    nnx_model = NNXModel(
      dtype=dtype,
      param_dtype=param_dtype,
      use_fast_variance=use_fast_variance,
      rngs=rngs,
    )
    nnx_model.linear.kernel.value = variables['params']['linear']['kernel']
    nnx_model.linear.bias.value = variables['params']['linear']['bias']

    nnx_out = nnx_model(x, mask=mask)
    assert isinstance(linen_out, jax.Array)
    np.testing.assert_array_equal(linen_out, nnx_out)


if __name__ == '__main__':
  absltest.main()

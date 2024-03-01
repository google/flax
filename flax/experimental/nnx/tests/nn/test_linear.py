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
from jax.lax import Precision
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx
from flax.typing import Dtype, PrecisionLike


class TestLinearGeneral:
  def test_basic(self):
    module = nnx.LinearGeneral(2, 3, rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)
    assert module.kernel.value.shape == (2, 3)
    assert module.bias.value is not None
    assert module.bias.value.shape == (3,)

  def test_basic_multi_features(self):
    module = nnx.LinearGeneral(2, (3, 4), rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3, 4)
    assert module.kernel.value.shape == (2, 3, 4)
    assert module.bias.value is not None
    assert module.bias.value.shape == (3, 4)


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
  )
  def test_nnx_linen_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    OUT_FEATURES = 64

    x = jax.numpy.ones((1, IN_FEATURES))
    model_nnx = nnx.Linear.create_abstract(
      IN_FEATURES,
      OUT_FEATURES,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      rngs=rngs,
    )
    model = linen.Dense(
      OUT_FEATURES,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
    )
    variables = model.init(key, x)
    model_nnx.kernel.value = variables['params']['kernel']
    if use_bias:
      model_nnx.bias.value = variables['params']['bias']

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)

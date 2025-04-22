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
from jax.lax import Precision
import numpy as np

from flax import linen
from flax import nnx
from flax.typing import Dtype, PrecisionLike, Shape


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
  def test_nnx_linear_equivalence(
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
    model_nnx = nnx.eval_shape(
      lambda rngs: nnx.Linear(
        IN_FEATURES,
        OUT_FEATURES,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        rngs=rngs,
      ),
      rngs,
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
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  @parameterized.product(
    einsum_str=['defab,bcef->adefc', 'd...ab,bc...->ad...c'],
    bias_shape=[None, (6, 7, 5)],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
  )
  def test_nnx_einsum_equivalence(
    self,
    einsum_str,
    bias_shape: tp.Optional[Shape],
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    INPUT_SHAPE = (8, 6, 7, 3, 4)
    KERNEL_SHAPE = (4, 5, 6, 7)

    x = jax.random.normal(key, INPUT_SHAPE)
    model_nnx = nnx.Einsum(
      einsum_str,
      KERNEL_SHAPE,
      bias_shape,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      rngs=rngs,
    )
    model = linen.Einsum(
      KERNEL_SHAPE,
      einsum_str,
      use_bias=True if bias_shape is not None else False,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
    )

    variables = model.init(key, x)
    variables['params']['kernel'] = model_nnx.kernel.value
    if bias_shape is not None:
      assert model_nnx.bias is not None
      variables['params']['bias'] = model_nnx.bias.value
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key, x)
    model_nnx.kernel.value = variables['params']['kernel']
    if bias_shape is not None:
      assert model_nnx.bias is not None
      model_nnx.bias.value = variables['params']['bias']
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  def test_einsum_op(self):
    def custom_einsum(*args, **kwargs):
      out = jnp.einsum(*args, **kwargs)
      return out.reshape((1, *out.shape))
    model = nnx.Einsum('ab,bc->ac', (3, 4), einsum_op=custom_einsum,
                       rngs=nnx.Rngs(42))
    y = model(jnp.ones((2, 3)))
    assert y.shape == (1, 2, 4)


if __name__ == '__main__':
  absltest.main()

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

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from flax import linen
from flax import nnx


class TestPReLU(parameterized.TestCase):
  def test_positive_inputs(self):
    prelu = nnx.PReLU()
    x = jnp.array([1.0, 2.0, 3.0])
    y = prelu(x)
    np.testing.assert_allclose(y, x, rtol=1e-6)

  def test_negative_inputs(self):
    prelu = nnx.PReLU(negative_slope_init=0.1)
    x = jnp.array([-1.0, -2.0, -3.0])
    y = prelu(x)
    expected = jnp.array([-0.1, -0.2, -0.3])
    np.testing.assert_allclose(y, expected, rtol=1e-6)

  def test_mixed_inputs(self):
    prelu = nnx.PReLU(negative_slope_init=0.25)
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = prelu(x)
    expected = jnp.array([-0.5, -0.25, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(y, expected, rtol=1e-6)

  def test_scalar_input(self):
    prelu = nnx.PReLU(negative_slope_init=0.1)
    x = jnp.array(-2.0)
    y = prelu(x)
    np.testing.assert_allclose(y, jnp.array(-0.2), rtol=1e-6)

  def test_multidimensional_input(self):
    prelu = nnx.PReLU(negative_slope_init=0.1)
    x = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
    y = prelu(x)
    expected = jnp.array([[-0.1, 2.0], [3.0, -0.4]])
    np.testing.assert_allclose(y, expected, rtol=1e-6)

  def test_negative_slope_init(self):
    prelu_default = nnx.PReLU()
    np.testing.assert_allclose(prelu_default.negative_slope[...], 0.01, rtol=1e-6)
    prelu_custom = nnx.PReLU(negative_slope_init=0.5)
    np.testing.assert_allclose(prelu_custom.negative_slope[...], 0.5, rtol=1e-6)

  def test_negative_slope_is_param(self):
    prelu = nnx.PReLU()
    self.assertIsInstance(prelu.negative_slope, nnx.Param)

  def test_zero_negative_slope_behaves_like_relu(self):
    prelu = nnx.PReLU(negative_slope_init=0.0)
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = prelu(x)
    expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(y, expected, rtol=1e-6)

  def test_negative_slope_of_one_behaves_like_identity(self):
    prelu = nnx.PReLU(negative_slope_init=1.0)
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = prelu(x)
    np.testing.assert_allclose(y, x, rtol=1e-6)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_dtypes(self, dtype, param_dtype):
    prelu = nnx.PReLU(dtype=dtype, param_dtype=param_dtype)
    self.assertEqual(prelu.negative_slope[...].dtype, param_dtype)
    x = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    y = prelu(x)
    self.assertEqual(y.dtype, dtype)

  def test_dtype_none_casts_param_to_input_dtype(self):
    prelu = nnx.PReLU(param_dtype=jnp.float32)
    x = jnp.array([-1.0, 1.0], dtype=jnp.float16)
    y = prelu(x)
    self.assertEqual(y.dtype, jnp.float16)

  def test_custom_promote_dtype(self):
    called_with = {}

    def custom_promote(arrays, *, dtype=None, **kwargs):
      called_with['dtype'] = dtype
      return tuple(jnp.asarray(a, dtype=dtype) for a in arrays)

    prelu = nnx.PReLU(
      negative_slope_init=0.1, dtype=jnp.float32, promote_dtype=custom_promote
    )
    x = jnp.array([-1.0, 1.0], dtype=jnp.float16)
    y = prelu(x)
    self.assertIn('dtype', called_with, 'custom_promote was never called')
    self.assertEqual(called_with['dtype'], jnp.float32)
    self.assertEqual(y.dtype, jnp.float32)

  def test_negative_slope_metadata(self):
    prelu = nnx.PReLU(negative_slope_metadata={'my_tag': 'test'})
    self.assertEqual(prelu.negative_slope.my_tag, 'test')

  def test_as_submodule(self):
    class MLP(nnx.Module):
      def __init__(self):
        self.linear = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
        self.act = nnx.PReLU(negative_slope_init=0.1)

      def __call__(self, x):
        return self.act(self.linear(x))

    model = MLP()
    x = jnp.ones((1, 3))
    y = model(x)
    self.assertEqual(y.shape, (1, 2))
    self.assertIsInstance(model.act.negative_slope, nnx.Param)


class TestPReLUConsistency(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_equivalence(self, dtype, param_dtype):
    key = jax.random.key(42)
    x = jnp.linspace(-10, 10, 20, dtype=dtype)
    negative_slope_init = 0.02
    # Linen PReLU does not accept a dtype argument, so both models
    # are created without it and rely on input dtype for casting.
    nnx_prelu = nnx.PReLU(
      negative_slope_init=negative_slope_init, param_dtype=param_dtype
    )
    linen_prelu = linen.PReLU(
      negative_slope_init=negative_slope_init, param_dtype=param_dtype
    )

    variables = linen_prelu.init(key, x)
    # Both use the same constant initializer (negative_slope_init=0.02),
    # verify the values match before comparing outputs.
    rtol = 1e-3 if dtype == jnp.float16 or param_dtype == jnp.float16 else 1e-6
    np.testing.assert_allclose(
      variables['params']['negative_slope'],
      nnx_prelu.negative_slope[...],
      rtol=rtol,
    )
    expected = linen_prelu.apply(variables, x)
    output = nnx_prelu(x)
    np.testing.assert_allclose(output, expected, rtol=rtol)

    # Check gradients
    @jax.jit
    def nnx_loss_function(model):
      return model(x).mean()

    @jax.jit
    def linen_loss_function(variables):
      return linen_prelu.apply(variables, x).mean()

    expected_loss, expected_grads = jax.value_and_grad(linen_loss_function)(
      variables
    )
    loss, grads = jax.value_and_grad(nnx_loss_function)(nnx_prelu)

    np.testing.assert_allclose(loss, expected_loss, rtol=rtol)
    np.testing.assert_allclose(
      expected_grads['params']['negative_slope'],
      grads.negative_slope[...],
      rtol=rtol,
    )


if __name__ == '__main__':
  absltest.main()

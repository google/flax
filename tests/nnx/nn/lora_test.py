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

from flax import nnx


class TestLoRA(parameterized.TestCase):
  def test_basic(self):
    module = nnx.LoRA(3, 2, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = module(x)

    self.assertEqual(y.shape, (1, 4))
    self.assertEqual(module.lora_a.shape, (3, 2))
    self.assertEqual(module.lora_b.shape, (2, 4))
    np.testing.assert_allclose(
      y, x @ module.lora_a[...] @ module.lora_b[...], rtol=1e-6
    )

  def test_lora_base_module(self):
    rngs = nnx.Rngs(0)
    linear = nnx.Linear(3, 4, use_bias=False, rngs=rngs)
    module = nnx.LoRA(3, 2, 4, base_module=linear, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = module(x)

    self.assertEqual(y.shape, (1, 4))
    self.assertIs(module.base_module, linear)
    self.assertEqual(module.base_module.kernel.shape, (3, 4))
    self.assertIsNone(module.base_module.bias)
    self.assertEqual(module.lora_a.shape, (3, 2))
    self.assertEqual(module.lora_b.shape, (2, 4))
    # dtype=None: all inputs are float32, so promote_dtype is a no-op
    np.testing.assert_allclose(
      y,
      x @ linear.kernel[...] + x @ module.lora_a[...] @ module.lora_b[...],
      rtol=1e-6,
    )

  def test_layer_swap_lora(self):
    class MLP(nnx.Module):
      def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

      def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)

    rngs = nnx.Rngs(0)
    model = MLP(3, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = model(x)

    # Replace one of the linear layers as LoRA linear layer.
    model.linear2 = nnx.LoRA(3, 4, 3, base_module=model.linear2, rngs=rngs)
    lora_y = model(x)

    self.assertEqual(y.shape, (1, 3))
    self.assertEqual(lora_y.shape, (1, 3))
    # lora_b is zero-initialized, so LoRA delta is zero
    np.testing.assert_allclose(y, lora_y)

  def test_layer_swap_loralinear(self):
    class MLP(nnx.Module):
      def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

      def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)

    rngs = nnx.Rngs(0)
    model = MLP(3, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = model(x)

    # Replace one of the linear layers as LoRA linear layer.
    _, state = nnx.split(
      model.linear2
    )  # To keep the kernel and bias of linear2
    model.linear2 = nnx.LoRALinear(3, 3, lora_rank=4, rngs=rngs)
    nnx.update(model.linear2, state)
    lora_y = model(x)

    self.assertEqual(y.shape, (1, 3))
    self.assertEqual(lora_y.shape, (1, 3))
    # lora_b is zero-initialized, so LoRA delta is zero
    np.testing.assert_allclose(y, lora_y)

  def test_lora_param_type(self):
    rngs = nnx.Rngs(0)
    model = nnx.LoRA(3, 4, 2, lora_param_type=nnx.LoRAParam, rngs=rngs)
    _, lora_params, params = nnx.split(model, nnx.LoRAParam, nnx.Param)
    self.assertFalse(params)
    self.assertIn('lora_a', lora_params)
    self.assertIn('lora_b', lora_params)
    np.testing.assert_allclose(lora_params['lora_a'][...], model.lora_a[...])
    np.testing.assert_allclose(lora_params['lora_b'][...], model.lora_b[...])

    model = nnx.LoRA(3, 4, 2, lora_param_type=nnx.Param, rngs=rngs)
    _, params, lora_params = nnx.split(model, nnx.Param, nnx.LoRAParam)
    self.assertIn('lora_a', params)
    self.assertIn('lora_b', params)
    np.testing.assert_allclose(params['lora_a'][...], model.lora_a[...])
    np.testing.assert_allclose(params['lora_b'][...], model.lora_b[...])
    self.assertFalse(lora_params)

  def test_rank_one(self):
    module = nnx.LoRA(4, 1, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 4))
    y = module(x)

    self.assertEqual(module.lora_a.shape, (4, 1))
    self.assertEqual(module.lora_b.shape, (1, 4))
    self.assertEqual(y.shape, (2, 4))
    np.testing.assert_allclose(
      y, x @ module.lora_a[...] @ module.lora_b[...], rtol=1e-6
    )

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_dtypes(self, dtype, param_dtype):
    rngs = nnx.Rngs(0)
    model = nnx.LoRA(
      3, 4, 2, dtype=dtype, param_dtype=param_dtype, rngs=rngs
    )
    self.assertEqual(model.lora_a[...].dtype, param_dtype)
    self.assertEqual(model.lora_b[...].dtype, param_dtype)

    x = jnp.ones((1, 3), dtype=jnp.float32)
    y = model(x)
    self.assertEqual(y.dtype, dtype)

    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )
    # promote_dtype casts all arrays to the explicit dtype
    x_p = x.astype(dtype)
    lora_a_p = model.lora_a[...].astype(dtype)
    lora_b_p = model.lora_b[...].astype(dtype)
    expected = x_p @ lora_a_p @ lora_b_p
    np.testing.assert_allclose(y, expected, rtol=rtol)

  def test_initial_output_zero(self):
    module = nnx.LoRA(4, 2, 3, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(42), (2, 4))
    y = module(x)

    # lora_b is zeros so output is exactly zero
    np.testing.assert_array_equal(y, jnp.zeros((2, 3)))

  def test_gradient_flow(self):
    # b_initializer=ones so lora_b != 0, ensuring lora_a also gets
    # non-zero gradient (grad_lora_a depends on lora_b value).
    model = nnx.LoRA(
      3,
      2,
      4,
      b_initializer=nnx.initializers.ones,
      rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (1, 3))

    def loss_fn(model):
      return jnp.sum(model(x))

    _, grad = nnx.value_and_grad(loss_fn)(model)
    self.assertTrue(jnp.any(grad.lora_a[...] != 0))
    self.assertTrue(jnp.any(grad.lora_b[...] != 0))

  def test_gradient_flow_with_frozen_base(self):
    rngs = nnx.Rngs(0)
    linear = nnx.Linear(3, 4, use_bias=False, rngs=rngs)
    model = nnx.LoRA(
      3,
      2,
      4,
      base_module=linear,
      b_initializer=nnx.initializers.ones,
      rngs=rngs,
    )
    x = jax.random.normal(jax.random.key(0), (1, 3))

    def loss_fn(model):
      return jnp.sum(model(x))

    # Full gradient: both LoRA and base params get gradients
    _, full_grad = nnx.value_and_grad(loss_fn)(model)
    self.assertTrue(jnp.any(full_grad.lora_a[...] != 0))
    self.assertTrue(jnp.any(full_grad.lora_b[...] != 0))
    # Guard: base kernel receives gradient in the full (unfrozen) case
    self.assertTrue(jnp.any(full_grad.base_module.kernel[...] != 0))

    # DiffState filtered gradient: only LoRAParam is differentiated
    diff_filter = nnx.DiffState(0, nnx.LoRAParam)
    _, filtered_grad = nnx.value_and_grad(
      loss_fn, argnums=diff_filter
    )(model)
    # LoRA params still receive gradients
    self.assertTrue(jnp.any(filtered_grad.lora_a[...] != 0))
    self.assertTrue(jnp.any(filtered_grad.lora_b[...] != 0))
    # Base kernel is excluded from gradient state (frozen)
    self.assertFalse(hasattr(filtered_grad, 'base_module'))

  @parameterized.product(
    lora_dtype=[jnp.float32, jnp.float16],
    lora_param_dtype=[jnp.float32, jnp.float16],
  )
  def test_lora_linear_dtypes(self, lora_dtype, lora_param_dtype):
    model = nnx.LoRALinear(
      3,
      4,
      lora_rank=2,
      lora_dtype=lora_dtype,
      lora_param_dtype=lora_param_dtype,
      rngs=nnx.Rngs(0),
    )
    self.assertEqual(model.lora.lora_a[...].dtype, lora_param_dtype)
    self.assertEqual(model.lora.lora_b[...].dtype, lora_param_dtype)

    x = jnp.ones((1, 3), dtype=jnp.float32)
    y = model(x)
    self.assertEqual(y.shape, (1, 4))

    # Verify LoRA contribution has correct dtype
    lora_out = model.lora(x)
    self.assertEqual(lora_out.dtype, lora_dtype)

  def test_noncallable_base_module_raises(self):
    model = nnx.LoRA(
      3, 2, 4, base_module=object(), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((1, 3))
    with self.assertRaisesRegex(ValueError, 'must be callable'):
      model(x)


if __name__ == '__main__':
  absltest.main()

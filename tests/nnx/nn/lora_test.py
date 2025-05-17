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
from absl.testing import absltest
import numpy as np

from flax import nnx
from jax import numpy as jnp


class TestLora(absltest.TestCase):
  def test_basic(self):
    module = nnx.LoRA(3, 2, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = module(x)

    assert y.shape == (1, 4)
    assert module.lora_a.value.shape == (3, 2)
    assert module.lora_b.value.shape == (2, 4)
    np.testing.assert_allclose(y, x @ module.lora_a.value @ module.lora_b.value)

  def test_lora_base_module(self):
    rngs = nnx.Rngs(0)
    linear = nnx.Linear(3, 4, use_bias=False, rngs=rngs)
    module = nnx.LoRA(3, 2, 4, base_module=linear, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (1, 3))
    y = module(x)

    assert y.shape == (1, 4)
    assert module.base_module == linear
    assert module.base_module.kernel.value.shape == (3, 4)
    assert module.base_module.bias == None
    assert module.lora_a.value.shape == (3, 2)
    assert module.lora_b.value.shape == (2, 4)
    np.testing.assert_allclose(
      y, x @ linear.kernel.value + x @ module.lora_a.value @ module.lora_b.value
    )

  def test_layer_swap_lora(self):
    class MLP(nnx.Module):
      __data__ = 'auto'
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

    assert y.shape == (1, 3)
    assert lora_y.shape == (1, 3)
    np.testing.assert_allclose(y, lora_y)
    a, b = model.linear2.lora_a.value, model.linear2.lora_b.value
    np.testing.assert_allclose(y + model.linear1(x) @ a @ b, lora_y)

  def test_layer_swap_loralinear(self):
    class MLP(nnx.Module):
      __data__ = 'auto'
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

    assert y.shape == (1, 3)
    assert lora_y.shape == (1, 3)
    np.testing.assert_allclose(y, lora_y)
    a, b = model.linear2.lora.lora_a.value, model.linear2.lora.lora_b.value
    np.testing.assert_allclose(y + model.linear1(x) @ a @ b, lora_y)

  def test_lora_param_type(self):
    rngs = nnx.Rngs(0)
    model = nnx.LoRA(3, 4, 2, lora_param_type=nnx.LoRAParam, rngs=rngs)
    _, lora_params, params = nnx.split(model, nnx.LoRAParam, nnx.Param)
    assert params == {}
    assert ('lora_a' in lora_params) and ('lora_b' in lora_params)
    np.testing.assert_allclose(lora_params['lora_a'].value, model.lora_a.value)

    model = nnx.LoRA(3, 4, 2, lora_param_type=nnx.Param, rngs=rngs)
    _, params, lora_params = nnx.split(model, nnx.Param, nnx.LoRAParam)
    assert ('lora_a' in params) and ('lora_b' in params)
    np.testing.assert_allclose(params['lora_a'].value, model.lora_a.value)
    assert lora_params == {}

  def test_dtype(self):
    rngs = nnx.Rngs(0)
    model = nnx.LoRA(3, 4, 2, dtype=jnp.float16, param_dtype=jnp.float32,
                     rngs=rngs)
    assert model.lora_a.value.dtype == jnp.float32
    y = model(jnp.ones((1, 3)).astype(jnp.float32))
    assert y.dtype == jnp.float16


if __name__ == '__main__':
  absltest.main()

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

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

class TestEMA(parameterized.TestCase):

  def test_ema_initialization_and_update(self):
    model = nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(0))
    initial_kernel = jnp.copy(model.kernel[...])

    ema = nnx.EMA(model, decay=0.9)

    np.testing.assert_allclose(ema.params.kernel, initial_kernel)

    def double(param):
      param[...] *= 2.0

    jax.tree.map(double, model, is_leaf=lambda x: isinstance(x, nnx.Variable))

    ema.update(model)
    expected = 0.9 * initial_kernel + 0.1 * (2.0 * initial_kernel)

    np.testing.assert_allclose(ema.params.kernel[...], expected)

  def test_ema_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('At least 4 devices required')

    mesh = jax.make_mesh(
        (2, 2), ('row', 'col'),
        axis_types=(jax.sharding.AxisType.Auto,
                    jax.sharding.AxisType.Auto),
    )
    with jax.set_mesh(mesh):
      model = nnx.Linear(
          4, 2, rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(),
              sharding=('row', 'col'),
          ),
          use_bias=False,
      )
      ema = nnx.EMA(model, decay=0.9)

    # EMA params should have the same sharding as the model variables.
    self.assertTrue(
        ema.params.kernel.sharding.is_equivalent_to(
            model.kernel.sharding,
            ndim=2,
        )
    )

  def test_ema_example(self):
    def loss_fn(model, x, y):
      return jnp.mean((model(x) - y) ** 2)

    rngs = nnx.Rngs(0)
    x = rngs.normal((1, 2))
    y = rngs.normal((1, 3))

    model = nnx.Linear(2, 3, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    ema = nnx.EMA(model, decay=0.9)
    ema_model = ema.apply_to(model)
    original_kernel = ema_model.kernel[...]

    @nnx.jit
    def train_step(model, optimizer, ema, x, y):
      loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
      optimizer.update(model, grads)
      ema.update(model)
      return loss

    train_step(model, optimizer, ema, x, y)

    self.assertIsInstance(ema_model.kernel, nnx.Param)
    self.assertIs(ema_model.kernel, ema.params.kernel)
    self.assertIsNot(ema_model.kernel, model.kernel)
    self.assertFalse(jnp.allclose(ema_model.kernel[...], original_kernel))

  def test_ema_apply_to(self):
    model = nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(0))
    ema = nnx.EMA(model, decay=0.9)
    ema_model = ema.apply_to(model)

    def double(param):
      param[...] *= 2.0

    jax.tree.map(double, model, is_leaf=lambda x: isinstance(x, nnx.Variable))

    ema.update(model)

    np.testing.assert_allclose(ema_model.kernel[...], ema.params.kernel[...])
    self.assertFalse(jnp.allclose(ema_model.kernel[...], model.kernel[...]))

  def test_ema_warmup(self):
    model = nnx.Linear(
        1, 1, use_bias=False, kernel_init=nnx.initializers.ones,
        rngs=nnx.Rngs(0),
    )
    ema = nnx.EMA(model, decay=0.9, warmup_steps=4)

    @nnx.jit
    def update(ema, model):
      ema.update(model)

    rates = []
    for _ in range(6):
      # Reset the EMA to 0 so that after the update it equals the rate.
      ema.params.kernel[...] = jnp.zeros((1, 1))
      update(ema, model)
      rates.append(ema.params.kernel[0, 0])

    # k = 0.9 / (0.1 * 4) = 2.25, rate = max(0.1, 1 / (1 + 2.25 * count)).
    np.testing.assert_allclose(
        rates, [1 / 3.25, 1 / 5.5, 1 / 7.75, 0.1, 0.1, 0.1], rtol=1e-6
    )
    self.assertEqual(ema.count[...], 6)

  def test_ema_invalid_args(self):
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    for decay in (-0.1, 1.0, 1.5):
      with self.assertRaisesRegex(ValueError, 'decay'):
        nnx.EMA(model, decay=decay)
    with self.assertRaisesRegex(ValueError, 'warmup_steps'):
      nnx.EMA(model, decay=0.9, warmup_steps=-1)
    schedule = optax.constant_schedule(0.9)
    with self.assertRaisesRegex(
        ValueError, 'Cannot use a callable `decay` and `warmup_steps` together.'
    ):
      nnx.EMA(model, decay=schedule, warmup_steps=1)

  def test_ema_schedule(self):
    @nnx.jit
    def update(ema, model):
      ema.update(model)

    model = nnx.Linear(
        1, 2, use_bias=False, kernel_init=nnx.initializers.zeros,
        rngs=nnx.Rngs(0),
    )
    schedule = optax.linear_schedule(
        0.5, 0.75, transition_steps=1, transition_begin=1
    )
    ema = nnx.EMA(model, decay=schedule)

    model.kernel[...] = jnp.array([[4.0, 8.0]])
    update(ema, model)  # decay = 0.5
    step1 = ema.params.kernel[...]

    model.kernel[...] = jnp.array([[6.0, 12.0]])
    update(ema, model)  # decay = 0.5
    step2 = ema.params.kernel[...]

    model.kernel[...] = jnp.array([[8.0, 16.0]])
    update(ema, model)  # decay = 0.75
    step3 = ema.params.kernel[...]

    np.testing.assert_allclose(step1, [[2.0, 4.0]])
    np.testing.assert_allclose(step2, [[4.0, 8.0]])
    np.testing.assert_allclose(step3, [[5.0, 10.0]])
    self.assertEqual(ema.count[...], 3)

  @parameterized.parameters(jnp.bfloat16, jnp.float16)
  def test_ema_schedule_preserves_dtype(self, dtype):
    model = nnx.Linear(
        1, 2, use_bias=False, param_dtype=dtype, rngs=nnx.Rngs(0)
    )
    model.kernel[...] = jnp.array([[0.0, 4.0]], dtype=dtype)
    ema = nnx.EMA(
        model, decay=optax.linear_schedule(0.5, 0.25, transition_steps=1)
    )

    model.kernel[...] = jnp.array([[4.0, 8.0]], dtype=dtype)
    ema.update(model)  # decay = 0.5

    self.assertEqual(ema.params.kernel[...].dtype, dtype)
    np.testing.assert_allclose(ema.params.kernel[...], [[2.0, 6.0]])


if __name__ == '__main__':
  absltest.main()

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
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

class TestEMA(absltest.TestCase):

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

  def test_ema_default_tracks_inexact_variables(self):
    class Model(nnx.Module):

      def __init__(self, rngs):
        self.linear = nnx.Linear(2, 2, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.temperature = nnx.Param(1.0)
        self.step = nnx.Variable(jnp.array(0, jnp.int32))
        self.flag = nnx.Variable(jnp.array(False))

      def __call__(self, x):
        return self.dropout(self.linear(x)) * self.temperature

    model = Model(nnx.Rngs(0))
    initial_kernel = jnp.copy(model.linear.kernel[...])
    ema = nnx.EMA(model, decay=0.9)
    ema_model = ema.apply_to(model)

    self.assertEqual(
        [path for path, _ in nnx.to_flat_state(ema.params)],
        [('linear', 'kernel'), ('temperature',)],
    )

    @nnx.jit
    def train_step(model, ema, x):
      model(x)
      model.linear.kernel[...] *= 2.0
      model.temperature[...] = 2.0
      model.step[...] += 5
      model.flag[...] = True
      ema.update(model)

    train_step(model, ema, jnp.ones((1, 2)))

    self.assertEqual(model.dropout.rngs.count[...], 1)
    np.testing.assert_allclose(
        ema.params['linear']['kernel'][...], 1.1 * initial_kernel
    )
    np.testing.assert_allclose(ema.params['temperature'][...], 1.1)
    self.assertIs(ema_model.dropout.rngs.count, model.dropout.rngs.count)
    self.assertIs(ema_model.step, model.step)
    self.assertEqual(ema_model.step[...], 5)
    self.assertTrue(ema_model.flag[...])

  def test_ema_default_skips_non_array_variables(self):
    class Model(nnx.Module):

      def __init__(self, rngs):
        self.linear = nnx.Linear(2, 2, rngs=rngs)
        self.cache = nnx.Variable((jnp.zeros(2), jnp.ones(2)))

    model = Model(nnx.Rngs(0))
    ema = nnx.EMA(model, decay=0.9)
    ema.update(model)

    self.assertEqual(
        [path for path, _ in nnx.to_flat_state(ema.params)],
        [('linear', 'bias'), ('linear', 'kernel')],
    )

if __name__ == '__main__':
  absltest.main()

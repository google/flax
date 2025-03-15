# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.pytreelib import Pytree
import jax.experimental.ode


class TestPytree(absltest.TestCase):
  def test_linear_model(self):
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    jax_model = Pytree(model)
    y0 = jnp.ones((2,))

    def dy_dt(y, t, jax_model):
      return jax_model(y)

    t = jnp.linspace(0, 1, 10)
    y = jax.experimental.ode.odeint(dy_dt, y0, t, jax_model)
    self.assertEqual(y.shape, (10, 2))

  def test_model_with_dropout(self):
    class Model(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 16, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.linear(x)))

    nnx_model = Model(nnx.Rngs(0))
    jax_model = Pytree(nnx_model)

    @jax.jit
    def f(jax_model, x):
      return jax_model(x)

    x = jnp.ones((1, 2))
    y = f(jax_model, x)
    self.assertEqual(y.shape, (1, 16))

  def test_state_updates_discarded(self):
    class Model(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 16, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.linear(x)))

    nnx_model = Model(nnx.Rngs(0))
    jax_model = Pytree(nnx_model)

    @jax.jit
    def f(jax_model, x):
      return jax_model(x)

    x = jnp.ones((1, 2))
    y1 = f(jax_model, x)
    y2 = f(jax_model, x)
    self.assertTrue(jnp.allclose(y1, y2))

  def test_state_updates_kept(self):
    class Model(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 16, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)

      def __call__(self, x):
        return nnx.relu(self.dropout(self.linear(x)))

    nnx_model = Model(nnx.Rngs(0))
    jax_model = Pytree(nnx_model)

    @jax.jit
    def g(jax_model, x):
      nnx_model = jax_model.merge()
      y = nnx_model(x)
      return y, Pytree(nnx_model)

    x = jnp.ones((1, 2))
    y1, jax_model = g(jax_model, x)
    y2, jax_model = g(jax_model, x)
    self.assertFalse(jnp.allclose(y1, y2))

  def test_nested_methods(self):
    class MyModule(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(2, 3, rngs=rngs)
        self.decoder = nnx.Linear(3, 2, rngs=rngs)

      def encode(self, x):
        return self.encoder(x)

      def decode(self, x):
        return self.decoder(x)

    nnx_model = MyModule(nnx.Rngs(0))
    jax_model = Pytree(nnx_model)

    @jax.jit
    def encode(jax_model, x):
      return jax_model.encode(x)

    @jax.jit
    def decode(jax_model, x):
      return jax_model.decode(x)

    x = jnp.ones((1, 2))
    z = encode(jax_model, x)
    self.assertEqual(z.shape, (1, 3))
    y = decode(jax_model, z)
    self.assertEqual(y.shape, (1, 2))


if __name__ == '__main__':
  absltest.main()

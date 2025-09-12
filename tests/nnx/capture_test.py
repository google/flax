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
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np


class TestCapture(absltest.TestCase):
  def test_fwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.capture_fwd('pre_act', x)
        return jnp.sin(x)
    model, x = Foo(8), jnp.ones((4, 8))

    # Normal
    with nnx.capture_intms(model) as intms:
      y = model(x)
    np.testing.assert_array_equal(jnp.sin(intms['pre_act']), y)

    # jit outside
    @jax.jit
    def run(model, x):
      with nnx.capture_intms(model) as intms:
        y = model(x)
      return y, intms
    y, intms = run(model, x)
    np.testing.assert_array_equal(jnp.sin(intms['pre_act']), y)

    # jit inside
    with nnx.capture_intms(model) as intms:
      y = jax.jit(lambda m, x: m(x))(model, x)
    np.testing.assert_array_equal(jnp.sin(intms['pre_act']), y)


  def test_bwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w1 = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
        self.w2 = nnx.Param(jax.random.normal(jax.random.key(1), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w1
        x = self.capture_bwd('before_w2', x)
        x = x @ self.w2
        x = self.capture_bwd('after_w2', x)
        return jnp.sin(x)
    def loss(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean()
    model, x, y = Foo(8), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    with nnx.capture_intms(model) as intms:
      _ = jax.grad(loss)(model, x, y)
    np.testing.assert_array_equal(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T)

    # jit outside
    @jax.jit
    def run(model, x, y):
      with nnx.capture_intms(model) as intms:
        grads = jax.grad(loss)(model, x, y)
      return intms, grads
    intms, grads = run(model, x, y)
    np.testing.assert_array_equal(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T)

    # jax inside
    with nnx.capture_intms(model) as intms:
      _ = jax.jit(jax.grad(loss))(model, x, y)
    np.testing.assert_array_equal(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T)

  def test_fwd_and_bwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w1 = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
        self.w2 = nnx.Param(jax.random.normal(jax.random.key(1), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w1
        x = self.capture_bwd('before_w2', x)
        x = x @ self.w2
        x = self.capture_bwd('after_w2', x)
        self.capture_fwd('pre_act', x)
        return jnp.sin(x)
    def loss_and_y(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean(), preds
    model, x, y = Foo(8), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    with nnx.capture_intms(model) as intms:
      (_, actual_y), _ = jax.value_and_grad(loss_and_y, has_aux=True)(
        model, x, y)

    np.testing.assert_array_equal(jnp.sin(intms['pre_act']), actual_y)
    np.testing.assert_array_equal(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T)

  def test_nested_modules(self):
    class Foo(nnx.Module):
      def __init__(self, dim, rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.capture_fwd('pre_act', x)
        x = self.capture_bwd('pre_act', x)
        return jnp.sin(x)
    class Bar(nnx.Module):
      def __init__(self, num, rngs):
        self.foos = nnx.data([Foo(8, rngs=rngs) for _ in range(num)])
      def __call__(self, x):
        for block in self.foos:
          x = block(x)
        return x
    model, x, y = Bar(2, nnx.Rngs(0)), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    def loss_and_y(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean(), preds
    with nnx.capture_intms(model) as intms:
      (_, actual_y), _ = jax.value_and_grad(loss_and_y, has_aux=True)(
        model, x, y)

    self.assertEqual(intms['foos'][0]['pre_act'].shape, (4, 8))
    self.assertEqual(intms['foos'][0]['grad_pre_act'].shape, (4, 8))



if __name__ == '__main__':
  absltest.main()

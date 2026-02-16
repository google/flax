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
import threading

class TestCapture(absltest.TestCase):
  def test_fwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        nnx.capture_fwd('pre_act', x)
        return jnp.sin(x)
    model, x = Foo(8), jnp.ones((4, 8))

    # Normal
    with nnx.capture_intermediates():
      y = model(x)
      intms = nnx.get_intermediates()
    np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)

    # jit outside
    @jax.jit
    def run(model, x):
      with nnx.capture_intermediates():
        y = model(x)
        return y, nnx.get_intermediates()
    y, intms = run(model, x)
    np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)

    # jit inside
    with nnx.capture_intermediates():
      y = jax.jit(lambda m, x: m(x))(model, x)
      intms = nnx.get_intermediates()
    np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)


  def test_bwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w1 = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
        self.w2 = nnx.Param(jax.random.normal(jax.random.key(1), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w1
        x = nnx.capture_bwd('grad_before_w2', x)
        x = x @ self.w2
        x = nnx.capture_bwd('grad_after_w2', x)
        return jnp.sin(x)
    def loss(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean()
    model, x, y = Foo(8), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    with nnx.capture_intermediates():
      _ = jax.grad(loss)(model, x, y)
      intms = nnx.get_intermediates()
    np.testing.assert_allclose(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T, rtol=1e-6)

    # jit outside
    @jax.jit
    def run(model, x, y):
      with nnx.capture_intermediates():
        grads = jax.grad(loss)(model, x, y)
        intms = nnx.get_intermediates()
      return intms, grads
    intms, grads = run(model, x, y)
    np.testing.assert_allclose(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T, rtol=1e-6)

    # jax inside
    with nnx.capture_intermediates():
      _ = jax.jit(jax.grad(loss))(model, x, y)
      intms = nnx.get_intermediates()
    np.testing.assert_allclose(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T, rtol=1e-6)

  def test_fwd_and_bwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w1 = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
        self.w2 = nnx.Param(jax.random.normal(jax.random.key(1), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w1
        x = nnx.capture_bwd('grad_before_w2', x)
        x = x @ self.w2
        x = nnx.capture_bwd('grad_after_w2', x)
        nnx.capture_fwd('pre_act', x)
        return jnp.sin(x)
    def loss_and_y(model, inputs, targets):
      preds = model(inputs)
      return jnp.square(preds - targets).mean(), preds
    model, x, y = Foo(8), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    with nnx.capture_intermediates():
      (_, actual_y), _ = jax.value_and_grad(loss_and_y, has_aux=True)(
        model, x, y)
      intms = nnx.get_intermediates()

    np.testing.assert_allclose(jnp.sin(intms['pre_act']), actual_y)
    np.testing.assert_allclose(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T, rtol=1e-6)

  def test_nested_modules(self):
    class Foo(nnx.Module):
      def __init__(self, dim, rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.capture_fwd('pre_act', x)
        x = self.capture_bwd('grad_pre_act', x)
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
    with nnx.capture_intermediates(model):
      (_, actual_y), _ = jax.value_and_grad(loss_and_y, has_aux=True)(
        model, x, y)
      intms = nnx.get_intermediates()
    self.assertEqual(intms['foos/0/pre_act'].shape, (4, 8))
    self.assertEqual(intms['foos/0/grad_pre_act'].shape, (4, 8))



  def test_capture_fwd_outside_context(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        nnx.capture_fwd('pre_act', x)
        return jnp.sin(x)
    model, x = Foo(8), jnp.ones((4, 8))

    with self.assertRaises((AttributeError, ValueError)):
      model(x)

  def test_thread_safety(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        nnx.capture_fwd('pre_act', x)
        return jnp.sin(x)

    results = {}
    errors = {}

    def run_in_thread(thread_id):
      try:
        model = Foo(8)
        x = jnp.ones((4, 8)) * thread_id
        with nnx.capture_intermediates():
          y = model(x)
          intms = nnx.get_intermediates()
        results[thread_id] = (y, intms)
      except Exception as e:
        errors[thread_id] = e

    threads = [threading.Thread(target=run_in_thread, args=(i,)) for i in range(5)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(len(errors), 0, f"Errors in threads: {errors}")
    self.assertEqual(len(results), 5)

    for thread_id, (y, intms) in results.items():
      self.assertIn('pre_act', intms)
      np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)

  def test_method_outputs_single_module(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w
      def helper(self, x):
        return jnp.sin(x)

    model = Foo(8)
    x = jnp.ones((4, 8))

    with nnx.capture_intermediates(model, method_outputs=True):
      y = model(x)
      z = model.helper(y)
      intms = nnx.get_intermediates()

    self.assertIn('', intms)
    self.assertIn('helper', intms)
    np.testing.assert_allclose(intms[''], y)
    np.testing.assert_allclose(intms['helper'], z)

  def test_method_outputs_nested_modules(self):
    class Inner(nnx.Module):
      def __init__(self, dim, rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)))
      def __call__(self, x):
        return x @ self.w
      def process(self, x):
        return jnp.sin(x)

    class Outer(nnx.Module):
      def __init__(self, rngs):
        self.inner1 = Inner(8, rngs)
        self.inner2 = Inner(8, rngs)
      def __call__(self, x):
        x = self.inner1(x)
        x = self.inner2.process(x)
        return x

    model = Outer(nnx.Rngs(0))
    x = jnp.ones((4, 8))

    with nnx.capture_intermediates(model, method_outputs=True):
      y = model(x)
      intms = nnx.get_intermediates()

    self.assertIn('', intms)
    self.assertIn('inner1/', intms)
    self.assertIn('inner2/process', intms)
    self.assertEqual(intms['inner1/'].shape, (4, 8))
    self.assertEqual(intms['inner2/process'].shape, (4, 8))

  def test_method_outputs_with_jit(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w

    model = Foo(8)
    x = jnp.ones((4, 8))

    @jax.jit
    def run(model, x):
      with nnx.capture_intermediates(model, method_outputs=True):
        y = model(x)
        return y, nnx.get_intermediates()

    y, intms = run(model, x)
    self.assertIn('', intms)
    np.testing.assert_allclose(intms[''], y)

  def test_method_outputs_without_module_arg(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w

    model = Foo(8)
    x = jnp.ones((4, 8))

    with nnx.capture_intermediates(method_outputs=True):
      y = model(x)
      intms = nnx.get_intermediates()

    self.assertEqual(len(intms), 0)

  def test_method_outputs_mixed_with_capture_fwd(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.capture_fwd('intermediate', x)
        return jnp.sin(x)

    model = Foo(8)
    x = jnp.ones((4, 8))

    with nnx.capture_intermediates(model, method_outputs=True):
      y = model(x)
      intms = nnx.get_intermediates()

    self.assertIn('', intms)
    self.assertIn('intermediate', intms)
    np.testing.assert_allclose(intms[''], y)
    np.testing.assert_allclose(jnp.sin(intms['intermediate']), y)

if __name__ == '__main__':
  absltest.main()

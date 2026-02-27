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


from absl.testing import absltest, parameterized
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np

class TestCapture(parameterized.TestCase):
  @parameterized.parameters('none', 'outside', 'inside')
  def test_fwd(self, jit_placement):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))

      def __call__(self, x):
        x = x @ self.w
        self.sow(nnx.Capture, 'pre_act', x)
        return jnp.sin(x)

    model, x = Foo(8), jnp.ones((4, 8))

    if jit_placement == 'none':
      y, intms = nnx.capture_intermediates(model, x)
    elif jit_placement == 'outside':
      @nnx.jit
      def run(model, x):
        return nnx.capture_intermediates(model, x)
      y, intms = nnx.jit(run)(model, x)
    else:  # inside
      def inner(m, x):
        return nnx.capture_intermediates(m, x)
      y, intms = nnx.jit(inner)(model, x)
    np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)

  @parameterized.parameters('none', 'outside', 'inside')
  def test_vmap(self, jit_placement):
    axes = nnx.StateAxes({nnx.Intermediate: 0, nnx.Param: None})

    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))

      def __call__(self, x):
        x = x @ self.w
        self.sow(nnx.Capture, 'pre_act', x)
        return jnp.sin(x)

      def run(self, x):
        return nnx.vmap(lambda self, xi: self(xi), in_axes=(axes, 0))(self, x)

      @nnx.jit
      def run_jit(self,x):
        return self.run(x)

    model, x = Foo(8), jnp.ones((4, 8))

    if jit_placement == 'none':
      y, intms = nnx.capture_intermediates(model, x, method='run')
    elif jit_placement == 'outside':
      def run(model, x):
        return nnx.capture_intermediates(model, x, method='run')
      y, intms = nnx.jit(run)(model, x)
    else:  # inside
      y, intms = nnx.capture_intermediates(model, x, method='run_jit')
    np.testing.assert_allclose(jnp.sin(intms['pre_act']), y)

  @parameterized.parameters('none', 'outside', 'inside')
  def test_bwd(self, jit_placement):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w1 = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
        self.w2 = nnx.Param(jax.random.normal(jax.random.key(1), (dim, dim)))

      def __call__(self, x):
        x = x @ self.w1
        x = self.perturb('grad_before_w2', x)
        x = x @ self.w2
        x = self.perturb('grad_after_w2', x)
        return jnp.sin(x)

      def run(self, x, y):
        def loss(model, inputs, targets):
          preds = model(inputs)
          return jnp.square(preds - targets).mean()
        nnx.grad(loss)(self, x, y)
        return None

      @nnx.jit
      def run_jit(self, x, y):
          return self.run(x, y)

    model, x, y = Foo(8), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5

    if jit_placement == 'none':
      _, intms = nnx.capture_intermediates(model, x, y, method='run')
    elif jit_placement == 'outside':
      def run(model, x, y):
        return nnx.capture_intermediates(model, x, y, method='run')
      _, intms = nnx.jit(run)(model, x, y)
    else:
      _, intms = nnx.capture_intermediates(model, x, y, method='run_jit')
    np.testing.assert_allclose(intms['grad_before_w2'],
                                  intms['grad_after_w2'] @ model.w2.T, rtol=1e-6)

  def test_nested_modules(self):
    class Foo(nnx.Module):
      def __init__(self, dim, rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.sow(nnx.Capture, 'pre_act', x)
        x = self.perturb('grad_pre_act', x)
        return jnp.sin(x)
    class Bar(nnx.Module):
      def __init__(self, num, rngs):
        self.foos = nnx.data([Foo(8, rngs=rngs) for _ in range(num)])
      def __call__(self, x):
        for block in self.foos:
          x = block(x)
        return x
      def run(self, x, y):
        graphdef, params, intms = nnx.split(self, nnx.Param, nnx.Intermediate)
        def loss_and_y(params, inputs, targets, graphdef, intms):
          model = nnx.merge(graphdef, params, intms)
          preds = model(inputs)
          return jnp.square(preds - targets).mean()
        return nnx.value_and_grad(loss_and_y)(params, x, y, graphdef, intms)
    model, x, y = Bar(2, nnx.Rngs(0)), jnp.ones((4, 8)), jnp.ones((4, 8)) * 0.5
    _, intms = nnx.capture_intermediates(model, x, y, method='run')
    self.assertEqual(intms['foos'][0]['pre_act'].shape, (4, 8))
    self.assertEqual(intms['foos'][0]['grad_pre_act'].shape, (4, 8))

  def test_method_outputs_single_module(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w
      def helper(self, x):
        return jnp.sin(x)
      def run(self, x):
        y = self(x)
        z = self.helper(y)
        return (y, z)

    model = Foo(8)
    x = jnp.ones((4, 8))

    (y, z), intms = nnx.capture_intermediates(model, x, method='run', method_outputs=True)

    self.assertIn('__call__', intms)
    self.assertIn('helper', intms)
    np.testing.assert_allclose(intms['__call__'], y)
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

    y, intms = nnx.capture_intermediates(model, x, method_outputs=True)

    self.assertIn('__call__', intms)
    self.assertIn('inner1', intms)
    self.assertIn('process', intms['inner2'])
    self.assertEqual(intms['inner1']['__call__'].shape, (4, 8))
    self.assertEqual(intms['inner2']['process'].shape, (4, 8))

  def test_method_outputs_with_jit(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w

    model = Foo(8)
    x = jnp.ones((4, 8))

    def run(model, x):
      return nnx.capture_intermediates(model, x, method_outputs=True)
    y, intms = jax.jit(run)(model, x)
    self.assertIn('__call__', intms)
    np.testing.assert_allclose(intms['__call__'], y)

  def test_method_outputs_without_module_arg(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        return x @ self.w
      def run(self, x):
        return self(x)

    model = Foo(8)
    x = jnp.ones((4, 8))

    y, intms = nnx.capture_intermediates(model, x, method='run', method_outputs=True)

    # Both run and __call__ method outputs are captured
    self.assertEqual(len(intms), 2)
    self.assertIn('__call__', intms)
    self.assertIn('run', intms)
    np.testing.assert_allclose(intms['__call__'], y)
    np.testing.assert_allclose(intms['run'], y)

  def test_method_outputs_mixed_with_sow(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.sow(nnx.Capture, 'intermediate', x)
        return jnp.sin(x)

    model = Foo(8)
    x = jnp.ones((4, 8))

    y, intms = nnx.capture_intermediates(model, x, method_outputs=True)

    self.assertIn('__call__', intms)
    self.assertIn('intermediate', intms)
    np.testing.assert_allclose(intms['__call__'], y)
    np.testing.assert_allclose(jnp.sin(intms['intermediate']), y)

if __name__ == '__main__':
  absltest.main()

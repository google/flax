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
import flax
from jax import numpy as jnp
import numpy as np
from contextlib import contextmanager

@contextmanager
def set_graph_mode(mode):
  old_mode = flax.config._read('nnx_graph_mode')
  try:
    flax.config.update('nnx_graph_mode', mode)
    yield
  finally:
    flax.config.update('nnx_graph_mode', old_mode)

class TestCapture(parameterized.TestCase):

  def test_vmap(self):

    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), dim))

      def __call__(self, x):
        x = self.perturb('grad_of_x', x)
        x = jnp.dot(x, self.w)
        self.sow(nnx.Intermediate, 'x', x)
        return x

      def pre_run(self, x):
        graphdef, intms, params = nnx.split(model, nnx.Intermediate, nnx.Param)
        def run(intms, params, x):
          return nnx.merge(graphdef, intms, params)(x)
        nnx.vmap(run, in_axes=(0, None, 0))(intms, params, x)

    @nnx.jit
    def train_step(model, perturbations, x):
      def loss_grad(model, perturbations, x):
        def loss(model, perturbations, x):
          return nnx.capture_intermediates(model, x, state=perturbations, filter=nnx.Not(nnx.Perturbation))
        (grads, perturbations), sowed = nnx.grad(loss, argnums=(0,1), has_aux=True)(model, perturbations, x)
        return nnx.merge_state(sowed, perturbations)
      return nnx.vmap(loss_grad, in_axes=(None, 0, 0))(model, perturbations,x)

    model, x = Foo(4), jnp.ones((3, 4))
    _, perturbations = nnx.capture_intermediates(model, x, method='pre_run', filter=nnx.Perturbation)
    metrics = train_step(model, perturbations, x)
    np.testing.assert_allclose(metrics['grad_of_x'].get_value(),
      jnp.broadcast_to(model.w.get_value()[None, :], (3, 4)))
    self.assertEqual(metrics['x'].get_value()[0].shape, (3,))


  @parameterized.parameters(True, False)
  def test_fwd_bwd(self, graph_mode):
    with set_graph_mode(graph_mode):

        class Foo(nnx.Module):
          @nnx.jit
          def __call__(self, x):
            x = self.perturb('grad_of_x', x)
            y = 3 * x
            self.sow(nnx.Intermediate, 'y', y)
            return y

        model = Foo()

        @nnx.jit
        def train_step(model, perturbations, x):
          def loss(model, perturbations, x):
            return nnx.capture_intermediates(model, x, state=perturbations, filter=nnx.Not(nnx.Perturbation))

          (grads, perturbations), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
          return nnx.merge_state(sowed, perturbations)

        x = 1.0
        _, perturbations = nnx.capture_intermediates(model, x, filter=nnx.Perturbation)
        metrics = train_step(model, perturbations, x)
        self.assertEqual(metrics['grad_of_x'], 3)
        self.assertEqual(metrics['y'][0], 3)

  @parameterized.parameters(True, False)
  def test_nested_modules(self, graph_mode):
    with set_graph_mode(graph_mode):

      class Foo(nnx.Module):
        def __call__(self, x):
          x = self.perturb('grad_of_x', x)
          y = 3 * x
          self.sow(nnx.Intermediate, 'y', y)
          return y
      class Bar(nnx.Module):
        def __init__(self):
          self.foos = nnx.data([Foo() for _ in range(3)])
        def __call__(self, x):
          for block in self.foos:
            x = block(x)
          return x

      model = Bar()

      @nnx.jit
      def train_step(model, perturbations, x):
        def loss(model, perturbations, x):
          return nnx.capture_intermediates(model, x, state=perturbations, filter=nnx.Not(nnx.Perturbation))
        (grads, perturbations), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
        return nnx.merge_state(sowed, perturbations)

      x = 1.0
      _, perturbations = nnx.capture_intermediates(model, x, filter=nnx.Perturbation)
      metrics = train_step(model, perturbations, x)
      for i in range(3):
        self.assertEqual(metrics['foos'][i]['grad_of_x'], 3**(3-i))
        self.assertEqual(metrics['foos'][i]['y'][0], 3**(i+1))

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

    (y, z), intms = nnx.capture_intermediates(model, x, method='run', method_output_type=nnx.Intermediate)

    self.assertIn('__call__', intms)
    self.assertIn('helper', intms)
    np.testing.assert_allclose(intms['__call__'][0], y)
    np.testing.assert_allclose(intms['helper'][0], z)

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

    y, intms = nnx.capture_intermediates(model, x, method_output_type=nnx.Intermediate)

    self.assertIn('__call__', intms)
    self.assertIn('inner1', intms)
    self.assertIn('process', intms['inner2'])
    self.assertEqual(intms['inner1']['__call__'][0].shape, (4, 8))
    self.assertEqual(intms['inner2']['process'][0].shape, (4, 8))

  def test_method_outputs_mixed_with_sow(self):
    class Foo(nnx.Module):
      def __init__(self, dim):
        self.w = nnx.Param(jax.random.normal(jax.random.key(0), (dim, dim)))
      def __call__(self, x):
        x = x @ self.w
        self.sow(nnx.Intermediate, 'intermediate', x)
        return jnp.sin(x)

    model = Foo(8)
    x = jnp.ones((4, 8))

    y, intms = nnx.capture_intermediates(model, x, method_output_type=nnx.Intermediate)

    self.assertIn('__call__', intms)
    self.assertIn('intermediate', intms)
    np.testing.assert_allclose(intms['__call__'][0], y)
    np.testing.assert_allclose(jnp.sin(intms['intermediate'][0]), y)

if __name__ == '__main__':
  absltest.main()

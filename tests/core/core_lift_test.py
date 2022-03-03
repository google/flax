# Copyright 2022 The Flax Authors.
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

import operator
from flax import errors
from flax.core import Scope, init, apply, lift, nn, FrozenDict, unfreeze

import jax
from jax import random
from jax import numpy as jnp

import numpy as np


from absl.testing import absltest

class LiftTest(absltest.TestCase):

  def test_aliasing(self):
    def f(scope):
      a = scope.push('a')

      def g(scopes, _):
        scope, a = scopes
        self.assertEqual(a.parent, scope)
      
      lift.vmap(g, variable_axes={}, split_rngs={})((scope, a), jnp.ones((1,)))

    init(f)(random.PRNGKey(0))

  def test_undefined_param(self):
    def f(scope):
      dense = lift.vmap(nn.dense, 
                        in_axes=(0, None), out_axes=0,
                        variable_axes={'params': 0},
                        split_rngs={'params': True})
      dense(scope.push('dense'), np.ones((3, 2)), 2)

    msg = r'No parameter named "kernel" exists in "/vmap\(dense\)".'
    with self.assertRaisesRegex(errors.ScopeParamNotFoundError, msg):
      apply(f)({'params': {'dense': {'abc': np.ones((3, 3))}}})

  def test_jit_cache(self):
    compiles = 0
    @lift.jit
    def f(scope, x):
      nonlocal compiles
      compiles += 1
      if scope.is_mutable_collection('intermediates') and not scope.is_mutable_collection('params'):
        scope.put_variable('intermediates', 'x', x + 1)
      return nn.dense(scope, x, 1)

    x = np.ones((3, 2))
    _, params = init(f)(random.PRNGKey(0), x)
    init(f)(random.PRNGKey(0), x)
    self.assertEqual(compiles, 1)
    apply(f)(params, x)
    self.assertEqual(compiles, 2)  # apply should cause a compile
    apply(f)(params, x)
    self.assertEqual(compiles, 2)  # applying again should not
    # edge case where only the implicit return of the jitted functions changes.
    # this should not use the previously cached apply.
    _, state = apply(f, mutable='intermediates')(params, x)
    self.assertEqual(compiles, 3)  # applying again should not
    self.assertEqual(state['intermediates']['x'].sum(), 3 * 2 * 2)


  def test_vjp(self):
    def g(scope, x):
      p = scope.param('test', nn.initializers.zeros, ())
      scope.variable('state', 'counter', lambda: 0)
      return p * x

    def f(scope, x):
      y, bwd = lift.vjp(g, scope, x)
      params_grad, x_grad = bwd(jnp.ones(y.shape))
      return params_grad, x_grad
    
    x = jnp.ones((3,))
    _, params = init(f)(random.PRNGKey(0), x)
    x_grad, params_grad = apply(f)(params, x)
    self.assertEqual(params_grad, {
      'params': FrozenDict({'test': 3.}),
    })
    np.testing.assert_allclose(x_grad, 0. * x)

  def test_jvp(self):
    def g(scope, x):
      p = scope.param('test', nn.initializers.zeros, ())
      scope.variable('state', 'counter', lambda: 0)
      return p * x

    def f(scope, x):
      vars_t = jax.tree_map(jnp.ones_like, scope.variables().get('params', {}))
      _, out_t = lift.jvp(g, scope, (x,), (jnp.zeros_like(x),), {'params': vars_t})
      return out_t
    
    x = jnp.ones((3,))
    _, params = init(f)(random.PRNGKey(0), x)
    y_t = apply(f)(params, x)
    np.testing.assert_allclose(y_t, jnp.ones_like(x))
  
  def test_while_loop(self):
    def f(scope, x):
      scope.param('inc', lambda _: 1)
      scope.put_variable('state', 'acc', 0)
      scope.put_variable('state', 'rng_params', jnp.zeros((2, 2), jnp.uint32))
      scope.put_variable('state', 'rng_loop', jnp.zeros((2, 2), jnp.uint32))

      def cond_fn(scope, c):
        acc = scope.get_variable('state', 'acc')
        return acc < x
      def body_fn(scope, c):
        i = scope.get_variable('state', 'acc')
        p_rng = scope.make_rng('params')
        l_rng = scope.make_rng('loop')
        scope.put_variable('state', 'rng_params', scope.get_variable('state', 'rng_params').at[i].set(p_rng))
        scope.put_variable('state', 'rng_loop', scope.get_variable('state', 'rng_loop').at[i].set(l_rng))
        inc = scope.get_variable('params', 'inc')
        scope.put_variable('state', 'acc', i + inc)
        return c + 2
      return lift.while_loop(cond_fn, body_fn, scope, 0, carry_variables='state', split_rngs={'params': False, 'loop': True})
    x = 2
    c, vars = apply(f, mutable=True)({}, x, rngs={'params': random.PRNGKey(0), 'loop': random.PRNGKey(1)})
    self.assertEqual(vars['state']['acc'], x)
    self.assertEqual(c, 2 * x)
    np.testing.assert_array_equal(vars['state']['rng_params'][0], vars['state']['rng_params'][1])
    np.testing.assert_array_compare(operator.__ne__, vars['state']['rng_loop'][0], vars['state']['rng_loop'][1])

if __name__ == '__main__':
  absltest.main()

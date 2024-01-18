# Copyright 2023 The Flax Authors.
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

import jax
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp
from jax import random

from flax import errors
from flax.core import FrozenDict, apply, copy, init, lift, nn


class LiftTest(absltest.TestCase):
  def test_aliasing(self):
    def f(scope):
      a = scope.push('a')

      def g(scopes, _):
        scope, a = scopes
        self.assertEqual(a.parent, scope)

      lift.vmap(g, variable_axes={}, split_rngs={})((scope, a), jnp.ones((1,)))

    init(f)(random.key(0))

  def test_undefined_param(self):
    def f(scope):
      dense = lift.vmap(
        nn.dense,
        in_axes=(0, None),
        out_axes=0,
        variable_axes={'params': 0},
        split_rngs={'params': True},
      )
      dense(scope.push('dense'), np.ones((3, 2)), 2)

    msg = r'Could not find parameter named "kernel" in scope "/vmap\(dense\)".'
    with self.assertRaisesRegex(errors.ScopeParamNotFoundError, msg):
      apply(f)({'params': {'dense': {'abc': np.ones((3, 3))}}})

  def test_jit_cache(self):
    compiles = 0

    @lift.jit
    def f(scope, _module_hash, x):
      nonlocal compiles
      compiles += 1
      if scope.is_mutable_collection(
        'intermediates'
      ) and not scope.is_mutable_collection('params'):
        scope.put_variable('intermediates', 'x', x + 1)
      return nn.dense(scope, x, 1)

    x = np.ones((3, 2))
    module_hash = 1
    _, params = init(f)(random.key(0), module_hash, x)
    init(f)(random.key(0), module_hash, x)
    self.assertEqual(compiles, 1)
    apply(f)(params, module_hash, x)
    self.assertEqual(compiles, 2)  # apply should cause a compile
    apply(f)(params, module_hash, x)
    self.assertEqual(compiles, 2)  # applying again should not
    # edge case where only the implicit return of the jitted functions changes.
    # this should not use the previously cached apply.
    _, state = apply(f, mutable='intermediates')(params, module_hash, x)
    self.assertEqual(compiles, 3)  # applying again should not
    self.assertEqual(state['intermediates']['x'].sum(), 3 * 2 * 2)

  def test_vjp(self):
    def g(scope, x, y):
      p = scope.param('test', nn.initializers.constant(0.5), ())
      scope.variable('state', 'counter', lambda: 0)
      return p * x * y

    def f(scope, x, y):
      z, bwd = lift.vjp(g, scope, x, y)
      return bwd(jnp.ones(y.shape))

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    _, params = init(f)(random.key(0), x, y)
    params_grad, x_grad, y_grad = apply(f)(params, x, y)
    self.assertEqual(
      params_grad,
      {
        'params': FrozenDict({'test': 32.0}),
      },
    )
    np.testing.assert_allclose(x_grad, [2.0, 2.5, 3.0])
    np.testing.assert_allclose(y_grad, [0.5, 1.0, 1.5])

  def test_jvp(self):
    def g(scope, x):
      p = scope.param('test', nn.initializers.zeros_init(), ())
      scope.variable('state', 'counter', lambda: 0)
      return p * x

    def f(scope, x):
      vars_t = jax.tree_util.tree_map(
        jnp.ones_like, scope.variables().get('params', {})
      )
      _, out_t = lift.jvp(
        g, scope, (x,), (jnp.zeros_like(x),), {'params': vars_t}
      )
      return out_t

    x = jnp.ones((3,))
    _, params = init(f)(random.key(0), x)
    y_t = apply(f)(params, x)
    np.testing.assert_allclose(y_t, jnp.ones_like(x))

  def test_while_loop(self):
    def f(scope, x):
      key_zero = random.key(0)
      key_zero = jnp.broadcast_to(key_zero, (2, *key_zero.shape))
      scope.param('inc', lambda _: 1)
      scope.put_variable('state', 'acc', 0)
      scope.put_variable('state', 'rng_params', key_zero)
      scope.put_variable('state', 'rng_loop', key_zero)

      def cond_fn(scope, c):
        acc = scope.get_variable('state', 'acc')
        return acc < x

      def body_fn(scope, c):
        i = scope.get_variable('state', 'acc')
        p_rng = scope.make_rng('params')
        l_rng = scope.make_rng('loop')
        scope.put_variable(
          'state',
          'rng_params',
          scope.get_variable('state', 'rng_params').at[i].set(p_rng),
        )
        scope.put_variable(
          'state',
          'rng_loop',
          scope.get_variable('state', 'rng_loop').at[i].set(l_rng),
        )
        inc = scope.get_variable('params', 'inc')
        scope.put_variable('state', 'acc', i + inc)
        return c + 2

      return lift.while_loop(
        cond_fn,
        body_fn,
        scope,
        0,
        carry_variables='state',
        split_rngs={'params': False, 'loop': True},
      )

    x = 2
    c, vars = apply(f, mutable=True)(
      {}, x, rngs={'params': random.key(1), 'loop': random.key(2)}
    )
    self.assertEqual(vars['state']['acc'], x)
    self.assertEqual(c, 2 * x)
    np.testing.assert_array_equal(
      vars['state']['rng_params'][0], vars['state']['rng_params'][1]
    )
    np.testing.assert_array_compare(
      operator.__ne__,
      vars['state']['rng_loop'][0],
      vars['state']['rng_loop'][1],
    )

  def test_cond(self):
    def f(scope, x, pred):
      scope.variable('state', 'true_count', lambda: 0)
      scope.variable('state', 'false_count', lambda: 0)

      def true_fn(scope, x):
        scope.variable('state', 'true_count').value += 1
        return scope.child(nn.dense)(x, 2)

      def false_fn(scope, x):
        scope.variable('state', 'false_count').value += 1
        return -scope.child(nn.dense)(x, 2)

      return lift.cond(pred, true_fn, false_fn, scope, x)

    x = jnp.ones((1, 3))
    y1, vars = init(f)(random.key(0), x, True)
    self.assertEqual(vars['state'], {'true_count': 1, 'false_count': 0})
    y2, vars = apply(f, mutable='state')(vars, x, False)
    self.assertEqual(vars['state'], {'true_count': 1, 'false_count': 1})
    np.testing.assert_allclose(y1, -y2)

  def test_switch(self):
    def f(scope, x, index):
      scope.variable('state', 'a_count', lambda: 0)
      scope.variable('state', 'b_count', lambda: 0)
      scope.variable('state', 'c_count', lambda: 0)

      def a_fn(scope, x):
        scope.variable('state', 'a_count').value += 1
        return scope.child(nn.dense)(x, 2)

      def b_fn(scope, x):
        scope.variable('state', 'b_count').value += 1
        return -scope.child(nn.dense)(x, 2)

      def c_fn(scope, x):
        scope.variable('state', 'c_count').value += 1
        return scope.child(nn.dense)(x, 2)

      return lift.switch(index, [a_fn, b_fn, c_fn], scope, x)

    x = jnp.ones((1, 3))
    y1, vars = init(f)(random.key(0), x, 0)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 0, 'c_count': 0})
    y2, updates = apply(f, mutable='state')(vars, x, 1)
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 1, 'c_count': 0})
    np.testing.assert_allclose(y1, -y2)
    y3, updates = apply(f, mutable='state')(vars, x, 2)
    vars = copy(vars, updates)
    self.assertEqual(vars['state'], {'a_count': 1, 'b_count': 1, 'c_count': 1})
    np.testing.assert_allclose(y1, y3)

  def test_subscope_var_aliasing(self):
    def test(scope, x):
      subscope = scope.push(name='a')
      subscope.put_variable('state', 'x', 0.0)
      _ = lift.while_loop(
        lambda scope, x: False,
        lambda scope, x: x,
        scope,
        jnp.array(0, jnp.int32),
        carry_variables=['state'],
      )
      subscope.put_variable('state', 'x', 1.0)
      val0 = scope.variables()['state']['a']['x']
      val1 = subscope.variables()['state']['x']
      self.assertEqual(val0, val1)
      return x

    init(test)(random.key(0), 1.0)


if __name__ == '__main__':
  absltest.main()

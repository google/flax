# Copyright 2021 The Flax Authors.
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

from flax import errors
from flax.core import Scope, init, apply, lift, nn

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
      apply(f)({})


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


if __name__ == '__main__':
  absltest.main()

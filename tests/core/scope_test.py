# Copyright 2020 The Flax Authors.
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

from flax.core import Scope, scope, init, apply, nn

from jax import random

import numpy as np


from absl.testing import absltest

class ScopeTest(absltest.TestCase):

  def test_rng(self):
    def f(scope):
      self.assertTrue(scope.has_rng('params'))
      self.assertFalse(scope.has_rng('dropout'))
      rng = scope.make_rng('params')
      self.assertTrue(np.all(rng == random.fold_in(random.PRNGKey(0), 1)))
    init(f)(random.PRNGKey(0))

  def test_in_filter(self):
    filter_true = lambda x, y : self.assertTrue(scope.in_filter(x, y))
    filter_false = lambda x, y : self.assertFalse(scope.in_filter(x, y))

    filter_true(True, 'any_string1')
    filter_false(False, 'any_string2')
    filter_true('exact_match', 'exact_match')
    filter_false('no_match1', 'no_match2')
    filter_true(['one', 'two'], 'one')
    filter_false(['one', 'two'], 'three')
    filter_false([], 'one')
    filter_false([], None)

  def test_group_collections(self):
    params = { 'dense1': { 'x': [10, 20] } }
    batch_stats = { 'dense1': { 'ema': 5 } }
    xs = { 'params': params, 'batch_stats': batch_stats }

    # Retrieve all keys only once.
    group = scope.group_collections(xs, ['params', 'params'])
    self.assertEqual(group, ({'params': params}, {}))

    # Ignore non-existing keys.
    self.assertEqual(scope.group_collections(xs, ['vars']), ({},))

    # False gets nothing and True retrieves all keys once.
    self.assertEqual(scope.group_collections(xs, [False, True, True]), 
                                             ({}, xs, {}))

  def test_inconsistent_param_shapes(self):
    def f(scope):
      scope.param('test', nn.initializers.ones, (4,))
    
    msg = 'Inconsistent shapes between value and initializer for parameter "test": (2,), (4,)'
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      apply(f)({'params': {'test': np.ones((2,))}})


if __name__ == '__main__':
  absltest.main()

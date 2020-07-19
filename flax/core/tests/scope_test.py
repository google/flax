from flax.core import Scope, init, apply

from jax import random

import numpy as np


from absl.testing import absltest

class ScopeTest(absltest.TestCase):

  def test_rng(self):
    def f(scope):
      self.assertTrue(scope.has_rng('param'))
      self.assertFalse(scope.has_rng('dropout'))
      rng = scope.make_rng('param')
      self.assertTrue(np.all(rng == random.fold_in(random.PRNGKey(0), 1)))

    init(f)(random.PRNGKey(0))


if __name__ == '__main__':
  absltest.main()

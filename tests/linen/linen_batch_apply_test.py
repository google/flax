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

"""Tests for flax.linen.batch_apply."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from flax import linen as nn

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class BatchApplyTest(parameterized.TestCase):
  @parameterized.parameters(
    {'fn': lambda a, b: a + b.reshape(1, -1)},
    {'fn': lambda a, b: jnp.dot(a, b)},
  )
  def test_batchapply(self, fn):
    a = jax.random.normal(jax.random.key(0), [2, 3, 4])
    b = jax.random.normal(jax.random.key(1), [4])

    def raises(a, b):
      if len(a.shape) != 2:
        raise ValueError('a must be shape 2')
      if len(b.shape) != 1:
        raise ValueError('b must be shape 1')
      return fn(a, b)

    out = nn.BatchApply(raises)(a, b)
    expected_merged_leading = raises(a.reshape(2 * 3, 4), b)
    expected = expected_merged_leading.reshape(
      (2, 3) + expected_merged_leading.shape[1:]
    )
    np.testing.assert_array_equal(out, expected)

  def test_batchapply_accepts_float(self):
    def raises(a, b):
      if len(a.shape) != 2:
        raise ValueError('a must be shape 2')
      return a + b

    out = nn.BatchApply(raises)(jnp.ones([2, 3, 4]), 2.0)
    np.testing.assert_array_equal(out, 3 * jnp.ones([2, 3, 4]))

  def test_batchapply_accepts_none(self):
    def raises(a, b):
      if a is not None:
        raise ValueError('a must be None.')
      if len(b.shape) != 2:
        raise ValueError('b must be shape 2')
      return 3 * b

    out = nn.BatchApply(raises)(None, jnp.ones([2, 3, 4]))
    np.testing.assert_array_equal(out, 3 * jnp.ones([2, 3, 4]))

  def test_batchapply_raises(self):
    with self.assertRaisesRegex(ValueError, 'requires at least one input'):
      nn.BatchApply(lambda: 1)()


if __name__ == '__main__':
  absltest.main()

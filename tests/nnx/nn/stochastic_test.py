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

import itertools

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import numpy as np

from flax import nnx


class TestDropout(parameterized.TestCase):
  def test_dropout_internal_rngs(self):
    n = 0
    m1 = nnx.Dropout(
      rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0)
    )
    m2 = nnx.Dropout(rate=0.5, deterministic=False)
    rngs2 = nnx.Rngs(dropout=0).fork()

    @nnx.jit
    def f(m, x, rngs=None):
      nonlocal n
      n += 1
      return m(x, rngs=rngs)

    x = jnp.ones((1, 10))
    self.assertIsNotNone(m1.rngs)
    self.assertEqual(m1.rngs.count[...], 0)

    y1 = f(m1, x)
    self.assertEqual(n, 1)
    self.assertEqual(m1.rngs.count[...], 1)
    y2 = f(m2, x, rngs=rngs2)
    self.assertEqual(n, 2)
    self.assertEqual(rngs2.dropout.count[...], 1)
    np.testing.assert_allclose(y1, y2)

    y1 = f(m1, x)
    self.assertEqual(m1.rngs.count[...], 2)
    y2 = f(m2, x, rngs=rngs2)
    self.assertEqual(rngs2.dropout.count[...], 2)
    np.testing.assert_allclose(y1, y2)

    self.assertEqual(n, 2)

  def test_dropout_rng_override(self):
    m1 = nnx.Dropout(
      rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0)
    )
    m2 = nnx.Dropout(
      rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=1)
    )
    x = jnp.ones((10, 10))

    y1 = m1(x)
    y2 = m2(x)
    self.assertFalse(
      np.array_equal(y1, y2),
      'Different RNG seeds should produce different masks',
    )

    # Override m2's seed with m1's seed -- outputs should match
    y2 = m2(x, rngs=nnx.Rngs(dropout=0).fork())
    np.testing.assert_allclose(y1, y2)

  def test_dropout_arg_override(self):
    m = nnx.Dropout(rate=0.5)
    x = jnp.ones((10, 10))

    # deterministic call arg provided
    y_det = m(x, deterministic=True)
    np.testing.assert_array_equal(y_det, x)
    # deterministic constructor arg provided
    m.set_attributes(deterministic=True)
    y = m(x)
    # call-time deterministic=False overrides constructor deterministic=True
    self.assertFalse(
      np.array_equal(
        y, m(x, deterministic=False, rngs=nnx.Rngs(dropout=0))
      ),
      'deterministic output should differ from stochastic output',
    )
    # no rng arg provided
    m.set_attributes(deterministic=False)
    with self.assertRaisesRegex(
      ValueError,
      r'`deterministic` is False.*no `rngs` argument',
    ):
      m(x)

  def test_dropout_arg_override_view(self):
    m = nnx.Dropout(rate=0.5)
    x = jnp.ones((10, 10))

    # deterministic via view
    new_m = nnx.view(m, deterministic=True)
    y = new_m(x)
    np.testing.assert_array_equal(y, x)
    # call-time deterministic=False overrides view deterministic=True
    self.assertFalse(
      np.array_equal(
        y,
        new_m(
          x, deterministic=False, rngs=nnx.Rngs(dropout=0)
        ),
      ),
      'deterministic output should differ from stochastic output',
    )
    # no rng arg provided
    new_m = nnx.view(m, deterministic=False)
    with self.assertRaisesRegex(
      ValueError,
      r'`deterministic` is False.*no `rngs` argument',
    ):
      new_m(x)

  def test_deterministic_passthrough(self):
    m = nnx.Dropout(rate=0.5, deterministic=True)
    x = jnp.ones((20, 20))
    y = m(x)
    np.testing.assert_array_equal(y, x)

  def test_rate_zero(self):
    m = nnx.Dropout(
      rate=0.0,
      deterministic=False,
      rngs=nnx.Rngs(dropout=0),
    )
    x = jnp.ones((20, 20))
    y = m(x)
    np.testing.assert_array_equal(y, x)

  def test_rate_one(self):
    m = nnx.Dropout(
      rate=1.0,
      deterministic=False,
      rngs=nnx.Rngs(dropout=0),
    )
    x = jnp.ones((20, 20))
    y = m(x)
    np.testing.assert_array_equal(y, jnp.zeros_like(x))

  def test_rate_one_gradient_not_nan(self):
    m = nnx.Dropout(
      rate=1.0,
      deterministic=False,
      rngs=nnx.Rngs(dropout=0),
    )
    x = jnp.ones((20, 20))
    grad = jax.grad(lambda x: jnp.sum(m(x)))(x)
    self.assertFalse(jnp.any(jnp.isnan(grad)))
    np.testing.assert_array_equal(grad, jnp.zeros_like(x))

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16, jnp.bfloat16],
  )
  def test_dtypes(self, dtype):
    m = nnx.Dropout(
      rate=0.5,
      deterministic=False,
      rngs=nnx.Rngs(dropout=0),
    )
    x = jnp.ones((10, 10), dtype=dtype)
    y = m(x)
    self.assertEqual(y.dtype, dtype)

  def test_rngs_as_jax_array(self):
    m = nnx.Dropout(rate=0.5, deterministic=False)
    x = jnp.ones((10, 10))
    key = random.key(0)
    y = m(x, rngs=key)
    self.assertTrue(jnp.any(y == 0.0))
    self.assertTrue(jnp.any(y > 0.0))
    # Kept values should be scaled by 1/keep_prob = 2.0
    np.testing.assert_allclose(
      y[y > 0.0], 2.0, rtol=1e-6
    )

  def test_rngs_as_nnx_rngs_in_call(self):
    m = nnx.Dropout(rate=0.5, deterministic=False)
    x = jnp.ones((10, 10))
    y = m(x, rngs=nnx.Rngs(dropout=0))
    self.assertTrue(jnp.any(y == 0.0))
    self.assertTrue(jnp.any(y > 0.0))
    np.testing.assert_allclose(
      y[y > 0.0], 2.0, rtol=1e-6
    )

  def test_custom_rng_collection(self):
    m = nnx.Dropout(
      rate=0.5,
      deterministic=False,
      rng_collection='my_dropout',
      rngs=nnx.Rngs(my_dropout=0),
    )
    x = jnp.ones((10, 10))
    y = m(x)
    self.assertTrue(jnp.any(y == 0.0))
    self.assertTrue(jnp.any(y > 0.0))

  def test_invalid_rngs_type_constructor(self):
    with self.assertRaisesRegex(
      TypeError,
      r'rngs must be a Rngs, RngStream or None',
    ):
      nnx.Dropout(rate=0.5, rngs='invalid')

  def test_invalid_rngs_type_call(self):
    m = nnx.Dropout(rate=0.5, deterministic=False)
    x = jnp.ones((10, 10))
    with self.assertRaisesRegex(
      TypeError,
      r'rngs must be a Rngs, RngStream or jax\.Array',
    ):
      m(x, rngs='invalid')

  @parameterized.parameters(
    {
      'num_dims': 2,
      'broadcast_dims': (1,),
      'slice_fn': lambda out, i: out[i, :],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 2,
      'broadcast_dims': (0,),
      'slice_fn': lambda out, i: out[:, i],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 3,
      'broadcast_dims': (1, 2),
      'slice_fn': lambda out, i: out[i, :, :],
      'summed_total': 2 * 10 * 10,
    },
    {
      'num_dims': 3,
      'broadcast_dims': (1,),
      'slice_fn': lambda out, i, j: out[i, :, j],
      'summed_total': 2 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (0, 2, 3),
      'slice_fn': lambda out, i: out[:, i, :, :],
      'summed_total': 2 * 10 * 10 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (0, 1),
      'slice_fn': lambda out, i, j: out[:, :, i, j],
      'summed_total': 2 * 10 * 10,
    },
    {
      'num_dims': 4,
      'broadcast_dims': (3,),
      'slice_fn': lambda out, i, j, k: out[i, j, k, :],
      'summed_total': 2 * 10,
    },
  )
  def test_broadcast_dims(
    self, num_dims, broadcast_dims, slice_fn, summed_total
  ):
    m = nnx.Dropout(
      rate=0.5,
      broadcast_dims=broadcast_dims,
      deterministic=False,
      rngs=nnx.Rngs(dropout=0),
    )
    x = jnp.ones((10,) * num_dims)
    out = m(x)

    n_free = num_dims - len(broadcast_dims)
    for indices in itertools.product(range(10), repeat=n_free):
      self.assertIn(
        float(slice_fn(out, *indices).sum()),
        (0, summed_total),
      )

  def test_rate_stats(self):
    n_trials = 10
    rootkey = random.key(0)
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
      rootkey, subkey = random.split(rootkey)
      m = nnx.Dropout(rate=rate, deterministic=False)
      nonzero_counts = 0
      for key in random.split(subkey, n_trials):
        y = m(
          jnp.ones((100, 100)),
          rngs=nnx.Rngs(dropout=key),
        )
        nonzero_counts += np.sum(y > 0.0)
      all_counts = np.prod((100, 100, n_trials))
      frac = nonzero_counts / all_counts
      keep_rate = 1.0 - rate
      # check within 4 sigma
      delta = (
        4
        * np.sqrt(rate * keep_rate)
        / np.sqrt(all_counts)
      )
      self.assertTrue(
        keep_rate - delta < frac < keep_rate + delta
      )


if __name__ == '__main__':
  absltest.main()

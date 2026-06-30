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

"""Tests for flax.nnx.nn.dtypes."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import enable_x64
from jax import numpy as jnp
import numpy as np

from flax.nnx.nn import dtypes


class CanonicalizeDtypeTest(parameterized.TestCase):
  @parameterized.parameters(
    jnp.int8, jnp.int16, jnp.int32,
    jnp.uint8, jnp.uint16, jnp.uint32,
  )
  def test_int_input_promoted_to_float(self, int_dtype):
    x = int_dtype(1)
    self.assertEqual(dtypes.canonicalize_dtype(x), jnp.float32)

  @parameterized.parameters(jnp.int64, jnp.uint64)
  def test_x64_int_promoted_to_float(self, int_dtype):
    with enable_x64():
      x = int_dtype(1)
      self.assertEqual(dtypes.canonicalize_dtype(x), jnp.float32)

  def test_float32_unchanged(self):
    x = jnp.float32(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(x), jnp.float32)

  @parameterized.parameters(jnp.float16, jnp.bfloat16)
  def test_low_precision_float_preserved(self, float_dtype):
    x = float_dtype(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(x), float_dtype)

  @parameterized.parameters(
    (jnp.float16, jnp.float32, jnp.float32),
    (jnp.bfloat16, jnp.float32, jnp.float32),
    (jnp.bfloat16, jnp.float16, jnp.float32),
  )
  def test_mixed_float_promotion(self, dtype_a, dtype_b, expected):
    x = dtype_a(1.0)
    y = dtype_b(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(x, y), expected)

  def test_complex_input(self):
    x = jnp.complex64(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(x), jnp.complex64)

  def test_inexact_false_preserves_int(self):
    x = jnp.int32(1)
    self.assertEqual(dtypes.canonicalize_dtype(x, inexact=False), jnp.int32)

  def test_x64_int64_preserved_when_not_inexact(self):
    with enable_x64():
      x = jnp.int64(1)
      self.assertEqual(dtypes.canonicalize_dtype(x, inexact=False), jnp.int64)

  def test_x64_float64_preserved(self):
    with enable_x64():
      x = jnp.float64(1.0)
      self.assertEqual(dtypes.canonicalize_dtype(x), jnp.float64)

  def test_explicit_dtype_override(self):
    x = jnp.float32(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(x, dtype=jnp.float16), jnp.float16)

  def test_explicit_dtype_overrides_multiple_args(self):
    x = jnp.float16(1.0)
    y = jnp.float32(2.0)
    self.assertEqual(dtypes.canonicalize_dtype(x, y, dtype=jnp.bfloat16), jnp.bfloat16)

  def test_explicit_exact_dtype_raises(self):
    with self.assertRaisesRegex(ValueError, 'Dtype must be inexact'):
      dtypes.canonicalize_dtype(dtype=jnp.int32, inexact=True)

  def test_explicit_exact_dtype_inexact_false(self):
    self.assertEqual(dtypes.canonicalize_dtype(dtype=jnp.int32, inexact=False), jnp.int32)

  def test_none_args_filtered(self):
    x = jnp.float32(1.0)
    self.assertEqual(dtypes.canonicalize_dtype(None, x, None), jnp.float32)

  def test_all_none_args_raises(self):
    with self.assertRaises(ValueError):
      dtypes.canonicalize_dtype(None, None)

  def test_no_args_raises(self):
    with self.assertRaises(ValueError):
      dtypes.canonicalize_dtype()


class PromoteDtypeTest(parameterized.TestCase):
  def test_single_element(self):
    x = jnp.float32(1.0)
    result = dtypes.promote_dtype((x,))
    self.assertIsInstance(result, tuple)
    self.assertEqual(result[0].dtype, jnp.float32)

  def test_none_preserved(self):
    x = jnp.float32(1.0)
    result = dtypes.promote_dtype((x, None))
    self.assertEqual(result[0].dtype, jnp.float32)
    self.assertIsNone(result[1])

  def test_multiple_nones_with_value(self):
    x = jnp.float16(1.0)
    result = dtypes.promote_dtype((None, x, None))
    self.assertIsNone(result[0])
    self.assertEqual(result[1].dtype, jnp.float16)
    self.assertIsNone(result[2])

  def test_explicit_downcast(self):
    x = jnp.float32(1.0)
    (result,) = dtypes.promote_dtype((x,), dtype=jnp.float16)
    self.assertEqual(result.dtype, jnp.float16)

  def test_cast_preserves_values(self):
    x = jnp.array([1.5, 2.5], dtype=jnp.float32)
    (result,) = dtypes.promote_dtype((x,), dtype=jnp.float16)
    np.testing.assert_allclose(result, [1.5, 2.5], rtol=1e-3)

  def test_int_promoted_to_float(self):
    x = jnp.int32(1)
    (result,) = dtypes.promote_dtype((x,))
    self.assertEqual(result.dtype, jnp.float32)

  def test_inexact_false_preserves_int(self):
    x = jnp.int32(1)
    (result,) = dtypes.promote_dtype((x,), inexact=False)
    self.assertEqual(result.dtype, jnp.int32)

  def test_explicit_exact_dtype_raises(self):
    x = jnp.int32(1)
    with self.assertRaisesRegex(ValueError, 'Dtype must be inexact'):
      dtypes.promote_dtype((x,), dtype=jnp.int32, inexact=True)

  def test_mixed_dtypes_promoted(self):
    x = jnp.float16(1.0)
    y = jnp.float32(2.0)
    result = dtypes.promote_dtype((x, y))
    self.assertEqual(result[0].dtype, jnp.float32)
    self.assertEqual(result[1].dtype, jnp.float32)

  def test_complex_dtype(self):
    x = jnp.complex64(1.0j)
    (result,) = dtypes.promote_dtype((x,))
    self.assertEqual(result.dtype, jnp.complex64)
    np.testing.assert_allclose(result, x)

  def test_none_only_with_explicit_dtype(self):
    # When dtype is explicitly provided, the inference branch is skipped
    # entirely (the `if dtype is None` block in canonicalize_dtype is not
    # entered), so None-only args don't cause an error.
    result = dtypes.promote_dtype((None,), dtype=jnp.float32)
    self.assertIsNone(result[0])

  def test_empty_tuple_raises(self):
    with self.assertRaises(ValueError):
      dtypes.promote_dtype(())

  def test_all_none_tuple_raises(self):
    with self.assertRaises(ValueError):
      dtypes.promote_dtype((None, None))


if __name__ == '__main__':
  absltest.main()

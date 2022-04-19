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

"""Tests for flax.linen.dtypes."""

import functools
from multiprocessing.sharedctypes import Value

from absl.testing import absltest

from flax import linen as nn
from flax.linen import dtypes

import jax
from jax import numpy as jnp

default_float_dtype = jnp.result_type(1.)

class DtypesTest(absltest.TestCase):

  def test_no_inexact_dtype(self):
    i32 = jnp.int32(1.)
    self.assertEqual(dtypes.canonicalize_dtype(i32, inexact=False), jnp.int32)

  def test_inexact_dtype(self):
    with jax.experimental.enable_x64():
      i64 = jnp.int64(1)
      self.assertEqual(dtypes.canonicalize_dtype(i64), jnp.float32)
    i32 = jnp.int32(1)
    self.assertEqual(dtypes.canonicalize_dtype(i32), jnp.float32)
    i16 = jnp.int16(1.)
    self.assertEqual(dtypes.canonicalize_dtype(i16), jnp.float32)

  def test_explicit_downcast(self):
    f32 = jnp.float32(1.)
    x, = dtypes.promote_dtype(f32, dtype=jnp.float16)
    self.assertEqual(x.dtype, jnp.float16)


if __name__ == '__main__':
  absltest.main()

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


import jax.numpy as jnp
import numpy as np

from flax import nnx

import pytest


class TestStochastic:
  def test_dropout_internal_rngs(self):
    n = 0
    m1 = nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0))
    m2 = nnx.Dropout(rate=0.5, deterministic=False)
    rngs2 = nnx.Rngs(dropout=0)

    @nnx.jit
    def f(m, x, rngs=None):
      nonlocal n
      n += 1
      return m(x, rngs=rngs)

    x = jnp.ones((1, 10))
    assert m1.rngs is not None and m1.rngs.dropout.count.value == 0

    y1 = f(m1, x)
    assert n == 1
    assert m1.rngs.dropout.count.value == 1
    y2 = f(m2, x, rngs=rngs2)
    assert n == 2
    assert rngs2.dropout.count.value == 1
    np.testing.assert_allclose(y1, y2)

    y1 = f(m1, x)
    assert m1.rngs.dropout.count.value == 2
    y2 = f(m2, x, rngs=rngs2)
    assert rngs2.dropout.count.value == 2
    np.testing.assert_allclose(y1, y2)

    assert n == 2

  def test_dropout_rng_override(self):
    m1 = nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0))
    m2 = nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=1))
    x = jnp.ones((1, 10))

    y1 = m1(x)
    y2 = m2(x)
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(y1, y2)

    y2 = m2(x, rngs=nnx.Rngs(dropout=0))
    np.testing.assert_allclose(y1, y2)

  def test_dropout_arg_override(self):
    m = nnx.Dropout(rate=0.5)
    x = jnp.ones((1, 10))

    # deterministic call arg provided
    m(x, deterministic=True)
    # deterministic constructor arg provided
    m.set_attributes(deterministic=True)
    y = m(x)
    # both deterministic call and constructor arg provided
    with pytest.raises(AssertionError):
      np.testing.assert_allclose(
        y, m(x, deterministic=False, rngs=nnx.Rngs(dropout=0))
      )
    # no rng arg provided
    m.set_attributes(deterministic=False)
    with pytest.raises(
      ValueError,
      match='`deterministic` is False, but no `rngs` argument was provided to Dropout',
    ):
      m(x)

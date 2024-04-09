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

from flax.experimental import nnx


class TestStochastic:
  def test_dropout_internal_rngs(self):
    n = 0
    m = nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0))

    @nnx.jit
    def f(m, x):
      nonlocal n
      n += 1
      return m(x)

    x = jnp.ones((1, 10))
    assert m.rngs is not None and m.rngs.dropout.count.value == 0

    y = f(m, x)
    assert n == 1
    assert m.rngs.dropout.count.value == 1

    y = f(m, x)
    assert n == 1
    assert m.rngs.dropout.count.value == 2

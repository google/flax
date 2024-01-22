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

import jax

from flax import linen
from flax.experimental import nnx


class TestCompatibility:
  def test_functional(self):
    # Functional API for NNX Modules
    functional = nnx.compatibility.functional(nnx.Linear)(32, 64)
    state = functional.init(rngs=nnx.Rngs(0))
    x = jax.numpy.ones((1, 32))
    y, updates = functional.apply(state)(x)

  def test_linen_wrapper(self):
    ## Wrapper API for Linen Modules
    linen_module = linen.Dense(features=64)
    x = jax.numpy.ones((1, 32))
    module = nnx.compatibility.LinenWrapper(
      linen_module, x, rngs=nnx.Rngs(0)
    )  # init
    y = module(x)  # apply

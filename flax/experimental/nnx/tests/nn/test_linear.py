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

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.lax import Precision
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx


class TestLinearGeneral:
  def test_basic(self):
    module = nnx.LinearGeneral(2, 3, rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)
    assert module.kernel.shape == (2, 3)
    assert module.bias is not None
    assert module.bias.shape == (3,)

  def test_basic_multi_features(self):
    module = nnx.LinearGeneral(2, (3, 4), rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3, 4)
    assert module.kernel.shape == (2, 3, 4)
    assert module.bias is not None
    assert module.bias.shape == (3, 4)


class TestLinenConsistency(parameterized.TestCase):

  @parameterized.product(
      use_bias = [True, False],
      dtype = [jnp.float32, jnp.float16],
      param_dtype = [jnp.float32, jnp.float16],
      precision = [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
  )
  def test_nnx_linen_equivalence(self, **kwargs):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    OUT_FEATURES = 64

    x = jax.numpy.ones((1, IN_FEATURES))
    model_nnx = nnx.Linear.create_abstract(IN_FEATURES, OUT_FEATURES, **kwargs, rngs=rngs)
    model = linen.Dense(OUT_FEATURES, **kwargs)
    variables = model.init(key, x)
    model_nnx.kernel = variables['params']['kernel']
    if kwargs["use_bias"]:
      model_nnx.bias = variables['params']['bias']

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)

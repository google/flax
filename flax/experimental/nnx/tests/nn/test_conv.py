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

from collections.abc import Sequence

import jax
from absl.testing import parameterized
from jax import numpy as jnp
from jax.lax import Precision
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx


class TestConvLinenConsistency(parameterized.TestCase):

  @parameterized.product(
      strides = [None, (2, 3)],
      padding = ['VALID', (4, 2)],
      input_dilation = [(2, 3)],
      kernel_dilation = [(2, 3)],
      feature_group_count = [3],
      use_bias = [True, False],
      dtype = [jnp.float32],
      param_dtype = [jnp.float16],
      precision = [Precision.HIGHEST],
  )
  def test_nnx_linen_equivalence(self, **kwargs):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 3
    OUT_FEATURES = 6
    INPUT_SHAPE = (24, 9, IN_FEATURES)
    kwargs["kernel_size"] = (7, 4)

    # Cannot use string padding specification for transpose conv
    if isinstance(kwargs["input_dilation"], Sequence) or kwargs["input_dilation"] > 1:
      kwargs["padding"] = (4, 2)

    x = jax.numpy.ones(INPUT_SHAPE)
    model_nnx = nnx.Conv.create_abstract(IN_FEATURES, OUT_FEATURES, **kwargs, rngs=rngs)
    model = linen.Conv(OUT_FEATURES, **kwargs)
    variables = model.init(key, x)
    model_nnx.kernel = variables['params']['kernel']
    if kwargs["use_bias"]:
      model_nnx.bias = variables['params']['bias']

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)

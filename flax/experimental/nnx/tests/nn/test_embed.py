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

import typing as tp

import jax
from absl.testing import parameterized
from jax import numpy as jnp
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx
from flax.typing import Dtype


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    input_dtype=[jnp.int16, jnp.int32],
    num_embeddings=[1, 7],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_nnx_linen_equivalence(
    self,
    input_dtype: tp.Optional[Dtype],
    num_embeddings: int,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    NUM_EMBEDDINGS = num_embeddings

    x = jax.numpy.arange(NUM_EMBEDDINGS, dtype=input_dtype)
    model_nnx = nnx.Embed.create_abstract(
      NUM_EMBEDDINGS,
      IN_FEATURES,
      dtype=dtype,
      param_dtype=param_dtype,
      rngs=rngs,
    )
    model = linen.Embed(
      NUM_EMBEDDINGS, IN_FEATURES, dtype=dtype, param_dtype=param_dtype
    )
    variables = model.init(key, x)
    model_nnx.embedding.value = variables['params']['embedding']

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)

    x = jax.numpy.ones((10,), dtype=input_dtype) * 10
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)
    assert_array_equal(jax.numpy.isnan(out).all(), jax.numpy.array([True]))

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
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np

from flax import linen
from flax import nnx
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
    model_nnx = nnx.eval_shape(
      lambda rngs: nnx.Embed(
        NUM_EMBEDDINGS,
        IN_FEATURES,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
      ),
      rngs,
    )
    model = linen.Embed(
      NUM_EMBEDDINGS, IN_FEATURES, dtype=dtype, param_dtype=param_dtype
    )
    variables = model.init(key, x)
    model_nnx.embedding.set_value(variables['params']['embedding'])

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

    x = jax.numpy.ones((10,), dtype=input_dtype) * 10
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)


  @parameterized.product(
    param_dtype=[jnp.float32, jnp.float16],
    query_dtype=[jnp.float32, jnp.float16],
  )
  def test_attend_default_dtype_matches_linen(
    self, param_dtype: Dtype, query_dtype: Dtype
  ):
    # With dtype=None, linen promotes the query and the table together.
    model_nnx = nnx.Embed(4, 8, param_dtype=param_dtype, rngs=nnx.Rngs(0))
    model = linen.Embed(4, 8, param_dtype=param_dtype)
    variables = model.init(jax.random.key(0), jnp.arange(4))
    model_nnx.embedding.set_value(variables['params']['embedding'])
    query = jnp.ones((2, 8), query_dtype)

    out_nnx = model_nnx.attend(query)
    out = model.apply(variables, query, method=model.attend)
    self.assertEqual(out_nnx.dtype, out.dtype)
    np.testing.assert_array_equal(out, out_nnx)

  def test_default_dtype_follows_the_embedding_after_a_cast(self):
    # "default: same as embedding" holds for the embedding the module has now.
    graphdef, state = nnx.split(nnx.Embed(4, 8, rngs=nnx.Rngs(0)))
    state = jax.tree.map(lambda a: a.astype(jnp.bfloat16), state)
    model = nnx.merge(graphdef, state)

    self.assertEqual(model(jnp.arange(4)).dtype, jnp.bfloat16)
    self.assertEqual(model.attend(jnp.ones((2, 8), jnp.bfloat16)).dtype, jnp.bfloat16)


if __name__ == '__main__':
  absltest.main()

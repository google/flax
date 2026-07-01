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
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from flax import linen
from flax import nnx


class TestEmbed(parameterized.TestCase):
  def test_output_shape(self):
    model = nnx.Embed(num_embeddings=10, features=8, rngs=nnx.Rngs(0))
    x = jnp.array([0, 1, 2, 3])
    out = model(x)
    self.assertEqual(out.shape, (4, 8))

  def test_output_shape_2d(self):
    model = nnx.Embed(num_embeddings=10, features=8, rngs=nnx.Rngs(0))
    x = jnp.array([[0, 1], [2, 3]])
    out = model(x)
    self.assertEqual(out.shape, (2, 2, 8))

  def test_embedding_is_param(self):
    model = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    self.assertIsInstance(model.embedding, nnx.Param)

  def test_embedding_shape(self):
    model = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    self.assertEqual(model.embedding[...].shape, (5, 3))

  def test_invalid_input_dtype_raises(self):
    model = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    x = jnp.array([0.0, 1.0, 2.0])
    with self.assertRaises(ValueError):
      model(x)

  def test_num_embeddings_1_broadcast(self):
    model = nnx.Embed(num_embeddings=1, features=4, rngs=nnx.Rngs(0))
    x = jnp.array([0, 0, 0])
    out = model(x)
    self.assertEqual(out.shape, (3, 4))
    expected = jnp.broadcast_to(model.embedding[...][0], (3, 4))
    np.testing.assert_allclose(out, expected, rtol=1e-6)

  def test_custom_embedding_init(self):
    model = nnx.Embed(
      num_embeddings=5,
      features=3,
      embedding_init=nnx.initializers.zeros_init(),
      rngs=nnx.Rngs(0),
    )
    np.testing.assert_array_equal(
      model.embedding[...], jnp.zeros((5, 3))
    )

  def test_negative_indexing(self):
    model = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    embedding = model.embedding[...]
    out_last = model(jnp.array([-1]))
    np.testing.assert_allclose(out_last[0], embedding[-1], rtol=1e-6)
    out_second_last = model(jnp.array([-2]))
    np.testing.assert_allclose(
      out_second_last[0], embedding[-2], rtol=1e-6
    )

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_dtypes(self, dtype, param_dtype):
    model = nnx.Embed(
      num_embeddings=5,
      features=3,
      dtype=dtype,
      param_dtype=param_dtype,
      rngs=nnx.Rngs(0),
    )
    self.assertEqual(model.embedding[...].dtype, param_dtype)
    x = jnp.array([0, 1, 2])
    out = model(x)
    self.assertEqual(out.dtype, dtype)

  @parameterized.product(
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_dtype_none_defaults_to_param_dtype(self, param_dtype):
    model = nnx.Embed(
      num_embeddings=5,
      features=3,
      param_dtype=param_dtype,
      rngs=nnx.Rngs(0),
    )
    self.assertIsNotNone(model.dtype)
    self.assertEqual(model.dtype, param_dtype)
    x = jnp.array([0, 1, 2])
    out = model(x)
    self.assertEqual(out.dtype, param_dtype)


class TestEmbedAttend(parameterized.TestCase):
  def test_attend_output_shape(self):
    model = nnx.Embed(num_embeddings=10, features=8, rngs=nnx.Rngs(0))
    query = jnp.ones((4, 8))
    out = model.attend(query)
    self.assertEqual(out.shape, (4, 10))

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_attend_computation(self, dtype, param_dtype):
    model = nnx.Embed(
      num_embeddings=5,
      features=3,
      dtype=dtype,
      param_dtype=param_dtype,
      rngs=nnx.Rngs(0),
    )
    query = jnp.ones((2, 3), dtype=dtype)
    out = model.attend(query)
    self.assertEqual(out.dtype, dtype)
    embedding = model.embedding[...].astype(dtype)
    expected = jnp.dot(query, embedding.T)
    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )
    np.testing.assert_allclose(out, expected, rtol=rtol)


class TestLinenConsistency(parameterized.TestCase):
  def _make_models(self, num_embeddings, features, dtype, param_dtype):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    x_init = jnp.arange(num_embeddings, dtype=jnp.int32)
    model_nnx = nnx.eval_shape(
      lambda rngs: nnx.Embed(
        num_embeddings,
        features,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
      ),
      rngs,
    )
    model_linen = linen.Embed(
      num_embeddings, features, dtype=dtype, param_dtype=param_dtype
    )
    variables = model_linen.init(key, x_init)
    model_nnx.embedding.set_value(variables['params']['embedding'])
    np.testing.assert_array_equal(
      model_nnx.embedding[...], variables['params']['embedding']
    )
    return model_nnx, model_linen, variables

  @parameterized.product(
    input_dtype=[jnp.int16, jnp.int32],
    num_embeddings=[1, 7],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_nnx_linen_equivalence(
    self,
    input_dtype,
    num_embeddings,
    dtype,
    param_dtype,
  ):
    model_nnx, model_linen, variables = self._make_models(
      num_embeddings, 32, dtype, param_dtype
    )
    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )

    x = jnp.arange(num_embeddings, dtype=input_dtype)
    out_nnx = model_nnx(x)
    out_linen = model_linen.apply(variables, x)
    np.testing.assert_allclose(out_linen, out_nnx, rtol=rtol)

    # Test broadcast behavior for num_embeddings=1
    if num_embeddings == 1:
      x_broadcast = jnp.zeros((10,), dtype=input_dtype)
      out_nnx = model_nnx(x_broadcast)
      out_linen = model_linen.apply(variables, x_broadcast)
      np.testing.assert_allclose(out_linen, out_nnx, rtol=rtol)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_attend_linen_consistency(self, dtype, param_dtype):
    model_nnx, model_linen, variables = self._make_models(
      7, 16, dtype, param_dtype
    )
    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )

    query = jnp.ones((3, 16), dtype=dtype)
    out_nnx = model_nnx.attend(query)
    out_linen = model_linen.apply(variables, query, method='attend')
    np.testing.assert_allclose(out_nnx, out_linen, rtol=rtol)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_gradient_consistency(self, dtype, param_dtype):
    model_nnx, model_linen, variables = self._make_models(
      7, 16, dtype, param_dtype
    )
    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )

    query = jnp.ones((3, 16), dtype=dtype)

    def nnx_loss(model):
      return model.attend(query).sum()

    def linen_loss(params):
      return model_linen.apply(
        {'params': params}, query, method='attend'
      ).sum()

    val_nnx, grad_nnx = nnx.value_and_grad(nnx_loss)(model_nnx)
    val_linen, grad_linen = jax.value_and_grad(linen_loss)(
      variables['params']
    )

    np.testing.assert_allclose(
      float(val_nnx), float(val_linen), rtol=rtol
    )
    np.testing.assert_allclose(
      grad_nnx.embedding[...],
      grad_linen['embedding'],
      rtol=rtol,
    )


if __name__ == '__main__':
  absltest.main()

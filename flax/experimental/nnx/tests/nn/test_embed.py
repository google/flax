import jax
from absl.testing import parameterized
from jax import numpy as jnp
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx


class TestLinenConsistency(parameterized.TestCase):

  @parameterized.product(
      input_dtype = [jnp.int16, jnp.int32],
      dtype = [jnp.float32, jnp.float16],
      param_dtype = [jnp.float32, jnp.float16],
  )
  def test_nnx_linen_equivalence(self, input_dtype, **kwargs):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    NUM_EMBEDDINGS = 7

    x = jax.numpy.ones((10,), dtype=input_dtype)
    model_nnx = nnx.Embed.create_abstract(NUM_EMBEDDINGS, IN_FEATURES, **kwargs, rngs=rngs)
    model = linen.Embed(NUM_EMBEDDINGS, IN_FEATURES, **kwargs)
    variables = model.init(key, x)
    model_nnx.embedding = variables['params']['embedding']

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert_array_equal(out, out_nnx)

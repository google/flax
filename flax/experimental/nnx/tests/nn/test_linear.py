import jax
from absl.testing import parameterized
from jax import numpy as jnp
from jax.lax import Precision
from numpy.testing import assert_array_equal

from flax import linen
from flax.experimental import nnx


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

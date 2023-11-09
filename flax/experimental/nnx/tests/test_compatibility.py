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

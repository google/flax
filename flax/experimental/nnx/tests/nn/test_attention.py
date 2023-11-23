import jax.numpy as jnp

from flax.experimental import nnx


class TestMultiHeadAttention:
  def test_basic(self):
    module = nnx.MultiHeadAttention(2, 3, 6, rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 7, 3)))
    assert y.shape == (1, 7, 6)

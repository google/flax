
import jax.numpy as jnp

from flax.experimental import nnx


class TestStochastic:
  def test_dropout_internal_rngs(self):
    n = 0
    m = nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(dropout=0))

    @nnx.jit
    def f(m, x):
      nonlocal n
      n += 1
      return m(x)

    x = jnp.ones((1, 10))
    assert m.rngs is not None and m.rngs.dropout.count.value == 0

    y = f(m, x)
    assert n == 1
    assert m.rngs.dropout.count.value == 1

    y = f(m, x)
    assert n == 1
    assert m.rngs.dropout.count.value == 2

import jax
import jax.numpy as jnp

from flax.nnx.sow import sow, capture

def test_multiple_sows_same_name_in_order():
  def f(x):
    sow(x, name="a")
    sow(x + 1, name="a")
    return x

  _, collected = jax.jit(capture(f))(jnp.float32(10.0))
  assert collected["a"] == (10.0, 11.0)


def test_pytree_value():
  def f(x):
    sow({"lo": x, "hi": x + 1}, name="pair")
    return x

  _, collected = jax.jit(capture(f))(jnp.float32(5.0))
  (d,) = collected["pair"]
  assert d["lo"] == 5.0 and d["hi"] == 6.0

def test_plain_jit_and_grad_still_work():
  def f(x):
    sow(x, name="x")
    return jnp.sin(x)

  assert jnp.allclose(jax.jit(f)(jnp.float32(1.0)), jnp.sin(1.0))
  assert jnp.allclose(jax.grad(f)(jnp.float32(1.0)), jnp.cos(1.0))
  xs = jnp.arange(3, dtype=jnp.float32)
  assert jnp.allclose(jax.vmap(f)(xs), jnp.sin(xs))

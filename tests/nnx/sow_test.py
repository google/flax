import jax
import jax.numpy as jnp

from flax.nnx.sow import sow, perturb, capture

def test_sow_vmap():
  @jax.vmap
  def f(x):
    sow(x, name="a")
    return x

  _, collected = capture(f)(jnp.arange(5))
  print(collected)


def test_bare_statement_collected():
  # No use of sow's return value -> effect keeps it alive.
  def f(x):
    sow(x * 2, name="doubled")
    return x + 1

  out, collected = capture(f)(jnp.float32(3.0))
  assert out == 4.0
  assert set(collected) == {"doubled"}
  assert len(collected["doubled"]) == 1
  assert collected["doubled"][0] == 6.0


def test_multiple_sows_same_name_in_order():
  def f(x):
    sow(x, name="a")
    sow(x + 1, name="a")
    return x

  _, collected = capture(f)(jnp.float32(10.0))
  assert collected["a"] == (10.0, 11.0)


def test_pytree_value():
  def f(x):
    sow({"lo": x, "hi": x + 1}, name="pair")
    return x

  _, collected = capture(f)(jnp.float32(5.0))
  (d,) = collected["pair"]
  assert d["lo"] == 5.0 and d["hi"] == 6.0


def test_output_unchanged_vs_plain():
  def f(x):
    sow(x, name="x")
    return jnp.sin(x) + 2

  x = jnp.float32(1.5)
  out, _ = capture(f)(x)
  assert jnp.allclose(out, jnp.sin(x) + 2)


def test_plain_jit_and_grad_still_work():
  # sow is a no-op identity under normal execution / jit / grad.
  def f(x):
    sow(x, name="x")
    return jnp.sin(x)

  assert jnp.allclose(jax.jit(f)(jnp.float32(1.0)), jnp.sin(1.0))
  assert jnp.allclose(jax.grad(f)(jnp.float32(1.0)), jnp.cos(1.0))
  xs = jnp.arange(3, dtype=jnp.float32)
  assert jnp.allclose(jax.vmap(f)(xs), jnp.sin(xs))


def test_sow_inside_jit_is_collected():
  @jax.jit
  def inner(x):
    sow(x * 3, name="inner")
    return x

  def f(x):
    return inner(x) + 1

  out, collected = capture(f)(jnp.float32(2.0))
  out, collected = capture(f)(jnp.float32(2.0))
  assert out == 3.0
  assert collected["inner"] == (6.0,)


def test_sown_is_jittable():
  # The whole transform runs under jit; collected values become jit outputs.
  def f(x):
    sow(x * 2, name="m")
    return x + 1

  out, collected = jax.jit(capture(f))(jnp.float32(4.0))
  assert out == 5.0
  assert jnp.allclose(collected["m"][0], 8.0)


def test_sow_inside_scan_is_stacked():
  # A per-iteration sow is stacked along the scan axis as one entry.
  def f(x):
    def body(c, _):
      sow(c, name="step")
      return c + 1, c
    c, _ = jax.lax.scan(body, x, None, length=4)
    return c

  out, collected = capture(f)(jnp.float32(0.0))
  assert out == 4.0
  (steps,) = collected["step"]
  assert jnp.array_equal(steps, jnp.arange(4, dtype=jnp.float32))


def test_sow_inside_cond():
  # Both branches sow the same name/shape -> collected is the taken branch's.
  def f(x, pred):
    def t(v):
      sow(v * 10, name="branch")
      return v
    def fl(v):
      sow(v * 100, name="branch")
      return v
    return jax.lax.cond(pred, t, fl, x)

  _, collected = capture(f)(jnp.float32(2.0), True)
  assert collected["branch"] == (20.0,)
  _, collected = capture(f)(jnp.float32(2.0), False)
  assert collected["branch"] == (200.0,)


def test_sow_inside_scan_under_jit():
  def f(x):
    def body(c, _):
      sow(c, name="step")
      return c + 1, c
    c, _ = jax.lax.scan(body, x, None, length=3)
    return c

  out, collected = jax.jit(capture(f))(jnp.float32(0.0))
  assert out == 3.0
  assert jnp.array_equal(collected["step"][0], jnp.arange(3, dtype=jnp.float32))


def test_pytree_sow_inside_scan():
  def f(x):
    def body(c, _):
      sow({"v": c, "sq": c * c}, name="d")
      return c + 1, c
    c, _ = jax.lax.scan(body, x, None, length=3)
    return c

  _, collected = capture(f)(jnp.float32(0.0))
  (d,) = collected["d"]
  assert jnp.array_equal(d["v"], jnp.arange(3, dtype=jnp.float32))
  assert jnp.array_equal(d["sq"], jnp.arange(3, dtype=jnp.float32) ** 2)


def test_sow_in_custom_vjp_backward():
  # A custom_vjp whose backward rule sows: sown(grad(...)) harvests it because
  # the bwd's sow is staged into the gradient jaxpr at trace time.
  @jax.custom_vjp
  def h(x):
    return x ** 2

  def h_fwd(x):
    return h(x), x

  def h_bwd(x, g):
    gx = 2 * x * g
    sow(gx, name="grad_h")
    return (gx,)

  h.defvjp(h_fwd, h_bwd)

  def loss(x):
    return jnp.sum(h(x))

  x = jnp.arange(3.0)
  grad, collected = capture(jax.grad(loss))(x)
  assert jnp.array_equal(grad, 2 * x)
  assert jnp.array_equal(collected["grad_h"][0], 2 * x)


def test_sow_inside_remat():
  # differentiated=False: the body is inlined, exactly as jax lowers it.
  def f(x):
    y = jnp.sin(x)
    sow(y, name="y")
    return y * 2

  out, collected = capture(jax.checkpoint(f))(jnp.float32(1.0))
  assert jnp.allclose(out, jnp.sin(1.0) * 2)
  assert jnp.allclose(collected["y"][0], jnp.sin(1.0))


def test_sow_inside_remat_under_grad_fires_once():
  # The known half is inlined by remat's partial eval, so the forward sow lands
  # at the top level of the gradient jaxpr and is collected exactly once.
  def f(x):
    y = jnp.sin(x)
    sow(y, name="y")
    return jnp.sum(y * 2)

  x = jnp.float32(1.0)
  g, collected = capture(jax.grad(jax.checkpoint(f)))(x)
  assert jnp.allclose(g, jnp.cos(x) * 2)
  assert len(collected["y"]) == 1
  assert jnp.allclose(collected["y"][0], jnp.sin(x))


def test_perturb_inside_remat_keeps_barrier():
  # differentiated=True: grad stages the cotangent sow INSIDE the remat body.
  # We rebind remat_p rather than inline, so the remat2 eqn must survive.
  def f(x):
    return jnp.sum(perturb(jnp.sin(x), name="dx") ** 2)

  x = jnp.float32(2.0)
  g, collected = capture(jax.grad(jax.checkpoint(f)))(x)
  assert jnp.allclose(g, 2 * jnp.sin(x) * jnp.cos(x))
  assert jnp.allclose(collected["dx"][0], 2 * jnp.sin(x))
  assert "remat2" in str(jax.make_jaxpr(capture(jax.grad(jax.checkpoint(f))))(x))


def test_pytree_sow_inside_remat_under_grad():
  def f(x):
    d = perturb({"a": jnp.sin(x), "b": jnp.cos(x)}, name="dp")
    return jnp.sum(d["a"] ** 2 + d["b"] * 3)

  x = jnp.float32(0.5)
  _, collected = capture(jax.grad(jax.checkpoint(f)))(x)
  (d,) = collected["dp"]
  assert jnp.allclose(d["a"], 2 * jnp.sin(x))
  assert jnp.allclose(d["b"], 3.0)


def test_sow_inside_while_raises():
  import pytest

  def f(x):
    def cond(c):
      return c < 3
    def body(c):
      sow(c, name="w")
      return c + 1
    return jax.lax.while_loop(cond, body, x)

  with pytest.raises(NotImplementedError):
    capture(f)(jnp.float32(0.0))

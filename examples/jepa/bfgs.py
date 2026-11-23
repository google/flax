import jax
import jax.numpy as jnp
from jax import lax

def make_bfgs(fg, hess_fn, n, max_iter=60, max_step=0.5, damp=1e-2,
              c1=0.3, c2=0.5, ls_eps=1e-3):
    I = jnp.eye(n)

    def h2h_jit(hessian):
        def cond(H):
            return jnp.linalg.matrix_rank(H) < n
        return lax.while_loop(cond, lambda H: H + I, hessian)

    def wolfe(theta, dk, f0, g0, Xb, theta_tgt):
        dt = dk.dtype
        slope = jnp.dot(g0, dk)

        def cond(c):
            lo, hi, alpha, done = c
            return (~done) & (hi - lo > ls_eps) & (lo < hi)

        def body(c):
            lo, hi, alpha, _ = c
            f1, g1 = fg(theta + alpha * dk, Xb, theta_tgt)
            armijo = f1 <= f0 + c1 * alpha * slope
            curvature = jnp.dot(g1, dk) >= c2 * slope
            done = armijo & curvature
            lo2 = jnp.where(armijo, alpha, lo)
            hi2 = jnp.where(armijo, hi, alpha)
            a2 = 0.5 * (lo2 + hi2)
            return (lo2, hi2, jnp.where(done, alpha, a2), done)

        init = (jnp.array(0., dt), jnp.array(2., dt),
                jnp.array(1., dt), jnp.array(False))
        return lax.while_loop(cond, body, init)[2]

    @jax.jit
    def solve(theta, Xb, theta_tgt, epsilon):
        f0, g0 = fg(theta, Xb, theta_tgt)
        H = h2h_jit(hess_fn(theta, Xb, theta_tgt))
        H = jnp.where(jnp.all(jnp.isfinite(H)), H, I)
        Hd = H + damp * I
        dk = -jnp.linalg.solve(Hd, g0)
        dk = jnp.clip(dk, -max_step, max_step)
        dk = jnp.where(jnp.dot(g0, dk) >= 0,
                       -jnp.clip(g0, -max_step, max_step), dk)

        def cond(c):
            _, _, d, _, _, it = c
            return (it < max_iter) & (jnp.linalg.norm(d) >= epsilon)

        def body(c):
            th, H, d, f, g, it = c
            alpha = wolfe(th, d, f, g, Xb, theta_tgt)
            s = alpha * d
            th2 = th + s
            f1, g1 = fg(th2, Xb, theta_tgt)
            yk = g1 - g
            sTy = jnp.dot(s, yk)
            Hs = H @ s
            sTHs = jnp.dot(s, Hs)
            ok = (jnp.all(jnp.isfinite(yk)) & jnp.isfinite(sTy) & (sTy > 1e-12)
                  & jnp.all(jnp.isfinite(Hs)) & jnp.isfinite(sTHs)
                  & (sTHs > 1e-12))

            safe1 = jnp.where(sTy > 1e-12, sTy, 1.0)
            safe2 = jnp.where(sTHs > 1e-12, sTHs, 1.0)
            H2 = jnp.where(ok,
                           H + jnp.outer(yk, yk) / safe1
                             - jnp.outer(Hs, Hs) / safe2,
                           I)
            H2 = jnp.where(jnp.all(jnp.isfinite(H2)), H2, I)
            d2 = -jnp.linalg.solve(H2 + damp * I, g1)
            d2 = jnp.clip(d2, -max_step, max_step)
            d2 = jnp.where(jnp.dot(g1, d2) >= 0,
                           -jnp.clip(g1, -max_step, max_step), d2)
            return (th2, H2, d2, f1, g1, it + 1)

        carry = (theta, H, dk, f0, g0, jnp.array(0, jnp.int32))
        out = lax.while_loop(cond, body, carry)
        return out[0], f0, out[3]

    return solve
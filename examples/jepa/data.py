import jax
import jax.numpy as jnp

def make_series(n_windows=512, ctx_steps=8, tgt_steps=3, d=4, seed=7):
    key = jax.random.PRNGKey(seed)
    k_phase, k_mix, k_noise = jax.random.split(key, 3)
    total = n_windows + ctx_steps + tgt_steps
    t = jnp.linspace(0, 60, total)[:, None]
    phases = jax.random.uniform(k_phase, (1, d), minval=0., maxval=2 * jnp.pi,
                                dtype=jnp.float64)
    freqs = jnp.linspace(0.35, 0.95, d)[None, :]
    base = (jnp.sin(freqs * t + phases)
            + 0.35 * jnp.sin(2.3 * freqs * t + 0.7 * phases))
    mix = jax.random.normal(k_mix, (d, d), dtype=jnp.float64) / jnp.sqrt(d)
    noise = 0.02 * jax.random.normal(k_noise, base.shape, dtype=jnp.float64)
    signal = base @ mix + noise
    L = ctx_steps + tgt_steps
    starts = jnp.arange(n_windows)
    return jax.vmap(lambda i: jax.lax.dynamic_slice(signal, (i, 0), (L, d)))(starts)
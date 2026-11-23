import jax.numpy as jnp

def linear_probe(model, theta, theta0, X):
    ctx = X[:, :model.ctx_steps, :]
    y = X[:, model.ctx_steps:model.ctx_steps + model.tgt_steps, :]
    y = y.reshape(X.shape[0], -1)

    def r2(A):
        A = jnp.concatenate([A, jnp.ones((A.shape[0], 1))], axis=1)
        W = jnp.linalg.solve(A.T @ A + 1e-3 * jnp.eye(A.shape[1]), A.T @ y)
        pred = A @ W
        return 1.0 - jnp.sum((y - pred) ** 2) / jnp.sum((y - y.mean(axis=0)) ** 2)

    Z = jnp.mean(model.encode(theta[:model.n_enc], ctx), axis=1)
    Z0 = jnp.mean(model.encode(theta0[:model.n_enc], ctx), axis=1)
    raw = X[:, model.ctx_steps - 1, :]
    return float(r2(Z)), float(r2(Z0)), float(r2(raw))
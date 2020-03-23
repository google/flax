import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Union
import jax.ops as ops


def _diag_shift(mat: jnp.ndarray,
                val: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """ Shifts the diagonal of mat by val. """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)


def multivariate_gaussian_kl(q, p):
    """ KL-divergence between multivariate Gaussian distributions defined as

        ∫ N(q.mean, q.scale) log{ N(q.mean, q.scale) / N (p.mean, p.scale) }.

    Args:
        q: `MultivariateNormal` object
        p: `MultivariateNormal` object

    Returns:
        kl: Python `float` the KL-divergence between `q` and `p`.
    """
    m_diff = q.mean - p.mean
    return .5*(2*jnp.log(jnp.diag(p.scale)).sum() - 2*jnp.log(jnp.diag(q.scale)).sum()
               - q.mean.shape[-1]
               + jnp.trace(jscipy.linalg.cho_solve((p.scale, True), q.scale) @ q.scale.T)
               + jnp.sum(m_diff * jscipy.linalg.cho_solve((p.scale, True), m_diff)))

import jax.numpy as jnp
import jax.scipy as jscipy
import abc
from jax import ops, random
from flax import struct
from typing import Callable
import kernels


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


def _diag_shift(mat, val):
    """

    Args:
        mat: `array_like`, a square matrix of shape `[..., N, N]`
        val: `array_like`, the value to be added to the diagonal
          of mat.

    Returns:
        mat: `array_like` the original `mat` with the diagonal
          indices updated.

    """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)


def linear_gaussian_conditional(
        y,
        lin_op, shift, noise_cov,
        prior_dist):
    """

    Args:
        y:
        lin_op:
        shift:
        noise_cov:
        prior_dist:

    Returns:

    """
    noise_cov_chol = jnp.linalg.cholesky(noise_cov)
    # intermed. expr
    lin_op_matmul_prior_scale = lin_op @ prior_dist.scale
    expr = noise_cov + lin_op_matmul_prior_scale @ lin_op_matmul_prior_scale.T
    expr = lin_op @ jnp.linalg.cholesky(expr)
    expr = jscipy.linalg.cho_solve((prior_dist.scale, True), expr)

    cond_cov = prior_dist.cov - expr @ expr.T
    cond_mean = cond_cov @ (
            jscipy.linalg.cho_solve((noise_cov_chol, True), y - shift)
            + jscipy.linalg.cho_solve((prior_dist.scale, prior_dist.mean)))

    return MultivariateNormalFull(cond_mean, cond_cov)


@struct.dataclass
class MultivariateNormal:

    @abc.abstractmethod
    def log_prob(self, x):
        pass


@struct.dataclass
class MultivariateNormalDiag(MultivariateNormal):
    mean: jnp.ndarray
    scale_diag: jnp.ndarray

    def log_prob(self, x):
        return jnp.sum(
        jscipy.stats.norm.logpdf(
            x, loc=self.mean, scale=self.scale_diag))

    def sample(self, key, shape=()):
        return random.normal(key, shape=shape) * self.scale_diag + self.mean


@struct.dataclass
class MultivariateNormalTriL(MultivariateNormal):
    mean: jnp.ndarray
    scale: jnp.ndarray

    def log_prob(self, x):
        dim = x.shape[-1]
        dev = x - self.mean
        maha = jnp.sum(dev *
                       jscipy.linalg.cho_solve((self.scale, True), dev))
        log_2_pi = jnp.log(2 * jnp.pi)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(self.scale)))
        return -0.5 * (dim * log_2_pi + log_det_cov + maha)

    def sample(self, key, shape=()):
        full_shape = shape + self.mean.shape
        std_normals = random.normal(key, full_shape)
        return self.mean
        #return jnp.tensordot(std_normals, self.scale, [-1, 1]) + self.mean

    @property
    def covariance(self):
        return self.scale @ self.scale.T


@struct.dataclass
class MultivariateNormalFull(MultivariateNormal):
    mean: jnp.ndarray
    covariance: jnp.ndarray

    def log_prob(self, x):
        scale = jnp.linalg.cholesky(self.covariance)
        dim = x.shape[-1]
        dev = x - self.mean
        maha = jnp.sum(dev *
                       jscipy.linalg.cho_solve((scale, True), dev))
        log_2_pi = jnp.log(2 * jnp.pi)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(scale)))
        return -0.5 * (dim * log_2_pi + log_det_cov + maha)

    def sample(self, key, shape=()):
        return random.multivariate_normal(
            key, self.mean, self.covariance, shape)

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import random
from flax import struct
from typing import Callable


@struct.dataclass
class GaussianProcess:
    index_points: jnp.ndarray
    mean_function: Callable = struct.field(pytree_node=False)
    kernel_function: Callable = struct.field(pytree_node=False)
    jitter: float

    def marginal(self):
        kxx = self.kernel_function(self.index_points)
        chol_kxx = jnp.linalg.cholesky(kxx)
        mean = self.mean_function(self.index_points)
        return MultivariateNormalTriL(mean, chol_kxx)

    def posterior_predictive(self, y, xnew):
        """ Returns p(f(xnew) | y, index_points)"""
        k11 = self.kernel_function(xnew)
        k12 = self.kernel_function(xnew, self.index_points)
        p = self.marginal()

        cond_mean = (self.mean_function(xnew)
                     + k12 @ jscipy.linalg.cho_solve((p.scale, True), y - p.mean))
        cond_cov = k11 - k12 @ jscipy.linalg.cho_solve((p.scale, True), k12.T)

        # don't return the TriL parameterised version because
        # predictive index points are often very dense and cholesky will fail
        # ToDo(dan): optionally return only the diag of cond_cov
        return MultivariateNormal(cond_mean, cond_cov)


@struct.dataclass
class MultivariateNormal:
    mean: jnp.ndarray
    covariance: jnp.ndarray


@struct.dataclass
class MultivariateNormalTriL:
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
        std_gaussians = random.normal(key, full_shape)
        return jnp.tensordot(std_gaussians, self.scale, [-1, 1]) + self.mean
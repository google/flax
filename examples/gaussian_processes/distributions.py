import jax.numpy as jnp
import jax.scipy as jscipy
import abc
from jax import random
from flax import struct
from typing import Callable
import kernels


# multivariate normal utility function
def base_conditional(
        kmn: jnp.ndarray,
        kmm: jnp.ndarray,
        knn: jnp.ndarray,
        f: jnp.ndarray,
        *,
        full_cov: bool = False):
    """ For Gaussian distributions x ~ N(0, kmm)
    and y|x = N(
    """
    kmm_chol = jnp.linalg.cholesky(kmm)
    return 1., 2.

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
        return jnp.tensordot(std_normals, self.scale, [-1, 1]) + self.mean

    @property
    def covariance(self):
        return self.scale @ self.scale.T


@struct.dataclass
class MultivariateNormalFull(MultivariateNormal):
    mean: jnp.ndarray
    covariance: jnp.ndarray

    def log_prob(self, x):
        return jscipy.stats.multivariate_normal.logpdf(x, self.mean, self.covariance)

    def sample(self, key, shape=()):
        return random.multivariate_normal(
            key, self.mean, self.covariance, shape)


@struct.dataclass
class GaussianProcess:
    index_points: jnp.ndarray
    mean_function: Callable = struct.field(pytree_node=False)
    kernel_function: Callable = struct.field(pytree_node=False)
    jitter: float

    def marginal(self):
        kxx = self.kernel_function(self.index_points, self.index_points)
        chol_kxx = jnp.linalg.cholesky(kxx)
        mean = self.mean_function(self.index_points)
        return MultivariateNormalTriL(mean, chol_kxx)

    def posterior_gp(self, y, x_new, observation_noise_variance, jitter=None):
        """ Returns a new GP conditional on y. """
        cond_kernel_fn, _ = kernels.SchurComplementKernelProvider.init(
            None,
            self.kernel_function,
            self.index_points,
            observation_noise_variance)

        k_xnew_x = self.kernel_function(x_new, self.index_points)
        marginal = self.marginal()

        def cond_mean_fn(x):
            return (self.mean_function(x_new)
                    + k_xnew_x @ jscipy.linalg.cho_solve(
                        (marginal.scale, True), y - marginal.mean))

        jitter = jitter if jitter else self.jitter
        return GaussianProcess(x_new,
                               cond_mean_fn,
                               cond_kernel_fn,
                               jitter)

    def posterior_predictive(self,
                             y,
                             x_new,
                             py):
        """ Returns p(f(x_new) | y, index_points)

        Args:
            y: `jnp.ndarray` the observed values of the output
            x_new: `jnp.ndarray` index points to predict at.
            py: `Distribution` object
        """
        if not isinstance(py, MultivariateNormal):
            raise NotImplementedError(
                'Analytic posterior only implemented for multivariate ',
                'Gaussian observation distributions')

        cov_x_new = self.kernel_function(x_new)
        cov_x_new_x = self.kernel_function(x_new, self.index_points)
        p = self.marginal()

        cond_mean, cond_cov = base_conditional(
            cov_x_new_x,
            p.scale @ p.scale.T + py.covariance,
            cov_x_new,
            y - p.mean,
            full_cov=True)

        cond_mean = cond_mean + self.mean_function(x_new)

        # don't return the TriL parameterised version because
        # predictive index points are often very dense and cholesky will fail
        # ToDo(dan): optionally return only the diag of cond_cov
        return MultivariateNormalFull(cond_mean, cond_cov)
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


@struct.dataclass
class GaussianProcess:
    index_points: jnp.ndarray
    mean_function: Callable = struct.field(pytree_node=False)
    kernel_function: Callable = struct.field(pytree_node=False)
    jitter: float

    def marginal(self):
        kxx = self.kernel_function(self.index_points, self.index_points)
        chol_kxx = jnp.linalg.cholesky(_diag_shift(kxx, self.jitter))
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
                        (cond_kernel_fn.divisor_matrix_cholesky, True),
                        y - marginal.mean))

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

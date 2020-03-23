import jax.numpy as jnp
import jax.scipy as jscipy
from flax import struct, nn
import dists
from utils import _diag_shift, multivariate_gaussian_kl
from typing import Any, Callable
import kernels


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
        return dists.MultivariateNormalTriL(mean, chol_kxx)

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


@struct.dataclass
class VariationalGaussianProcess(GaussianProcess):
    """ ToDo(dan): ugly `Any` typing to avoid circular dependency with GP
          inside of inducing_variables. Ideally break this by lifting
          variational GPs into their own module.
    """
    inducing_variable: Any

    def prior_kl(self):
        if self.inducing_variable.whiten:
            return self.prior_kl_whiten()
        else:
            qu = self.inducing_variable.variational_distribution
            pu = self.inducing_variable.prior_distribution
            return multivariate_gaussian_kl(qu, pu)

    def prior_kl_whiten(self):
        qu = self.inducing_variable.variational_distribution
        log_det = 2*jnp.sum(jnp.log(jnp.diag(qu.scale)))
        dim = qu.mean.shape[-1]
        return -.5*(log_det + 0.5*dim - jnp.sum(qu.mean**2) - jnp.sum(qu.scale**2))


class SVGPProvider(nn.Module):
    def apply(self,
              index_points,
              mean_fn,
              kernel_fn,
              inducing_var,
              jitter=1e-4):
        """

        Args:
            index_points: the nd-array of index points of the GP model.
            mean_fn: callable mean function of the GP model.
            kernel_fn: callable kernel function.
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`.
            jitter: float `jitter` term to add to the diagonal of the covariance
              function before computing Cholesky decompositions.

        Returns:
            svgp: A sparse Variational GP model.
        """
        z = inducing_var.locations
        qu = inducing_var.variational_distribution
        qu_mean = qu.mean
        qu_scale = qu.scale

        # cholesky of the base kernel function applied at the inducing point
        # locations.
        kzz_chol = jnp.linalg.cholesky(
            _diag_shift(kernel_fn(z, z), jitter))

        if inducing_var.whiten:
            qu_mean = kzz_chol @ qu_mean
            qu_scale = kzz_chol @ qu_scale

        z = inducing_var.locations

        var_kern = kernels.VariationalKernel(
            kernel_fn, z, qu_scale)

        def var_mean(x_):
            kxz = kernel_fn(x_, z)
            dev = (qu_mean - mean_fn(z))[..., None]
            return (mean_fn(x_)[..., None]
                    + kxz @ jscipy.linalg.cho_solve(
                        (kzz_chol, True), dev))[..., 0]

        return VariationalGaussianProcess(
            index_points, var_mean, var_kern, jitter, inducing_var)

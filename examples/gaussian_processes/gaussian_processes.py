import jax.numpy as jnp
import jax.scipy as jscipy
from flax import struct
import dists
from inducing_variables import InducingVariable
from utils import _diag_shift, multivariate_gaussian_kl
from typing import Callable
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
    inducing_variable: InducingVariable

    def prior_kl(self):
        qu = self.inducing_variable.variational_distribution
        pu = self.inducing_variable.prior_distribution
        return multivariate_gaussian_kl(qu, pu)

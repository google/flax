import jax
import jax.numpy as jnp
from flax import struct, nn
from jax import random
from distributions import MultivariateNormalDiag, MultivariateNormalTriL
from gaussian_processes import GaussianProcess
from typing import Union, Callable


@struct.dataclass
class InducingVariable:
    variational_distribution: MultivariateNormalTriL
    prior_distribution: MultivariateNormalTriL


@struct.dataclass
class InducingPointsVariable(InducingVariable):
    locations: jnp.ndarray
    whiten: bool = False


class InducingPointsProvider(nn.Module):
    """ Handles parameterisation of an inducing points variable. """
    def apply(self,
              index_points: jnp.ndarray,
              kernel_fun: Callable,
              num_inducing_points: int,
              inducing_locations_init: Union[Callable, None] = None,
              fixed_locations: bool = False,
              whiten: bool = False,
              jitter: float = 1e-4,
              dtype: jnp.dtype = jnp.float64) -> InducingPointsVariable:
        """

        Args:
            index_points: the nd-array of index points of the GP model.
            kernel_fun: callable kernel function.
            num_inducing_points: total number of inducing points.
            inducing_locations_init: initializer function for the inducing
              variable locations.
            fixed_locations: boolean specifying whether to optimise the inducing
              point locations (default True).
            whiten: boolean specifying whether to apply the whitening transformation.
              (default False)
            jitter: float `jitter` term to add to the diagonal of the covariance
              function of the GP prior of the inducing variable, only used if no
              whitening transform applied.
            dtype: the data-type of the computation (default: float64)

        Returns:
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`

        """
        n_features = index_points.shape[-1]
        z_shape = (num_inducing_points, n_features)
        if inducing_locations_init is None:
            inducing_locations_init = lambda key, shape: random.normal(key, z_shape)

        if fixed_locations:
            _default_key = random.PRNGKey(0)
            z = inducing_locations_init(_default_key, z_shape)
        else:
            z = self.param('locations',
                           (num_inducing_points, n_features),
                           inducing_locations_init)

        qu_mean = self.param('mean', (num_inducing_points,),
                             lambda key, shape: jax.nn.initializers.zeros(
                                 key, z_shape[0], dtype=dtype))

        qu_scale = self.param(
            'scale',
            (num_inducing_points, num_inducing_points),
            lambda key, shape: jnp.eye(num_inducing_points, dtype=dtype))

        if whiten:
            prior = MultivariateNormalDiag(
                mean=jnp.zeros(index_points.shape[-1]),
                scale_diag=jnp.ones(index_points.shape[-2]))

        else:
            prior = GaussianProcess(
                    z,
                    lambda x_: jnp.zeros(x_.shape[:-1]),
                    kernel_fun,
                    jitter).marginal()

        return InducingPointsVariable(
            variational_distribution=MultivariateNormalTriL(
                qu_mean, jnp.tril(qu_scale)),
            prior_distribution=prior,
            locations=z,
            whiten=whiten)

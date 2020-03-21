import jax
import jax.numpy as jnp
from flax import struct, nn
from jax import random
from distributions import MultivariateNormalTriL
from gaussian_processes import GaussianProcess
from typing import Union, Callable


@struct.dataclass
class InducingVariable:
    variational_distribution: MultivariateNormalTriL
    prior_distribution: MultivariateNormalTriL


@struct.dataclass
class InducingPointsVariable(InducingVariable):
    locations: jnp.ndarray


class InducingPointsProvider(nn.Module):
    """ Handles parameterisation of an inducing points variable. """
    def apply(self,
              index_points: jnp.ndarray,
              kernel_fun: Callable,
              inducing_locations_init: Union[Callable, None] = None,
              num_inducing_points: int = 5,
              fixed_locations: bool = False,
              dtype: jnp.dtype = jnp.float64) -> InducingPointsVariable:
        """

        Args:
            index_points: the nd-array of index points of the GP model.
            kernel_fun: callable kernel function.
            inducing_locations_init: initializer function for the inducing
              variable locations.
            num_inducing_points: total number of inducing points.
            fixed_locations: boolean specifying whether to opimise the inducing
              point locations (default True).
            dtype: the data-type of the computation (default: float64)

        Returns:
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`

        """
        n_features = index_points.shape[-1]
        z_shape = (num_inducing_points, n_features)

        if inducing_locations_init is None:
            inducing_locations_init = lambda key, shape : random.normal(key, z_shape)

        if fixed_locations:
            z = inducing_locations_init(None, z_shape)
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
        print(z)
        prior = GaussianProcess(
            z,
            lambda x_: jnp.zeros(x_.shape[:-1]),
            kernel_fun,
            1e-6).marginal()

        return InducingPointsVariable(
            variational_distribution=MultivariateNormalTriL(
                qu_mean, jnp.tril(qu_scale)),
            prior_distribution=prior,
            locations=z)

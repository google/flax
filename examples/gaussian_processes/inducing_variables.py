import jax.numpy as jnp
import jax
from flax import struct
from jax import random
from flax import nn
from distributions import MultivariateNormalTriL

import kernels


@struct.dataclass
class InducingVariable:
    variational_distribution: MultivariateNormalTriL
    prior_distribution: MultivariateNormalTriL


@struct.dataclass
class InducingPointsVariable(InducingVariable):
    locations: jnp.ndarray


class InducingPoints(nn.Module):
    """ Inducing variables as points in the original input domain
    """
    def apply(self,
              index_points: jnp.ndarray,
              num_inducing_points: int =5,
              dtype=jnp.float64) -> (jnp.ndarray, MultivariateNormalTriL):
        """

        Args:
            index_points:
            num_inducing_points:
            dtype:

        Returns:
            z: array_like of the inducing points locations of shape
              `[num_inducing_points, n_features]`.
            qu: `MultivariateNormalTriL` object giving the distribution
              of the inducing variables at locations `z`.
        """
        n_features = index_points.shape[-1]
        minval = jnp.min(index_points, axis=-2)
        maxval = jnp.max(index_points, axis=-2)

        z_init = lambda key, shape: random.uniform(
            key,
            shape=(num_inducing_points, n_features),
            minval=jnp.atleast_2d(minval),
            maxval=jnp.atleast_2d(maxval),
            dtype=dtype)

        z = self.param('locations',
                       (num_inducing_points, n_features), z_init)

        qu_mean = self.param('mean', (num_inducing_points, n_features),
                             lambda key, shape: jax.nn.initializers.zeros(
                                 key, shape, dtype=dtype))

        qu_scale = self.param(
            'scale',
            (num_inducing_points, num_inducing_points),
            lambda key, shape: jnp.eye(num_inducing_points, dtype=dtype))

        return z, MultivariateNormalTriL(qu_mean, jnp.tril(qu_scale))


class SVGPLayer(nn.Module):
    def apply(self, x, kernel_fn, z, qz):
        """

        Args:
            x: index_points
            z: inducing_variables

        Returns:

        """
        var_kern = kernels.VariationalKernel(
            kernel_fn,
            z,
            qz.scale)

        return var_kern


class MyModel(nn.Module):
    def apply(self, x):
        kern_fun = RBFKernelProvider()

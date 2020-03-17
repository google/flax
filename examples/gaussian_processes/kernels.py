from typing import Callable
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Kernel:
    """ Kernel valid as a pytree, see
        https://jax.readthedocs.io/en/latest/notebooks/JAX_pytrees.html
        so it can be composed in jax functions
    """
    kernel_fn: Callable = struct.field(pytree_node=False)

    def apply(self, x, x2):
        return self.kernel_fn(x, x2)

    def __call__(self, x, x2=None):
        x2 = x if x2 is None else x2
        return self.apply(x, x2)


def rbf_kernel_fun(x, x2, amplitude, lengthscale):
    """ Functional definition of an RBF kernel. """
    pwd_dists = (x[..., None, :] - x2[..., None, :, :]) / lengthscale
    kernel_matrix = jnp.exp(-.5 * jnp.sum(pwd_dists ** 2, axis=-1))
    return amplitude**2 * kernel_matrix

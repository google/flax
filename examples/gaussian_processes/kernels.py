from typing import Callable
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import ops
from flax import nn
from flax import struct


def _diag_shift(mat, val):
    """ Shifts the diagonal of mat by val. """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)


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


@struct.dataclass
class SchurComplementKernel(Kernel):
    fixed_inputs: jnp.ndarray
    divisor_matrix_cholesky: jnp.ndarray

    def apply(self, x1, x2):
        k12 = self.kernel_fn(x1, x2)
        k1z = self.kernel_fn(x1, self.fixed_inputs)
        kz2 = self.kernel_fn(self.fixed_inputs, x2)

        return k12 - k1z @ jscipy.linalg.cho_solve(
            (self.divisor_matrix_cholesky, True), kz2)


class SchurComplementKernelProvider(nn.Module):
    """ Provides a schur complement kernel. """
    def apply(self,
              base_kernel_fun: Callable,
              fixed_index_points: jnp.ndarray,
              diag_shift: jnp.ndarray = jnp.zeros([1])) -> SchurComplementKernel:
        """

        Args:
            kernel_fun:
            fixed_index_points:
            diag_shift: Python `float`

        Returns:

        """
        # compute the "divisor-matrix"
        divisor_matrix = base_kernel_fun(
            fixed_index_points, fixed_index_points)
        divisor_matrix_cholesky = jnp.linalg.cholesky(
            _diag_shift(divisor_matrix, diag_shift))

        return SchurComplementKernel(base_kernel_fun,
                                     fixed_index_points,
                                     divisor_matrix_cholesky)

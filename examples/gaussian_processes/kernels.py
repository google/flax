from typing import Callable
import jax.numpy as jnp
import jax.scipy as jscipy
import jax
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
    #pwd_dists = (jnp.expand_dims(x, -2) - jnp.expand_dims(x2, -3)) / lengthscale
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
        return (k12
                - k1z @ jscipy.linalg.cho_solve(
                    (self.divisor_matrix_cholesky, True), kz2))


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


@struct.dataclass
class VariationalKernel(Kernel):
    fixed_inputs: jnp.ndarray
    variational_scale: jnp.ndarray

    def apply(self, x1, x2):
        z = self.fixed_inputs
        kxy = self.kernel_fn(x1, x2)
        kxz = self.kernel_fn(x1, z)
        kzy = self.kernel_fn(z, x2)
        kzz = self.kernel_fn(z, z)
        kzz_cholesky = jnp.linalg.cholesky(
            kzz + 1e-6 * jnp.eye(z.shape[-2]))

        kzz_chol_qu_scale = jscipy.linalg.cho_solve(
            (kzz_cholesky, True), self.variational_scale)

        return (kxy
                - kxz @ jscipy.linalg.cho_solve((kzz_cholesky, True), kzy)
                + kxz @ (kzz_chol_qu_scale @ kzz_chol_qu_scale.T) @ kzy)


class RBFKernelProvider(nn.Module):
    """ Provides an RBF kernel function.

    The role of a kernel provider is to handle initialisation, and
    parameter storage of a particular kernel function. Allowing
    functionally defined kernels to be slotted into more complex models
    built using the Flax functional api.
    """
    def apply(self,
              index_points: jnp.ndarray,
              amplitude_init: Callable = jax.nn.initializers.ones,
              length_scale_init: Callable = jax.nn.initializers.ones) -> Callable:
        """

        Args:
            index_points: The nd-array of index points to the kernel. Only used for
              feature shape finding.
            amplitude_init: initializer function for the amplitude parameter.
            length_scale_init: initializer function for the length-scale parameter.

        Returns:
            rbf_kernel_fun: Callable kernel function.
        """
        amplitude = jax.nn.softplus(
            self.param('amplitude',
                       (1,),
                       amplitude_init)) + jnp.finfo(float).tiny

        length_scale = jax.nn.softplus(
            self.param('length_scale',
                       (index_points.shape[-1],),
                       length_scale_init)) + jnp.finfo(float).tiny

        return Kernel(
            lambda x_, y_: rbf_kernel_fun(x_, y_, amplitude, length_scale))

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class GaussianLogLik:
    qu_mean: jnp.ndarray
    qu_scale: jnp.ndarray
    observation_noise_scale: jnp.ndarray

    def variational_expectation(self, y):
        return -.5 * jnp.squeeze(
            (jnp.sum(jnp.square(self.qu_mean - y))
             + jnp.trace(self.qu_scale @ self.qu_scale.T))
            / self.observation_noise_scale ** 2
            + y.shape[-1] * jnp.log(self.observation_noise_scale ** 2)
            + jnp.log(2 * jnp.pi))

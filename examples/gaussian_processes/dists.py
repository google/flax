import jax.numpy as jnp
from flax import struct
from flax import nn
import abc
import jax.scipy as jscipy
from jax import random


class MultivariateNormal(abc.ABC):
    """ Base multivariate normal distribution. """

    @abc.abstractmethod
    def log_prob(self, x):
        pass


@struct.dataclass
class MultivariateNormalDiag(MultivariateNormal):
    mean: jnp.ndarray
    scale_diag: jnp.ndarray

    def log_prob(self, x):
        return jnp.sum(
        jscipy.stats.norm.logpdf(
            x, loc=self.mean, scale=self.scale_diag))

    def sample(self, key, shape=()):
        return random.normal(key, shape=shape) * self.scale_diag + self.mean


@struct.dataclass
class MultivariateNormalTriL(MultivariateNormal):
    mean: jnp.ndarray
    scale: jnp.ndarray

    @property
    def covariance(self):
        return self.scale @ self.scale.T

    def log_prob(self, x):
        dim = x.shape[-1]
        dev = x - self.mean
        maha = jnp.sum(dev *
                       jscipy.linalg.cho_solve((self.scale, True), dev))
        log_2_pi = jnp.log(2 * jnp.pi)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(self.scale)))
        return -0.5 * (dim * log_2_pi + log_det_cov + maha)

    def sample(self, key, shape=()):
        return random.multivariate_normal(
            key, self.mean, self.covariance, shape)


class MyModel(nn.Module):
    def apply(self, x):
        return MultivariateNormalTriL(
            jnp.zeros(x.shape[:-1]),
            jnp.eye(x.shape[-2]))


def create_model(rng):
    _, params = MyModel.init_by_shape(rng, [((3, 1), jnp.float64), ])
    return nn.Model(MyModel, params)


rng = random.PRNGKey(0)
model = create_model(rng)
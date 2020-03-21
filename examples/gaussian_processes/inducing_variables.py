import jax.numpy as jnp
from flax import nn, struct
from distributions import MultivariateNormalTriL


@struct.dataclass
class InducingVariable:
    variational_distribution: MultivariateNormalTriL
    prior_distribution: MultivariateNormalTriL


@struct.dataclass
class InducingPointsVariable(InducingVariable):
    locations: jnp.ndarray

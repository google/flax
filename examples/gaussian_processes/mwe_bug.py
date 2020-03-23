import jax.numpy as jnp
import flax.nn as nn
from jax import random


class Layer(nn.Module):
    def apply(self, x):
        x2 = x
        pwd_dists = x[..., None, :] - x2[..., None, :, :]
        return jnp.sum(pwd_dists)

class MyModel(nn.Module):
    def apply(self, x):
        x = Layer(x)
        return x

x = jnp.linspace(-1, 1)[:, None]

def loss(model):
    return model.module.call(model.params, x)

rng = random.PRNGKey(0)
_, par = MyModel.init(rng, x)
model = nn.Model(MyModel, par)

loss(model)
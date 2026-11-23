import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import xavier_normal, zeros
from jax.flatten_util import ravel_pytree

class MLP(nn.Module):
    layer_dims: tuple
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x):
        dims = [int(d) for d in self.layer_dims]
        n_layers = len(dims) - 1
        for i, (_, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            x = nn.Dense(
                out_dim,
                kernel_init=xavier_normal(),
                bias_init=zeros,
                dtype=self.dtype,
                param_dtype=self.dtype,
                name=f'dense_{i}',
            )(x)
            if i < n_layers - 1:
                x = jnp.tanh(x)
        return x


def _init_mlp(module, key, in_dim, dtype):
    dummy = jnp.zeros((1, int(in_dim)), dtype)
    variables = flax.core.unfreeze(module.init(key, dummy))
    theta0, unravel = ravel_pytree(variables)

    def apply(theta, x):
        return module.apply(unravel(theta), x)

    return theta0, apply, int(theta0.size), variables, unravel
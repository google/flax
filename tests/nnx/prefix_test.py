import flax
from flax import nnx
import jax
import jax.numpy as jnp

def test_prefix_works():
  with flax.config.temp_flip_flag('graph_mode', False, prefix='nnx'):
    with flax.config.temp_flip_flag('graph_updates', False, prefix='nnx'):

      class Model(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
          self.linear = nnx.Linear(20, 10, rngs=rngs)
          self.drop = nnx.Dropout(0.1, rngs=rngs)

      rngs = nnx.Rngs(0, dropout=jax.random.key(1))
      rngs = rngs.split({f'dropout': 5})
      model = jax.vmap(Model, in_axes=(nnx.prefix(rngs, {'dropout': 0}),))(rngs)
      assert model.drop.rngs.key[...].shape == (5,)
      assert model.drop.rngs.count[...].shape == (5,)
      bias = model.linear.bias[...]
      assert all(jnp.allclose(x,y) for (x,y) in zip(bias, bias[1:]))

    # Problem: need nnx.vmap to work with prefix
    # Currently, vmap flattens to a graphdef first. We must disable this.

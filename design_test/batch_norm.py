from flax.core import Scope, init, apply, nn

from jax import random

# batch norm is in nn/normalization.py

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (2, 3))
  y, params = init(nn.batch_norm)(random.PRNGKey(1), x)
  print(y)
  print(params)

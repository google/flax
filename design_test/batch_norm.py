from flax.core import Scope, init, apply

from flax import nn
from jax import random


x = random.normal(random.PRNGKey(0), (2, 3))
y, params = init(nn.batch_norm)(random.PRNGKey(1), x)
print(y)
print(params)

import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np
from pprint import pprint
from dense import Dense

# Many NN layers and blocks are best described by a single function with inline variables.
# In this case, variables are initialized during the first call.
class MLP(Module):
  def __call__(self, x):
    return Dense(self, features=1)(nn.relu(Dense(self, features=2)(x)))

# Return an initialized instance of MLP by calling `__call__` with an input batch,
# initializing all variables.
#
# Variable shapes depend on the input shape passed in.
rngkey = jax.random.PRNGKey(10)
mlp = MLP(parent=None).initialized({'param': rngkey}, jnp.zeros((1, 3)))

pprint(mlp.variables)
# {'param': {'Dense_0': {'bias': DeviceArray([0.], dtype=float32),
#                        'kernel': DeviceArray([[-0.04267037],
#              [-0.51097125]], dtype=float32)},
#            'Dense_1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                        'kernel': DeviceArray([[-6.3845289e-01,  6.0373604e-01],
#              [-5.9814966e-01,  5.1718324e-01],
#              [-6.2220657e-01,  5.8988278e-04]], dtype=float32)}}}
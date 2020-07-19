import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np
from pprint import pprint
from dense import Dense

# Here submodules are explicitly defined during init, but still materialized
# lazily only once a first input is passed through and shapes are known.
class MLP(Module):
  def setup(self):
    self.dense1 = Dense(self, features=2)
    self.dense2 = Dense(self, features=1)

    # shapes aren't yet known, so variables aren't materialized
    print(self.dense2.variables)
    # FrozenDict({})

  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))

# Return an initialized instance of MLP by calling `__call__` with an input batch,
# initializing all variables.
#
# Variable shapes depend on the input shape passed in.
rngkey = jax.random.PRNGKey(10)
mlp = MLP(parent=None).initialized({'param': rngkey}, jnp.zeros((1, 3)))

pprint(mlp.variables)
# {'param': {'dense1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.18307537, -0.38739476],
#              [-0.902451  , -0.5190721 ],
#              [ 0.51552075,  1.1169153 ]], dtype=float32)},
#            'dense2': {'bias': DeviceArray([0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.6704609 ],
#              [-0.90477365]], dtype=float32)}}}


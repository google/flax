import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module
import numpy as np
from pprint import pprint
from dense import Dense

# Add `in_features` to the built-in Dense layer that normally works
# via shape inference.
class DenseExplicit(Dense):
  in_features: Optional[int] = None

  def setup(self):
    # We feed a fake batch through the module, which initialized parameters.
    # Assuming we're in a jit, should use no FLOPs -- "just shape inference".
    self.__call__(jnp.zeros((1, self.in_features, )))

class MLP(Module):
  def setup(self):
    self.dense1 = DenseExplicit(in_features=3, features=2)
    self.dense2 = DenseExplicit(in_features=2, features=1)

    # explicit instances are materialized immediately at init
    pprint(self.dense2.variables)
    # {'param': {'bias': DeviceArray([0.], dtype=float32),
    #            'kernel': DeviceArray([[ 0.6704609 ],
    #              [-0.90477365]], dtype=float32)}}


  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))

# Return an initialized instance of MLP by only calling `setup`.
rngkey = jax.random.PRNGKey(10)
init_variables = MLP().init({'param': rngkey}, jnp.ones((1, 3)))

pprint(init_variables)
# {'param': {'dense1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.18307537, -0.38739476],
#              [-0.902451  , -0.5190721 ],
#              [ 0.51552075,  1.1169153 ]], dtype=float32)},
#            'dense2': {'bias': DeviceArray([0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.6704609 ],
#              [-0.90477365]], dtype=float32)}}}
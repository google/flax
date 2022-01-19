# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import numpy as jnp, random, lax
from flax import linen as nn
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact
import numpy as np
from pprint import pprint
from dense import Dense


# Many NN layers and blocks are best described by a single function with inline variables.
# In this case, variables are initialized during the first call.
class MLP(Module):
  sizes: Iterable[int]

  @compact
  def __call__(self, x):
    for size in self.sizes[:-1]:
        x = Dense(size)(x)
        x = nn.relu(x)
    return Dense(self.sizes[-1])(x)

# Return an initialized instance of MLP by calling `__call__` with an input batch,
# initializing all variables.
#
# Variable shapes depend on the input shape passed in.
rngkey = jax.random.PRNGKey(10)
model = MLP((2, 1))
x = jnp.ones((1, 3))
mlp_variables = model.init(rngkey, x)
print(mlp_variables)
# {'params': {'Dense_0': {'bias': DeviceArray([0.], dtype=float32),
#                        'kernel': DeviceArray([[-0.04267037],
#              [-0.51097125]], dtype=float32)},
#            'Dense_1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                        'kernel': DeviceArray([[-6.3845289e-01,  6.0373604e-01],
#              [-5.9814966e-01,  5.1718324e-01],
#              [-6.2220657e-01,  5.8988278e-04]], dtype=float32)}}}
print(model.apply(mlp_variables, x))

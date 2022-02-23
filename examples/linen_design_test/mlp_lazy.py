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
from flax.linen import Module
import numpy as np
from pprint import pprint
from dense import Dense


# Here submodules are explicitly defined during init, but still materialized
# lazily only once a first input is passed through and shapes are known.
class MLP(Module):
  def setup(self):
    self.dense1 = Dense(features=2)
    self.dense2 = Dense(features=1)

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
mlp_variables = MLP().init(rngkey, jnp.zeros((1, 3)))

pprint(mlp_variables)
# {'params': {'dense1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.18307537, -0.38739476],
#              [-0.902451  , -0.5190721 ],
#              [ 0.51552075,  1.1169153 ]], dtype=float32)},
#            'dense2': {'bias': DeviceArray([0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.6704609 ],
#              [-0.90477365]], dtype=float32)}}}


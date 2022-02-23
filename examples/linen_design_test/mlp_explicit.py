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

from pprint import pprint
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.deprecated import nn
from flax.deprecated.nn import initializers
from dense import Dense
from flax.linen import Module
import jax
from jax import lax, numpy as jnp, random
import numpy as np


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
    # {'params': {'bias': DeviceArray([0.], dtype=float32),
    #            'kernel': DeviceArray([[ 0.6704609 ],
    #              [-0.90477365]], dtype=float32)}}


  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))

# Return an initialized instance of MLP by only calling `setup`.
rngkey = jax.random.PRNGKey(10)
init_variables = MLP().init({'params': rngkey}, jnp.ones((1, 3)))

pprint(init_variables)
# {'params': {'dense1': {'bias': DeviceArray([0., 0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.18307537, -0.38739476],
#              [-0.902451  , -0.5190721 ],
#              [ 0.51552075,  1.1169153 ]], dtype=float32)},
#            'dense2': {'bias': DeviceArray([0.], dtype=float32),
#                       'kernel': DeviceArray([[ 0.6704609 ],
#              [-0.90477365]], dtype=float32)}}}

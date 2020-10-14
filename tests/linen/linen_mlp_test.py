# Copyright 2020 The Flax Authors.
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
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module
import numpy as np
from pprint import pprint
from dense import Dense
from absl.testing import absltest

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()
jax.config.parse_flags_with_absl()

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
    # pprint(self.dense2.variables)
    # {'params': {'bias': DeviceArray([0.], dtype=float32),
    #            'kernel': DeviceArray([[ 0.6704609 ],
    #              [-0.90477365]], dtype=float32)}}


  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))


class MLPTest(absltest.TestCase):
    def mlp_test(self):
        rngkey = jax.random.PRNGKey(10)
        init_variables = MLP().init({'params': rngkey}, jnp.ones((1, 3)))
        dense1 = init_variables['params']['dense1']
        dense1_param_shapes = unfreeze(jax.tree_map(jnp.shape, dense1))
        dense2 = init_variables['params']['dense2']
        dense2_param_shapes = unfreeze(jax.tree_map(jnp.shape, dense2))
        self.assertEqual(dense1_param_shapes, {
            'kernel': (3, 2),
            'bias': (2,),
        })
        self.assertEqual(dense2_param_shapes, {
            'kernel': (3, 2),
            'bias': (2,),
        })

if __name__ == '__main__':
  absltest.main()

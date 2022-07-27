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
from dense import Dense


# TODO(avital, levskaya): resurrect this example once interactive api is restored.

# class TiedAutoEncoder(Module):
#   def setup(self):
#     self.encoder = Dense(features=4, use_bias=False)

#   @property
#   def decoder(self):
#     return self.encoder.detached().attached(variables={
#       'params': {"kernel": self.encoder.variables['params']['kernel'].T}})

#   def __call__(self, x):
#     z = self.encoder(x)
#     x = self.decoder(z)
#     return x

# tae = TiedAutoEncoder(parent=None)
# tae = tae.initialized(
#   {'params': random.PRNGKey(42)},
#   jnp.ones((1, 16)))
# print("reconstruct", jnp.shape(tae(jnp.ones((1, 16)))))
# print("var shapes", jax.tree_util.tree_map(jnp.shape, tae.variables))

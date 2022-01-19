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

from dataclasses import dataclass
import jax
from jax import numpy as jnp, random, lax, jit
from flax import linen as nn
from flax.core.scope import Scope
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact
import numpy as np
from dense import Dense
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict


def standardize(x, axis, eps=1e-8):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x

# TODO(avital, levskaya): resurrect this example once interactive api is restored.

# A wrapper that calls through a simple module with standardized parameters.
#
# Note that StdWeight is /not/ a module, hence it doesn't add another layer
# of depth in the variable dict (i.e. this is a "transparent module")
# @dataclass
# class StdWeight:
#   module: Module

#   def __call__(self, x):
#     # TODO: Think about how this modifies other state
#     if not 'params' in self.module.variables:
#       # initialize parameters
#       self.module(x)

#     param = self.module.variables['params']
#     # Make a copy because `param` is (and should be) frozen. We're only transforming
#     # the parameters, not mutating them.
#     std_param = param.copy(kernel=standardize(param['kernel'], axis=[0, 1]))
#     return self.module.clone(parent=None).apply({'params': std_param}, x)

# class MyModule(Module):
#   def __call__(self, x):
#     module = Dense(self, 3)
#     std_module = StdWeight(module)
#     return std_module(x)

# m_variables = MyModule().init({'params': jax.random.PRNGKey(10)}, jnp.ones((1, 4)))
# print(m_variables)

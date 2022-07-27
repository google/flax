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
from jax import numpy as jnp, random, lax, jit
from flax import linen as nn
from dense import Dense


X = jnp.ones((1, 10))
Y = jnp.ones((5,))

model = Dense(features=5)

@jit
def predict(params):
  return model.apply({'params': params}, X)

@jit
def loss_fn(params):
  return jnp.mean(jnp.abs(Y - predict(params)))

@jit
def init_params(rng):
  mlp_variables = model.init({'params': rng}, X)
  return mlp_variables['params']

# Get initial parameters
params = init_params(jax.random.PRNGKey(42))
print("initial params", params)

# Run SGD.
for i in range(50):
  loss, grad = jax.value_and_grad(loss_fn)(params)
  print(i, "loss = ", loss, "Yhat = ", predict(params))
  lr = 0.03
  params = jax.tree_util.tree_map(lambda x, d: x - lr * d, params, grad)

import jax
from jax import numpy as jnp, random, lax, jit
from flax import linen as nn
from dense import Dense

X = jnp.ones((1, 10))
Y = jnp.ones((5,))

model = Dense(features=5)

@jit
def predict(params):
  return model.apply({'param': params}, X)

@jit
def loss_fn(params):
  return jnp.mean(jnp.abs(Y - predict(params)))

@jit
def init_params(rng):
  mlp_variables = model.init({'param': rng}, X)
  return mlp_variables['param']

# Get initial parameters
params = init_params(jax.random.PRNGKey(42))
print("initial params", params)

# Run SGD.
for i in range(50):
  loss, grad = jax.value_and_grad(loss_fn)(params)
  print(i, "loss = ", loss, "Yhat = ", predict(params))
  lr = 0.03
  params = jax.tree_multimap(lambda x, d: x - lr * d, params, grad)

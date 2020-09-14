import jax
import numpy as onp

@jax.jit
def policy_action(model, state):
  """Forward pass of the network.
  Potentially the random choice of the action from probabilities can be moved
  here with additional rng_key parameter.
  """
  out = model(state)
  return out
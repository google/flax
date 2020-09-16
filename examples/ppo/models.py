import flax
from flax import nn
import jax.numpy as jnp

class ActorCritic(flax.nn.Module):
  '''
  Architecture from "Human-level control through deep reinforcement learning."
  Nature 518, no. 7540 (2015): 529-533.
  Note that this is different than the one from  "Playing atari with deep
  reinforcement learning." arxiv.org/abs/1312.5602 (2013)
  '''
  def apply(self, x):
    x = x.astype(jnp.float32) / 255.
    dtype = jnp.float32
    x = nn.Conv(x, features=32, kernel_size=(8, 8),
                strides=(4, 4), name='conv1',
                dtype=dtype)
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4),
                strides=(2, 2), name='conv2',
                dtype=dtype)
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3),
                strides=(1, 1), name='conv3',
                dtype=dtype)
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, name='hidden', dtype=dtype)
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    # network used to both estimate policy (logits) and expected state value
    # see github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
    logits = nn.Dense(x, features=4, name='logits', dtype=dtype)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(x, features=1, name='value', dtype=dtype)
    return policy_log_probabilities, value

def create_model(key):
  input_dims = (1, 84, 84, 4) #(minibatch, height, width, stacked frames)
  _, initial_par = ActorCritic.init_by_shape(key, [(input_dims, jnp.float32)])
  model = flax.nn.Model(ActorCritic, initial_par)
  return model

def create_optimizer(model, learning_rate):
  optimizer_def = flax.optim.Adam(learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer

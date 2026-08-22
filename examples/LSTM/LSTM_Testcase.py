import sys

sys.path.append(".")

from LSTM_JAX import LSTM_seq

import jax
import jax.numpy as jnp
from flax.core import init, unfreeze
from flax.linen import Module

batch_size = 2
seq_len = 3
input_size = 4
hidden_size = 5

x = jnp.ones((batch_size, seq_len, input_size))

model = LSTM_seq(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len)

rng = jax.random.PRNGKey(0)
params = model.init(rng, x)

output = model.apply(params, x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output values:\n", output)

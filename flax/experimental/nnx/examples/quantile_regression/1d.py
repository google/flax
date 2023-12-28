# %%
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from flax.experimental import nnx


# %%
def quantile_loss(q: float, y_true: jax.Array, y_pred: jax.Array):
  e = y_true - y_pred
  return jnp.maximum(q * e, (q - 1.0) * e)


def plot_quantiles(values: jax.Array, quantiles: jax.Array):
  widths = values[..., 1:] - values[..., :-1]
  density = quantiles[..., 1:] - quantiles[..., :-1]
  heights = density / widths
  plt.bar(values[..., :-1], heights, widths, align='edge')


# %%


class LinearQuantiles(nnx.Module):
  def __init__(self, din: int, quantiles: Sequence[float], *, rngs: nnx.Rngs):
    assert len(quantiles) > 2
    self.din = din
    self.quantiles = tuple(float(q) for q in quantiles)
    self.scores = nnx.Linear(din, len(quantiles) - 1, rngs=rngs)
    self.width = nnx.Linear(din, 1, rngs=rngs)
    self.min = nnx.Linear(din, 1, rngs=rngs)

  def __call__(self, x: jax.Array):
    min = self.min(x)  # learn the lowest quantile
    width = nnx.softplus(self.width(x))  # max - min
    # intermediate quantiles are computed via softmax scores defined as the proportion between
    # the difference of adjacent quantiles to the total width, this induces a monotonic ordering
    scores = nnx.softmax(self.scores(x))
    scores = jnp.concatenate([jnp.zeros_like(min), scores], axis=-1)
    proportions = jnp.cumsum(scores, axis=-1)
    values = min + width * proportions
    return values

  def loss(self, x: jax.Array, y: jax.Array):
    values = self(x)
    quantiles = jnp.broadcast_to(jnp.asarray(self.quantiles), values.shape)
    loss = quantile_loss(quantiles, y, values)
    return loss.mean()


quantiles = [0.01, *np.linspace(0.02, 0.98, 6), 0.99]
model = LinearQuantiles(1, quantiles, rngs=nnx.Rngs(0))


x = np.asarray(jnp.zeros((1000, 1)))
y = np.asarray(jax.random.exponential(jax.random.key(0), x.shape) * 0.1)
params = model.extract(nnx.Param)
tx = optax.adamw(1e-3)
opt_state = tx.init(params)

quantiles = np.asarray(quantiles).squeeze()
real_values = np.quantile(y, quantiles)


@nnx.value_and_grad
def loss_fn(model: LinearQuantiles, x: jax.Array, y: jax.Array):
  return model.loss(x, y)


@nnx.jit
def train_step(model: LinearQuantiles, opt_state, x: jax.Array, y: jax.Array):
  loss, grads = loss_fn(model, x, y)
  params = model.extract(nnx.Param)
  updates, opt_state = tx.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  model.update(params)
  return loss, opt_state


losses = []
for i in range(5_000 + 1):
  loss, opt_state = train_step(model, opt_state, x, y)
  if i % 100 == 0:
    losses.append(float(loss))
    print(loss)
    values = model(jnp.zeros((1,)))
    df = pd.DataFrame(
      {'quantiles': quantiles, 'values': values, 'real': real_values}
    )
    df['error'] = (df['values'] - df['real']).abs()
    print(df)

fig = plt.figure()
plot_quantiles(values, quantiles)
plt.hist(y, bins=100, density=True, alpha=0.5, color='red')
fig = plt.figure()
plt.plot(losses)
plt.show()


# %%

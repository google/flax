# %%

import os
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from flax.experimental import nnx

plt.rcParams['figure.dpi'] = int(os.environ.get('FIGURE_DPI', 150))
plt.rcParams['figure.facecolor'] = os.environ.get('FIGURE_FACECOLOR', 'white')
np.random.seed(69)


def create_data(multimodal: bool):
  x = np.random.uniform(0.3, 10, 1000)
  y = np.log(x) + np.random.exponential(0.1 + x / 20.0)

  if multimodal:
    x = np.concatenate([x, np.random.uniform(5, 10, 500)])
    y = np.concatenate([y, np.random.normal(6.0, 0.3, 500)])

  return x[..., None], y[..., None]


multimodal: bool = False

x, y = create_data(multimodal)

fig = plt.figure()
plt.scatter(x[..., 0], y[..., 0], s=20, facecolors='none', edgecolors='k')

# %%


def quantile_loss(q: float, y_true: jax.Array, y_pred: jax.Array):
  e = y_true - y_pred
  return jnp.maximum(q * e, (q - 1.0) * e)


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


class QuantileRegression(nnx.Module):
  def __init__(self, quantiles: Sequence[float], *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(1, 128, rngs=rngs)
    self.linear2 = nnx.Linear(128, 64, rngs=rngs)
    self.quantiles = LinearQuantiles(64, quantiles, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.linear1(x)
    x = nnx.relu(x)
    x = self.linear2(x)
    x = nnx.relu(x)
    x = self.quantiles(x)
    return x

  def loss(self, x: jax.Array, y: jax.Array):
    x = self.linear1(x)
    x = nnx.relu(x)
    x = self.linear2(x)
    x = nnx.relu(x)
    loss = self.quantiles.loss(x, y)
    return loss


quantiles = np.linspace(0.05, 0.95, 6)
model = QuantileRegression(quantiles, rngs=nnx.Rngs(0))
params = model.extract(nnx.Param)
tx = optax.adamw(1e-3)
opt_state = tx.init(params)


@nnx.value_and_grad
def loss_fn(model: QuantileRegression, x: jax.Array, y: jax.Array):
  return model.loss(x, y)


@nnx.jit
def train_step(
  model: QuantileRegression, opt_state, x: jax.Array, y: jax.Array
):
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

# %%
x_test = np.linspace(x.min(), x.max(), 100)
y_pred = model(x_test[..., None])

fig = plt.figure()
plt.scatter(x, y, s=20, facecolors='none', edgecolors='k')

for i, q_values in enumerate(np.split(y_pred, len(quantiles), axis=-1)):
  plt.plot(x_test, q_values[:, 0], linewidth=2, label=f'Q({quantiles[i]:.2f})')

plt.legend()

# %%

median_closests = list(
  sorted(enumerate(quantiles), key=lambda t: abs(t[1] - 0.5))
)
median_idx = median_closests[0][0]
q = quantiles[median_idx]

fig = plt.figure()
plt.fill_between(x_test, y_pred[:, -1], y_pred[:, 0], alpha=0.5, color='b')
plt.scatter(x, y, s=20, facecolors='none', edgecolors='k')
plt.plot(
  x_test,
  y_pred[:, median_idx],
  color='r',
  linestyle='dashed',
  label=f'Q({q:.2f})',
)
plt.legend()


# %%
def plot_quantiles(values: jax.Array, quantiles: jax.Array):
  widths = values[..., 1:] - values[..., :-1]
  density = quantiles[..., 1:] - quantiles[..., :-1]
  heights = density / widths
  plt.bar(values[..., :-1], heights, widths, align='edge')


xi = jnp.array([7.0])
values = model(xi)

fig = plt.figure()
plot_quantiles(values, quantiles)
plt.show()

# %%

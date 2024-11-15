---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Performance Considerations
Currently `nnx.jit` traverses the object graph in Python. This is slow and primarily affects the small model regime, as the Python overhead starts to disappear as the model's width grows. To solve this in general, we will be developing a Rust extension called `flaxlib` (see first steps in #4196) to speedup some of the traversal logic in `graph.py`, similar to how JAX solved the same issue with `jaxlib` for standard pytrees. 

Meanwhile, there is a pattern you can use to remove the python overhead using regular `jax.jit` + `nnx.split` / `nnx.merge` to stage out the traversal logic. Take this code that uses `nnx.jit` as an example:

```{code-cell}
from flax import nnx
import jax
import jax.numpy as jnp
import optax

class Model(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.2, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)

model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)

@nnx.jit  # <== currently slow
def train_step(model, optimizer, metrics, x, y):
  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates
  metrics.update(loss=loss)

  return loss
  
for _ in range(10):
  x, y = jnp.ones((32, 2)), jnp.zeros((32, 3))
  loss = train_step(model, optimizer, metrics, x, y)
```

To speed it up you can use `nnx.split` before starting the training loop to create a `graphdef` and `state` for the NNX objects which are fast to traverse, and then call `merge` + `split` inside the `jax.jit`-decorated function so they only run once during tracing:

```{code-cell}
model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))  # reference sharing
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
# split before training loop
graphdef, state = nnx.split((model, optimizer, metrics))

@jax.jit  # regular JAX
def train_step(graphdef, state, x, y):
  # merge at the beginning of the function
  model, optimizer, metrics = nnx.merge(graphdef, state)

  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  metrics.update(loss=loss)

  # split at the end of the function
  _, state = nnx.split((model, optimizer, metrics))

  # return new state
  return state, loss

for _ in range(10):
  x, y = jnp.ones((32, 2)), jnp.zeros((32, 3))
  state, loss = train_step(graphdef, state, x, y)

# update objects after training
nnx.update((model, optimizer, metrics), state)
```

After the training loop is done (or whenever need) `nnx.update` can be used to update `model`, `optimizer`, and `metrics` to a new `state`.

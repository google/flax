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
Currently `nnx.jit` traverses the object graph in pure Python, this is slow and adds overhead. To solve this in general we will be developing a Rust extension called `flaxlib` (see first steps in #4196) to speedup some of the traversal logic in `graph.py`, similar to how JAX solved the same issue with `jaxlib` for standard pytrees. However, there's two things to consider:

* The overhead is only relevant for small models. See [Asynchronous dispatch](#asynchronous-dispatch).
* You can remove the overhead by using `jax.jit` + `nnx.split` / `nnx.merge` to stage out the traversal logic. See [Lowering the Python Overhead](#lowering-the-python-overhead).


## Asynchronous dispatch
In [benchmarks/nnx_simple_training.py](https://github.com/google/flax/blob/main/benchmarks/nnx_simple_training.py) we are increasing the layer width (features per layer) and measuring the total training time for the same model trained both with `nnx.jit` and `jax.jit`. As you can see in the graph below, after a certain model size the time spent in the traversal is completely absorbed by async dispatch. This happens when Python is able to finish the current for loop step, and reach the next `train_step` and JAX is still not done with the previous `train_step`. 

![performance-graph](images/performance-graph.png)

This means that you only need to worry about the `nnx.jit` overhead for small models. If you are working with a small model, check out the next section to see how you can remove the overhead.

## Lowering the Python Overhead
To remove the python overhead you can use regular `jax.jit` in combination with `nnx.split` and `nnx.merge` to stage out the traversal logic. To learn how to do this, lets first create this simple model:

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
```

Lets say we have this `train_step` function that is using `nnx.jit` and takes in a `model`, `optimizer`, and `metrics`, all of which are Flax NNX objects:

```{code-cell}
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

To speed it up, before starting the training loop we can use `nnx.split` over the all the Flax NNX objects that are inputs to `train_step` to create a `graphdef` and `state` pytrees that are fast to traverse. Next we change `train_step` so accept `graphdef` and `state` and use `nnx.merge` and `nnx.split` at the beginning and end of `train_step` to switch back and forth between the objects and their pytree representations. Even though `nnx.split` and `nnx.merge` are slow it doesn't matter because they will only run once during tracing. With this in place, we can change the `train_step` function to use `jax.jit` instead of `nnx.jit`:

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

Notice that we only do this for `jit`, you can still use other transforms like `nnx.value_and_grad` shown in the example since their overhead is already absorbed by the outer `jit`. Also, after the training loop is done (or whenever need) `nnx.update` can be used to update Flax NNX objects like `model`, `optimizer`, and `metrics` to a new `state`.

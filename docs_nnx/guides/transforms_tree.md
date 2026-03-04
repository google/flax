---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Transforms

NNX transforms (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.scan`, ...) are thin wrappers over JAX transforms that provide the same APIs. Their main feature is **automatic state propagation**: the state of input `Variable`s is tracked and updated automatically.

This guide builds on the concepts introduced in the [NNX Basics](https://flax.readthedocs.io/en/latest/nnx_basics_tree.html) guide. We'll use a simple `Linear` layer throughout and progressively show how to use `nnx.jit`, `nnx.grad`, `nnx.vmap`, and `nnx.scan`.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import optax

import flax
flax.config.update("nnx_graph_mode", False)
jax.config.update("jax_num_cpu_devices", 8)
```

## Model definition

We'll use a `Linear` layer with:
- A weight matrix (`Param`).
- A call counter (`Count`) — a custom `Variable` type for non-differentiable state.
- An `rngs` argument in `__call__` to add noise.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Linear(nnx.Pytree):
  def __init__(self, din, dout, *, rngs):
    self.din, self.dout = din, dout
    self.w = nnx.Param(rngs.normal((din, dout)))
    self.count = Count(jnp.array(0))

  def __call__(self, x: jax.Array, *, rngs: nnx.Rngs):
    self.count[...] += 1
    y = x @ self.w
    return y + rngs.normal(y.shape) * 0.1  # noise
```

## jit — forward pass

`nnx.jit` compiles and caches the function just like `jax.jit`. Variable updates made
inside the function are automatically propagated back.

```{code-cell} ipython3
model = Linear(2, 5, rngs=nnx.Rngs(0))
rngs = nnx.Rngs(1)
x = jnp.ones((3, 2))

@nnx.jit
def forward(model, x, rngs):
  return model(x, rngs=rngs)

y = forward(model, x, rngs)
print(f'{y.shape = }')
print(f'{model.count[...] = !s}')  # called once
```

## jit + grad — training step

`nnx.grad` differentiates with respect to `nnx.Param` variables by default, treating all other state as non-differentiable. The `wrt` argument accepts any [Filter](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) to select which Variable types to differentiate. It handles `split`/`merge`/`clone` internally, so you only need to write the loss function.

`nnx.Optimizer` wraps an [Optax](https://optax.readthedocs.io/) optimizer and provides a simple `update(model, grads)` method that performs in-place updates to both the optimizer state and model parameters.

```{code-cell} ipython3
x, y = jnp.ones((3, 2)), jnp.ones((3, 5))
model = Linear(2, 5, rngs=nnx.Rngs(0))
rngs = nnx.Rngs(1)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y, rngs):
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)
  def loss_fn(params, nondiff, rngs):
    model = nnx.merge(graphdef, params, nondiff)
    return jnp.mean((model(x, rngs=rngs) - y) ** 2)

  grads = nnx.grad(loss_fn)(params, nondiff, rngs)
  optimizer.update(model, grads)

train_step(model, optimizer, x, y, rngs)

print(f'{model.count[...] = !s}')    # called once
print(f'{optimizer.step[...] = !s}') # one optimizer step
```

## vmap — vectorized forward pass

`nnx.vmap` vectorizes a function over an axis dimension. NNX objects participate in
`in_axes` / `out_axes` just like any other pytree. Here we broadcast the model and `rngs`
(via `None`) and vectorize only the data across a batch of inputs.

```{code-cell} ipython3
rngs = nnx.Rngs(1)
model = Linear(3, 3, rngs=rngs)
x = jnp.ones((1, 3, 10))

@nnx.vmap(in_axes=(None, 2, None), out_axes=0)
def batched_forward(model, x, rngs): # model & rngs broadcast, x vectorized
  return model(x, rngs=rngs)  

y = batched_forward(model, x, rngs)
print(f'{y.shape = !s}')             # (1, 5, 10)
print(f'{model.count[...] = !s}')    # called once (broadcast)
```

Because the model is passed with `in_axes=None`, it is broadcast — the same weights
are shared across all vectorized inputs. The same applies to `rngs`, so every input
sees identical noise.

+++

## vmap + scan — scan over layers

A common pattern is to stack many identical layers and apply them sequentially.
We use `nnx.vmap` to initialize a stack of layers, then `nnx.scan` to iterate
over them. The hidden state `x` and `rngs` are threaded through the **carry**,
while the layer stack is the **scan argument** (scanned over axis 0).

```{code-cell} ipython3
# --- initialize a stack of layers with vmap ---
@nnx.vmap(in_axes=0, out_axes=0)
def create_stack(rngs):
  return Linear(3, 3, rngs=rngs)

stack = create_stack(nnx.Rngs(0).fork(split=5))
print(f'{stack.w.shape = }')  # (5, 3, 3) — one weight per layer

# --- scan over the layer stack ---
@nnx.scan
def apply_stack(carry, layer):
  x, rngs = carry
  y = layer(x, rngs=rngs)
  return (y, rngs), None

x = jnp.ones((1, 3))
rngs = nnx.Rngs(1)
(y, _), _ = apply_stack((x, rngs), stack)

print(f'{y.shape = !s}')                  # (1, 3) — final output after all layers
print(f'{stack.count[...] = !s}')         # each layer called once
print(f'{rngs.default.count[...] = !s}')  # used 5 times (one per layer)
```

Because `rngs` is part of the carry, it automatically advances its internal state
on every iteration — each layer sees different noise without needing to manually
split keys. Variable updates inside each iteration are propagated back into
`stack` automatically.

+++

## Graph mode APIs

The examples above all use the default **tree mode** (`graph=False`), which treats modules as plain JAX pytrees with automatic Variable state propagation.

NNX also supports a **graph mode** (`graph=True`) that extends transforms with richer capabilities such as reference sharing, graph structure updates, and special "lift types" for fine-grained control. While more powerful, graph mode introduces complexities that tree mode avoids. This section briefly introduces the graph-mode-only APIs so you know what they are.

### StateAxes

`nnx.StateAxes` lets you specify **per-Variable-type axis behavior** inside `nnx.vmap` and `nnx.scan`. It maps [Filters](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) (Variable types or path predicates) to axis indices or `None` (broadcast).

For example, you might want to vectorize the `Param` weights on axis 0 but broadcast
the `Count` state:

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel, count):
    self.kernel = nnx.Param(kernel)
    self.count = Count(count)

weights = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  count=jnp.array(0),  # single scalar, not vectorized
)
x = jax.random.normal(random.key(1), (10, 2))

state_axes = nnx.StateAxes({nnx.Param: 0, Count: None})  # broadcast Count

@nnx.graph.vmap(in_axes=(state_axes, 0), out_axes=1)
def forward(weights, x):
  weights.count[...] += 1
  return x @ weights.kernel

y = forward(weights, x)
print(f'{y.shape = !s}')
print(f'{weights.count[...] = !s}')
```

`StateAxes` is only needed in graph mode — in tree mode, the entire object is treated
as a standard pytree prefix and axis specifications apply uniformly to all leaves.

### DiffState

`nnx.DiffState` lets you control which sub-state of an argument participates in
differentiation with `nnx.grad`. It wraps an argument index and a filter:

```{code-cell} ipython3
m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

# only differentiate m1's kernel and m2's bias
@nnx.graph.grad(argnums=(
  nnx.DiffState(0, nnx.PathContains('kernel')),
  nnx.DiffState(1, nnx.PathContains('bias')),
))
def loss_fn(m1, m2):
  return jnp.mean(m1.kernel * m2.kernel) + jnp.mean(m1.bias * m2.bias)

grads_m1, grads_m2 = loss_fn(m1, m2)
print(f'grads_m1: {jax.tree.map(jnp.shape, grads_m1)}')
print(f'grads_m2:   {jax.tree.map(jnp.shape, grads_m2)}')
```

In tree mode, you achieve the same effect using `nnx.split` to separate the parts
you want to differentiate, then pass them to `grad` directly.

### StateSharding

`nnx.StateSharding` maps Variable types to JAX shardings for use with `nnx.jit`. It
has the same structure as `StateAxes` but values are sharding specs instead of axis
indices:

```{code-cell} ipython3
rngs = nnx.Rngs(1)
mesh = jax.make_mesh((8,), ('devices',))

def sharding(*args):
  return jax.sharding.NamedSharding(mesh, jax.P(*args))

# Create weights outside mesh context so arrays are uncommitted
weights = Weights(
  kernel=rngs.uniform((16, 16)),
  count=jnp.array(0),
)
x = jnp.ones((16, 16))

# Define sharding for different Variable types
state_sharding = nnx.StateSharding({
  nnx.Param: sharding(None, 'devices'),  # shard Param on second axis
  Count: sharding(),                     # replicate Count
})

@nnx.graph.jit(in_shardings=(state_sharding, sharding('devices')))
def forward(weights, x):
  weights.count[...] += 1
  return x @ weights.kernel

y = forward(weights, x)
print(f'{y.shape = }')
print(f'{weights.count[...] = }')
```

In tree mode, you can use standard pytree-based `in_shardings` / `out_shardings` with `nnx.jit` or `jax.jit` directly.

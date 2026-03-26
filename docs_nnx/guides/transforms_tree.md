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

NNX transforms (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.scan`, ...) are thin wrappers over JAX transforms that provide the same APIs. Their main feature is **automatic state propagation**: input `Variable`'s state is tracked and automatically updated. Here is a sketch of how they work:

```python
def transform_wrapper(*args):
  if graph: args = to_tree(args)
  check_no_aliases(args=args)
  
  @jax_transform
  def transformed_f(*args):
    updates, snapshot = updates_and_snapshot(args)
    if graph: args = from_tree(args)
    out = f(*args)
    if graph: out = to_tree(out)
    check_no_aliases(args=updates, out=out)
    updates = mask_variable_updates(updates, snapshot)
    return out, updates
  
  out, updates = transformed_f(*args)
  apply_variable_updates(args, updates)
  if graph: out = from_tree(out)
  return out
```

The transformed function tracks input Variable `updates`, applies `f`, and masks Variable updates (no updates for Variables that didn’t change). It also checks that there are no Variable aliases between the inputs and outputs (no shared references), and returns the user output plus the Variable updates. The wrapper function calls the transformed function, applies the Variable updates to the input Variables, and returns the user output. To support graphs theres some back forth conversion between object and tree representations at various points.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from flax import nnx
import optax

nnx.set_graph_mode(False)
nnx.set_graph_updates(False)
jax.config.update("jax_num_cpu_devices", 8)
```

## Model definition
Throughout this guide we'll use a simple `Linear` layer and show how to use it with various transforms. This layer includes:
- A weight matrix (`w: Param`).
- A call counter (`count: Count`) — a custom `Variable` type with non-differentiable state.
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
over them. The hidden state `x` is passed as a **Carry**, while the layer stack and the split `rngs` are the **scanned** over axis 0. The new state `x` is returned as a Carry for the next iteration.

```{code-cell} ipython3
rngs = nnx.Rngs(0)
# --- initialize a stack of layers with vmap ---
@nnx.vmap(in_axes=0, out_axes=0)
def create_stack(rngs):
  return Linear(3, 3, rngs=rngs)

stack = create_stack(rngs.split(5))
print(f'{stack.w.shape = }')  # (5, 3, 3) — one weight per layer

# --- scan over the layer stack ---
@nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=nnx.Carry)
def apply_stack(x, layer, rngs):
  x = layer(x, rngs=rngs)
  return x

x = jnp.ones((1, 3))
y = apply_stack(x, stack, rngs.split(5))

print(f'{y.shape = !s}')                  # (1, 3) — final output after all layers
print(f'{stack.count[...] = !s}')         # each layer called once
print(f'{rngs.default.count[...] = !s}')  # rngs used 2 times (one per split)
```

Updates to `count` Variables are propagated out automatically.

+++

## Graph Mode

Setting `graph=True` on any NNX transform allows passing NNX objects with shared references — that is, inputs that form a graph rather than a strict tree. By default transforms require each leaf to appear exactly once; passing the same `Variable` in two arguments violates that constraint and raises an error. With `graph=True`, the transform detects shared `Variable`s, handles them correctly, and propagates updates back to the original object. Sharing not only applies to `Variable`s, but also to any `nnx.Pytree` which is the base type for `Module`, `Optimizer`, `Metric`, etc.

The example below shares a single `Variable` between two arguments:

```{code-cell} ipython3
@nnx.jit(graph=True)
def f(v1, v2):
  assert v1 is v2  # relative identities are preserved in graph mode
  v1[...] += 1

v = nnx.Variable(jnp.array(0))
f(v, v)

print(f'{v[...] = !s}')  # v is updated in-place, so should be 1
```

Graph mode does have one important limitation: aliased `Variable`s must be treated consistently across all arguments. For example, if the same `Variable` is passed to two arguments that have different `in_axes`, the transform cannot resolve the conflict and will raise an error:

```{code-cell} ipython3
@nnx.vmap(in_axes=(None, 0), graph=True)
def f(v1, v2):
  pass

v = nnx.Variable(jnp.array(0))

try:
  f(v, v)
except Exception as e:
  print(f'Error: {e}')
```

This is roughly saying that the same Variable (`v`) received `in_axes` of `None` on the first argument and `0` on the second argument, which is a conflict.

+++

## Legacy: Graph Updates and Prefix Filters

NNX transforms also supports a legacy **graph updates** mode which requires setting `graph=True` and `graph_updates=True` on each transform. In this mode updates to the graph objects (e.g. Modules) are also tracked and propagated. In this mode **prefix filters** like `StateAxes`, `DiffState`, `StateSharding` can be used to specify how graph substates are treated by transforms. For convenience the legacy behavior of the transforms can used via the `nnx.compat` module, this simply sets the `graph` and `graph_updates` to `True` on each transform.

In this section we will explain how to use prefix filters for users that still rely on the behavior.

### StateAxes

`nnx.StateAxes` lets you specify substate axis behavior inside `nnx.vmap`, `nnx.scan`, and `nnx.pmap`. It maps [Filters](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) like Variable types or path predicates to axis indices or `None` (broadcast).

For example, you might want to vectorize the `Param` weights on axis 0 but broadcast
the `Count` state:

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel, count):
    self.kernel = nnx.Param(kernel)
    self.count = Count(count)

rngs = nnx.Rngs(0)
weights = Weights(
  kernel=rngs.uniform((10, 2, 3)),
  count=jnp.array(0),  # single scalar, not vectorized
)
x = rngs.normal((10, 2))

state_axes = nnx.StateAxes({nnx.Param: 0, Count: None})  # broadcast Count

@nnx.compat.vmap(in_axes=(state_axes, 0), out_axes=1)
def forward(weights, x):
  weights.count[...] += 1
  return x @ weights.kernel

y = forward(weights, x)
print(f'{y.shape = !s}')
print(f'{weights.count[...] = !s}')
```

### DiffState

`nnx.DiffState` lets you control which sub-state of an argument participates in
differentiation with `nnx.grad`. It wraps an argument index and a filter:

```{code-cell} ipython3
m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))

# only differentiate m1's kernel and m2's bias
@nnx.compat.grad(argnums=(
  nnx.DiffState(0, nnx.PathContains('kernel')),
  nnx.DiffState(1, nnx.PathContains('bias')),
))
def loss_fn(m1, m2):
  return jnp.mean(m1.kernel * m2.kernel) + jnp.mean(m1.bias * m2.bias)

grads_m1, grads_m2 = loss_fn(m1, m2)
print(f'grads_m1: {jax.tree.map(jnp.shape, grads_m1)}')
print(f'grads_m2:   {jax.tree.map(jnp.shape, grads_m2)}')
```

Without graph updates, you achieve the same effect using `nnx.split` to separate the parts
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

@nnx.compat.jit(in_shardings=(state_sharding, sharding('devices')))
def forward(weights, x):
  weights.count[...] += 1
  return x @ weights.kernel

y = forward(weights, x)
print(f'{y.shape = }')
print(f'{weights.count[...] = !s}')
```

Without graph updates, you can use standard pytree-based `in_shardings` / `out_shardings` with `nnx.jit` or `jax.jit` directly.

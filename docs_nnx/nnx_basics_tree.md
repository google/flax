---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# NNX Basics

NNX is a Neural Networks library for JAX. NNX provides the tools to structure modeling code as [JAX pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) so it can work with transforms, `jax.tree.*` utilities, and all standard JAX APIs. This guide covers the core concepts you need to get started.

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp

nnx.graphlib.set_graph_mode(False)
nnx.graphlib.set_graph_updates(False)
```

NNX's main build blocks are:

- **`nnx.Pytree`**: Base class for pytree-compatible objects. Defines the tree structure of your model.
- **`nnx.Variable`**: Wraps array data and tracks mutable state. Subclasses like `nnx.Param` categorize different kinds of state.
- **State APIs** (`nnx.{state, split, merge, update}`): Extract, partition, reconstruct, and apply state updates.
- **NNX Transforms** (`nnx.{jit, grad, scan, ...}`): Thin wrappers over JAX transforms that automate state propagation.

+++

## Pytrees and Variables

`nnx.Pytree` and `nnx.Variable` are two orthogonal systems. **Pytrees** define the structure of your model as a JAX-compatible tree. **Variables** wrap array data and enable expressing state updates via in-place mutation. 

`Pytree`s are python objects that define its tree structure dynamically through its attributes, these are split into two categories: **Static attributes** (e.g. `int`, `str`) are embedded in the tree structure definition and are not traced by JAX. **Data attributes** (e.g. `nnx.Variable`, `jax.Array`) are the leaves of the tree and are traced by JAX. For more details see the [Pytree guide](https://flax.readthedocs.io/en/latest/guides/pytree.html).

Here's a typical layer definition:

```{code-cell} ipython3
class Count(nnx.Variable): pass  # custom Variable types

class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.din, self.dout = din, dout                                # static attributes
    self.w = nnx.Param(rngs.uniform((din, dout)))  # data attribute
    self.count = Count(jnp.array(0))                            # data attribute

  def __call__(self, x: jax.Array):
    self.count[...] += 1  # inplace state updates
    return x @ self.w     # Variable are Array-like

model = Linear(2, 5, rngs=nnx.Rngs(0))

nnx.display(model)
```

> **Note:** Most user code uses `nnx.Module`, which is a subclass of `nnx.Pytree` with additional features such as sopport for metric reporting.

As we can see above, Variables are array-like; they support arithmetic operators, indexing, and can be used directly in JAX expressions. You can update their value in-place using `variable[...] = new_value`. Since NNX Pytrees are standard JAX pytrees, you can use `jax.tree.*` functions directly on them:

```{code-cell} ipython3
x = jnp.ones((3, 2))
y = model(x)
print(f'{y.shape = }, {model.count[...] = }')

# jax.tree.map works directly on NNX Pytrees
doubled_model = jax.tree.map(lambda x: x * 2, model)
print(f'\nmodel.w sum:   {model.w.sum():.4f}')
print(f'doubled.w sum: {doubled_model.w.sum():.4f}')

# jax.tree.leaves_with_path shows the full tree structure
print('\nPytree leaves:')
for path, value in jax.tree.leaves_with_path(model):
  print(f'{jax.tree_util.keystr(path)}: {value!r}')
```

Here `jax.tree.map` was first used create a new model with each leaf Array doubled, and then `jax.tree.flatten_with_path` was used to show how JAX sees the tree structure. Notice that because Variables are also JAX pytrees containing a single element (their inner value) we see `value` as part of the leaf path.

+++

## Rngs
`nnx.Rngs` simplify managing [JAX PRNG state](https://jax.readthedocs.io/en/latest/random-numbers.html). It is itself an `nnx.Pytree` that stores a seed `key` and an incrementing `counter` in `Variable`s internally. By calling it, `Rngs` can produce new PRNG keys:

```{code-cell} ipython3
rngs = nnx.Rngs(0)  # seeded with 0

key1 = rngs()       # get a raw key
key2 = rngs()       # different key (counter incremented)
arr = rngs.normal((2, 3))  # draw samples directly

print(f'{key1 = }')
print(f'{key2 = }')
print(f'{arr = }')
print(rngs)
```

As we've seen so far, `Rngs` conveniently exposes every `jax.random.*` distribution as a method (e.g. `rngs.uniform(...)`, `rngs.normal(...)`) without requiring the `key` argument and returning different random values every time they are called, this highly simplifies the user experience. In general `Rngs` can hold multiple keys and counters in structures called `RngStream`s, above we see that the `default` stream is being used. For more information check out the [Randomness guide](https://flax.readthedocs.io/en/latest/guides/randomness.html).

+++

## Nested Modules

Pytree subclasses compose naturally, you can assign one as an attribute of another to build nested models. The example below builds a simple `MLP` from two `Linear` layers:

```{code-cell} ipython3
class MLP(nnx.Pytree):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.din, self.dmid, self.dout = din, dmid, dout  # static attributes
    self.linear1 = Linear(din, dmid, rngs=rngs)             # data attribute
    self.linear2 = Linear(dmid, dout, rngs=rngs)            # data attribute

  def __call__(self, x: jax.Array):
    x = nnx.relu(self.linear1(x))
    return self.linear2(x)

mlp = MLP(2, 16, 5, rngs=nnx.Rngs(0))
y = mlp(jnp.ones((3, 2)))
print(f'{y.shape = }')

nnx.display(mlp)
```

Because the entire model is a single pytree, all the `jax.tree.*` functions, JAX transforms, and NNX state APIs work on the full nested structure at once. For more info check out the [Pytree guide](https://flax.readthedocs.io/en/latest/guides/pytree.html).

+++

## JAX Transforms

NNX models can be passed directly to JAX transforms like `jax.jit`. However, JAX transforms create pure functions, meaning that they won't propagate side effects such as Variable state updates back to the caller:

```{code-cell} ipython3
model = Linear(2, 5, rngs=nnx.Rngs(0))

@jax.jit
def forward(model, x): # pure function
  y = model(x)
  return y

y = forward(model, x)

print(model.count[...]) # no state update
```

Here `count` was not updated because inside `jax.jit` new Variable copies are created so any updates inside will not be reflected outside. To propagate updates we can use two NNX helpers. `nnx.state(obj, *filters)` extracts the current state of all Variables in `obj` as a nested `State` dict; you can pass **filters** to select specific Variable types, for example `nnx.state(model, Count)` extracts only `Count` Variables (see the [Filters guide](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) for details). `nnx.update(obj, state)` writes a `State` back into the corresponding Variables of `obj`.

```{code-cell} ipython3
model = Linear(2, 5, rngs=nnx.Rngs(0))

@jax.jit
def forward(model, x):
  y = model(x)
  return y, nnx.state(model, Count)  # propagate state

y, updates = forward(model, x)
nnx.update(model, updates)  # apply state updates

print(model.count[...])  # updated successfully
```

In this example we could've also chosen to return the entire `model` and replace its reference outside, however the use `nnx.state/update` is preferred as NNX promotes preserving existing Variable references.

+++

### Training step with JAX transforms

For a full training step we also need to differentiate with respect to some parameters while keeping the rest non-differentiable. `nnx.split` and `nnx.merge` let us partition and reconstruct the model. `nnx.split(obj, *filters)` returns a structure definition (`GraphDef`) followed by one `State` group per filter, where the catch-all filter `...` matches everything not yet matched by a previous filter (see the [Filters guide](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) for the full filter language). `nnx.merge(graphdef, *states)` reconstructs a copy of the object from its definition and state groups. We will use these to select the differentiable parameters when passing them to `jax.grad`.

The example below shows a complete training step using raw JAX transforms. `nnx.Optimizer` wraps an [Optax](https://optax.readthedocs.io/) optimizer and stores its internal state as Variables, providing a simple `update(model, grads)` method that performs in-place updates to both the optimizer state and model parameters:

```{code-cell} ipython3
import optax

x, y = jnp.ones((3, 2)), jnp.ones((3, 5))
model = Linear(2, 5, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@jax.jit
def train_step(model, optimizer, x, y):
  # use same filter as Optimizer's `wrt`
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, nondiff):
    nondiff = nnx.clone(nondiff) # refresh trace state
    model = nnx.merge(graphdef, params, nondiff)
    loss = jnp.mean((model(x) - y) ** 2)
    return loss, nnx.state(model, Count)  # propagate state

  grads, updates = jax.grad(loss_fn, has_aux=True)(params, nondiff)
  nnx.update(model, updates)
  optimizer.update(model, grads)

  return nnx.state((model, optimizer))

updates = train_step(model, optimizer, x, y)
nnx.update((model, optimizer), updates)

print(f'{model.count[...] = }')
print(f'{optimizer.step[...] = }')
```

A few things to note. The state of the `model` and `optimizer` is extracted at once by packing them in a tuple (or any pytree), and `nnx.update` accepts the same structure. By default `jax.grad` differentiates with respect to the first positional argument only, `params` in our case. Finally, `nnx.clone` is needed because `jax.grad` passes non differentiable inputs (here `nondiff`) directly without tracing them, so we must manually clone them to refresh the trace state of their Variables - preventing tracer leakage. Omitting `nnx.clone` raises an error.

+++

## NNX Transforms

NNX transforms (`nnx.jit`, `nnx.grad`, ...) are thin wrappers over JAX transforms that provide the exact same APIs. Their main feature is **automatic state propagation**: the state of all input Variables is tracked and updated automatically behind the scenes. This removes the need for the `nnx.state/update` boilerplate and the use of `nnx.clone`:

```{code-cell} ipython3
x, y = jnp.ones((3, 2)), jnp.ones((3, 5))
model = Linear(2, 5, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # automatic state propagation
def train_step(model, optimizer, x, y):
  # use same filter as Optimizer's `wrt`
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, nondiff):
    model = nnx.merge(graphdef, params, nondiff)
    loss = jnp.mean((model(x) - y) ** 2)
    return loss

  grads = nnx.grad(loss_fn)(params, nondiff)
  optimizer.update(model, grads)

train_step(model, optimizer, x, y)

print(f'{model.count[...] = }')
print(f'{optimizer.step[...] = }')
```

Notice that `train_step` doesn't need to return anthing as `nnx.jit` propagates all Variable updates (model parameters, optimizer state, counts) automatically.

+++

## Graph Mode

Certain programs are easier to express by sharing references between objets on different parts of a structure, however this is not compatible with JAX's pytree model. If we create a simple model that shares a reference to the same Variable in two different attributes, NNX transforms and most other APIs will raise an error as sharing can result in inconsistencies:

```{code-cell} ipython3
@nnx.dataclass
class Foo(nnx.Module):
  a: nnx.Param
  b: nnx.Param

p = nnx.Param(jnp.array(1.0))
model = Foo(p, p)  # shared Param

@nnx.jit
def forward(model, x):
  model.a[...] += 1.0
  return model.a * x + model.b

try:
  forward(model, jnp.array(1.0))
except ValueError as e:
  print(f'Error: {e}')
```

However, at the cost of some python overhead, `graph=True` can be passed to NNX APIs to enable **graph mode**. In graph mode, general graphs structures are allowed as long as they Variables are transformed consistently. We can fix the above example by enabling graph mode in `nnx.jit`:

```{code-cell} ipython3
@nnx.jit(graph=True)
def forward(model, x):
  model.a[...] += 1.0
  return model.a * x + model.b

y = forward(model, jnp.array(1.0))

print(f'{y = !s}, {model.a[...] = !s}, {model.b[...] = !s}')
```

## Hijax (experimental)

JAX's experimental **Hijax** API allows custom mutable types whose state updates propagate automatically through JAX transforms. When enabled via `nnx.var_default(hijax=True)`, plain JAX transforms like `jax.jit` handle state propagation of `Variable`s without any manual `nnx.state` / `nnx.update` calls. As a bonus, in hijax mode Variables can also be passed as captures, further simplifying the loss function:

```{code-cell} ipython3
with nnx.var_defaults(hijax=True): # enables Hijax Variables
  x, y = jnp.ones((3, 2)), jnp.ones((3, 5))
  model = Linear(2, 5, rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

print(model)  # display Hijax Variables

@jax.jit  # automatic state propagation
def train_step(model, optimizer, x, y):
  # use same filter as Optimizer's `wrt`
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params):
    model = nnx.merge(graphdef, params, nondiff)
    loss = jnp.mean((model(x) - y) ** 2)
    return loss

  grads = jax.grad(loss_fn)(nnx.vars_as(params, hijax=False))  # disable hijax for param grads
  optimizer.update(model, grads)

train_step(model, optimizer, x, y)

print(f'{model.count[...] = }')
print(f'{optimizer.step[...] = }')
```

As a temporary limitation, `jax.grad` does not yet handle mutable Hijax types. We work around this by converting `params` to regular Variables via `nnx.vars_as(params, hijax=False)` before passing them to `grad`. Hijax can also be enabled on a per-Variable basis by passing `hijax=True` to the constructor:

```{code-cell} ipython3
v = nnx.Variable(jnp.array(1), hijax=True)

@jax.jit
def inc(v):
  v[...] += 1

print(f'{v[...] = !s}')
inc(v)
print(f'{v[...] = !s}')
```

---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Flax basics

Flax NNX is a new simplified API that is designed to make it easier to create, inspect, debug, and analyze neural networks in [JAX](https://jax.readthedocs.io/). It achieves this by adding first class support for Python reference semantics. This allows users to express their models using regular Python objects, which are modeled as PyGraphs (instead of pytrees), enabling reference sharing and mutability. Such API design should make PyTorch or Keras users feel at home.

To begin, install Flax with `pip` and import necessary dependencies:

```{code-cell} ipython3
:tags: [skip-execution]

# ! pip install -U flax
```

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
```

## The Flax NNX Module system

The main difference between the Flax[`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) and other `Module` systems in [Flax Linen](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html) or [Haiku](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html#Built-in-Haiku-nets-and-nested-modules) is that in NNX everything is **explicit**. This  means, among other things, that the [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) itself holds the state (such as parameters) directly, the [PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) state is threaded by the user, and all shape information must be provided on initialization (no shape inference).

Let's begin by creating a `Linear` [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html). As shown next, dynamic state is usually stored in `nnx.Param`s, and static state (all types not handled by NNX) such as integers or strings are stored directly. Attributes of type `jax.Array` and `numpy.ndarray` are also treated as dynamic states, although storing them inside `nnx.Variable`s, such as `Param`, is preferred. Also the [`nnx.Rngs`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html#flax.nnx.Rngs) object can be used to get new unique keys based on a root PRNG key passed to the constructor.

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b
```

Also note that the inner values of `nnx.Variable`s can be accessed using the `value` property, but for convenience they implement all numeric operators and can be used directly in arithmetic expressions (as shown in the code above).

To initialize a Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html), you just call the constructor, and all the parameters of a `Module` are usually created eagerly. Since [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s hold their own state methods, you can call them directly without the need for a separate `apply` method.
This can be very convenient for debugging, allowing you to directly inspect the entire structure of the model.

```{code-cell} ipython3
model = Linear(2, 5, rngs=nnx.Rngs(params=0))
y = model(x=jnp.ones((1, 2)))

print(y)
nnx.display(model)
```

The above visualization by `nnx.display` is generated using the awesome
[Treescope](https://treescope.readthedocs.io/en/stable/index.html#) library.

+++

### Stateful computation

Implementing layers, such as [`nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm), requires performing state updates during a forward pass. In Flax NNX, you just need to create a [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable) and update its `.value` during the forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Counter(nnx.Module):
  def __init__(self):
    self.count = Count(jnp.array(0))

  def __call__(self):
    self.count += 1

counter = Counter()
print(f'{counter.count.value = }')
counter()
print(f'{counter.count.value = }')
```

Mutable references are usually avoided in JAX. But Flax NNX provides sound mechanisms
to handle them, as demonstrated in later sections of this guide.

+++

### Nested Modules

Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s can be used to compose other `Module`s in a nested structure. These can be assigned directly as attributes, or inside an attribute of any (nested) pytree type, such as a `list`, `dict`, `tuple`, and so on.

The example below shows how to define a simple `MLP` by subclassing [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html). The model consists of two `Linear` layers, an [`nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout) layer, and an [`nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) layer:

```{code-cell} ipython3
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
    return self.linear2(x)

model = MLP(2, 16, 5, rngs=nnx.Rngs(0))

y = model(x=jnp.ones((3, 2)))

nnx.display(model)
```

In Flax, [`nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout) is a stateful module that stores an [`nnx.Rngs`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html#flax.nnx.Rngs) object, so that it can generate new masks during the forward pass without the need for the user to pass a new key each time.

+++

### Model surgery

Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s are mutable by default. This means that their structure can be changed at any time, which makes [model surgery](https://flax.readthedocs.io/en/latest/guides/surgery.html) quite easy, as any sub-`Module` attribute can be replaced with anything else, such as new `Module`s, existing shared `Module`s, `Module`s of different types, and so on. Moreover, [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable)s can also be modified or replaced/shared.

The following example shows how to replace the `Linear` layers in the `MLP` model from the previous example with `LoraLinear` layers:

```{code-cell} ipython3
class LoraParam(nnx.Param): pass

class LoraLinear(nnx.Module):
  def __init__(self, linear: Linear, rank: int, rngs: nnx.Rngs):
    self.linear = linear
    self.A = LoraParam(jax.random.normal(rngs(), (linear.din, rank)))
    self.B = LoraParam(jax.random.normal(rngs(), (rank, linear.dout)))

  def __call__(self, x: jax.Array):
    return self.linear(x) + x @ self.A @ self.B

rngs = nnx.Rngs(0)
model = MLP(2, 32, 5, rngs=rngs)

# Model surgery.
model.linear1 = LoraLinear(model.linear1, 4, rngs=rngs)
model.linear2 = LoraLinear(model.linear2, 4, rngs=rngs)

y = model(x=jnp.ones((3, 2)))

nnx.display(model)
```

## Flax transformations

[Flax NNX transformations (transforms)](https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html) extend [JAX transforms](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations) to support [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s and other objects. They serve as supersets of their equivalent JAX counterparts with the addition of being aware of the object's state and providing additional APIs to transform it.

One of the main features of Flax Transforms is the preservation of reference semantics, meaning that any mutation of the object graph that occurs inside the transform is propagated outside as long as it is legal within the transform rules. In practice this means that Flax programs can be express using imperative code, highly simplifying the user experience.

In the following example, you define a `train_step` function that takes a `MLP` model, an [`nnx.Optimizer`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/optimizer.html#module-flax.nnx.optimizer), and a batch of data, and returns the loss for that step. The loss and the gradients are computed using the [`nnx.value_and_grad`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad) transform over the `loss_fn`. The gradients are passed to the optimizer's [`nnx.Optimizer.update`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/optimizer.html#flax.nnx.optimizer.Optimizer.update) method to update the `model`'s parameters.

```{code-cell} ipython3
import optax

# An MLP containing 2 custom `Linear` layers, 1 `nnx.Dropout` layer, 1 `nnx.BatchNorm` layer.
model = MLP(2, 16, 10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

@nnx.jit  # Automatic state management
def train_step(model, optimizer, x, y):
  def loss_fn(model: MLP):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # In place updates.

  return loss

x, y = jnp.ones((5, 2)), jnp.ones((5, 10))
loss = train_step(model, optimizer, x, y)

print(f'{loss = }')
print(f'{optimizer.step.value = }')
```

There are two things happening in this example that are worth mentioning:

1. The updates to each of the [`nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) and [`nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout) layer's state is automatically propagated from within `loss_fn` to `train_step` all the way to the `model` reference outside.
2. The `optimizer` holds a mutable reference to the `model` - this relationship is preserved inside the `train_step` function making it possible to update the model's parameters using the optimizer alone.

> **Note**<br> `nnx.jit` has performance overhead for small models, check the [Performance Considerations](https://flax.readthedocs.io/en/latest/guides/performance.html) guide for more information.

### Scan over layers

The next example uses Flax [`nnx.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.vmap) to create a stack of multiple MLP layers and [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.scan) to iteratively apply each layer of the stack to the input.

In the code below notice the following:

1. The custom `create_model` function takes in a key and returns an `MLP` object, since you create five keys and use [`nnx.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.vmap) over `create_model` a stack of 5 `MLP` objects is created.
2. The [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.Scan) is used to iteratively apply each `MLP` in the stack to the input `x`.
3. The [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.Scan) (consciously) deviates from [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) and instead mimics [`nnx.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.vmap), which is more expressive. [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.Scan) allows specifying multiple inputs, the scan axes of each input/output, and the position of the carry.
4. [`State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) updates for the [`nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) and [`nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout) layers are automatically propagated by [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.Scan).

```{code-cell} ipython3
@nnx.vmap(in_axes=0, out_axes=0)
def create_model(key: jax.Array):
  return MLP(10, 32, 10, rngs=nnx.Rngs(key))

keys = jax.random.split(jax.random.key(0), 5)
model = create_model(keys)

@nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
def forward(model: MLP, x):
  x = model(x)
  return x

x = jnp.ones((3, 10))
y = forward(model, x)

print(f'{y.shape = }')
nnx.display(model)
```

How do Flax NNX transforms achieve this? To understand how Flax NNX objects interact with JAX transforms, the next section explains the Flax NNX Functional API.

+++

## The Flax Functional API

The Flax NNX Functional API establishes a clear boundary between reference/object semantics and value/pytree semantics. It also allows the same amount of fine-grained control over the state that Flax Linen and Haiku users are used to. The Flax NNX Functional API consists of three basic methods:  [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split), [`nnx.merge`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.merge), and [`nnx.update`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.update).

Below is an example of of `StatefulLinear` [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) that uses the Functional API. It contains:

- Some [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable)s; and
- A custom `Count()` [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable) type, which is used to track the integer scalar state that increases on every forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class StatefulLinear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = Count(jnp.array(0, dtype=jnp.uint32))

  def __call__(self, x: jax.Array):
    self.count += 1
    return x @ self.w + self.b

model = StatefulLinear(din=3, dout=5, rngs=nnx.Rngs(0))
y = model(jnp.ones((1, 3)))

nnx.display(model)
```

### State and GraphDef

A Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) can be decomposed into [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) and [`nnx.GraphDef`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.GraphDef) using the [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split) function:

- [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) is a `Mapping` from strings to [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable)s or nested  [`State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State)s.
- [`nnx.GraphDef`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.GraphDef) contains all the static information needed to reconstruct a [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) graph, it is analogous to [JAX's `PyTreeDef`](https://jax.readthedocs.io/en/latest/pytrees.html#internal-pytree-handling).

```{code-cell} ipython3
graphdef, state = nnx.split(model)

nnx.display(graphdef, state)
```

### Split, merge, and update

Flax's [`nnx.merge`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.merge) is the reverse of [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split). It takes the [`nnx.GraphDef`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.GraphDef) + [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) and reconstructs the [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html). The example below demonstrates this as follows:

- By using [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split) and [`nnx.merge`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.merge) in sequence any `Module` can be lifted to be used in any JAX transform.
- [`nnx.update`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.update) can update an object in place with the content of a given [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State).
- This pattern is used to propagate the state from a transform back to the source object outside.

```{code-cell} ipython3
print(f'{model.count.value = }')

# 1. Use `nnx.split` to create a pytree representation of the `nnx.Module`.
graphdef, state = nnx.split(model)

@jax.jit
def forward(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> tuple[jax.Array, nnx.State]:
  # 2. Use `nnx.merge` to create a new model inside the JAX transformation.
  model = nnx.merge(graphdef, state)
  # 3. Call the `nnx.Module`
  y = model(x)
  # 4. Use `nnx.split` to propagate `nnx.State` updates.
  _, state = nnx.split(model)
  return y, state

y, state = forward(graphdef, state, x=jnp.ones((1, 3)))
# 5. Update the state of the original `nnx.Module`.
nnx.update(model, state)

print(f'{model.count.value = }')
```

The key insight of this pattern is that using mutable references is fine within a transform context (including the base eager interpreter) but it is necessary to use the Functional API when crossing boundaries.

**Why aren't Modules just pytrees?** The main reason is that it is very easy to lose track of shared references by accident this way, for example if you pass two [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s that have a shared `Module` through a JAX boundary, you will silently lose that sharing. Flax's Functional API makes this behavior explicit, and thus it is much easier to reason about.

+++

### Fine-grained State control

Experienced [Flax Linen](https://flax-linen.readthedocs.io/) or [Haiku](https://dm-haiku.readthedocs.io/) API users may recognize that having all the states in a single structure is not always the best choice as there are cases in which you may want to handle different subsets of the state differently. This a common occurrence when interacting with [JAX transforms](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations).

For example:

- Not every model state can or should be differentiated when interacting with [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad).
- Or, sometimes, there is a need to specify what part of the model's state is a carry and what part is not when using [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan).

To address this, the Flax NNX API has [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split), which allows you to pass one or more [`nnx.filterlib.Filter`s](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) to partition the [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable)s into mutually exclusive [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State)s. Flax NNX uses `Filter` create `State` groups in APIs (such as [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split), [`nnx.state()`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.state), and many of NNX transforms).

The example below shows the most common `Filter`s:

```{code-cell} ipython3
# Use `nnx.Variable` type `Filter`s to split into multiple `nnx.State`s.
graphdef, params, counts = nnx.split(model, nnx.Param, Count)

nnx.display(params, counts)
```

**Note:** [`nnx.filterlib.Filter`s](https://flax.readthedocs.io/en/latest/guides/filters_guide.html)s must be exhaustive, if a value is not matched an error will be raised.

As expected, the [`nnx.merge`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.merge) and [`nnx.update`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.update) methods naturally consume multiple [`State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State)s:

```{code-cell} ipython3
# Merge multiple `State`s
model = nnx.merge(graphdef, params, counts)
# Update with multiple `State`s
nnx.update(model, params, counts)
```

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

The main difference between the Flax `Module` and other Module systems in [Flax Linen](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html) or [Haiku](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html#Built-in-Haiku-nets-and-nested-modules) is that in NNX everything is **explicit**. This  means, among other things, that the NNX Module itself holds the state (such as parameters) directly, the [PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) state is threaded by the user, and all shape information must be provided on initialization (no shape inference).

Let's begin by creating a Linear `Module`. As shown next, dynamic state is usually stored in `Param`s, and static state (all types not handled by NNX) such as integers or strings are stored directly. Attributes of type `jax.Array` and `numpy.ndarray` are also treated as dynamic states, although storing them inside Variables, such as Param, is preferred. Also the `Rngs` object can be used to get new unique keys based on a root PRNG key passed to the constructor.

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(rngs.params.uniform((din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b[None]
```

Also note that the inner values of `Variable`s can be accessed using the `value` property, but for convenience they implement all numeric operators and can be used directly in arithmetic expressions (as shown in the code above).

To initialize a Flax `Module`, you just call the constructor, and all the parameters of a Module are usually created eagerly. Since Modules hold their own state methods, you can call them directly without the need for a separate apply method.
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

Implementing layers, such as `BatchNorm`, requires performing state updates during a forward pass. In Flax NNX, you just need to create a `Variable` and update its `.value` during the forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Counter(nnx.Module):
  def __init__(self):
    self.count = Count(jnp.array(0))

  def __call__(self):
    self.count[...] += 1

counter = Counter()
print(f'{counter.count[...] = }')
counter()
print(f'{counter.count[...] = }')
```

Mutable references are usually avoided in JAX. But Flax NNX provides sound mechanisms
to handle them, as demonstrated in later sections of this guide.

+++

### Nested Modules

Flax `Module`s can be used to compose other Modules in a nested structure. These can be assigned directly as attributes, or inside an attribute of any (nested) pytree type, such as a `list`, `dict`, `tuple`, and so on.

The example below shows how to define a simple `MLP` by subclassing `Module`. The model consists of two `Linear` layers, a `Dropout` layer, and a `BatchNorm` layer. Note that we need to pass the `__call__` method the RNG state that we want the `Dropout` layer to use.

```{code-cell} ipython3
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array, rngs: nnx.Rngs):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x)), rngs=rngs))
    return self.linear2(x)

model = MLP(2, 16, 5, rngs=nnx.Rngs(0))

y = model(x=jnp.ones((3, 2)), rngs=nnx.Rngs(1))

nnx.display(model)
```

### Model surgery

Flax `Module`s are mutable by default. This means that their structure can be changed at any time, which makes [model surgery](https://flax.readthedocs.io/en/latest/guides/surgery.html) quite easy, as any sub-Module attribute can be replaced with anything else, such as new Modules, existing shared Modules, Modules of different types, and so on. Moreover, `Variable`s can also be modified or replaced/shared.

The following example shows how to replace the `Linear` layers in the `MLP` model from the previous example with `LoraLinear` layers:

```{code-cell} ipython3
class LoraParam(nnx.Param): pass

class LoraLinear(nnx.Module):
  def __init__(self, linear: Linear, rank: int, rngs: nnx.Rngs):
    self.linear = linear
    self.A = LoraParam(rngs.normal((linear.din, rank)))
    self.B = LoraParam(rngs.normal((rank, linear.dout)))

  def __call__(self, x: jax.Array):
    return self.linear(x) + x @ self.A @ self.B

rngs = nnx.Rngs(0)
model = MLP(2, 32, 5, rngs=rngs)

# Model surgery.
model.linear1 = LoraLinear(model.linear1, 4, rngs=rngs)
model.linear2 = LoraLinear(model.linear2, 4, rngs=rngs)

y = model(x=jnp.ones((3, 2)), rngs=rngs)

nnx.display(model)
```

## Flax transformations

[Flax NNX transformations (transforms)](https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html) extend [JAX transforms](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations) to support `Module`s and other objects. They serve as supersets of their equivalent JAX counterparts with the addition of being aware of the object's state and providing additional APIs to transform it.

One of the main features of Flax Transforms is the preservation of reference semantics, meaning that any mutation of the object graph that occurs inside the transform is propagated outside as long as it is legal within the transform rules. In practice this means that Flax programs can be expressed using imperative code, highly simplifying the user experience.

In the following example, you define a `train_step` function that takes a `MLP` model, an `Optimizer`, and a batch of data, and returns the loss for that step. The loss and the gradients are computed using the `nnx.value_and_grad` transform over the `loss_fn`. The gradients are passed to the optimizer's `update` method to update the model's parameters.

```{code-cell} ipython3
import optax

# An MLP containing 2 custom `Linear` layers, 1 `nnx.Dropout` layer, 1 `nnx.BatchNorm` layer.
model = MLP(2, 16, 10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # Automatic state management
def train_step(model, optimizer, x, y, rngs):
  def loss_fn(model: MLP, rngs: nnx.Rngs):
    y_pred = model(x, rngs)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model, rngs)
  optimizer.update(model, grads)  # In place updates.

  return loss

x, y = jnp.ones((5, 2)), jnp.ones((5, 10))
loss = train_step(model, optimizer, x, y, rngs)

print(f'{loss = }')
print(f'{optimizer.step.value = }')
```

There are two things happening in this example that are worth mentioning:

1. The updates to each of the `BatchNorm` and `Dropout` layer's state is automatically propagated from within `loss_fn` to `train_step` all the way to the `model` reference outside.
2. The `optimizer` holds a mutable reference to the model - this relationship is preserved inside the train_step function making it possible to update the model's parameters using the optimizer alone.

> **Note**<br> `nnx.jit` has performance overhead for small models, check the [Performance Considerations](https://flax.readthedocs.io/en/latest/guides/performance.html) guide for more information.

### Scan over layers

The next example uses Flax `nnx.vmap` to create a stack of multiple MLP layers and `nnx.scan` to iteratively apply each layer of the stack to the input.

In the code below notice the following:

1. The custom `create_model` function takes in a key and returns an `MLP` object, since you create five keys and use `nnx.vmap` over `create_model` a stack of 5 `MLP` objects is created.
2. The `nnx.scan` is used to iteratively apply each `MLP` in the stack to the input `x`.
3. The nnx.scan (consciously) deviates from `jax.lax.scan` and instead mimics nnx.vmap, which is more expressive. nnx.scan allows specifying multiple inputs, the scan axes of each input/output, and the position of the carry.
4. `State` updates for `BatchNorm` layers are automatically propagated by nnx.scan.
5. The `rngs` object is split into separate streams for each layer using the `fork` method.

```{code-cell} ipython3
@nnx.vmap(in_axes=0, out_axes=0)
def create_model(rngs):
  return MLP(10, 32, 10, rngs=rngs)

@nnx.scan(in_axes=(0, 0, nnx.Carry), out_axes=nnx.Carry)
def forward(model: MLP, rngs: nnx.Rngs, x):
  x = model(x, rngs)
  return x
    
param_rngs = nnx.Rngs(0).fork(split=5)
model = create_model(param_rngs)
```

```{code-cell} ipython3
x = jnp.ones((3, 10))
dropout_rngs = nnx.Rngs(1).fork(split=5)
y = forward(model, dropout_rngs, x)

print(f'{y.shape = }')
nnx.display(model)
```

How do Flax NNX transforms achieve this? To understand how Flax NNX objects interact with JAX transforms, the next section explains the Flax NNX Functional API.

+++

## The Flax Functional API

The Flax NNX Functional API establishes a clear boundary between reference/object semantics and value/pytree semantics. It also allows the same amount of fine-grained control over the state that Flax Linen and Haiku users are used to. The Flax NNX Functional API consists of three basic methods:  `nnx.split`, `nnx.merge`, and `nnx.update`.

Below is an example of of `StatefulLinear` `Module` that uses the Functional API. It contains:

- Some `Param` Variables; and
- A custom `Count` Variable type, which is used to track the integer scalar state that increases on every forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class StatefulLinear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(rngs.uniform((din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = Count(jnp.array(0, dtype=jnp.uint32))

  def __call__(self, x: jax.Array):
    self.count.value += 1
    return x @ self.w + self.b

model = StatefulLinear(din=3, dout=5, rngs=nnx.Rngs(0))
y = model(jnp.ones((1, 3)))

nnx.display(model)
```

### State and GraphDef

A Flax `Module` can be decomposed into `State` and `GraphDef` using the `nnx.split` function:

- `State` is a `Mapping` from strings to `Variable`s or nested  `State`s.
- `GraphDef` contains all the static information needed to reconstruct a `Module` graph, it is analogous to [JAX's `PyTreeDef`](https://jax.readthedocs.io/en/latest/pytrees.html#internal-pytree-handling).

```{code-cell} ipython3
graphdef, state = nnx.split(model)

nnx.display(graphdef, state)
```

### Split, merge, and update

Flax's `nnx.merge` is the reverse of `nnx.split`. It takes the `GraphDef` + `State` and reconstructs the `Module`. The example below demonstrates this as follows:

- By using `nnx.split` and `nnx.merge` in sequence any `Module` can be lifted to be used in any JAX transform.
- `nnx.update` can update an object in place with the content of a given `State`.
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

**Why aren't Modules just pytrees?** The main reason is that it is very easy to lose track of shared references by accident this way, for example if you pass two `Module`s that have a shared Module through a JAX boundary, you will silently lose that sharing. Flax's Functional API makes this behavior explicit, and thus it is much easier to reason about.

+++

### Fine-grained State control

Experienced [Flax Linen](https://flax-linen.readthedocs.io/) or [Haiku](https://dm-haiku.readthedocs.io/) API users may recognize that having all the states in a single structure is not always the best choice as there are cases in which you may want to handle different subsets of the state differently. This is a common occurrence when interacting with [JAX transforms](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations).

For example:

- Not every model state can or should be differentiated when interacting with `jax.grad`.
- Or, sometimes, there is a need to specify what part of the model's state is a carry and what part is not when using `jax.lax.scan`.

To address this, the Flax NNX API has `nnx.split`, which allows you to pass one or more `Filter`s to partition the `Variable`s into mutually exclusive `State`s. Flax NNx uses `Filter` create `State` groups in APIs (such as `nnx.split`, `nnx.state`, and many of NNX transforms).

The example below shows the most common `Filter`s:

```{code-cell} ipython3
# Use `nnx.Variable` type `Filter`s to split into multiple `nnx.State`s.
graphdef, params, counts = nnx.split(model, nnx.Param, Count)

nnx.display(params, counts)
```

**Note:** `Filter`s must be exhaustive, if a value is not matched an error will be raised.

As expected, the `nnx.merge` and `nnx.update` methods naturally consume multiple `State`s:

```{code-cell} ipython3
# Merge multiple `State`s
model = nnx.merge(graphdef, params, counts)
# Update with multiple `State`s
nnx.update(model, params, counts)
```

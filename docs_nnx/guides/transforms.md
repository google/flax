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
JAX transformations in general operate on [Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) of arrays
and abide by value semantics, this presents a challenge for Flax NNX which represents Modules as regular Python objects
that follow reference semantics. To address this, Flax NNX introduces its own set of transformations that extend JAX
transformations to allow Modules and other Flax NNX objects to be passed in and out of transformations while preserving
reference semantics.

Flax NNX transformations should feel quite familar to those who have used JAX transformations before as they use the
same APIs and behave like the JAX transformations when only working with Pytrees of arrays. However, when working with
Flax NNX objects, they allow Python's reference semantics to be preserved for these objects, this includes:
* Preserving shared references across multiple objects in the inputs and outputs of the transformation.
* Propagating any state changes made to the objects inside the transformation to the objects outside the transformation.
* Enforcing consistency of how objects are transformed when aliases are present across multiple inputs and outputs.

```{code-cell} ipython3
import jax
from jax import numpy as jnp, random
from flax import nnx
```

Throughout this guide we will use `nnx.vmap` as a case study to demonstrate how Flax NNX transformations work but the principles
outlined here extend to all transformations.

## Basic Example
To begin, let's look at a simple example of using `nnx.vmap` to extend an elementwise `vector_dot` function to work on
batched inputs. We will define a `Weights` Module with no methods to hold some parameters, these weights will be passed
as an input to the `vector_dot` function along with some data. Both the weights and data will be batched on axis `0` and we will use
`nnx.vmap` to apply `vector_dot` to each batch element, and the result will be a batched on axis `1`:

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)

self = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  bias=jnp.zeros((10, 3)),
)
x = jax.random.normal(random.key(1), (10, 2))

def vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  return x @ weights.kernel + weights.bias

y = nnx.vmap(vector_dot, in_axes=0, out_axes=1)(self, x)

print(f'{y.shape = }')
nnx.display(self)
```

Notice that `in_axes` interacts naturally with the `Weights` Module, treating it as if it where a Pytree of arrays. Prefix patterns are also allowed, `in_axes=(0, 0)` would've also worked in this case.

Objects are also allowed as outputs of Flax NNX transformations, this can be useful to transform initializers. For example,
we can define a `create_weights` function to create an single `Weights` Module and use `nnx.vmap` to create a stack of
`Weights` with the same shapes as before:

```{code-cell} ipython3
def create_weights(seed: jax.Array):
  return Weights(
    kernel=random.uniform(random.key(seed), (2, 3)),
    bias=jnp.zeros((3,)),
  )

seeds = jnp.arange(10)
self = nnx.vmap(create_weights)(seeds)
nnx.display(self)
```

## Transforming Methods
Methods in Python are just functions that take the instance as the first argument, this means that you can decorate methods from `Module` and other Flax NNX subtypes. For example, we can refactor `Weights` from the previous example and decorate `__init__` with `vmap` to do the work of `create_weights`, and add a `__call__` method and decorate it with `vmap` to do the work of `vector_dot`:

```{code-cell} ipython3
class WeightStack(nnx.Module):
  @nnx.vmap
  def __init__(self, seed: jax.Array):
    self.kernel = nnx.Param(random.uniform(random.key(seed), (2, 3)))
    self.bias = nnx.Param(jnp.zeros((3,)))

  @nnx.vmap(in_axes=0, out_axes=1)
  def __call__(self, x: jax.Array):
    assert self.kernel.ndim == 2, 'Batch dimensions not allowed'
    assert x.ndim == 1, 'Batch dimensions not allowed'
    return x @ self.kernel + self.bias

weights = WeightStack(jnp.arange(10))

x = jax.random.normal(random.key(1), (10, 2))
y = weights(x)

print(f'{y.shape = }')
nnx.display(weights)
```

Throughout the rest of the guide we will focus on transforming individual functions, however, note all examples can easily be written in this method style.

+++

## State propagation
So far our functions have been stateless. However, the real power of Flax NNX transformations comes when we have stateful functions since one of their main features is to propagate state changes to preserve reference semantics. Let's update our example by adding
a `count` attribute to `Weights` and incrementing it in the new `stateful_vector_dot` function.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

self = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  bias=jnp.zeros((10, 3)),
  count=jnp.arange(10),
)
x = jax.random.normal(random.key(1), (10, 2))

def stateful_vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  weights.count += 1
  return x @ weights.kernel + weights.bias


y = nnx.vmap(stateful_vector_dot, in_axes=0, out_axes=1)(self, x)

self.count
```

After running `stateful_vector_dot` once we verify that the `count` attribute was correctly updated. Because Weights is vectorized, `count` was initialized as an `arange(10)`, and all of its elements were incremented by 1 inside the transformation. The most important part is that updates were propagated to the original `Weights` object outside the transformation. Nice!

+++

### Graph updates propagation
JAX transformations see inputs as pytrees of arrays, and Flax NNX see inputs pytrees of arrays and Python references, where references form a graph. Flax NNX's state propagation machinery can track arbitrary updates to the objects as long as they're local to the inputs (updates to globals inside transforms are not supported). This means that you can modify graph structure as needed, including updating existing attributes, adding/deleting attributes, swapping attributes, sharing (new) references between objects, sharing Variables between objects, etc. The sky is the limit!

The following example demonstrates performing some arbitrary updates to the `Weights` object inside `nnx.vmap` and verifying that the updates are correctly propagated to the original `Weights` object outside the transformation.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

self = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  bias=jnp.zeros((10, 3)),
  count=jnp.arange(10),
)
x = jax.random.normal(random.key(1), (10, 2))

def crazy_vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  weights.count += 1
  y = x @ weights.kernel + weights.bias
  weights.some_property = ['a', 2, False] # add attribute
  del weights.bias # delete attribute
  weights.new_param = weights.kernel # share reference
  return y

y = nnx.vmap(crazy_vector_dot, in_axes=0, out_axes=1)(self, x)

nnx.display(self)
```

> With great power comes great responsibility.
> <br> \- Uncle Ben

While this feature is very powerful, it must be used with care as it can clash with JAX's underlying assumptions for certain transformations. For example, `jit` expects the structure of the inputs to be stable in order to cache the compiled function, so changing the graph structure inside a `nnx.jit`-ed function cause continuous recompilations and performance degradation. On the other hand, `scan` only allows a fixed `carry` structure, so adding/removing substates declared as carry will cause an error.

+++

## Transforming Substates (Lift Types)

Certain JAX transformation allow the use of pytree prefixes to specify how different parts of the inputs/outputs should be transformed. Flax NNX supports pytree prefixes for pytree structures but currently it doesn't have the notion of a prefix for graph objects. Instead, Flax NNX introduces the concept of `Lift Types` which allow specifying how different substates of an object should be transformed. Different transformations support different Lift Types, here is the list of currently supported Lift Types for each transformation:

| Lift Type        | Transforms                              |
|------------------|-----------------------------------------|
| `StateAxes`      | `vmap`, `pmap`, `scan`                  |
| `StateSharding`  | `jit`, `shard_map`                      |
| `DiffState`      | `grad`, `value_and_grad`, `custom_vjp`  |

> NOTE: `shard_map` is not yet implemented.

If we want to specify how to vectorize different substates of an object in `nnx.vmap`, we create a `StateAxes` which maps a set of substates via [Filters](https://flax-nnx.readthedocs.io/en/latest/guides/filters_guide.html) to their corresponding axes, and pass the `StateAxes` to `in_axes` and `out_axes` as if it were a pytree prefix. Let's use the previous `stateful_vector_dot` example and
vectorize only the `Param` variables and broadcast the `count` variable so we only keep a single count for all the batch elements.
To do this we will define a `StateAxes` with a filter that matches the `Param` variables and maps them to axis `0`, and all the `Count` variables to `None`, and pass this `StateAxes` to `in_axes` for the `Weights` object.

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

self = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  bias=jnp.zeros((10, 3)),
  count=jnp.array(0),
)
x = jax.random.normal(random.key(1), (10, 2))


def stateful_vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  weights.count += 1
  return x @ weights.kernel + weights.bias

state_axes = nnx.StateAxes({nnx.Param: 0, Count: None}) # broadcast Count
y = nnx.vmap(stateful_vector_dot, in_axes=(state_axes, 0), out_axes=1)(self, x)

self.count
```

Here count is now a scalar since its not being vectorized. Also, note that `StateAxes` can only be used directly on Flax NNX objects, it cannot be used as a prefix for a pytree of objects.

+++

### Random State
In Flax NNX random state is just regular state. This means that its stored inside Modules that need it and its treated as any other type of state. This is a simplification over Flax Linen where random state was handled by a separate mechanism. In practice Modules simply need to keep a reference to a `Rngs` object that is passed to them during initialization, and use it to generate a unique key for each random operation. For the purposes of this guide, this means that random state can be transformed like any other type of state but we also need be aware of how the state is laid out so we can transform it correctly.

Let's suppose we want change things up a bit and apply the same weights to all elements in the batch but we want to add different random noise to each element. To do this we will add a `Rngs` attribute to `Weights`, created from a `seed` key argument passed during construction, this seed key must be `split` before hand so we can vectorize it succesfully. For pedagogical reasons, we will assign the seed key to a `noise` Stream and sample from it. To vectorize the RNG state we must configure `StateAxes` to map all `RngState` (base class for all variables in `Rngs`) to axis `0`, and `Param` and `Count` to `None`.

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel, bias, count, seed):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)
    self.rngs = nnx.Rngs(noise=seed)

self = Weights(
  kernel=random.uniform(random.key(0), (2, 3)),
  bias=jnp.zeros((3,)),
  count=jnp.array(0),
  seed=random.split(random.key(0), num=10),
)
x = random.normal(random.key(1), (10, 2))

def noisy_vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  weights.count += 1
  y = x @ weights.kernel + weights.bias
  return y + random.normal(weights.rngs.noise(), y.shape)

state_axes = nnx.StateAxes({nnx.RngState: 0, (nnx.Param, Count): None})
y1 = nnx.vmap(noisy_vector_dot, in_axes=(state_axes, 0))(self, x)
y2 = nnx.vmap(noisy_vector_dot, in_axes=(state_axes, 0))(self, x)

print(jnp.allclose(y1, y2))
nnx.display(self)
```

Because `Rngs`'s state is updated in place and automatically propagated by `nnx.vmap`, we will get a different result every time that `noisy_vector_dot` is called.

In the example above we manually split the random state during construction, this is fine as it makes the intention clear but it also doesn't let us use `Rngs` outside of `vmap` since its state is always split. To solve this we pass an unplit seed and use the `nnx.split_rngs` decorator before `vmap` to split the `RngState` right before each call to the function and then "lower" it back so its usable.

```{code-cell} ipython3
self = Weights(
  kernel=random.uniform(random.key(0), (2, 3)),
  bias=jnp.zeros((3,)),
  count=jnp.array(0),
  seed=0,
)
x = random.normal(random.key(1), (10, 2))

state_axes = nnx.StateAxes({nnx.RngState: 0, (nnx.Param, Count): None})

@nnx.split_rngs(splits=10)
@nnx.vmap(in_axes=(state_axes, 0))
def noisy_vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  weights.count += 1
  y = x @ weights.kernel + weights.bias
  return y + random.normal(weights.rngs.noise(), y.shape)

y1 = noisy_vector_dot(self, x)
y2 = noisy_vector_dot(self, x)

print(jnp.allclose(y1, y2))
nnx.display(self)
```

## Consistent aliasing
The main issue with allowing for reference semantics in transforms that references can be shared across inputs and outputs, this can be problematic if not taken care of because it would lead to ill-defined or inconsistent behavior. In the example below we have a single `Weights` Module `m` whose reference appears in multiple places in `arg1` and `arg2`. The problem is that we also specified we wanted to vectorize `arg1` in axis `0` and `arg2` in axis `1`, this is fine in JAX due to referential transparency of pytrees but its problematic in Flax NNX since we are trying to vectorize `m` in two different ways. NNX will enforce consistency by raising an error.

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, array: jax.Array):
    self.param = nnx.Param(array)

m = Weights(jnp.arange(10))
arg1 = {'a': {'b': m}, 'c': m}
arg2 = [(m, m), m]

@nnx.vmap(in_axes=(0, 1))
def f(arg1, arg2):
  ...

try:
  f(arg1, arg2)
except ValueError as e:
  print(e)
```

Inconsistent aliasing can also happen between inputs and outputs. In the next example we have a trivial function that accepts and immediately return `arg1`, however `arg1` is vectorized on axis `0` on the input and axis `1` on the output. As expected, this is problematic and Flax NNX will raise an error.

```{code-cell} ipython3
@nnx.vmap(in_axes=0, out_axes=1)
def f(arg1):
  return arg1

try:
  f(arg1)
except ValueError as e:
  print(e)
```

## Axes Metadata
Flax NNX Variables can have hold arbitrary metadata which can be added by simply passing them as keyword arguments to their constructor. This is often used to store `sharding` information which is used by the `nnx.spmd` APIs like `nnx.get_partition_spec` and `nnx.get_named_sharding`. However, its often important to keep this axes-related information in sync to what the actual state of the axes is when transforms are involved, for example, if we vectorize a variable on axis `1` we should remove the `sharding` information at position `1` when inside a `vmap` or `scan` to reflect the fact that the axes is temporarily removed. To achieve this Flax NNX transforms provide a non-standard `transform_metadata` dictionary argument, when the `nnx.PARTITION_NAME` key is present the `sharding` metadata will be updated as specified by `in_axes` and `out_axes`. Let's see an example of this in action:

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, array: jax.Array, sharding: tuple[str | None, ...]):
    self.param = nnx.Param(array, sharding=sharding)

m = Weights(jnp.ones((3, 4, 5)), sharding=('a', 'b', None))

@nnx.vmap(in_axes=1, transform_metadata={nnx.PARTITION_NAME: 'b'})
def f(m: Weights):
  print(f'Inner {m.param.shape = }')
  print(f'Inner {m.param.sharding = }')

f(m)
print(f'Outter {m.param.shape = }')
print(f'Outter {m.param.sharding = }')
```

Here we added a `sharding` metadata to the `Param` variables and used `transform_metadata` to update the `sharding` metadata to reflect the axes changes, specifically we can see that first axis `b` was removed from the `sharding` metadata when inside `vmap` and then added back when outside `vmap`.

We can verify that this also works when Modules are created inside the transformation, the new `sharding` axes will be added to the Module's Variables outside the transformation, matching the axes of the transformed Variables.

```{code-cell} ipython3
@nnx.vmap(out_axes=1, axis_size=4, transform_metadata={nnx.PARTITION_NAME: 'b'})
def init_vmap():
  return Weights(jnp.ones((3, 5)), sharding=('a', None))

m = init_vmap()
print(f'Outter {m.param.shape = }')
print(f'Outter {m.param.sharding = }')
```

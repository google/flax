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

# Transformations

In general, JAX transformations (transforms) operate on [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) of `jax.Array`s
and abide by value semantics. This presents a challenge for Flax NNX, which represents `nnx.Module`s as regular Python objects
that follow reference semantics. To address this, Flax NNX introduced its own set of transforms that extend JAX
transforms to allow `nnx.Module`s and other Flax NNX objects to be passed in and out of transforms while preserving
reference semantics.

Flax NNX transforms should feel quite familiar if you have used JAX transforms before. They use the
same APIs and behave like the JAX transforms when only working with pytrees of `jax.Array`s. However, when working with
Flax NNX objects, they allow Python's reference semantics to be preserved for these objects, this includes:

* Preserving shared references across multiple objects in the inputs and outputs of the transformation.
* Propagating any state changes made to the objects inside the transformation to the objects outside the transformation.
* Enforcing consistency of how objects are transformed when aliases are present across multiple inputs and outputs.

```{code-cell} ipython3
import jax
from jax import numpy as jnp, random
from flax import nnx
```

Throughout this guide, `nnx.vmap` is used as a case study to demonstrate how Flax NNX transforms work. However, the principles
outlined in this document extends to all transforms.

## Basic example

To begin, let's look at a simple example of using `nnx.vmap` to extend an element wise `vector_dot` function to work on
batched inputs. We will define a `Weights` Module with no methods to hold some parameters, these weights will be passed
as an input to the `vector_dot` function along with some data. Both the weights and data will be batched on axis `0` and we will use
`nnx.vmap` to apply `vector_dot` to each batch element, and the result will be a batched on axis `1`:

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)

weights = Weights(
  kernel=random.uniform(random.key(0), (10, 2, 3)),
  bias=jnp.zeros((10, 3)),
)
x = jax.random.normal(random.key(1), (10, 2))

def vector_dot(weights: Weights, x: jax.Array):
  assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
  assert x.ndim == 1, 'Batch dimensions not allowed'
  return x @ weights.kernel + weights.bias

y = nnx.vmap(vector_dot, in_axes=0, out_axes=1)(weights, x)

print(f'{y.shape = }')
nnx.display(weights)
```

Notice that `in_axes` interacts naturally with the `Weights` Module, treating it as if it were a pytree of `jax.Array`s. Prefix patterns are also allowed, so `in_axes=(0, 0)` would have also worked in this case.

Objects are also allowed as outputs of Flax NNX transforms, which can be useful to transform initializers. For example,
you can define a `create_weights` function to create an single `Weights` `nnx.Module`, and use `nnx.vmap` to create a stack of
`Weights` with the same shapes as before:

```{code-cell} ipython3
def create_weights(seed: jax.Array):
  return Weights(
    kernel=random.uniform(random.key(seed), (2, 3)),
    bias=jnp.zeros((3,)),
  )

seeds = jnp.arange(10)
weights = nnx.vmap(create_weights)(seeds)
nnx.display(weights)
```

## Transforming methods

Methods in Python are just functions that take the instance as the first argument, this means that you can decorate methods from `Module` and other Flax NNX subtypes. For example, we can refactor `Weights` from the previous example and decorate `__init__` with `vmap` to do the work of `create_weights`, and add a `__call__` method and decorate it with `@nnx.vmap` to do the work of `vector_dot`:

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

The rest of the guide will focus on transforming individual functions. But do note that all examples can be written in this method style.

+++

## State propagation

So far our functions have been stateless. However, the real power of Flax NNX transforms comes when you have stateful functions, because one of their main features is to propagate state changes to preserve reference semantics. Let's update the previous example by adding
a `count` attribute to `Weights` and incrementing it in the new `stateful_vector_dot` function:

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

weights = Weights(
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


y = nnx.vmap(stateful_vector_dot, in_axes=0, out_axes=1)(weights, x)

weights.count
```

After running `stateful_vector_dot` once, you verified that the `count` attribute was correctly updated. Because `Weights` was vectorized, `count` was initialized as an `arange(10)`, and all of its elements were incremented by `1` inside the transformation. The most important part is that updates were propagated to the original `Weights` object outside the transformation. Nice!

+++

### Graph updates propagation

JAX transforms see inputs as pytrees of `jax.Array`s, and Flax NNX sees inputs as  pytrees of `jax.Array`s and Python references, where references form a graph. Flax NNX's state propagation machinery can track arbitrary updates to the objects as long as they're local to the inputs (updates to globals inside transforms are not supported).

This means that you can modify graph structure as needed, including updating existing attributes, adding/deleting attributes, swapping attributes, sharing (new) references between objects, sharing `nnx.Variable`s between objects, etc. Sky is the limit!

The following example demonstrates performing some arbitrary updates to the `Weights` object inside `nnx.vmap`, and verifying that the updates are correctly propagated to the original `Weights` object outside the transformation:

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

weights = Weights(
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

y = nnx.vmap(crazy_vector_dot, in_axes=0, out_axes=1)(weights, x)

nnx.display(weights)
```

> With great power comes great responsibility.
> <br> \- Uncle Ben

While this feature is very powerful, it must be used with care because it can clash with JAX's underlying assumptions for certain transforms. For example, `jit` expects the structure of the inputs to be stable in order to cache the compiled function, so changing the graph structure inside an `nnx.jit`-ed function causes continuous recompilations and performance degradation. On the other hand, `scan` only allows a fixed `carry` structure, so adding/removing sub-states declared as carry will cause an error.

+++

## Transforming sub-states (lift types)

Certain JAX transforms allow the use of pytree prefixes to specify how different parts of the inputs/outputs should be transformed. Flax NNX supports pytree prefixes for pytree structures but currently it doesn't have the notion of a prefix for graph objects. Instead, Flax NNX introduces the concept of “lift types” which allow specifying how different sub-states of an object should be transformed. Different transforms support different lift types, here is the list of currently supported FLax NNX lift types for each JAX transformation:

| Lift type        | JAX transforms                          |
|------------------|-----------------------------------------|
| `StateAxes`      | `vmap`, `pmap`, `scan`                  |
| `StateSharding`  | `jit`, `shard_map`*                     |
| `DiffState`      | `grad`, `value_and_grad`, `custom_vjp`  |

> **Note:** * Flax NNX `shard_map` has not been implemented yet at the time of writing this version of the document.

To specify how to vectorize different sub-states of an object in `nnx.vmap`, the Flax team created a `nnx.StateAxes`. `StateAxes` maps a set of sub-states via Flax NNX [Filters](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) to their corresponding axes, and you can pass the `nnx.StateAxes` to `in_axes` and `out_axes` as if it/they were a pytree prefix.

Let's use the previous `stateful_vector_dot` example and vectorize only the `nnx.Param` variables and broadcast the `count` variable so we only keep a single count for all the batch elements.
To do this we will define a `nnx.StateAxes` with a filter that matches the `nnx.Param` variables and maps them to axis `0`, and all the `Count` variables to `None`, and pass this `nnx.StateAxes` to `in_axes` for the `Weights` object.

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array, count: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)

weights = Weights(
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
y = nnx.vmap(stateful_vector_dot, in_axes=(state_axes, 0), out_axes=1)(weights, x)

weights.count
```

Here, `count` is now a scalar since it's not being vectorized. Also, note that `nnx.StateAxes` can only be used directly on Flax NNX objects, and it cannot be used as a prefix for a pytree of objects.

+++

### Random state

In Flax NNX, a random state is just a regular state. This means that it is stored inside `nnx.Module`s that need it, and it is treated as any other type of state. This is a simplification over Flax Linen, where a random state was handled by a separate mechanism. In practice `nnx.Module`s simply need to keep a reference to a `Rngs` object that is passed to them during initialization, and use it to generate a unique key for each random operation. For the purposes of this guide, this means that random state can be transformed like any other type of state but we also need to be aware of how the state is laid out so we can transform it correctly.

Suppose you want to change things up a bit and apply the same weights to all elements in the batch. But you also want to add different random noise to each element.

To do this, you will add an `Rngs` attribute to `Weights`, created from a `seed` key argument passed during construction. This seed key must be `split` beforehand, so that you can vectorize it successfully. For pedagogical reasons, you will assign the seed key to a `noise` “stream” and sample from it. To vectorize the PRNG state, you must configure `nnx.StateAxes` to map all `RngState`s (a base class for all variables in `Rngs`) to axis `0`, and `nnx.Param` and `Count` to `None`.

```{code-cell} ipython3
class Weights(nnx.Module):
  def __init__(self, kernel, bias, count, seed):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)
    self.count = Count(count)
    self.rngs = nnx.Rngs(noise=seed)

weights = Weights(
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
y1 = nnx.vmap(noisy_vector_dot, in_axes=(state_axes, 0))(weights, x)
y2 = nnx.vmap(noisy_vector_dot, in_axes=(state_axes, 0))(weights, x)

print(jnp.allclose(y1, y2))
nnx.display(weights)
```

Because `Rngs`'s state is updated in place and automatically propagated by `nnx.vmap`, we will get a different result every time that `noisy_vector_dot` is called.

In the example above, you manually split the random state during construction. This is fine, as it makes the intention clear, but it also doesn't let you use `Rngs` outside of `nnx.vmap` because its state is always split. To solve this, you can pass an unsplit seed and use the `nnx.split_rngs` decorator before `nnx.vmap` to split the `RngState` right before each call to the function, and then "lower" it back so that it becomes usable.

```{code-cell} ipython3
weights = Weights(
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

y1 = noisy_vector_dot(weights, x)
y2 = noisy_vector_dot(weights, x)

print(jnp.allclose(y1, y2))
nnx.display(weights)
```

## Rules and limitations
In this section we will cover some rules and limitations apply when using Modules inside transformations.

### Mutable Module cannot be passed by closure

While Python allows for passing objects as closures to functions, this is generally not supported by Flax NNX transforms. The reason is that because Modules are mutable it is very easy to capture tracer into a Module created outside of the transform, this is silent error in JAX. To avoid this, Flax NNX checks that the Modules and Variables being mutated are passed as arguments to the transformed function.

For example, if we have a stateful Module such as `Counter` that increments a counter every time it is called, and we try to pass it as a closure to a function decorated with `nnx.jit`, we would be leaking the tracer. However Flax NNX will raise an error instead to prevent this:

```{code-cell} ipython3
class Counter(nnx.Module):
  def __init__(self):
    self.count = nnx.Param(jnp.array(0))

  def increment(self):
    self.count += jnp.array(1)

counter = Counter()

@nnx.jit
def f(x):
  counter.increment()
  return 2 * x

try:
  y = f(3)
except Exception as e:
  print(e)
```

To solve this issue pass all Module as arguments to the functions being transformed. In this case `f` should accept `counter` as an argument.

+++

### Consistent aliasing

The main issue with allowing for reference semantics in transforms is that references can be shared across inputs and outputs. This can be problematic if it is not taken care of because it would lead to ill-defined or inconsistent behavior. In the example below you have a single `Weights` Module `m` whose reference appears in multiple places in `arg1` and `arg2`. The problem here is that you also specify that you want to vectorize `arg1` in axis `0` and `arg2` in axis `1`. This would be fine in JAX because of referential transparency of pytrees. But this would be problematic in Flax NNX because you are trying to vectorize `m` in two different ways. Flax NNX will enforce consistency by raising an error.

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

Inconsistent aliasing can also happen between inputs and outputs. In the next example you have a trivial function that accepts and immediately returns `arg1`. However, `arg1` is vectorized on axis `0` on the input, and axis `1` on the output. As expected, this is problematic and Flax NNX will raise an error.

```{code-cell} ipython3
@nnx.vmap(in_axes=0, out_axes=1)
def f(arg1):
  return arg1

try:
  f(arg1)
except ValueError as e:
  print(e)
```

## Axis metadata

Flax NNX `Variable`s can hold arbitrary metadata, which can be added by simply passing it as keyword arguments to its constructor. This is often used to store `sharding` information, as used by the `nnx.spmd` APIs (like `nnx.get_partition_spec` and `nnx.get_named_sharding`).

However, it is often important to keep this axes-related information in sync to what the actual state of the axes is when transforms are involved. For example, if you vectorize a variable on axis `1`, you should remove the `sharding` information at position `1` when inside a `vmap` or `scan` to reflect the fact that the axes are temporarily removed.

To achieve this, Flax NNX transforms provide a non-standard `transform_metadata` dictionary argument. And when the `nnx.PARTITION_NAME` key is present, the `sharding` metadata will be updated as specified by `in_axes` and `out_axes`.

Let's see an example of this in action:

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

Here, you added a `sharding` metadata to the `nnx.Param` variables, and used `transform_metadata` to update the `sharding` metadata to reflect the axis changes. Specifically, you can see that the first axis `b` was removed from the `sharding` metadata when inside of `nnx.vmap`, and then added back when outside of `nnx.vmap`.

You can verify that this also works when `nnx.Module`s are created inside the transformation - the new `sharding` axes will be added to the `nnx.Module` `nnx.Variable`s outside the transformation, matching the axes of the transformed `nnx.Variable`s.

```{code-cell} ipython3
@nnx.vmap(out_axes=1, axis_size=4, transform_metadata={nnx.PARTITION_NAME: 'b'})
def init_vmap():
  return Weights(jnp.ones((3, 5)), sharding=('a', None))

m = init_vmap()
print(f'Outter {m.param.shape = }')
print(f'Outter {m.param.sharding = }')
```

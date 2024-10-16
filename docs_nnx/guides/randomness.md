---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Randomness

Random state handling in Flax NNX is radically simplified compared to systems like Haiku and Flax Linen because Flax NNX _defines the random state as an object state_. In essence, this means that in Flax NNX, the random state is 1) just another type of state; 2) stored in `nnx.Variable`s; and 3) held by the models themselves.

The Flax NNX [pseudorandom number generator (PRNG)](https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences) system has the following main characteristics:

- It is **explicit**.
- It is **order-based**.
- It uses **dynamic counters**.

This is a bit different from [Flax Linen's PRNG system](https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/rng_guide.html), which is `(path + order)`-based, and uses static counters.

Let’ start with some necessary imports:

```{code-cell} ipython3
from flax import nnx
import jax
from jax import random, numpy as jnp
```

## `Rngs`, `RngStream`, and `RngState`

In Flax NNX, the `nnx.Rngs` type is the primary convenience API for managing the random state(s). Following Flax Linen's footsteps, `nnx.Rngs` have the ability to create multiple named PRNG key [streams](https://jax.readthedocs.io/en/latest/jep/263-prng.html), each with its own state, for the purpose of having tight control over randomness in the context of [JAX transformations (transforms)](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations).

Here are the main PRNG-related types in Flax NNX:

* **`nnx.Rngs`**: The main user interface. It defines a set of named `nnx.RngStream` objects.
* **`nnx.RngStream`**: An object that can generate a stream of PRNG keys. It holds a root `key` and a `count` inside an `nnx.RngKey` and `nnx.RngCount` `nnx.Variable`s, respectively. When a new key is generated, the count is incremented.
* **`nnx.RngState`**: The base type for all RNG-related states.
  * **`nnx.RngKey`**: NNX Variable type for holding PRNG keys. It includes a `tag` attribute containing the name of the PRNG key stream.
  * **`nnx.RngCount`**: NNX Variable type for holding PRNG counts. It includes a `tag` attribute containing the PRNG key stream name.

To create an `nnx.Rngs` object you can simply pass an integer seed or `jax.random.key` instance to any keyword argument of your choice in the constructor.

> **Note:** To learn more about random number generation in JAX, the `jax.random` API, and PRNG-generated sequences, check out this [JAX PRNG tutorial](https://jax.readthedocs.io/en/latest/random-numbers.html).

Here's an example:

```{code-cell} ipython3
rngs = nnx.Rngs(params=0, dropout=random.key(1))
nnx.display(rngs)
```

Notice that the `key` and `count` `nnx.Variable`s contain the PRNG key stream name in a `tag` attribute. This is primarily used for filtering as we'll see later.

To generate new keys, you can access one of the streams and use its `__call__` method with no arguments. This will return a new key by using `random.fold_in` with the current `key` and `count`. The `count` is then incremented so that subsequent calls will return new keys.

```{code-cell} ipython3
params_key = rngs.params()
dropout_key = rngs.dropout()

nnx.display(rngs)
```

Note that the `key` attribute does not change when new PRNG keys are generated.

### Standard PRNG key stream names

There are only two standard PRNG key stream names used by Flax NNX's built-in layers, shown in the table below:

| PRNG key stream name | Description                                   |
|----------------------|-----------------------------------------------|
| `params`             | Used for parameter initialization             |
| `dropout`            | Used by `nnx.Dropout` to create dropout masks |

- `params` is used by most of the standard layers (such as `nnx.Linear`, `nnx.Conv`, `nnx.MultiHeadAttention`, and so on) during the construction to initialize their parameters.
- `dropout` is used by `nnx.Dropout` and `nnx.MultiHeadAttention` to generate dropout masks.

Below is a simple example of a model that uses `params` and `dropout` PRNG key streams:

```{code-cell} ipython3
class Model(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(20, 10, rngs=rngs)
    self.drop = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    return nnx.relu(self.drop(self.linear(x)))

model = Model(nnx.Rngs(params=0, dropout=1))

y = model(x=jnp.ones((1, 20)))
print(f'{y.shape = }')
```

### Default PRNG key stream

One of the downsides of having named streams is that the user needs to know all the possible names that a model will use when creating the `nnx.Rngs` object. While this could be solved with some documentation, Flax NNX provides a `default` stream that can be
be used as a fallback when a stream is not found. To use the default PRNG key stream, you can simply pass an integer seed or `jax.random.key` as the first positional argument.

```{code-cell} ipython3
rngs = nnx.Rngs(0, params=1)

key1 = rngs.params() # Call params.
key2 = rngs.dropout() # Fallback to the default stream.
key3 = rngs() # Call the default stream directly.

# Test with the `Model` that uses `params` and `dropout`.
model = Model(rngs)
y = model(jnp.ones((1, 20)))

nnx.display(rngs)
```

As shown above, a PRNG key from the `default` stream can also be generated by calling the `nnx.Rngs` object itself.

> **Note**
> <br> For large projects it is recommended to use named streams to avoid potential conflicts. For small projects or quick prototyping just using the `default` stream is a good choice.

+++

## Filtering random state

Random state can be manipulated using [Filters](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) just like any other type of state. It can be filtered using types (`nnx.RngState`, `nnx.RngKey`, `nnx.RngCount`) or using strings corresponding to the stream names (refer to [the Flax NNX `Filter` DSL](https://flax.readthedocs.io/en/latest/guides/filters_guide.html#the-filter-dsl)). Here's an example using `nnx.state` with various filters to select different substates of the `Rngs` inside a `Model`:

```{code-cell} ipython3
model = Model(nnx.Rngs(params=0, dropout=1))

rng_state = nnx.state(model, nnx.RngState) # All random states.
key_state = nnx.state(model, nnx.RngKey) # Only PRNG keys.
count_state = nnx.state(model, nnx.RngCount) # Only counts.
rng_params_state = nnx.state(model, 'params') # Only `params`.
rng_dropout_state = nnx.state(model, 'dropout') # Only `dropout`.
params_key_state = nnx.state(model, nnx.All('params', nnx.RngKey)) # `Params` PRNG keys.

nnx.display(params_key_state)
```

## Reseeding

In Haiku and Flax Linen, random states are explicitly passed to `Module.apply` each time before you call the model. This makes it easy to control the randomness of the model when needed (for example, for reproducibility).

In Flax NNX, there are two ways to approach this:

1. By passing an `nnx.Rngs` object through the `__call__` stack manually. Standard layers like `nnx.Dropout` and `nnx.MultiHeadAttention` accept the `rngs` argument if you want to have tight control over the random state.
2. By using `nnx.reseed` to set the random state of the model to a specific configuration. This option is less intrusive and can be used even if the model is not designed to enable manual control over the random state.

`nnx.reseed` is a function that accepts an arbitrary graph node (this includes [pytrees](https://jax.readthedocs.io/en/latest/working-with-pytrees.html#working-with-pytrees) of `nnx.Module`s) and some keyword arguments containing the new seed or key value for the `nnx.RngStream`s specified by the argument names. `nnx.reseed` will then traverse the graph and update the random state of the matching `nnx.RngStream`s, this includes both setting the `key` to a possibly new value and resetting the `count` to zero.

Here's an example of how to use `nnx.reseed` to reset the random state of the `nnx.Dropout` layer and verify that the computation is identical to the first time the model was called:

```{code-cell} ipython3
model = Model(nnx.Rngs(params=0, dropout=1))
x = jnp.ones((1, 20))

y1 = model(x)
y2 = model(x)

nnx.reseed(model, dropout=1) # reset dropout RngState
y3 = model(x)

assert not jnp.allclose(y1, y2) # different
assert jnp.allclose(y1, y3)     # same
```

## Splitting `nnx.Rngs`

When interacting with [Flax NNX transforms](https://flax.readthedocs.io/en/latest/guides/transforms.html) like `nnx.vmap` or `nnx.pmap`, it is often necessary to split the random state such that each replica has its own unique state. This can be done in two ways:

- By manually splitting a key before passing it to one of the `nnx.Rngs` streams; or
- By using the `nnx.split_rngs` decorator which will automatically split the random state of any `nnx.RngStream`s found in the inputs of the function, and automatically "lower" them once the function call ends.

It is more convenient to use `nnx.split_rngs`, since it works nicely with Flax NNX transforms, so here’s one example:

```{code-cell} ipython3
rngs = nnx.Rngs(params=0, dropout=1)

@nnx.split_rngs(splits=5, only='dropout')
def f(rngs: nnx.Rngs):
  print('Inside:')
  # rngs.dropout() # ValueError: fold_in accepts a single key...
  nnx.display(rngs)

f(rngs)

print('Outside:')
rngs.dropout() # works!
nnx.display(rngs)
```

> **Note:** `nnx.split_rngs` allows passing an NNX [`Filter`](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) to the `only` keyword argument in order to select the `nnx.RngStream`s that should be split when inside the function. In such a case, you only need to split the `dropout` PRNG key stream.

+++

## Transforms

As stated before, in Flax NNX the random state is just another type of state. This means that there is nothing special about it when it comes to Flax NNX transforms, which means that you should be able to use the Flax NNX state handling APIs of each transform to get the results you want.

In this section, you will go through two examples of using the random state in Flax NNX transforms - one with `nnx.pmap`, where you will learn how to split the PRNG state, and another one with `nnx.scan`, where you will freeze the PRNG state.

### Data parallel dropout

In the first example, you’ll explore how to use `nnx.pmap` to call the `nnx.Model` in a data parallel context.
- Since the `nnx.Model` uses `nnx.Dropout`, you’ll need to split the random state of the `dropout` to ensure that each replica gets different dropout masks.
- `nnx.StateAxes` is passed to `in_axes` to specify that the `model`'s `dropout` PRNG key stream will be parallelized across axis `0`, and the rest of its state will be replicated.
- `nnx.split_rngs` is used to split the keys of the `dropout` PRNG key streams into N unique keys, one for each replica.

```{code-cell} ipython3
model = Model(nnx.Rngs(params=0, dropout=1))

num_devices = jax.local_device_count()
x = jnp.ones((num_devices, 16, 20))
state_axes = nnx.StateAxes({'dropout': 0, ...: None})

@nnx.split_rngs(splits=num_devices, only='dropout')
@nnx.pmap(in_axes=(state_axes, 0), out_axes=0)
def forward(model: Model, x: jnp.ndarray):
  return model(x)

y = forward(model, x)
print(y.shape)
```

### Recurrent dropout

Next, let’s explore how to implement an `RNNCell` that uses a recurrent dropout. To do this:

- First, you will create an `nnx.Dropout` layer that will sample PRNG keys from a custom `recurrent_dropout` stream.
- You will apply dropout (`drop`) to the hidden state `h` of the `RNNCell`.
- Then, define an `initial_state` function to create the initial state of the `RNNCell`.
- Finally, instantiate `RNNCell`.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class RNNCell(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.linear = nnx.Linear(dout + din, dout, rngs=rngs)
    self.drop = nnx.Dropout(0.1, rngs=rngs, rng_collection='recurrent_dropout')
    self.dout = dout
    self.count = Count(jnp.array(0, jnp.uint32))

  def __call__(self, h, x) -> tuple[jax.Array, jax.Array]:
    h = self.drop(h) # Recurrent dropout.
    y = nnx.relu(self.linear(jnp.concatenate([h, x], axis=-1)))
    self.count += 1
    return y, y

  def initial_state(self, batch_size: int):
    return jnp.zeros((batch_size, self.dout))

cell = RNNCell(8, 16, nnx.Rngs(params=0, recurrent_dropout=1))
```

Next, you will use `nnx.scan` over an `unroll` function to implement the `rnn_forward` operation.
- The key ingredient of recurrent dropout is to apply the same dropout mask across all time steps. Therefore, to achieve this you will pass `nnx.StateAxes` to `nnx.scan`'s `in_axes`, specifying that the `cell`'s `recurrent_dropout` PRNG stream will be broadcast, and the rest of the `RNNCell`'s state will be carried over.
- Also, the hidden state `h` will be the `nnx.scan`'s `Carry` variable, and the sequence `x` will be `scan`ned over its axis `1`.

```{code-cell} ipython3
@nnx.jit
def rnn_forward(cell: RNNCell, x: jax.Array):
  h = cell.initial_state(batch_size=x.shape[0])

  # Broadcast the 'recurrent_dropout' PRNG state to have the same mask on every step.
  state_axes = nnx.StateAxes({'recurrent_dropout': None, ...: nnx.Carry})
  @nnx.scan(in_axes=(state_axes, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
  def unroll(cell: RNNCell, h, x) -> tuple[jax.Array, jax.Array]:
    h, y = cell(h, x)
    return h, y

  h, y = unroll(cell, h, x)
  return y

x = jnp.ones((4, 20, 8))
y = rnn_forward(cell, x)

print(f'{y.shape = }')
print(f'{cell.count.value = }')
```

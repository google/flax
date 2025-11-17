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

Flax NNX uses the stateful `nnx.Rngs` class to simplify Jax's handling of random states. For example, the code below uses a `nnx.Rngs` object to define a simple linear model with dropout:

```{code-cell} ipython3
from flax import nnx

class Model(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.linear = nnx.Linear(20, 10, rngs=rngs)
    self.drop = nnx.Dropout(0.1)

  def __call__(self, x, *, rngs):
    return nnx.relu(self.drop(self.linear(x), rngs=rngs))

rngs = nnx.Rngs(0)
model = Model(rngs=rngs)  # pass rngs to initialize parameters
x = rngs.normal((32, 20))  # convenient jax.random methods
y = model(x, rngs=rngs)  # pass rngs for dropout masks
```

We always pass `nnx.Rngs` objects to models at initialization (to initialize parameters). For models with nondeterministic outputs like the one above, we also pass `nnx.Rngs` objects to the model's `__call__` method.

The Flax NNX [pseudorandom number generator (PRNG)](https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences) system has the following main characteristics:

- It is **explicit**.
- It is **order-based**.
- It uses **dynamic counters**.

> **Note:** To learn more about random number generation in JAX, the `jax.random` API, and PRNG-generated sequences, check out this [JAX PRNG tutorial](https://jax.readthedocs.io/en/latest/random-numbers.html).

## `Rngs`, `RngStream`, and `RngState`

In Flax NNX, the `nnx.Rngs` type is the primary convenience API for managing the random state(s). Following Flax Linen's footsteps, `nnx.Rngs` have the ability to create multiple named PRNG key [streams](https://jax.readthedocs.io/en/latest/jep/263-prng.html), each with its own state, for the purpose of having tight control over randomness in the context of [JAX transformations (transforms)](https://jax.readthedocs.io/en/latest/key-concepts.html#transformations).

Here are the main PRNG-related types in Flax NNX:

* **`nnx.Rngs`**: The main user interface. It defines a set of named `nnx.RngStream` objects.
* **`nnx.RngStream`**: An object that can generate a stream of PRNG keys. It holds a root `key` and a `count` inside an `nnx.RngKey` and `nnx.RngCount` `nnx.Variable`s, respectively. When a new key is generated, the count is incremented.
* **`nnx.RngState`**: The base type for all RNG-related states.
  * **`nnx.RngKey`**: NNX Variable type for holding PRNG keys. It includes a `tag` attribute containing the name of the PRNG key stream.
  * **`nnx.RngCount`**: NNX Variable type for holding PRNG counts. It includes a `tag` attribute containing the PRNG key stream name.

To create an `nnx.Rngs` object you can simply pass an integer seed or `jax.random.key` instance to any keyword argument of your choice in the constructor.

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

+++

### Using random state with flax Modules.

Almost all flax Modules require a random state for initialization. In a `Linear` layer, for example, we need to sample the weights and biases from the appropriate Normal distribution. Random state is provided using the `rngs` keyword argument at initialization.

```{code-cell} ipython3
linear = nnx.Linear(20, 10, rngs=rngs)
```

Specifically, this will use the RngSteam `rngs.params` for weight initialization. The `params` stream is also used for initialization of `nnx.Conv`, `nnx.ConvTranspose`, and `nnx.Embed`.

+++

The `nnx.Dropout` module also requires a random state, but it requires this state at *call* time rather than initialization. Once again, we can pass it random state using the `rngs` keyword argument.

```{code-cell} ipython3
dropout = nnx.Dropout(0.5)
```

```{code-cell} ipython3
import jax.numpy as jnp
dropout(jnp.ones(4), rngs=rngs)
```

The `nnx.Dropout` layer will use the rng's `dropout` stream. This also applies to Modules that use `Dropout` as a sub-Module, like `nnx.MultiHeadAttention`.

+++

To summarize, there are only two standard PRNG key stream names used by Flax NNX's built-in layers, shown in the table below:

| PRNG key stream name | Description                                   |
|----------------------|-----------------------------------------------|
| `params`             | Used for parameter initialization             |
| `dropout`            | Used by `nnx.Dropout` to create dropout masks |

+++

### Default PRNG key stream

One of the downsides of having named streams is that the user needs to know all the possible names that a model will use when creating the `nnx.Rngs` object. While this could be solved with some documentation, Flax NNX provides a `default` stream that can be
be used as a fallback when a stream is not found. To use the default PRNG key stream, you can simply pass an integer seed or `jax.random.key` as the first positional argument.

```{code-cell} ipython3
rngs = nnx.Rngs(0, params=1)

key1 = rngs.params() # Call params.
key2 = rngs.dropout() # Fallback to the default stream.
key3 = rngs() # Call the default stream directly.

nnx.display(rngs)
```

As shown above, a PRNG key from the `default` stream can also be generated by calling the `nnx.Rngs` object itself.

> **Note**
> <br> For large projects it is recommended to use named streams to avoid potential conflicts. For small projects or quick prototyping just using the `default` stream is a good choice.

+++

### jax.random shorthand methods
Since a very common pattern is to sample a key and immediately pass it to a function from `jax.random`, both `Rngs` and `RngStream` expose the same functions as methods with the same signature except they don't require a key:

```{code-cell} ipython3
import jax
rngs = nnx.Rngs(0, params=1)

# using jax.random
z1 = jax.random.normal(rngs(), (2, 3))
z2 = jax.random.bernoulli(rngs.params(), 0.5, (10,))

# shorthand methods
z1 = rngs.normal((2, 3))                 # generates key from rngs.default
z2 = rngs.params.bernoulli(0.5, (10,)) # generates key from rngs.params
```

## Forking random state

+++

Say you want to train a model that uses dropout on a batch of data. You don't want to use the same random state for every dropout mask in your batch.  Instead, you want to fork the random state into separate pieces for each layer. This can be accomplished with the `fork` method, as shown below.

```{code-cell} ipython3
class Model(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(20, 10, rngs=rngs)
    self.drop = nnx.Dropout(0.1)

  def __call__(self, x, rngs):
    return nnx.relu(self.drop(self.linear(x), rngs=rngs))
```

```{code-cell} ipython3
model =  Model(rngs=nnx.Rngs(0))
```

```{code-cell} ipython3
@nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
def model_forward(model, x, rngs):
  return model(x, rngs=rngs)
```

```{code-cell} ipython3
dropout_rngs = nnx.Rngs(1)
forked_rngs = dropout_rngs.fork(split=5)
(dropout_rngs, forked_rngs)
```

```{code-cell} ipython3
model_forward(model, jnp.ones((5, 20)), forked_rngs).shape
```

The output of `rng.fork` is another `Rng` with keys and counts that have an expanded shape. In the example above, the `RngKey` and `RngCount` of `dropout_rngs` have shape `()`, but in `forked_rngs` they have shape `(5,)`.

+++

# Implicit Random State

+++

So far, we have looked at passing random state directly to each Module when it gets called. But there's another way to handle call-time randomness in flax: we can bundle the random state into the Module itself. This makes the random state is just another type of Module state. Using implicit random state requires passing the `rngs` keyward argument when initializing the module rather than when calling it. For example, here is how we might construct the simple `Module` we defined earlier using an implicit style.

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

This implicit state handling style is less verbose than passing RNGs explicitly, and more closely resembles code in other deep learning frameworks like PyTorch. However, as we'll see in the following sections, using implicit state makes it less obvious how to apply jax transformations to your Modules. With explicit state, you can usually use tranforms like `jax.vmap` directly. With implicit state, you'll need to some extra tricks with `nnx.vmap` to make everything work. Because of this additional complexity, we recommend that new flax projects stick to the explicit style.

+++

## Filtering random state

Implicit random state can be manipulated using [Filters](https://flax.readthedocs.io/en/latest/guides/filters_guide.html) just like any other type of state. It can be filtered using types (`nnx.RngState`, `nnx.RngKey`, `nnx.RngCount`) or using strings corresponding to the stream names (refer to [the Flax NNX `Filter` DSL](https://flax.readthedocs.io/en/latest/guides/filters_guide.html#the-filter-dsl)). Here's an example using `nnx.state` with various filters to select different substates of the `Rngs` inside a `Model`:

```{code-cell} ipython3
model = Model(nnx.Rngs(params=0, dropout=1))

rng_state = nnx.state(model, nnx.RngState) # All random states.
key_state = nnx.state(model, nnx.RngKey) # Only PRNG keys.
count_state = nnx.state(model, nnx.RngCount) # Only counts.
rng_dropout_state = nnx.state(model, 'dropout') # Only `dropout`.

nnx.display(rng_dropout_state)
```

## Reseeding

In Haiku and Flax Linen, random states are explicitly passed to `Module.apply` each time before you call the model. This makes it easy to control the randomness of the model when needed (for example, for reproducibility).

In Flax NNX, there are two ways to approach this:

1. By passing an `nnx.Rngs` object through the `__call__` stack manually, as shown previously.
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

## Forking implicit random state

We saw above how to use `rng.fork` when passing explicit random state through [Flax NNX transforms](https://flax.readthedocs.io/en/latest/guides/transforms.html) like `nnx.vmap` or `nnx.pmap`. The decorator `nnx.fork_rngs` allows this for implicit random state. Consider the example below, which generates a batch of samples from the nondeterministic model we defined above.

```{code-cell} ipython3
rng_axes = nnx.StateAxes({'dropout': 0, ...: None})

@nnx.fork_rngs(split={'dropout': 5})
@nnx.vmap(in_axes=(rng_axes, None), out_axes=0)
def sample_from_model(model, x):
    return model(x)

print(sample_from_model(model, x).shape)
```

Here `sample_from_model` is modified by two decorators:
- The function we get from the `nnx.vmap` decorator expects that the random state of the `model` argument has already been split into 5 pieces. It runs the model once for each random key.
- The function we get from the `nnx.fork_rngs` decorator splits the random state of its `model` argument into five pieces before passing it on to the inner function.

+++

## Transforming implicit state

+++

In the previous section, we showed how to use `nnx.vmap` with a module that contained implicit random state. But we can use other `nnx` transformations too! Remember: implicit random state isn't different from any other type of Model state, and this applies to Flax NNX transforms too. This means you can use the Flax NNX state handling APIs of each transform to get the results you want. For a more involved example, let’s explore how to implement recurrent dropout on an `RNNCell` using `nnx.scan`.

We'll start by constructing the `RNNCell` class:

- First, create an `nnx.Dropout` layer that will sample PRNG keys from a custom `recurrent_dropout` stream.
- Apply dropout (`drop`) to the hidden state `h` of the `RNNCell`.
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
    self.count.value += 1
    return y, y

  def initial_state(self, batch_size: int):
    return jnp.zeros((batch_size, self.dout))

cell = RNNCell(8, 16, nnx.Rngs(params=0, recurrent_dropout=1))
```

Next, use `nnx.scan` over an `unroll` function to implement the `rnn_forward` operation:
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

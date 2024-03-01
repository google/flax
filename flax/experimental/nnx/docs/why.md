---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Why NNX?

<!-- open in colab button -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/flax/experimental/nnx/docs/why.ipynb)

Four years ago we developed the Flax "Linen" API to support modeling research on JAX, with a focus on scaling scaling and performance.  We've learned a lot from our users over these years.

We introduced some ideas that have proven to be good:
 - Organizing variables into [collections](https://flax.readthedocs.io/en/latest/glossary.html#term-Variable-collections) or types to support JAX transforms and segregation of different data types in training loops.
 - Automatic and efficient [PRNG management](https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences) (with support for splitting/broadcast control across map transforms)
 - [Variable Metadata](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.with_partitioning.html#flax.linen.with_partitioning) for SPMD annotations, optimizer metadata, and other uses.

However, one choice we made was to use functional "define by call" semantics for NN programming via the lazy initialization of parameters.  This made for concise (`compact`) implementation code, allowed for a single specification when transforming a layer, and aligned our API with  Haiku.  Lazy initialization meant that the semantics of modules and variables in Flax were non-pythonic and often surprising.  It also led to implementation complexity and obscured the core ideas of transformations on neural nets.

NNX is an attempt to keep the features that made Linen useful while introducing some new principles:

- Regular Python semantics for Modules, including (within JIT boundaries) support for mutability and shared references.
- A simple API to interact directly with the JAX, this includes the ability to easily implement custom lifted Modules and other purely functional tricks.

We'd love to hear from any of our users about their thoughts on these ideas.

[[nnx on github](https://github.com/google/flax/tree/main/flax/experimental/nnx)]
[[this doc on github](https://github.com/google/flax/blob/main/flax/experimental/nnx/docs/why.ipynb)]

```{code-cell}
! pip install -U git+https://github.com/google/flax.git
from functools import partial
import jax
from jax import random, numpy as jnp
from flax.experimental import nnx
```

### NNX is Pythonic
The main feature of NNX Module is that it adheres to Python semantics. This means that:

* fields are mutable so you can perform inplace updates
* Module references can be shared between multiple Modules
* Module construction implies parameter initialization
* Module methods can be called directly

```{code-cell}
:outputId: d8ef66d5-6866-4d5c-94c2-d22512bfe718

class Count(nnx.Variable):   # custom Variable types define the "collections"
  pass


class CounterLinear(nnx.Module):
  def __init__(self, din, dout, *, rngs): # explicit RNG threading
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.count = Count(jnp.zeros((), jnp.int32)) # typed Variable collections

  def __call__(self, x):
    self.count.value += 1  # in-place stateful updates
    return self.linear(x)


model = CounterLinear(4, 4, rngs=nnx.Rngs(0))  # no special `init` method
y = model(jnp.ones((2, 4)))  # call methods directly

print(f'{model = }')
```

Because NNX Modules contain their own state, they are very easily to inspect:

```{code-cell}
:outputId: 10a46b0f-2993-4677-c26d-36a4ddf33449

print(f'{model.count = }')
print(f'{model.linear.kernel = }')
```

#### Intuitive Surgery

In NNX surgery can be done at the Module level by simply updating / replacing existing fields.

```{code-cell}
:outputId: e6f86be8-3537-4c48-f471-316ee0fb6c45

# pretend this came from a checkpoint or elsewhere:
pretrained_weight = random.uniform(random.key(0), (4, 4))

# you can replace weights directly
model.linear.kernel = pretrained_weight
y = model(jnp.ones((2, 4)))
y
```

```{code-cell}
:outputId: 5190ac7b-12f7-4400-d5bb-f91b97a557b6

def load_pretrained_fragment():
  # pretend this inits / loads some fragment of a model
  replacement = nnx.Linear(4, 4, rngs=nnx.Rngs(1))
  return replacement

# you can replace modules directly
model.linear = load_pretrained_fragment()
y = model(jnp.ones((2, 4)))
y
```

Not only is this easier than messing with dictionary structures and aligning that with code changes, but one can even replace a field with a completely different Module type, or even change the architecture (e.g. share two Modules that were not shared before).

```{code-cell}
rngs = nnx.Rngs(0)
model = nnx.Sequence(
  [
    nnx.Conv(1, 16, [3, 3], padding='SAME', rngs=rngs),
    partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),
    nnx.Conv(16, 32, [3, 3], padding='SAME', rngs=rngs),
    partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),
    lambda x: x.reshape((x.shape[0], -1)),  # flatten
    nnx.Linear(32 * 7 * 7, 10, rngs=rngs),
  ]
)

y = model(jnp.ones((2, 28, 28, 1)))

# Do some weird surgery of the stack:
for i, layer in enumerate(model):
  if isinstance(layer, nnx.Conv):
    model[i] = nnx.Linear(layer.in_features, layer.out_features, rngs=rngs)

y = model(jnp.ones((2, 28, 28, 1)))
```

Note that here we are replacing `Conv` with `Linear` as a silly example, but in reality you would do things like replacing a layer with its quantized version, or changing a layer with an optimized version, etc.

+++

### Interacting with JAX is easy

While NNX Modules inherently follow reference semantics, they can be easily converted into a pure functional representation that can be used with JAX transformations and other value-based, functional code.

NNX has two very simple APIs to interact with JAX: `split` and `merge`.

The `Module.split` method allows you to convert into a `State` dict-like object that contains the dynamic state of the Module, and a `GraphDef` object that contains the static structure of the Module.

```{code-cell}
:outputId: 9a3f378b-739e-4f45-9968-574651200ede

model = CounterLinear(4, 4, rngs=nnx.Rngs(0))

state, static = model.split()

# state is a dictionary-like JAX pytree
print(f'{state = }')

# static is also a JAX pytree, but containing no data, just metadata
print(f'\n{static = }')
```

The `GraphDef.merge` method allows you to take a `GraphDef` and one or more `State` objects and merge them back into a `Module` object.

Using `split` and `merge` in conjunction allows you to carry your Module in and out of any JAX transformation. Here is a simple jitted `forward` function as an example:

```{code-cell}
:outputId: 0007d357-152a-449e-bcb9-b1b5a91d2d8d

@jax.jit
def forward(static: nnx.GraphDef, state: nnx.State, x: jax.Array):
  model = static.merge(state)
  y = model(x)
  state, _ = model.split()
  return y, state

x = jnp.ones((2, 4))
y, state = forward(static,state, x)

print(f'{y.shape = }')
print(f'{state["count"] = }')
```

#### Custom lifting and transformation

By using the same mechanism inside Module methods you can implement lifted Modules, that is, Modules that use a JAX transformation to have a distinct behavior.

One of Linen's current pain points is that it is not easy to interact with JAX transformations that are not currently supported by the framework. NNX makes it very easy to implement custom lifted Modules or bespoke custom functional transforms for specific use cases.

As an example here we will create a `LinearEnsemble` Module that uses `jax.vmap` both during `__init__` and `__call__` to vectorize the computation over multiple `CounterLinear` models (defined above). The example is a little bit longer, but notice how each method conceptually very simple.

It uses the single additional method `update` to locally modify model state.

```{code-cell}
:outputId: fdd212d7-4994-4fa5-d922-5a7d7cfad3e3

class LinearEnsemble(nnx.Module):
  def __init__(self, din, dout, *, num_models, rngs: nnx.Rngs):
    # get raw rng seeds
    keys = rngs.fork(num_models) # split all keys into `num_models`

    # define pure init fn and vmap
    def vmap_init(keys):
      return CounterLinear(din, dout, rngs=nnx.Rngs(keys)).split(
        nnx.Param, Count
      )
    params, counts, static = jax.vmap(
      vmap_init, in_axes=(0,), out_axes=(0, None, None)
    )(keys)

    # update wrapped submodule reference
    self.models = static.merge(params, counts)

  def __call__(self, x):
    # get module values, define pure fn,
    # notice that we split the data into two collections by their types.
    params, counts, static = self.models.split(nnx.Param, Count)

    # define pure init fn and vmap
    def vmap_apply(x, params, counts, static):
      model = static.merge(params, counts)
      y = model(x)
      params, counts, static = model.split(nnx.Param, Count)
      return y, params, counts, static

    y, params, counts, static = jax.vmap(
        vmap_apply,
        in_axes=(None, 0, None, None),
        out_axes=(0, 0, None, None)
    )(x, params, counts, static)

    # update wrapped module
    # uses `update` to integrate the new state
    self.models.update(params, counts, static)
    return y

x = jnp.ones((4,))
ensemble = LinearEnsemble(4, 4, num_models=8, rngs=nnx.Rngs(0))

# forward pass
y = ensemble(x)

print(f'{y.shape = }')
print(f'{ensemble.models.count = }')
print(f'state = {jax.tree_map(jnp.shape, ensemble.get_state())}')
```

#### Convenience lifted transforms

+++

Like linen, for convenience we still provide simple lifted transforms for standard JAX transforms, usable as class transforms and decorators.  We've endeavored to simplify the API for scan and vmap compared to the flax specifications.

```{code-cell}
:outputId: c4800a49-efd1-4ee5-e703-6e63e18da4cb

# class transform:
ScannedLinear = nnx.Scan(nnx.Linear, variable_axes={nnx.Param: 0}, length=4)

scanned = ScannedLinear(2, 2, rngs=nnx.Rngs(0))
scanned.get_state()
```

```{code-cell}
:outputId: 9efd6e71-d180-4674-ade0-2b02057a400b

# method decorators:

class ScannedLinear(nnx.Module):

  @partial(nnx.scan, variable_axes={nnx.Param: 0}, length=4)
  def __init__(self, din, dout, *, rngs: nnx.Rngs):
    self.model = nnx.Linear(din, dout, rngs=nnx.Rngs(rngs))

  @partial(nnx.scan, variable_axes={nnx.Param: 0}, length=4)
  def __call__(self, x):
    return self.model(x)

scanned = ScannedLinear(2, 2, rngs=nnx.Rngs(0))
scanned.get_state()
```

#### Aside: Why aren't Modules Pytrees?

A common questions is why aren't NNX Modules registered as Pytrees? (in the style of Equinox, Treex, PytreeClass, etc.)  It _is_ trivial to define a pytree registration in terms of `split`/`merge`.

The problem is that Pytrees impose value semantics (referencial transparency) while Modules assume reference semantics, and therefore it is dangerous in general to automatically treat Modules as Pytrees.

As an example, lets take a look at what would happen if we allowed this very simple program to be valid:

```{code-cell}
@jax.jit
def f(m1: nnx.Module, m2: nnx.Module):
  return m1, m2
```

Here we are just creating a jitted function `f` that takes in two Modules `(m1, m2)` and returns them as is. What could go wrong?

There are two main problems with this:
* Shared references are not maintained, that is, if `m1.shared` is the same as `m2.shared` outside `f`, this will NOT be true both inside `f`, and at the output of `f`.
* Even if you accept this fact and added code to compensate for this, `f` would now behave differently depending on whether its being `jit`ted or not, this is an undesirable asymmetry and `jit` would no longer be a no-op.

+++

### Standardized "Hooks"

NNX introduces a standard getter/setter/creator interface for custom variables (similar to Haiku hooks).  This is used internally to support SPMD metadata for managing sharding information, but is available for user-defined applications.

```{code-cell}
:outputId: c4e6586a-bfe0-4f26-d05b-8c9e395971b2

class TransposedParam(nnx.Variable):
  def create_value(self, value):
    return value.T  # called on variable creation to transform initial value
  def get_value(self):
    return self.value.T  # called when value fetched via module getattr
  def set_value(self, value):
    return self.replace(value=value.T)  # called when setting value from module setattr


class OddLinear(nnx.Module):
  def __init__(self, din, dout, *, rngs):
    self.kernel = TransposedParam(random.uniform(rngs.params(), (din, dout)))
    self.bias = nnx.Param(jnp.zeros((dout,)))

  def __call__(self, x):
    print(f'{self.kernel.shape = }')
    return x @ self.kernel + self.bias


model = OddLinear(4, 8, rngs=nnx.Rngs(0))
y = model(jnp.ones((2, 4)))

print(f'outer kernel shape = {model.split()[0]["kernel"].shape}')
```

SPMD metadata is handled using `nnx.with_partitioning` helpers, but it's easy to add one's own metadata schema:

```{code-cell}
:outputId: ef312738-0f56-4c0e-9aaf-3319d131f1a2

class MetadataParam(nnx.Param):
  def __init__(self, *args, **kwargs):
    for key in kwargs:
      setattr(self, key, kwargs[key])
    super().__init__(*args)


class AnnotatedLinear(nnx.Module):
  def __init__(self, din, dout, *, rngs):
    self.kernel = TransposedParam(random.uniform(rngs.params(), (din, dout)), meta='foo', other_meta=0)
    self.bias = TransposedParam(jnp.zeros((dout,)), meta='bar', other_meta=1)

  def __call__(self, x):
    return x @ self.kernel + self.bias


model = AnnotatedLinear(4, 8, rngs=nnx.Rngs(0))
y = model(jnp.ones((2, 4)))

state, static = model.split()

print(f"{state.variables['kernel'].meta=}\n{state.variables['kernel'].other_meta=}")
print(f"{state.variables['bias'].meta=}\n{state.variables['bias'].other_meta=}")
```

## Shape Inference

Shape inference is still possible in NNX using abstract evaluation when it's really needed, it just isn't automatic.

```{code-cell}
:outputId: 942a3788-bcbf-426d-87e6-c5a041172c64

def batched_flatten(x):
  return jnp.reshape(x, (x.shape[0], -1))

class Example(nnx.Module):
  def __init__(self, *,
               in_filters=3,
               out_filters=4,
               input_shape=None,  # provide an example input size
               rngs):
      self.encoder = nnx.Conv(in_filters, out_filters,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="SAME",
                              rngs=rngs)
      # calculate the flattened shape post-conv using jax.eval_shape
      encoded_shape = jax.eval_shape(
          lambda x: batched_flatten(self.encoder(x)),
          jax.ShapeDtypeStruct(input_shape, jnp.float32)
      ).shape
      # use this shape information to continue initializing
      self.linear = nnx.Linear(encoded_shape[-1], 4, rngs=rngs)

  def __call__(self, x):
    x = self.encoder(x)
    x = batched_flatten(x)
    return self.linear(x)

model = Example(in_filters=3,
                out_filters=4,
                input_shape=(2, 6, 6, 3),
                rngs=nnx.Rngs(0))

state, static = model.split()
jax.tree_map(jnp.shape, state)
```

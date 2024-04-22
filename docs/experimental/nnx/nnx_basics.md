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

NNX is a **N**eural **N**etworks JA**X** library that embraces Python’s object-oriented 
programming model to provide an intuitive and highly simplified user experience. It
represents objects as PyGraphs (instead of PyTrees), which allows NNX to handle reference
sharing and mutability, making model code be regular Python code that users from frameworks
like Pytorch will be familiar with.be familiar with.

NNX is also designed to support 
all the patterns that allowed Linen to scale to large code bases while having a much simpler
implementation.

```{code-cell} ipython3
from flax.experimental import nnx
import jax
import jax.numpy as jnp
```

## The Module System
To begin lets see how to create a `Linear` Module using NNX. The main noticeable
difference between NNX and Module systems like Haiku or Linen is that in NNX everything is
**explicit**. This means among other things that 1) the Module itself holds the state
(e.g. parameters) directly, 2) the RNG state is threaded by the user, and 3) all shape information
must be provided on initialization (no shape inference).

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w.value + self.b.value
```

As shown above dynamic state is usually stored in `nnx.Param`s,
and static state (all types not handled by NNX) such as integers or strings 
are stored directly. Attributes of type `jax.Array` and `numpy.ndarray` are also treated as dynamic state,
although storing them inside `nnx.Variable`s is preferred. Also, the `nnx.Rngs` object by can be used to
get new unique keys based on a root key passed to the constructor (see below).

To actually initialize a Module is very easy: simply call the constructor. All the
parameters of a Module will be created right then and there, and are immediately available
for inspection using regular Python attribute access.

```{code-cell} ipython3
model = Linear(din=2, dout=3, rngs=nnx.Rngs(params=0))

print(f'{model = }')
print(f'{model.w.value = }')
print(f'{model.b.value = }')
```

This is very handy for debugging as it allows accessing the entire structure or
modifying it. Similarly, computations can be ran directly.

```{code-cell} ipython3
x = jnp.ones((1, 2))

model(x)
```

Since Modules hold their own state there is no need for a separate `apply` method, as in
Linen or Haiku.

+++

### Stateful Computation

When implementing layers like Batch Normalization or Multi Head Attention with 
autoregressive decoding you often need to store and update state inside a Module 
during the forward pass. The way to do this in NNX is simply to store the state 
inside a `Variable` and update it in-place when need it.

```{code-cell} ipython3
class Counter(nnx.Module):
  def __init__(self):
    self.count = nnx.Variable(0)

  def __call__(self):
    self.count.value += 1

counter = Counter()
print(f'{counter.count.value = }')
counter()
print(f'{counter.count.value = }')
```

JAX frameworks have avoided mutable references until now. The key innovations which 
allows their usage in NNX is that 1) there is a clear boundary between code that uses 
reference semantics and code that uses value semantics, defined by 
[The Functional API](#the-functional-api), and 2) there are guards in place to avoid 
updating NNX objects from a `MainTrace`, thus preventing tracer leakage.

+++

### Nested Modules

As expected, Modules can be used to compose other Modules in a nested
structure, including standard Modules such as `nnx.Linear`,
`nnx.Conv`, etc., or any custom Module created by users. Modules can
be assigned as attributes of a Module, but as shown by `MLP.blocks` in the
example below, they can also be stored in attributes of type `list`, `dict`, `tuple`, 
or in nested structures of the same.

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, dim: int, *, rngs: nnx.Rngs):
    self.linear = nnx.Linear(dim, dim, rngs=rngs)
    self.bn = nnx.BatchNorm(dim, use_running_average=True, rngs=rngs)

  def __call__(self, x: jax.Array):
    return nnx.relu(self.bn(self.linear(x)))
  
class MLP(nnx.Module):
  def __init__(self, num_layers: int, dim: int, *, rngs: nnx.Rngs):
    self.blocks = [Block(dim, rngs=rngs) for _ in range(num_layers)]
  
  def __call__(self, x: jax.Array):
    for block in self.blocks:
      x = block(x)
    return x
  
model = MLP(num_layers=5, dim=2, rngs=nnx.Rngs(0))
print(f'{model = }'[:500] + '...')
```

One of the benefits of NNX is that nested Modules as easy to inspect and
static analyzers, e.g., code completion, can help you while doing so.

```{code-cell} ipython3
print(f'{model.blocks[1].linear.kernel.value = }')
print(f'{model.blocks[0].bn.scale.value = }')
```

#### Model Surgery
NNX Modules are mutable by default, this means their structure can be changed
at any time. Also, NNX's Module system supports reference sharing of Modules and
Variables.

This makes Model Surgery quite easy as any submodule could be replaced by
e.g., a pretrained Module, a shared Module, or even just a Module/function that
uses the same signature. More over, Variables can also be modified or shared.

```{code-cell} ipython3
# Module replacement
pretrained = Block(dim=2, rngs=nnx.Rngs(42)) # imagine this is pretrained
model.blocks[0] = pretrained
# adhoc Module sharing
model.blocks[3] = model.blocks[1]
# monkey patching
def awesome_layer(x): return x
model.blocks[2] = awesome_layer

# Variable sharing (weight tying)
model.blocks[-1].linear.kernel = model.blocks[0].linear.kernel

model(jnp.ones((1, 2)))
```

## NNX Transforms

+++

## The Functional API

The Functional API establishes a clear boundary between reference/object semantics and
value/pytree semantics. It also allows same amount of fine-grained control over the 
state that Linen/Haiku users are used to. The Functional API consists of 3 basic methods:
`split`, `merge`, and `update`.

The `StatefulLinear` Module shown below will serve as an example for the use of the
Functional API. It contains some `nnx.Param` Variables and a custom `Count` Variable
type which is used to keep track of integer scalar state that increases on every 
forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class StatefulLinear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = Count(0)

  def __call__(self, x: jax.Array):
    self.count.value += 1
    return x @ self.w.value + self.b.value
  
model = StatefulLinear(din=2, dout=3, rngs=nnx.Rngs(0))
```

### State and GraphDef

A Module can be decomposed into `GraphDef` and `State` using the
`split` function. State is a Mapping from strings to Variables or nested 
States. GraphDef contains all the static information needed to reconstruct 
a Module graph, it is analogous to JAX's `PyTreeDef`.

```{code-cell} ipython3
graphdef, state = nnx.split(model)

print(f'{state = }\n')
print(f'{graphdef = }'[:200] + '...')
```

```{code-cell} ipython3
graphdef, state = nnx.split((model, model))

state
```

### Split, Merge, and Update

`merge` is the reverse of `split`, it takes the GraphDef + State and reconstructs
the Module. As shown in the example below, by using `split` and `merge` in sequence
any Module can be lifted to be used in any JAX transform. `update` can
update an object inplace with the content of a given State. This pattern is used to 
propagate the state from a transform back to the source object outside.

```{code-cell} ipython3
print(f'{model.count.value = }')

# 1. Use split to create a pytree representation of the Module
graphdef, state = nnx.split(model)

@jax.jit
def forward(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> tuple[jax.Array, nnx.State]:
  # 2. Use merge to create a new model inside the JAX transformation
  model = graphdef.merge(state)
  # 3. Call the Module
  y = model(x)
  # 4. Use split to propagate State updates
  _, state = nnx.split(model)
  return y, state

y, state = forward(graphdef, state, x=jnp.ones((1, 2)))
# 5. Update the state of the original Module
nnx.update(model, state)

print(f'{model.count.value = }')
```

The key insight of this pattern is that using mutable references is 
fine within a transform context (including the base eager interpreter)
but its necessary to use the Functional API when crossing boundaries.

**Why aren't Module's just Pytrees?** The main reason is that it is very
easy to lose track of shared references by accident this way, for example
if you pass two Module that have a shared Module through a JAX boundary
you will silently lose that sharing. The Functional API makes this
behavior explicit, and thus it is much easier to reason about.

+++

### Fine-grained State Control

Seasoned Linen and Haiku users might recognize that having all the state in
a single structure is not always the best choice as there are cases in which
you might want to handle different subsets of the state differently. This a
common occurrence when interacting with JAX transforms, for example, not all
the model's state can or should be differentiated when interacting which `grad`,
or sometimes there is a need to specify what part of the model's state is a
carry and what part is not when using `scan`.

To solve this, `split` allows you to pass one or more `Filter`s to partition
the Variables into mutually exclusive States. The most common Filter being
types as shown below.

```{code-cell} ipython3
# use Variable type filters to split into multiple States
graphdef, params, counts = nnx.split(model, nnx.Param, Count)

print(f'{params = }\n')
print(f'{counts = }')
```

**Note**: filters must be exhaustive, if a Variable is not matched an error will be raised.

As expected the `merge` and `update` methods naturally consume multiple States:

```{code-cell} ipython3
# merge multiple States
model = graphdef.merge(params, counts)
# update with multiple States
nnx.update(model, params, counts)
```

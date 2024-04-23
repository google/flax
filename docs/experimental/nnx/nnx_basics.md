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

NNX is a **N**eural **N**etwork library for JA**X** that focuses on providing the best 
development experience, so building and experimenting with neural networks is easy and
intuitive. It achieves this by representing objects as PyGraphs (instead of PyTrees), 
enabling reference sharing and mutability. This design allows your models to resemble 
familiar Python object-oriented code, particularly appealing to users of frameworks
like PyTorch.

Despite its simplified implementation, NNX supports the same powerful design patterns 
that have allowed Linen to scale effectively to large codebases.

```{code-cell} ipython3
from flax.experimental import nnx
import jax
import jax.numpy as jnp
```

## The Module System
To begin lets see how to create a `Linear` Module using NNX. The main difference between 
NNX and Module systems like Haiku or Linen is that in NNX everything is **explicit**. This 
means among other things that 1) the Module itself holds the state (e.g. parameters) directly, 
2) the RNG state is threaded by the user, and 3) all shape information must be provided on 
initialization (no shape inference).

As shown next, dynamic state is usually stored in `nnx.Param`s, and static state 
(all types not handled by NNX) such as integers or strings  are stored directly. 
Attributes of type `jax.Array` and `numpy.ndarray` are also treated as dynamic 
state,although storing them inside `nnx.Variable`s such as `Param` is preferred.
Also, the `nnx.Rngs` object by can be used to get new unique keys based on a root 
key passed to the constructor.

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

`nnx.Variable`'s inner values can be accessed using the `.value` property, however
for convenience they implement all numeric operators and can be used directly in
arithmetic expressions (as shown above). Additionally, Variables can passed
to any JAX function as they implement the `__jax_array__` protocol (as long as their
inner value is a JAX array).

To actually initialize a Module you simply call the constructor, all the parameters 
of a Module are usually created eagerly. Since Modules hold their own state methods 
can be called directly without the no need for a separate `apply` method, this is very 
convenient for debugging as entire structure of the model can be inspected directly.

```{code-cell} ipython3
model = Linear(2, 5, rngs=nnx.Rngs(params=0))
y = model(x=jnp.ones((1, 2)))

print(y)
nnx.display(model)
```

The above visualization by `nnx.display` is generated using the awesome [Penzai](https://penzai.readthedocs.io/en/stable/index.html#) library.

+++

### Stateful Computation

Implementing layers such as `BatchNorm` requires performing state updates during the 
forward pass. To implement this in NNX you just create a `Variable` and update its 
`.value` during the forward pass.

```{code-cell} ipython3
class Count(nnx.Variable): pass

class Counter(nnx.Module):
  def __init__(self):
    self.count = Count(jnp.array(0))

  def __call__(self):
    self.count.value += 1

counter = Counter()
print(f'{counter.count.value = }')
counter()
print(f'{counter.count.value = }')
```

Mutable references are usually avoided in JAX, however as we'll see in later sections
NNX provides sound mechanisms to handle them.

+++

### Nested Modules

As expected, Modules can be used to compose other Modules in a nested structure, these can 
be assigned directly as attributes, or inside an attribute of any (nested) pytree type e.g.
 `list`, `dict`, `tuple`, etc. In the example below we define a simple `MLP` Module that
consists of two `Linear` layers, a `Dropout` layer, and a `BatchNorm` layer.

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

In NNX `Dropout` is a stateful module that stores an `Rngs` object so that it can generate
new masks during the forward pass without the need for the user to pass a new key each time.

+++

#### Model Surgery
NNX Modules are mutable by default, this means their structure can be changed at any time, 
this makes model surgery quite easy as any submodule attribute can be replaced with anything
else e.g. new Modules, existing shared Modules, Modules of different types, etc. More over, 
`Variable`s can also be modified or replaced / shared.

The following example shows how to replace the `Linear` layers in the `MLP` model
from before with `LoraLinear` layers.

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

# model surgery
model.linear1 = LoraLinear(model.linear1, 4, rngs=rngs)
model.linear2 = LoraLinear(model.linear2, 4, rngs=rngs)

y = model(x=jnp.ones((3, 2)))

nnx.display(model)
```

## NNX Transforms

NNX Transforms extend JAX transforms to support Modules and other objects.
They are supersets of their equivalent JAX counterpart with the addition of
being aware of the object's state and providing additional APIs to transform 
it. One of the main features of NNX Transforms is the preservation of reference semantics, 
meaning that any mutation of the object graph that occurs inside the transform is
propagated outisde as long as its legal within the transform rules. In practice this
means that NNX programs can be express using imperative code, highly simplifying
the user experience.

In the following example we define a `train_step` function that takes a `MLP` model,
an `Optimizer`, and a batch of data, and returns the loss for that step. The loss
and the gradients are computed using the `nnx.value_and_grad` transform over the
`loss_fn`. The gradients are passed to the optimizer's `update` method to update
the `model`'s parameters.

```{code-cell} ipython3
import optax

# MLP contains 2 Linear layers, 1 Dropout layer, 1 BatchNorm layer
model = MLP(2, 16, 10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

@nnx.jit  # automatic state management
def train_step(model, optimizer, x, y):
  def loss_fn(model: MLP):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # inplace updates

  return loss

x, y = jnp.ones((5, 2)), jnp.ones((5, 10))
loss = train_step(model, optimizer, x, y)

print(f'{loss = }')
print(f'{optimizer.step.value = }')
```

Theres a couple of things happening in this example that are worth mentioning:
1. The updates to the `BatchNorm` and `Dropout` layer's state is automatically propagated
  from within `loss_fn` to `train_step` all the way to the `model` reference outside.
2. `optimizer` holds a mutable reference to `model`, this relationship is preserved
  inside the `train_step` function making it possible to update the model's parameters
  using the optimizer alone.

#### Scan over layers
Next lets take a look at a different example using `nnx.vmap` to create an
`MLP` stack and `nnx.scan` to iteratively apply each layer in the stack to the
input (scan over layers). 

Notice the following:
1. The `create_model` function creates a (single) `MLP` object that is lifted by
  `nnx.vmap` to have an additional dimension of size `axis_size`.
2. The `forward` function indexes the `MLP` object's state to get a different set of
  parameters at each step.
3. `nnx.scan` automatically propagates the state updates for the `BatchNorm` and 
`Dropout` layers from within `forward` to the `model` reference outside.

```{code-cell} ipython3
from functools import partial

@partial(nnx.vmap, axis_size=5)
def create_model(rngs: nnx.Rngs):
  return MLP(10, 32, 10, rngs=rngs)

model = create_model(nnx.Rngs(0))

@nnx.scan
def forward(x, model: MLP):
  x = model(x)
  return x, None

x = jnp.ones((3, 10))
y, _ = forward(x, model)

print(f'{y.shape = }')
nnx.display(model)
```

How do NNX transforms achieve this? To understand how NNX objects interact with
JAX transforms lets take a look at the Functional API.

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
  
model = StatefulLinear(din=3, dout=5, rngs=nnx.Rngs(0))
nnx.display(model)
```

### State and GraphDef

A Module can be decomposed into `GraphDef` and `State` using the
`split` function. State is a Mapping from strings to Variables or nested 
States. GraphDef contains all the static information needed to reconstruct 
a Module graph, it is analogous to JAX's `PyTreeDef`.

```{code-cell} ipython3
graphdef, state = nnx.split(model)

nnx.display(graphdef, state)
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
  model = nnx.merge(graphdef, state)
  # 3. Call the Module
  y = model(x)
  # 4. Use split to propagate State updates
  _, state = nnx.split(model)
  return y, state

y, state = forward(graphdef, state, x=jnp.ones((1, 3)))
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

nnx.display(params, counts)
```

Note that filters must be exhaustive, if a value is not matched an error will be raised.

As expected the `merge` and `update` methods naturally consume multiple States:

```{code-cell} ipython3
# merge multiple States
model = nnx.merge(graphdef, params, counts)
# update with multiple States
nnx.update(model, params, counts)
```

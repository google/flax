---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# NNX Demo

```{code-cell} ipython3
import jax
from jax import numpy as jnp
from flax.experimental import nnx
```

### [1] NNX is Pythonic

```{code-cell} ipython3
:outputId: d8ef66d5-6866-4d5c-94c2-d22512bfe718


class Block(nnx.Module):
  def __init__(self, din, dout, *, rngs):
    self.linear = nnx.Linear(din, dout, rngs=rngs,
                    kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal() , ('data', 'mp')))
    self.bn = nnx.BatchNorm(dout, rngs=rngs)

  def __call__(self, x, *, train: bool):
    x = self.linear(x)
    x = self.bn(x, use_running_average=not train)
    x = nnx.relu(x)
    return x


class MLP(nnx.Module):
  def __init__(self, nlayers, dim, *, rngs): # explicit RNG threading
    self.blocks = [
      Block(dim, dim, rngs=rngs) for _ in range(nlayers)
    ]
    self.count = Count(0)  # stateful variables are defined as attributes

  def __call__(self, x, *, train: bool):
    self.count += 1  # in-place stateful updates
    for block in self.blocks:
      x = block(x, train=train)
    return x

class Count(nnx.Variable):   # custom Variable types define the "collections"
  pass

model = MLP(5, 4, rngs=nnx.Rngs(0))  # no special `init` method
y = model(jnp.ones((2, 4)), train=False)  # call methods directly

print(f'{model = }'[:500] + '\n...')
```

Because NNX Modules contain their own state, they are very easily to inspect:

```{code-cell} ipython3
:outputId: 10a46b0f-2993-4677-c26d-36a4ddf33449

print(f'{model.count = }')
print(f'{model.blocks[0].linear.kernel = }')
# print(f'{model.blocks.sdf.kernel = }') # typesafe inspection
```

### [2] Model Surgery is Intuitive

```{code-cell} ipython3
:outputId: e6f86be8-3537-4c48-f471-316ee0fb6c45

# Module sharing
model.blocks[1] = model.blocks[3]
# Weight tying
model.blocks[0].linear.variables.kernel = model.blocks[-1].linear.variables.kernel
# Monkey patching
def my_optimized_layer(x, *, train: bool): return x
model.blocks[2] = my_optimized_layer

y = model(jnp.ones((2, 4)), train=False)  # still works
print(f'{y.shape = }')
```

### [3] Interacting with JAX is easy

```{code-cell} ipython3
:outputId: 9a3f378b-739e-4f45-9968-574651200ede

state, static = model.split()

# state is a dictionary-like JAX pytree
print(f'{state = }'[:500] + '\n...')

# static is also a JAX pytree, but containing no data, just metadata
print(f'\n{static = }'[:300] + '\n...')
```

```{code-cell} ipython3
:outputId: 0007d357-152a-449e-bcb9-b1b5a91d2d8d

state, static = model.split()

@jax.jit
def forward(static: nnx.GraphDef, state: nnx.State, x: jax.Array):
  model = static.merge(state)
  y = model(x, train=True)
  state, _ = model.split()
  return y, state

x = jnp.ones((2, 4))
y, state = forward(static,state, x)

model.update(state)

print(f'{y.shape = }')
print(f'{model.count = }')
```

```{code-cell} ipython3
params, batch_stats, counts, static = model.split(nnx.Param, nnx.BatchStat, Count)

@jax.jit
def forward(static: nnx.GraphDef, params, batch_stats, counts, x: jax.Array):
  model = static.merge(params, batch_stats, counts)
  y = model(x, train=True)
  params, batch_stats, counts, _ = model.split(nnx.Param, nnx.BatchStat, Count)
  return y, params, batch_stats, counts

x = jnp.ones((2, 4))
y, params, batch_stats, counts = forward(static, params, batch_stats, counts, x)

model.update(params, batch_stats, counts)

print(f'{y.shape = }')
print(f'{model.count = }')
```

```{code-cell} ipython3
class Parent(nnx.Module):

    def __init__(self, model: MLP):
        self.model = model

    def __call__(self, x, *, train: bool):

        params, batch_stats, counts, static = self.model.split(nnx.Param, nnx.BatchStat, Count)

        @jax.jit
        def forward(static: nnx.GraphDef, params, batch_stats, counts, x: jax.Array):
            model = static.merge(params, batch_stats, counts)
            y = model(x, train=True)
            params, batch_stats, counts, _ = model.split(nnx.Param, nnx.BatchStat, Count)
            return y, params, batch_stats, counts

        y, params, batch_stats, counts = forward(static, params, batch_stats, counts, x)

        self.model.update(params, batch_stats, counts)

        return y

parent = Parent(model)

y = parent(jnp.ones((2, 4)), train=False)

print(f'{y.shape = }')
print(f'{parent.model.count = }')
```

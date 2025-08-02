---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Mutable Arrays (experimental)

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import jax.experimental
import optax
```

## Basics

+++

### Mutable Arrays 101

```{code-cell} ipython3
m_array = jax.experimental.mutable_array(jnp.array([1, 2, 3]))

@jax.jit
def increment(m_array: jax.experimental.MutableArray):  # no return!
  array: jax.Array = m_array[...]  # access
  m_array[...] = array + 1         # update

print("[1] =", m_array); increment(m_array); print("[2] =", m_array)
```

```{code-cell} ipython3
@jax.jit
def inc(x):
  x[...] += 1

print(increment.lower(m_array).as_text())
```

### Mutable Variables

```{code-cell} ipython3
variable = nnx.Variable(jnp.array([1, 2, 3]), mutable=True)
print(f"{variable.mutable = }\n")

print("[1] =", variable); increment(variable); print("[2] =", variable)
```

```{code-cell} ipython3
with nnx.use_mutable_arrays(True):
  variable = nnx.Variable(jnp.array([1, 2, 3]))

print(f"{variable.mutable = }")
```

### Changing Status

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs(), (in_features, out_features)))
    self.bias = nnx.Param(jnp.zeros(out_features))

  def __call__(self, x):
    return x @ self.kernel + self.bias[None]

model = Linear(1, 3, rngs=nnx.Rngs(0)) # without mutable arrays
mutable_model = nnx.mutable(model) # convert to mutable arrays
frozen_model = nnx.freeze(mutable_model) # freeze mutable arrays again

print("nnx.mutable(model) =", mutable_model)
print("nnx.freeze(mutable_model) =", frozen_model)
```

## Examples

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)
    self.linear_out = Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.gelu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)
```

### Training Loop

```{code-cell} ipython3
with nnx.use_mutable_arrays(True):
  model = Block(2, 64, 3, rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@jax.jit
def train_step(model, optimizer, x, y):
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)
  def loss_fn(params):
    model =  nnx.merge(graphdef, params, nondiff)
    return ((model(x) - y) ** 2).mean()

  loss, grads = jax.value_and_grad(loss_fn)(nnx.freeze(params))  # freeze MutableArrays for jax.grad
  optimizer.update(model, grads)

  return loss

train_step(model, optimizer, x=jnp.ones((10, 2)), y=jnp.ones((10, 3)))
```

### Scan Over Layers

```{code-cell} ipython3
@nnx.vmap
def create_stack(rngs):
  return Block(2, 64, 2, rngs=rngs)

with nnx.use_mutable_arrays(True):
  block_stack = create_stack(nnx.Rngs(0).fork(split=8))

def scan_fn(x, block):
  x = block(x)
  return x, None

x = jax.random.uniform(jax.random.key(0), (3, 2))
y, _ = jax.lax.scan(scan_fn, x, block_stack)

print("y = ", y)
```

## Limitations

+++

### MutableArray Outputs

```{code-cell} ipython3
@jax.jit
def create_model(rngs):
  return Block(2, 64, 3, rngs=rngs)

try:
  with nnx.use_mutable_arrays(True):
    model = create_model(nnx.Rngs(0))
except Exception as e:
  print(f"Error:", e)
```

```{code-cell} ipython3
with nnx.use_mutable_arrays(False): # <-- disable mutable arrays
  model = create_model(nnx.Rngs(0))

model = nnx.mutable(model) # convert to mutable after creation

print("model.linear =", model.linear)
```

```{code-cell} ipython3
@nnx.jit
def create_model(rngs):
  return Block(2, 64, 3, rngs=rngs)

with nnx.use_mutable_arrays(True):
  model = create_model(nnx.Rngs(0))

print("model.linear =", model.linear)
```

### Reference Sharing (aliasing)

```{code-cell} ipython3
def get_error(f, *args):
  try:
    return f(*args)
  except Exception as e:
    return f"{type(e).__name__}: {e}"
  
x = jax.experimental.mutable_array(jnp.array(0))

@jax.jit
def f(a, b):
  ...

print(get_error(f, x, x))
```

```{code-cell} ipython3
class SharedVariables(nnx.Object):
  def __init__(self):
    self.a = nnx.Variable(jnp.array(0))
    self.b = self.a

class SharedModules(nnx.Object):
  def __init__(self):
    self.a = Linear(1, 1, rngs=nnx.Rngs(0))
    self.b = self.a

@jax.jit
def g(pytree):
  ...

with nnx.use_mutable_arrays(True):
  shared_variables = SharedVariables()
  shared_modules = SharedModules()

print("SharedVariables", get_error(g, shared_variables))
print("SharedModules", get_error(g, shared_modules))
```

```{code-cell} ipython3
@jax.jit
def h(graphdef, state):
  obj = nnx.merge(graphdef, state)
  obj.a[...] += 10

graphdef, state = nnx.split(shared_variables)
print(state) # split deduplicates the state

h(graphdef, state)

print("updated", shared_variables)
```

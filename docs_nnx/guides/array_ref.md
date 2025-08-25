---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Array Refs (experimental)

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import optax
```

## Basics

+++

### Array Refs 101

```{code-cell} ipython3
a_ref = nnx.array_ref(jnp.array([1, 2, 3]))

@jax.jit
def increment(a_ref: nnx.ArrayRef):  # no return!
  array: jax.Array = a_ref[...]  # access
  a_ref[...] = array + 1         # update

print("[1] =", a_ref); increment(a_ref); print("[2] =", a_ref)
```

```{code-cell} ipython3
@jax.jit
def inc(x):
  x[...] += 1

print(increment.lower(a_ref).as_text())
```

### Variables Refs

```{code-cell} ipython3
variable = nnx.Variable(jnp.array([1, 2, 3]), use_ref=True)
print(f"{variable.has_ref = }\n")

print("[1] =", variable); increment(variable); print("[2] =", variable)
```

```{code-cell} ipython3
with nnx.use_refs(True):
  variable = nnx.Variable(jnp.array([1, 2, 3]))

print(f"{variable.has_ref = }")
```

Mention `nnx.use_refs` can be used as global flag

+++

### Changing Status

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs(), (in_features, out_features)))
    self.bias = nnx.Param(jnp.zeros(out_features))

  def __call__(self, x):
    return x @ self.kernel + self.bias[None]

model = Linear(1, 3, rngs=nnx.Rngs(0)) # without array refs
refs_model = nnx.to_refs(model) # convert to array refs
arrays_model = nnx.to_arrays(refs_model) # convert to regular arrays

print("nnx.to_refs(model) =", refs_model)
print("nnx.to_arrays(refs_model) =", arrays_model)
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
with nnx.use_refs(True):
  model = Block(2, 64, 3, rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@jax.jit
def train_step(model, optimizer, x, y):
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)
  def loss_fn(params):
    model =  nnx.merge(graphdef, params, nondiff)
    return ((model(x) - y) ** 2).mean()

  loss, grads = jax.value_and_grad(loss_fn)(nnx.to_arrays(params))  # freeze ArrayRefs for jax.grad
  optimizer.update(model, grads)

  return loss

train_step(model, optimizer, x=jnp.ones((10, 2)), y=jnp.ones((10, 3)))
```

### Scan Over Layers

```{code-cell} ipython3
@nnx.vmap
def create_stack(rngs):
  return Block(2, 64, 2, rngs=rngs)

with nnx.use_refs(True):
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
  with nnx.use_refs(True):
    model = create_model(nnx.Rngs(0))
except Exception as e:
  print(f"Error:", e)
```

```{code-cell} ipython3
with nnx.use_refs(False): # <-- disable array refs
  model = create_model(nnx.Rngs(0))

model = nnx.to_refs(model) # convert to mutable after creation

print("model.linear =", model.linear)
```

```{code-cell} ipython3
@nnx.jit
def create_model(rngs):
  return Block(2, 64, 3, rngs=rngs)

with nnx.use_refs(True):
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
  
x = nnx.array_ref(jnp.array(0))

@jax.jit
def f(a, b):
  ...

print(get_error(f, x, x))
```

```{code-cell} ipython3
class SharedVariables(nnx.Pytree):
  def __init__(self):
    self.a = nnx.Variable(jnp.array(0))
    self.b = nnx.Variable(jnp.array(1))
    self.c = self.a

class SharedModules(nnx.Pytree):
  def __init__(self):
    self.d = Linear(1, 1, rngs=nnx.Rngs(0))
    self.e = Linear(1, 1, rngs=nnx.Rngs(0))
    self.f = self.d

@jax.jit
def g(pytree):
  ...

with nnx.use_refs(True):
  shared_variables = SharedVariables()
  shared_modules = SharedModules()

print("SharedVariables", get_error(g, shared_variables))
print("SharedModules", get_error(g, shared_modules))
```

```{code-cell} ipython3
if (duplicates := nnx.find_duplicates(shared_variables)):
  print("shared variables duplicates:", duplicates)

if (duplicates := nnx.find_duplicates(shared_modules)):
  print("shared modules duplicates:  ", duplicates)
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

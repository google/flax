---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Hijax (experimental)

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import optax
```

## Basics

+++

### Variables Modes

```{code-cell} ipython3
variable = nnx.Variable(
  value=jnp.array([1, 2, 3]),
  mode='ref' # | 'hijax' | 'lojax' (default)
)
print(f"{variable.mode = }\n")

@jax.jit
def increment(variable: nnx.Variable[jax.Array]):  # no return!
  new_value = variable + 1  # Array-like operations
  variable[...] = new_value        # in-place updates

print("Before =", variable); increment(variable); print("After =", variable)
```

```{code-cell} ipython3
value = jnp.array(0, dtype=jnp.int32)
print("hijax =", jax.make_jaxpr(increment)(nnx.Variable(value, mode='hijax')))
print("ref =", jax.make_jaxpr(increment)(nnx.Variable(value, mode='ref')))
```

```{code-cell} ipython3
nnx.variable_mode('ref')

variable = nnx.Variable(jnp.array([1, 2, 3]))

print(f"{variable.mode = }")
```

### Changing Status

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs(), (in_features, out_features)))

  def __call__(self, x):
    return x @ self.kernel

model = Linear(1, 3, rngs=nnx.Rngs(0))

ref_vars = nnx.as_ref(model)            # convert to ref Variables
hijax_vars = nnx.as_hijax(model)        # convert to hijax Variables
pytree_vars = nnx.as_lojax(hijax_vars)  # convert to lojax Variables

print("nnx.as_ref(model) =", ref_vars)
print("nnx.as_hijax(model) =", hijax_vars)
print("nnx.as_lojax(refs_model) =", pytree_vars)
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
# hijax Variables by default
model = Block(2, 64, 3, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@jax.jit
def train_step(model, optimizer, x, y):
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)
  def loss_fn(params):
    model =  nnx.merge(graphdef, params, nondiff)
    return ((model(x) - y) ** 2).mean()

  loss, grads = jax.value_and_grad(loss_fn)(nnx.as_lojax(params))  # lojax Variables for jax.grad
  optimizer.update(model, grads)

  return loss

for _ in range(3):
  loss = train_step(model, optimizer, x=jnp.ones((10, 2)), y=jnp.ones((10, 3)))
  print(f"{loss = !s}")
```

### Scan Over Layers

```{code-cell} ipython3
@nnx.vmap  # NOTE: uses nnx.vmap
def create_stack(rngs):
  return Block(2, 64, 2, rngs=rngs)

block_stack = nnx.as_ref(create_stack(nnx.Rngs(0).fork(split=8)))

def scan_fn(x, block):
  x = block(x)
  return x, None

x = jax.random.uniform(jax.random.key(0), (3, 2))
y, _ = jax.lax.scan(scan_fn, x, block_stack)

print("y = ", y)
```

## Limitations

+++

### Mutable Outputs

```{code-cell} ipython3
@jax.jit
def create_model(rngs):
  return Block(2, 64, 3, rngs=rngs)

try:
  model = create_model(nnx.Rngs(0))
except Exception as e:
  print(f"Error:", e)
```

```{code-cell} ipython3
@jax.jit
def create_model(rngs):
  return nnx.as_lojax(Block(2, 64, 3, rngs=rngs))

model = nnx.as_hijax(create_model(nnx.Rngs(0)))

print("model.linear =", model.linear)
```

```{code-cell} ipython3
@nnx.jit
def create_model(rngs):
  return Block(2, 64, 3, rngs=rngs)

model = create_model(nnx.Rngs(0))

print("model.linear =", model.linear)
```

### Reference Sharing (aliasing)

```{code-cell} ipython3
# TODO: why does this not fail?
def get_error(f, *args):
  try:
    return f(*args)
  except Exception as e:
    return f"{type(e).__name__}: {e}"

x = nnx.Variable(jnp.array(0))

@jax.jit
def f(a, b):
  ...

print(get_error(f, x, x))
```

```{code-cell} ipython3
class Shared(nnx.Pytree):
  def __init__(self):
    self.a = nnx.Variable(jnp.array(0))
    self.b = self.a
    self.c = Linear(1, 1, rngs=nnx.Rngs(0))
    self.d = self.c

@jax.jit
def g(pytree):
  ...

shared = Shared()

print(get_error(g, shared))
```

```{code-cell} ipython3
print("Duplicates found:")
if (all_duplicates := nnx.find_duplicates(shared)):
  for duplicates in all_duplicates:
    print("-", duplicates)
```

```{code-cell} ipython3
@jax.jit
def h(graphdef, state):
  obj = nnx.merge(graphdef, state)
  obj.a[...] += 10

graphdef, state = nnx.split(shared)
print("before:", state.a) # split deduplicates the state

h(graphdef, state)

print("after:", shared.a)
```

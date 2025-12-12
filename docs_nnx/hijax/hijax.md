---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Hijax

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import optax

current_mode = nnx.using_hijax() # ignore: only needed for testing
```

```{code-cell} ipython3
nnx.use_hijax(True)

rngs = nnx.Rngs(0)
model = nnx.Linear(2, 3, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(1e-2), wrt=nnx.Param)

@jax.jit
def train_step(x, y):
  loss_fn = lambda m: jnp.mean((m(x) - y) ** 2)
  loss, grads = jax.value_and_grad(loss_fn)(nnx.vars_as(model, is_mutable=False))  # tmp fix for jax.grad
  optimizer.update(model, grads)
  return loss

x, y = rngs.uniform((4, 2)), rngs.uniform((4, 3))
for _ in range(3):
  print(train_step(x, y))
```

## Hijax Variable

+++

State propagation:

```{code-cell} ipython3
v = nnx.Variable(jnp.array(0), is_hijax=True)

@jax.jit
def inc(v):
  v[...] += 1

print(v[...]); inc(v); print(v[...])
```

```{code-cell} ipython3
v = nnx.Variable(jnp.array(0), is_hijax=True)
print(jax.make_jaxpr(inc)(v))
```

Pytree values:

```{code-cell} ipython3
v = nnx.Variable({'a': jnp.array(0), 'b': jnp.array(2)}, is_hijax=True)

@jax.jit
def inc_and_double(v):
  v['a'] += 1
  v['b'] *= 2

print(v); inc_and_double(v); print(v)
```

Dynamic state structure:

```{code-cell} ipython3
rngs = nnx.Rngs(0)
x = rngs.uniform((4, 5))
w = rngs.normal((5, 3))
metrics = nnx.Variable({}, is_hijax=True)

@jax.jit
def linear(x, w, metrics: nnx.Variable):
  y = x @ w
  metrics['y_mean'] = jnp.mean(y)
  return y

print("Before:", metrics)
y = linear(x, w, metrics)
print("After:", metrics)
```

```{code-cell} ipython3
# set default Variable mode for the rest of the guide
nnx.use_hijax(True)

variable = nnx.Variable(jnp.array([1, 2, 3]))

print(variable)
```

### Mutability

```{code-cell} ipython3
class Linear(nnx.Module):
  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    self.kernel = nnx.Param(rngs.normal((in_features, out_features)))

  def __call__(self, x):
    return x @ self.kernel

model = Linear(1, 3, rngs=nnx.Rngs(0))

print(f"{nnx.vars_as(model, is_mutable=False) = !s}")
print(f"{nnx.vars_as(model, is_mutable=True) = !s}")
```

```{code-cell} ipython3
v = nnx.Variable(jnp.array(0))
v_immut = nnx.vars_as(v, is_mutable=False)
assert not v_immut.is_mutable

try:
  v_immut[...] += 1  # raises an error
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

### Ref support

```{code-cell} ipython3
v = nnx.Variable(jnp.array(0))
v_ref = nnx.vars_as(v, has_ref=True)
assert v_ref.has_ref
print(v_ref)
print(v_ref.get_raw_value())
```

```{code-cell} ipython3
v_immut = nnx.vars_as(v_ref, is_mutable=False)
assert not v_immut.has_ref
print("immutable =", v_immut)

v_ref = nnx.vars_as(v_immut, is_mutable=True)
assert v_ref.has_ref
print("mutable =", v_ref)
```

### Examples

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

#### Training Loop

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

  loss, grads = jax.value_and_grad(loss_fn)(nnx.vars_as(params, is_mutable=False))  # immutable for jax.grad
  optimizer.update(model, grads)

  return loss

for _ in range(3):
  loss = train_step(model, optimizer, x=jnp.ones((10, 2)), y=jnp.ones((10, 3)))
  print(f"{loss = !s}")
```

#### Scan Over Layers

```{code-cell} ipython3
# TODO: does not work with hijax yet
# @jax.vmap
# def create_stack(rngs):
#   return nnx.as_immutable_vars(Block(2, 64, 2, rngs=rngs))

# block_stack = nnx.as_mutable_vars(create_stack(nnx.Rngs(0).fork(split=8)))

# def scan_fn(x, block):
#   x = block(x)
#   return x, None

# x = jax.random.uniform(jax.random.key(0), (3, 2))
# y, _ = jax.lax.scan(scan_fn, x, block_stack)

# print("y = ", y)
```

### Limitations

+++

#### Mutable Outputs

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
  return nnx.vars_as((Block(2, 64, 3, rngs=rngs)), is_hijax=False)

model = nnx.vars_as(create_model(nnx.Rngs(0)), is_hijax=True)

print("model.linear =", model.linear)
```

#### Reference Sharing (aliasing)

```{code-cell} ipython3
# NOTE: doesn't currently fail on the jax side
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
# NOTE: doesn't currently fail on the jax side
class HasShared(nnx.Pytree):
  def __init__(self):
    self.a = nnx.Variable(jnp.array(0))
    self.b = self.a

@jax.jit
def g(has_shared):
  has_shared.a[...] = 5

has_shared = HasShared()

print(get_error(g, has_shared))
print(has_shared)  # updates don't propagate
```

```{code-cell} ipython3
print("Duplicates found:")
if (all_duplicates := nnx.find_duplicates(has_shared)):
  for duplicates in all_duplicates:
    print("-", duplicates)
```

```{code-cell} ipython3
@jax.jit
def h(graphdef, state):
  has_shared = nnx.merge(graphdef, state)
  has_shared.a[...] = 5

graphdef, state = nnx.split(has_shared)
h(graphdef, state)
print(has_shared)
```

```{code-cell} ipython3
# clean up for CI tests
_ = nnx.use_hijax(current_mode)
```

---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import dataclasses

#----------------------
# helper functions
#----------------------
def pytree_structure(pytree, title='pytree structure'):
  print(f"{title}:")
  for path, value in jax.tree.leaves_with_path(pytree):
    print(f"- pytree{jax.tree_util.keystr(path)} = {value!r}")
```

## Pytree

```{code-cell} ipython3
class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int):
    self.din, self.dout = din, dout
    self.w = jnp.ones((din, dout))
    self.b = jnp.zeros((dout,))

class MLP(nnx.Pytree):
  def __init__(self, num_layers, dim):
    self.num_layers = num_layers
    self.layers = nnx.List([Linear(dim, dim) for _ in range(num_layers)])

pytree = MLP(num_layers=2, dim=1)
pytree_structure(pytree)
```

### Attribute Annotations

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self, i: int):
    self.i = nnx.data(i)  # explicit data
    self.f = nnx.static(i * 0.5)  # explicit static
    self.x = jnp.array(42 * i)  # arrays are data
    self.s = "Hi" + "!" * i  # strings are static
    self.h = hash(i)  # weak types are static
    self.u = None  # empty pytrees are static

class Bar(nnx.Pytree):
  def __init__(self):
    self.ls = nnx.List([Foo(i) for i in range(2)])  # nnx.Pytrees are data
    self.shapes = (8, 16, 32)  # common pytrees are static

pytree = Bar()
pytree_structure(pytree)
```

```{code-cell} ipython3
print(f"""
# ------ DATA ------------
{nnx.is_data( jnp.array(0) ) = }                   # Arrays are data
{nnx.is_data( nnx.Param(1) ) = }                   # Variables are data
{nnx.is_data( nnx.Rngs(2) ) = }                    # nnx.Pytrees are data

# ------ STATIC ------------
{nnx.is_data( 'hello' ) = }                       # strings, arbitrary objects
{nnx.is_data( 42 ) = }                            # int, float, bool, complex, etc.
{nnx.is_data( [1, 2.0, 3j, jnp.array(1)] ) = }    # list, dict, tuple, pytrees
""")
```

* remove mixed
* error on nnx.data/nnx.static in pytrees

+++

### Class Annotations

```{code-cell} ipython3
@dataclasses.dataclass
class Foo(nnx.Pytree):
  i: nnx.Data[int]
  s: nnx.Static[str]
  x: jax.Array
  a: int

@dataclasses.dataclass
class Bar(nnx.Pytree):
  ls: nnx.Data[list[Foo]]
  shapes: list[int]

pytree = Bar(
  ls=[Foo(i, "Hi" + "!" * i, jnp.array(42 * i), hash(i)) for i in range(2)],
  shapes=[8, 16, 32]
)
pytree_structure(pytree)
```

#### When to use explicit annotations?

```{code-cell} ipython3
class Bar(nnx.Pytree):
  def __init__(self, x, use_bias: bool):
    self.x = nnx.data(x)  # constrain inputs (e.g. user could pass Array or ShapeDtypeStruct)
    self.y = nnx.data(42)  # force undefined types
    self.ls = nnx.data([jnp.array(i) for i in range(3)]) # on pytrees
    if use_bias:
      self.bias = nnx.Param(jnp.array(0.0))
    else:
      self.bias = nnx.data(None)  # on branches that cause mismatch

pytree = Bar(1.0, True)
pytree_structure(pytree)
```

### Attribute Updates

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.a = jnp.array(1.0)  # data
    self.b = "Hello, world!"               # static
    self.c = nnx.data(3.14)         # data

pytree = Foo()
pytree_structure(pytree, "original")

pytree.a = "🤔"  # static values don't change status on data attributes
pytree.b = nnx.data(42)     # annotation to override status
pytree.c = nnx.static(0.5)  # annotation to override status
pytree_structure(pytree, "updated")
```

### Attribute checks

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self, name):
    self.name = nnx.static(name)

try:
  foo = Foo(name=jnp.array(123))
except ValueError as e:
  print("ValueError:", e)
```

```{code-cell} ipython3
try:
  foo = Foo(name="mattjj")
  foo.name = jnp.array(123)
except ValueError as e:
  print("ValueError:", e)
```

* `nnx.check_pytree`

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.ls = []  # should be nnx.data([]), treated as static
    for i in range(5):
      self.ls.append(jnp.array(i))  # error: inserting arrays into static attribute

try:
  foo = Foo()  # nnx.check_pytree ran after __init__
except ValueError as e:
  print("ValueError:", e)
```

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.a = [nnx.data(1), nnx.data(2)]  # annotations in sub-pytree

try:
  foo = Foo()
except ValueError as e:
  print("ValueError:", e)
```

### Trace-level awareness

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.count = nnx.data(0)

foo = Foo()

@jax.vmap  # or jit, grad, shard_map, pmap, scan, etc.
def increment(n):
  foo.count += 1

try:
  increment(jnp.arange(5))
except Exception as e:
  print(f"Error: {e}")
```

## Module

+++

### set_attributes

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.mode = 1
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    return nnx.relu(self.dropout(self.bn(self.linear(x))))
  
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

print("train:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")

# Set attributes for evaluation
model.set_attributes(deterministic=True, use_running_average=True, mode=2)

print("eval:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")
```

```{code-cell} ipython3
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

model.eval(mode=2)  # .set_attributes(deterministic=True, use_running_average=True, mode=2)
print("eval:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")

model.train(mode=1)  # .set_attributes(deterministic=False, use_running_average=False, mode=1)
print("train:")
print(f"- {model.mode = }")
print(f"- {model.bn.use_running_average = }")
print(f"- {model.dropout.deterministic = }")
```

### sow

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    y = nnx.relu(self.dropout(self.bn(self.linear(x))))
    self.sow(nnx.Intermediate, "y_mean", jnp.mean(y))
    return y

class MLP(nnx.Module):
  def __init__(self, num_layers, dim, rngs: nnx.Rngs):
    self.blocks = nnx.data([Block(dim, dim, rngs) for _ in range(num_layers)])

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
    return x


model = MLP(num_layers=3, dim=20, rngs=nnx.Rngs(0))
x = jnp.ones((10, 20))
y = model(x)
intermediates = nnx.pop(model, nnx.Intermediate) # extract intermediate values
print(intermediates)
```

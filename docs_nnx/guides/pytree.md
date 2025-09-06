---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Module & Pytree

+++

Flax NNX's Modules are by default registered as JAX Pytrees, this allows using them throughout most of JAX APIs but in particular JAX transforms and the `jax.tree.*` functions. Thanks to the pytree protocol a simple NNX program might look like this:

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp

class Linear(nnx.Module):
  def __init__(self, din, dout, rngs: nnx.Rngs):
    self.din, self.dout = din, dout
    self.kernel = nnx.Param(rngs.normal((din, dout)))

rngs = nnx.Rngs(0)
weights = Linear(2, 3, rngs=rngs)

@jax.jit
def forward(weights, x):
  return x @ weights.kernel

y = forward(weights, x=rngs.uniform((5, 2)))
print(f"{y.shape = }")
```

Here `weights`, of type `Linear`, was able to be passed directly to the `jit`-ed function `forward`. Throughout the rest of this guide we will try to answer the questions:
1. What are pytrees? 
2. How does NNX implement pytrees?

+++

## Pytrees 101
JAX pytrees are tree structures made of python objects that can be recursively traversed in order to collect an ordered list of leaves and a definition of the tree structure, this is done via the `jax.tree.flatten` function. Most common pytrees are native python containers like `list`, `dict`, and `tuple`, but interestingly it also include `None`. The example bellow shows how to collect all the integer leaves from a nested structure using `flatten`:

```{code-cell} ipython3
pytree = [
  {'a': 1},
  {
    'b': 2,
    'c': (3, 4),
    'd': None,
  }
]

leaves, treedef = jax.tree.flatten(pytree)
print(f"leaves = {leaves}")
print(f"treedef = {treedef}")
```

Note that `None` is not a leaf because its defined as a pytree with no children. The main purpose of being able to flatten, apart from collecting the leaves, is being able reconstruct the pytree structure from the tree definition from any sequence of leaves of the same length via the `jax.tree.unflatten` function:

```{code-cell} ipython3
new_leaves = [x * 10 for x in leaves]
new_pytree = jax.tree.unflatten(treedef, new_leaves)

print(f"old pytree = {pytree}")
print(f"new pytree = {new_pytree}")
```

### Custom Pytrees
JAX allows us to register custom pytree node type by using the `jax.tree_util.register_pytree_node` utility. For any type we are able to define a flatten that decomposes the object into a a sequence of nodes / children and a static (hashable) structure, and a unflatten function which takes the sequence of nodes and the static structure and creates a new instance. In the following example we create a simple type `Foo` with the attributes `a`, `b`, and `c`, and define `a` and `b` as nodes and `c` as static.

```{code-cell} ipython3
class Foo:
  def __init__(self):
    self.a = 1
    self.b = 2
    self.c = "hi"

def flatten_foo(foo: Foo):
  nodes = [foo.a, foo.b]  # sequence of nodes
  static = (foo.c,) # hashable & equatable structure
  return nodes, static

def unflatten_foo(static, nodes):
  foo = object.__new__(Foo)  # create uninitialized instance
  foo.a = nodes[0]
  foo.b = nodes[1]
  foo.c = static[0]
  return foo

jax.tree_util.register_pytree_node(Foo, flatten_foo, unflatten_foo)

foo = Foo()
leaves, treedef = jax.tree.flatten(foo)
print(f"leaves = {leaves}")
print(f"treedef = {treedef}")
```

Notice that `'hi'` does not appear in the leaves because `c` is defined as static, but you can see it as part of the `PyTreeDef` structure.

+++

## nnx.Pytree
In general it would be cumbersome for users to manually register the pytree definition for every type they create. To automate this process NNX provides the `nnx.Pytree` base type that offers a simple API: users annotate attributes using either `nnx.static` or `nnx.data`, and Pytree will register some flatten and unflatten functions that will take the annotations into account.

```{code-cell} ipython3
class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int):
    self.din = nnx.static(din)
    self.dout = nnx.static(dout)
    self.w = nnx.data(jnp.ones((din, dout)))
    self.b = nnx.data(jnp.zeros((dout,)))

class MLP(nnx.Pytree):
  def __init__(self, num_layers, dim):
    self.num_layers = nnx.static(num_layers)
    self.layers = nnx.data([
      Linear(dim, dim) for _ in range(num_layers)
    ])

pytree = MLP(num_layers=2, dim=1)

def pytree_structure(pytree, title='pytree structure'):
  print(f"{title}:")
  path_leaves, treedef = jax.tree.flatten_with_path(pytree)
  for path, value in path_leaves:
    print(f" - pytree{jax.tree_util.keystr(path)} = {value!r}")

pytree_structure(pytree)
```

As you can see above, only the `data` paths appear in the leaves. However, its very verbose to have to define `static` and `data` for each attribute so `Pytree` has sensible defaults so you can mostly just remove them from them and it will work.

```{code-cell} ipython3
class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int):
    self.din = din # static
    self.dout = dout # static
    self.w = jnp.ones((din, dout)) # data
    self.b = jnp.zeros((dout,)) # data

class MLP(nnx.Pytree):
  def __init__(self, num_layers, dim):
    self.num_layers = num_layers # static
    self.layers = nnx.List([ # data
      Linear(dim, dim) for _ in range(num_layers)
    ])

pytree = MLP(num_layers=2, dim=1)
pytree_structure(pytree)
```

The main chain we had to do is use `nnx.List` to signal that `layers` contains `data`. The rules that determine if a value is data or not are the following:

* Its an `Array`, `Variable`, `ArrayRef`, or `Pytree` instance.
* Its a type registered as data using `nnx.register_data_type`.

If you are ever curious if a value is going to treated as data or as static use the `nnx.is_data` function which will return its status:

```{code-cell} ipython3
print(f"""
# ------ DATA ------------
{nnx.is_data( jnp.array(0) ) = }                   # Arrays are data
{nnx.is_data( nnx.Param(1) ) = }                   # Variables are data
{nnx.is_data( nnx.Rngs(2) ) = }                    # nnx.Pytrees are data

# ------ STATIC ------------
{nnx.is_data( 'hello' ) = }                       # strings, arbitrary objects
{nnx.is_data( 42 ) = }                            # int, float, bool, complex, etc.
{nnx.is_data( [1, 2.0, 3j, jnp.array(1)] ) = }    # list, dict, tuple, regular pytrees
""")
```

### When to use explicit annotations?

+++

There are cases were you do want to explicitely annotate the attributes to avoid ambiguety or protect yourself against possible edge cases. These include constraining input arguments which might have unexpected types, forcing attributes as data when their type is not treated as data by default, or using `nnx.static` as a way to assert the attribute should not contain data.

```{code-cell} ipython3
class Bar(nnx.Pytree):
  def __init__(self, x, use_bias: bool):
    self.x = nnx.data(x)  # constrain inputs (e.g. user could pass Array or float)
    self.y = nnx.data(42)  # force types that are not data by default
    self.ls = nnx.data([jnp.array(i) for i in range(3)]) # on pytrees
    self.bias = nnx.data(None)  # optional values that can be data
    if use_bias:
      self.bias = nnx.Param(jnp.array(0.0))

pytree = Bar(1.0, True)
pytree_structure(pytree)
```

### Class Annotations
Annotations can also be added at the type level via `nnx.Static` and `nnx.Data`, this is useful for creating `dataclasses` but the mechanism also works for regular classes.

```{code-cell} ipython3
import dataclasses

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

### pytree=False

```{code-cell} ipython3
class Foo(nnx.Pytree, pytree=False):
  def __init__(self):
    self.a = [jnp.array(1), jnp.array(2)]  # no checks
    self.b = "hello" 
    self.b = jnp.array(3) # no checks

foo = Foo()

@nnx.jit # can use in NNX transformations
def double(foo: Foo):
  foo.a = [x * 2 for x in foo.a]
  foo.b *= 2

double(foo)
print(f"{ nnx.state(foo) = }")  # can be used with NNX APIs
print(f"{ jax.tree_util.all_leaves([foo]) = }")  # not a pytree
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

## Object

```{code-cell} ipython3
class Foo(nnx.Object): # instead of Foo(nnx.Pytree, pytree=False)
  def __init__(self):
    self.a = [jnp.array(1), jnp.array(2)]  # no checks
    self.b = "hello" 
    self.b = jnp.array(3) # no checks

foo = Foo()

@nnx.jit # can use in NNX transformations
def double(foo: Foo):
  foo.a = [x * 2 for x in foo.a]
  foo.b *= 2

double(foo)
print(f"{ nnx.state(foo) = }")  # can be used with NNX APIs
print(f"{ jax.tree_util.all_leaves([foo]) = }")  # not a pytree
```

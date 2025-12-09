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
Most modern ML models have too many Arrays for users to pass around individually, to deal with this JAX developed a way to track Array data in nested structures that still allowed caching for compilation: Pytrees. JAX pytrees are tree structures made of python objects that can be recursively traversed in order to collect an ordered list of leaves and a definition of the tree structure, this is done via the `jax.tree.flatten` function. Most common pytrees are native python containers like `list`, `dict`, and `tuple`, but interestingly it also include `None`. The example bellow shows how to collect all the integer leaves from a nested structure using `flatten`:

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
In general it would be cumbersome for users to manually register the pytree definition for every type they create. To automate this process NNX provides the `nnx.Pytree` base type that offers a simple API: users annotate attributes using either `nnx.static` or `nnx.data`, and Pytree will register some flatten and unflatten functions that will take the annotations into account. The `nnx.data` and `nnx.static` annotations must only be assigned to `Pytree` attributes directly.

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

As you can see above, only the `data` paths appear in the leaves. However, its very verbose to have to define `static` and `data` for each attribute, so `Pytree` has sensible defaults. You can remove most of them and it will just work:

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

The only change we had to do here is use `nnx.List` to signal that `layers` contains `data`, the status of the rest of the attributes can be correctly inferred. The rules that determine if a value is data or not are the following:

* `Array`s, `Variable`s, `ArrayRef`s, and `nnx.Pytree`s are data.
* Types registered using `nnx.register_data_type` are data.
* All other types are static.

To check if a value is data use the `nnx.is_data` function which will return its status:

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

There are cases were you do want to explicitely annotate the attributes to avoid ambiguity or protect yourself against possible edge cases. These include constraining input arguments which might have unexpected types, forcing attributes as data when their type is not treated as data by default, or using `nnx.static` as a way to assert the attribute should not contain data.

```{code-cell} ipython3
class Bar(nnx.Pytree):
  def __init__(self, x, use_bias: bool):
    self.x = nnx.data(x)  # constrain inputs (e.g. user could pass Array or float)
    self.y = nnx.data(42)  # force types that are not data by default
    self.ls = nnx.List([jnp.array(i) for i in range(3)]) # use nnx.List for lists of data
    self.bias = nnx.data(None)  # optional values that can be data
    if use_bias:
      self.bias = nnx.Param(jnp.array(0.0))

pytree = Bar(1.0, True)
pytree_structure(pytree)
```

### Dataclasses
`nnx.Pytree` dataclasses can be created by using the `nnx.dataclass` decorator. To control the status of each field, `nnx.static` and `nnx.data` can be used as `field` specifiers.

```{code-cell} ipython3
import dataclasses

@nnx.dataclass
class Foo(nnx.Pytree):
  i: int = nnx.data()
  x: jax.Array
  a: int
  s: str = nnx.static(default='hi', kw_only=True)

@nnx.dataclass
class Bar(nnx.Pytree):
  ls: list[Foo] = nnx.data()
  shapes: list[int]

pytree = Bar(
  ls=[Foo(i, jnp.array(42 * i), hash(i)) for i in range(2)],
  shapes=[8, 16, 32]
)
pytree_structure(pytree)
```

`dataclasses.dataclass` can also be used directly, however type checkers will not handle `nnx.static` and `nnx.data` correctly. To solve this `dataclasses.field` can be used by setting `metadata` with the appropriate entry for `static`.

```{code-cell} ipython3
@dataclasses.dataclass
class Bar(nnx.Pytree):
  a: int = dataclasses.field(metadata={'static': False}) # data
  b: str = dataclasses.field(metadata={'static': True})  # static

pytree = Bar(a=10, b="hello")
pytree_structure(pytree, title='dataclass pytree structure')
```

### Attribute Updates

+++

The status of an attribute is defined during its first assignment and will not change upon reassignment. However, it is possible to override the status by explicitly using `nnx.data` or `nnx.static` on reassignment.

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.a = jnp.array(1.0)  # data
    self.b = "Hello, world!"               # static
    self.c = nnx.data(3.14)         # data

pytree = Foo()
pytree_structure(pytree, "original")

pytree.a = "🤔"  # data status doesn't change
pytree.b = nnx.data(42)     # explicit annotation overrides status to data
pytree.c = nnx.static(0.5)  # explicit annotation overrides status to static
pytree_structure(pytree, "updated")
```

### Attribute checks
`Pytree` has a variety of checks to prevent a common class of errors in JAX. This includes checking for Arrays being assigned to new `static` attributes:

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self, name):
    self.name = nnx.static(name)

try:
  foo = Foo(name=jnp.array(123))
except ValueError as e:
  print("ValueError:", e)
```

Checking for Arrays being assigned to known `static` attributes:

```{code-cell} ipython3
try:
  foo = Foo(name="mattjj")
  foo.name = jnp.array(123)
except ValueError as e:
  print("ValueError:", e)
```

Checking for Arrays after `__init__` on `static` attributes that could've been inserted via mutation. This check can be manually trigger via `nnx.check_pytree` at any time.

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.ls = []  # treated as static
    for i in range(5):
      self.ls.append(jnp.array(i))  # append arrays into static attribute

try:
  foo = Foo()  # error: Array found in static attribute after `__init__`
except ValueError as e:
  print("ValueError:", e)
```

Checking for `nnx.data` or `nnx.static` annotations stored inside nested structures that are not `nnx.Pytree` instances:

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.a = [nnx.data(1), nnx.static(2)]  # annotations in sub-pytree

try:
  foo = Foo()
except ValueError as e:
  print("ValueError:", e)
```

### Trace-level awareness
To prevent tracer leakage NNX will raise an error when trying to update the attribute of a `Pytree` or the value of a `Variable` on instances that are passed as captures to functions called by JAX transforms:

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.count = nnx.data(0)

foo = Foo()

@jax.vmap  # or jit, grad, shard_map, pmap, scan, etc.
def increment(n):
  # foo passed as capture
  foo.count += 1  # error!

try:
  increment(jnp.arange(5))
except Exception as e:
  print(f"Error: {e}")
```

### Reference Sharing

+++

As the name implies Pytrees should be trees. To check if a structure is a well-defined tree you can use the `nnx.find_duplicates` functions which will return a list of duplicates, where each duplicate is a list of path tuples. In the example below we see that `left` and `right` are shared references therefore `find_duplicates` returns a non-empty list with the paths:

```{code-cell} ipython3
class Shared(nnx.Pytree):
  def __init__(self):
    self.x = jnp.array(1.0)

class Parent(nnx.Pytree):
  def __init__(self):
    self.left = Shared()
    self.right = self.left  # reference sharing

m = Parent()

print(f"{nnx.find_duplicates(m) = }  # not a tree")
```

The main issue is that sharing is not preserved across pytree operations including JAX transforms, and this results in unintended state duplication:

```{code-cell} ipython3
m = Parent()
print(f"Before: {m.left is m.right = }")

@jax.jit
def f(m):
  print(f"Inside: {m.left is m.right = }")
  return m

m = f(m)
print(f"After:  {m.left is m.right = }")
```

Reference sharing is rare in most Machine Learning applications, however if it is required you can either use the `nnx.{split, merge, state, update}` APIs to move the deduplicated state and graph definiton across the JAX transforms:

```{code-cell} ipython3
m = Parent()
print(f"Before: {m.left is m.right = }")
graphdef, state = nnx.split(m)

@jax.jit
def f(graphdef, state):
  m = nnx.merge(graphdef, state)
  print(f"Inside: {m.left is m.right = }")
  return nnx.state(m)

state = f(graphdef, state)
nnx.update(m, state)

print(f"After:  {m.left is m.right = }")
print(f"{state = }") # deduplicated state
```

Or alternatively you can use the NNX transforms which preserve shared references:

```{code-cell} ipython3
m = Parent()
print(f"Before: {m.left is m.right = }")

@nnx.jit
def f(m):
  print(f"Inside: {m.left is m.right = }")
  return m

m = f(m)

print(f"After:  {m.left is m.right = }")
```

### Turning off pytree registration
`nnx.Pytree` allows you to turn off the pytree registration along with the attribute checks for subtypes by setting `pytree` type attribute option to `False`. This can be useful when upgrading to previous NNX code to newer Flax verions as you will still be able to use the NNX APIs or when creating types that should not be treated as pytree because e.g. they shared references.

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

## Module

+++

NNX Modules are `Pytree`s that have two additional methods for traking intermediate values: `sow` and `perturb`.

+++

### sow
`sow` receives a `Variable` type, a `name`, and a `value`, and stores it in the `Module` so it can be retrieved at a later time. As the following example shows, NNX APIs such as `nnx.state` or `nnx.pop` are a good way of retrieving the sowed state, however `pop` is recommended because it explicitly removes the temporary state from the Module.

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
    self.blocks = nnx.List([Block(dim, dim, rngs) for _ in range(num_layers)])

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

### perturb
`perturb` is similar to `sow` but it aims to capture the gradient of a value, currently this is a two step process although it might be simplified in the future:
1. Initialize the pertubation state by running the model once.
2. Pass the perturbation state as a differentiable target to `grad`.

As an example lets create a simple model and use `perturb` to get the intermediate gradient `xgrad` for the variable `x`, and initialize the perturbations:

```{code-cell} ipython3
import optax

class Model(nnx.Module):
  def __init__(self, rngs):
    self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    self.linear2 = nnx.Linear(3, 4, rngs=rngs)
  def __call__(self, x):
    x = nnx.gelu(self.linear1(x))
    x = self.perturb('xgrad', x)
    x = self.linear2(x)
    return x

rngs = nnx.Rngs(0)
model = Model(rngs)
optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-1), wrt=nnx.Param)
x, y = rngs.uniform((1, 2)), rngs.uniform((1, 4))
_ = model(x) # initialize perturbations
print(f"{nnx.state(model, nnx.Perturbation) = !s}")
```

Next we'll create a training step function that differentiates w.r.t. both the parameters of the model and the perturbations, the later will be the gradients for the intermediate values. `nnx.jit` and `nnx.value_and_grad` will be use to automatically propagate state updates. We'll return the `loss` function and the itermediate gradients.

```{code-cell} ipython3
@nnx.jit
def train_step(model, optimizer, x, y):
  graphdef, params, perturbations = nnx.split(model, nnx.Param, nnx.Perturbation)

  def loss_fn(params, perturbations):
    model = nnx.merge(graphdef, params, perturbations)
    return jnp.mean((model(x) - y) ** 2)

  loss, (grads, iterm_grads) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(params, perturbations)
  optimizer.update(model, grads)

  return loss, iterm_grads

for step in range(2):
  loss, iterm_grads = train_step(model, optimizer, x, y)
  print(f"{step = }, {loss = }, {iterm_grads = !s}")
```

## Object

+++

`Object` are NNX types that are **not** registered as JAX pytrees. Formally, any `Object` subclass is a `nnx.Pytree` with `pytree=False`.

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

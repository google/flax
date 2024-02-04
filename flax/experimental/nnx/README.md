[![codecov](https://codecov.io/gh/cgarciae/nnx/branch/main/graph/badge.svg?token=VqJjL474Z7)](https://codecov.io/gh/cgarciae/nnx)

# NNX

_**N**eural **N**etworks for JA**X**_

NNX is a JAX-based neural network library designed for simplicity and power. Its modular approach follows standard Python conventions, making it both intuitive and compatible with the broader JAX ecosystem.

* **Pythonic**: Modules are standard Python classes, promoting ease of use and a more familiar
  development experience.
* **Compatible**: Effortlessly convert between Modules and pytrees using the Functional API for maximum flexibility.
* **Control**: Manage a Module's state with precision using typed Variable collections, enabling fine-grained control
  on JAX transformations.
* **User-friendly**: NNX prioritizes simplicity for common use cases, building upon lessons learned from Linen
  to provide a streamlined experience.

#### Table of Contents
* [Installation](#installation)
* [Getting Started](#getting-started)
* [Examples](#examples)
* [FAQs](#faqs)
* [User Guide](#user-guide)

## Installation

To get started with `nnx`, install Flax from GitHub:
```
pip install git+https://github.com/google/flax.git
```

## Getting Started

The following example guides you through creating a basic `Linear` model with NNX and executing a forward pass. It also demonstrate how handle mutable state by showing how to keep track of the number of times the model has been called.

```python
from flax.experimental import nnx
import jax
import jax.numpy as jnp

class Count(nnx.Variable): pass # typed Variable collections

class Linear(nnx.Module):
  def __init__(self, din, dout, *, rngs: nnx.Rngs): # explicit RNG management
    key = rngs()
    # put dynamic state in Variable types
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = Count(0)
    # other types as treated as static
    self.din = din
    self.dout = dout

  def __call__(self, x):
    self.count += 1 # inplace stateful updates
    return x @ self.w + self.b

model = Linear(din=12, dout=2, rngs=nnx.Rngs(0)) # no special `init` method
x = jnp.ones((8, 12))
y = model(x) # call methods directly

assert model.count == 1
```

In this example `nnx.Rngs(0)` create a `random.key` for `params` with seed `0`, this is used by `rngs.<rng-name>()` inside `__init__` to generate a random key to initialize the parameters.

### Interacting with JAX

While NNX Modules inherently follow reference semantics, they can be easily converted into a pure functional representation that can be used with JAX transformations and other value-based, functional code.

NNX has two very simple APIs to interact with JAX: `split` and `merge`.

The `Module.split` method allows you to convert into a `State` dict-like object that contains the dynamic state of the Module, and a `ModuleDef` object that contains the static structure of the Module.

```python
state, static = model.split()
```
```
state = State({
  'b': Array(..., dtype=float32),
  'count': Array(1, dtype=int32),
  'w': Array(..., dtype=float32)
})
```

The `ModuleDef.merge` method allows you to take a `ModuleDef` and one or more `State` objects and merge them back into a `Module` object.

Using `split` and `merge` in conjunction allows you to carry your Module in and out of any JAX transformation. Here is a simple jitted `forward` function as an example:

```python
@jax.jit
def forward(static: nnx.ModuleDef, state: nnx.State, x: jax.Array):
  model = static.merge(state)
  y = model(x)
  state, _ = model.split()
  return y, state

x = jnp.ones((2, 4))
y, state = forward(static, state, x)
```
```
state["count"] = Array(2, dtype=int32)
```

For simple use cases, you can use `nnx.jit` which is a lifted transform that automatically splits, merges, and updates the outside Module for you:

```python
state, static = model.split()

@nnx.jit
def forward(model: Linear, x: jax.Array):
  return model(x)

y = forward(model, x=jnp.ones((2, 4)))

assert model.count == 3 # state automatically updated!
```

#### Training Example

Using `split` and `merge` (the [Functional API](#functional-api)) is the recommended way to use NNX as it provides tight control over the state, allows you to use regular JAX transformations, and it minimizes overhead. In this example we will create a simple training step that implements Stochastic Gradient Descent (SGD):

```python
params, counts, static = model.split(nnx.Param, Count)

@jax.jit
def train_step(params, counts, x, y):
  def loss_fn(params):
    model = static.merge(params, counts)
    y_pred = model(x)
    counts = model.extract(Count) # get updated Counts
    loss = jax.numpy.mean((y_pred - y) ** 2)
    return loss, counts

  # compute gradient
  grads, counts = jax.grad(loss_fn, has_aux=True)(params)
  # SGD update
  params = jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)

  return params, counts

# execute the training step
params, counts = train_step(params, counts, x, y)
model = static.merge(params, counts)
assert model.count == 4
```
Here `...` is a `Filter` (much like `nnx.Param`) that matches any node type, see the [Filters](#filters) section for more information.

#### Training with Lifted Transforms

[Lifted Transforms](#lifted-transforms) provide a convenient way interact with NNX Modules. In this example, we use the `nnx.jit` and `nnx.grad` lifted transforms to define the training step. The model is trained using Stochastic Gradient Descent (SGD). Because lifted transforms automatically update the Module's state, `train_step` doesn't require a return statement.

```python
@nnx.jit
def train_step(model, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jax.numpy.mean((y_pred - y) ** 2)

    # compute gradient
    grads: nnx.State = nnx.grad(loss_fn, wrt=nnx.Param)(model)
    # SGD update
    params, *_ = model.split(nnx.Param, ...)
    model.update(
        jax.tree_map(lambda w, g: w - 0.1 * g, params, grads)
    )

# execute the training step
train_step(model, x, y)
assert model.count == 3
```

**Note**: Using `nnx.jit` introduces some overhead when compared to using `jax.jit` directly. Use `nnx.jit` for simple prototypes, but for production code use `jax.jit` directly.

## Examples

* [Using the Functional API](https://github.com/cgarciae/nnx/blob/main/examples/01_functional_api.py): Shows how to train a simple model using the functional API.
* [Using Lifted Transforms](https://github.com/cgarciae/nnx/blob/main/examples/02_lifted_transforms.py): Shows how to train a simple model using lifted transforms.
* [Using TrainState](https://github.com/cgarciae/nnx/blob/main/examples/03_train_state.py): Shows how to train a simple model using the functional API with the help of `TrainState`.
* [Training a VAE](https://github.com/cgarciae/nnx/blob/main/examples/05_vae.py): Shows how to train a VAE on the binarized MNIST dataset, uses the functional API, `TrainState`, and shows how to use capture intermediate values to retrieve `kl_loss`.
* [Scan over layers](https://github.com/cgarciae/nnx/blob/main/examples/06_scan_over_layers.py): An contrived example that implements scan over layers with dropout and a share BatcNorm layer to showcase how lifted transforms can be implemented. It uses the functional API along with `jax.vmap` and `jax.lax.scan`.
* [Creating a Transformer](https://github.com/cgarciae/nnx/blob/main/examples/07_transformer.py): Shows how to create a Transformer with an auto-regressive decoder that uses scan over layers and a kv-cache for fast inference. Credits to @levskaya.

## FAQs

### Status
NNX is still in early development so expect bugs and breaking changes.

### How is it different from Flax?
NNX takes the best features that allow Flax to scale to large projects and integrates them into a much simpler Module system with pythonic semantics.

One place in which NNX strongly deviates from Flax is that (currently) it avoids shape inference in favor of static initialization. It is not a technical limitation but rather a design choice. This design both simplifies the internal implementation and makes it easier to reason about the code for the user, at the cost of being more verbose at times. On the other hand, Pytorch users will feel right at home.

## User Guide

### Modules

NNX Modules are normal python classes, they obey regular python semantics such as mutability and reference sharing, including reference cycles. They can contain 2 types of attributes: node attributes and static attributes. Node attributes include NNX `Variable`s (e.g. `nnx.Param`) and sub-Modules. All other types are treated as static attributes.

```python
class Foo(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    # node attributes
    self.param = nnx.Param(jnp.array(1))
    self.submodule = nnx.Linear(12, 3, rngs=rngs)
    self.container = [4, nnx.Linear(5, 6, rngs=rngs), 7]
    # static attributes
    self.int = 8
    self.float = 9.0
    self.str = 'hello'

  def __call__(self, x):
      return self.submodule(x + self.param)

  def some_method(self, x):
      return x + 1

model = Foo(rngs=nnx.Rngs(0))
```
As shown above, python container types such as `list`, `tuple`, and `dict` are treated as node attributes,
this means you can naturally have e.g. `list`s or `dict`s of Modules.

### Functional API

NNX Modules are not pytrees so they cannot be passed to JAX transformations. In order to interact with JAX, a Module must be partitioned into a `State` and `GraphDef` objects. The `State` object is a flat dictionary-like pytree structure that contains all the deduplicated node attributes, and the `GraphDef` contains the static attributes and structural information needed to reconstruct the Module.

```python
state, static = model.split()
```
```
State({
  'param': Array(1, dtype=int32, weak_type=True),
  'submodule': {
    'kernel': Array(..., dtype=float32),
    'bias': Array(..., dtype=float32)
  },
  'container': {
    '1': {
      'kernel': Array(..., dtype=float32),
      'bias': Array(..., dtype=float32)
    }
  }
})
```

`State` and `GraphDef` are pytrees so they can be passed to JAX transformations. More over, `GraphDef` provides 2 very important methods: `merge` and `apply`. The `merge` method can be used to create a new `Module` from a `State` object:

```python
model = static.merge(state)
```
This can be use to e.g. recreate a module inside a JAX transformation. The `apply` provides a functional interface to the module, it can be used call any method or submodule and get the output and the updated state:

```python
# run __call__
y, (state, static) = static.apply(state)(x)
# run some_method
y, (state, static) = static.apply(state).some_method(x)
# run submodule
y, (state, static) = static.apply(state).submodule(x)
```

`apply` can call any nested method or submodule as long as it can be accessed via the `.` or `[]` operators.

### Partitioning State
In NNX you can filter based on any node type, most commonly you will want to filter based on `nnx.Variable` subclasses such as `nnx.Param` or `nnx.BatchStat`.

Here are various examples of how you can use the `split` method to split a module into multiple substates:

```python
# split the module into the state with all the nodes and the static
state, static = model.split()
# verify that the state contains only params, else raise an error
params, static = model.split(nnx.Param)
# split the state into params and batch_stats, verify no nodes are left
params, batch_stats, static = model.split(nnx.Param, nnx.BatchStat)
# if there are any nodes left, use the `...` filter to capture them
params, batch_stats, rest, static = model.split(nnx.Param, nnx.BatchStat, ...)
# using `...` as the only filter is equivalent to not passing any filters
model.split(...) = model.split()
```
`split` will make sure all nodes are match by atleast one filter, else it will raise an error. You can use the `...` filter which will any (remaining) nodes. For a more general filter you can pass a predicate function that can use both the path and value of the node:

```python
(path: Tuple[str, ...], value: Any) -> bool
```
To reconstruct the module from a set of substates, you can use `merge` as usual but passing the substates as additional arguments:

```python
model = static.merge(params, batch_stats, rest)
```

The same is true for `apply`.

```python
y, (state, static) = static.apply(params, batch_stats, rest)(x)
```

 Note that `apply` will return a single `state` object, if you need to `split` the state you can use `State`'s own `split` method:

```python
params, batch_stats, rest = state.split(nnx.Param, nnx.BatchStat, ...)
```

Alternatively, if you are just interested in a subset of partitions, you can use the `State.extract` method which will not raise an error if some nodes are not matched by any filter:

```python
# only get params
params = state.extract(nnx.Param)
# get params and batch_stats
params, batch_stats = state.extract(nnx.Param, nnx.BatchStat)
```

### Filters

Filters let you select subsets of nodes based on some criteria. These are use throughout the API in methods like `split`, `extract`, and `pop`. There are 4 types of filters:

* `type`: matches all node instances of the given type.
* `...`: matches all nodes.
* `(path, any) -> bool`: a predicate function that takes a node path and value and returns a boolean.
* `Tuple[Filter, ...]`: a tuple of filters, matches all nodes that match any of the filters.

NNX also provides the following custom filters:

* `nnx.Not(filter)`: matches all nodes that do not match the given filter
* `nnx.All(*filters)`: matches nodes that match all filters

Here is an example of how to use `Not`:
```python
non_params = module.extract(nnx.Not(nnx.Param))
```


### Capturing Intermediate Values
In NNX you can easily propagate intemediate values by simply assigning them to an attribute at runtime. For convenience, you should assign them to a `Variable` attribute with a `collection` name by using `nnx.var` so you can easily retrieve them later.

Here is an example of how to create a `Linear` module that captures its output into a `Variable` attribute with the `intermediates` collection name:

```python
class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        y = x @ self.w + self.b
        self.y = nnx.Intermediate(y)
        return y

model = Linear(12, 2, rngs=nnx.Rngs(0))
```
Since `y` is only created when the module is called, it is not available upon initialization. However, once you call the module `y` will be created. It is recommended that you use `pop` to retrieve temporary collections like `Intermediate`:

```python
y = model(jnp.ones((8, 12)))
intermediates = model.pop(nnx.Intermediate)
```
`pop` will return a `State` object with the nodes that match the given filter and remove them from the module's attributes.

```
intermediates
```
```
State({
  'y: Intermediate(value=Array(...))
})
```

If you use the functional API to call the module instead, the `Intermediate` nodes will be present in the output `state`. To retrieve the `Intermediate` nodes and optionally separate them from the output `state` you can use `State.split`:

```python
state, static = model.split()
y, (state, static) = static.apply(state)(jnp.ones((8, 12)))
# "pop" the intermediates from the state
intermediates, state = state.split(nnx.Intermediate, ...)
```

Alternatively, you can use `State.extract` to retrieve the `Intermediate` nodes without removing them from the `state`.


### Lifted Transforms

NNX lifted transforms analogous versions of JAX transforms but they know how to work with Modules. They usually perform the following tasks:

* Handle the Module's substates and Rngs's RNG streams according to the transform's semantics.
* Properly propagating state in and out of the transform, including updating the input Module's state with updates that happen inside the transform.

Here's a diagram illustrating how lifted transformations work:

![lifted-transforms](https://raw.githubusercontent.com/cgarciae/nnx/main/docs/images/stateful-transforms.png)

Currently NNX provides the `jit`, `grad`, `scan`, and `remat`, lifted transforms.

#### Manual Lifting

In case you want to use JAX transforms directly you can always use the functional API
to manually lift your Modules.

Here we will create an example of how to implement an MLP that uses "scan over layers" to efficiently process a sequence of inputs assuming that each layer has the same parameters and input/output dimensions. The first thing we need to do is create a `Block` module that represents a single layer, this block with just contain a `Linear` layer, a `Dropout` layer, and a `GELU` activation function:

```python
class Block(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, train: bool, rngs: nnx.Rngs) -> jax.Array:
        x = self.linear(x)
        x = self.dropout(x, deterministic=not train, rngs=rngs)
        x = jax.nn.gelu(x)
        return x
```

Now we will define `ScanMLP`. During `__init__`, instead of creating a list of `Block`s, we will use `jax.vmap` to create a single `Block` whose parameters have an addtional `layer` axis. This will allow us to pass the parameters as inputs to scan so it will apply a layer at each step.

```python
class ScanMLP(nnx.Module):
    def __init__(self, dim: int, *, n_layers: int, rngs: nnx.Rngs):
        params_key = jax.random.split(rngs.params(), n_layers)
        self.n_layers = n_layers
        state, static = jax.vmap(
            lambda key: Block(dim, rngs=nnx.Rngs(params=key)).split()
        )(params_key)
        self.layers = static.merge(state)

```
Note that we split the `params` key into `n_layers` keys so each layer has different parameters.

Now we will define `__call__`. Here we need to split the `dropout` key into `n_layers` keys so each layer has a different dropout mask, and `split` the layers to get their `params`. Both `params` and `dropout_key` will be passed as inputs, `x` will be the carry value. Inside the `scan_fn` we will merge the `params` back into a `Block` module and
apply it to the input `x`, passing the sliced `dropout_key` as part of the `Rngs`.


```python
    def __call__(self, x: jax.Array, *, train: bool, rngs: nnx.Rngs) -> jax.Array:
        dropout_key = jax.random.split(rngs.dropout(), self.n_layers)
        params, static = self.layers.split(nnx.Param)

        def scan_fn(x, inputs):
            params, dropout_key = inputs
            module = static.merge(params)
            x = module(x, train=train, rngs=nnx.Rngs(dropout=dropout_key))
            return x, module.extract(nnx.Param)

        x, params = jax.lax.scan(scan_fn, x, (params, dropout_key))
        self.layers.update(params)
        return x
```
Finally we apply `jax.lax.scan`, update the `layers` state with the new `params`, and return the final `x` value.

Here is a simple way to test our `ScanMLP`:

```python
model = ScanMLP(10, n_layers=5, rngs=nnx.Rngs(0))

x = jnp.ones((3, 10))
y = model(x, train=True, rngs=nnx.Rngs(dropout=1))
```

For a more robust implementation with comments take a look at the [Scan over layers](https://github.com/cgarciae/nnx/blob/main/examples/06_scan_over_layers.py) example.

### Case Studies
#### Shared State

In NNX, you can create modules that share state between them. This is useful when designing complex neural network architectures, as it allows you to reuse certain layers and reduce the number of learnable parameters.

Here's an example of creating a module with shared state:

```python
class Block(nnx.Module):
    def __init__(self, linear: nnx.Linear, *, rngs: nnx.Rngs):
        self.linear = linear
        self.bn = nnx.BatchNorm(2, rngs=rngs)

    def __call__(self, x, *, rngs: nnx.Rngs):
        x = self.linear(x)
        x = self.bn(x, rngs=rngs)
        x = nnx.relu(x)
        return x

class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        shared = nnx.Linear(2, 2, rngs=rngs)
        self.block1 = Block(shared, rngs=rngs)
        self.block2 = Block(shared, rngs=rngs)

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
```

In this example, the `Model` module contains two instances of the `Block` module. Each instance shares the same `nnx.Linear` module. To run the model, you can use the Rngs `flags` argument to set the `use_running_average` flag for all `BatchNorm` modules.

Here's an example of computing the loss for a `Model` instance:

```python
def loss_fn(model: Model, x: jax.Array, y: jax.Array):
    with nnx.flags(use_running_average=True):
        y_pred = model(x)
    return jnp.mean((y - y_pred) ** 2)
```

It's important to note that the state for the shared `nnx.Linear` module will be kept in sync at all times on both `Block` instances, including during gradient updates.

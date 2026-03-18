# Tree Mode NNX

Mar 4, 2026
Cristian Garcia, Samuel Anklesaria, Flax Team

## Motivation

Current NNX APIs allow general graph structures and graph transformations, this includes:

1. Tracking Variable state updates  
2. Handling shared references (graphs)  
3. Supporting prefix filters (StateAxes, DiffState, StateSharding)  
4. Propagating graph updates (static state and structure changes)

While powerful, some of these capabilities (**3** and **4**) are beyond what JAX transform APIs offer and supporting them results in both internal complexity, harder to reason about code, and a larger set of APIs a user must learn.  We wish to tackle all these issues by simplifying NNX.

## Proposal

To do this we propose two things. First, the introduction of **Tree Mode NNX**: a reimplementation of the NNX APIs that only handles trees, assumes referential transparency, and has a more limited support for state updates. Concretely, this means:

* Automatic state updates only for Variables in NNX transforms.  
* Tree structure assumed and enforced on all APIs (no sharing)  
* Modules treated as stateless pytrees (no graph updates).  
* Full JAX transform compatibility (remove [prefix filters](#prefix-filters): StateAxes, DiffState, StateSharding).

Second, simplifying graph support. Graphs stand out as an important feature for some NNX users. However, we will be limiting support to **1** and **2**, meaning that prefix filters and graph updates will be dropped. This will make it such that tree and graph transforms can share the same underlying implementation and semantics while still allowing for a great deal of expressivity.

## Implementation

Tree mode will be implemented on top of the current APIs by introducing a `graph` argument, when `True` graph support is enabled, when `False` only trees are supported and internals rely on `jax.tree.*` APIs. Additionally, a `graph_updates` argument will be added to NNX transforms, when `False` transforms will no longer propagate graph structure update (**4**) or support prefix filters (**3**).

```py
def split(..., graph: bool | None = None)
...
def jit(..., graph: bool | None = None, graph_updates: bool | None = None)
...
```

If `graph` or `graph_updates` are not provided, their default values will be taken from the `nnx_graph_mode` and `nnx_graph_updates` config flags respectively. These can be easily fetched and updated via `set_graph_mode` and `set_graph_updates`.

```py
# status
print(nnx.set_graph_mode.current_value())
print(nnx.set_graph_updates.current_value())

# set value
nnx.set_graph_mode(True/False)
nnx.set_graph_updates(True/False)

# via env vars 
# NNX_GRAPH_MODE=true/false
# NNX_GRAPH_UPDATES=true/false

# context managers
with nnx.set_graph_mode(True/False):
  ...
with nnx.set_graph_updates(True/False):
  ...
```

The goal will be to have the default value for `nnx_graph_mode` and `nnx_graph_updates` to be set to `False`, thus enabling tree mode for new projects. Users that don’t want to migrate can use these flags to make sure their code continues to work with current features.

### Simple transforms

These new transforms are highly simplified compared to current transforms, they are easier to implement and optimize, while supporting both trees and graphs. Given a user function f, most simplified transforms follow this pattern:

```py
def transform_wrapper(*args):
  if graph: args = to_tree(args)
  check_no_aliases(args=args)
  
  @jax_transform
  def transformed_f(*args):
    updates, snapshot = updates_and_snapshot(args)
    if graph: args = from_tree(args)
    out = f(*args)
    if graph: out = to_tree(out)
    check_no_aliases(args=updates, out=out)
    updates = mask_variable_updates(updates, snapshot)
    return out, updates
  
  out, updates = transformed_f(*args)
  apply_variable_updates(args, updates)
  if graph: out = from_tree(out)
  return out
```

The transformed function tracks input Variable `updates`, applies  f, and masks Variable updates (no updates for Variables that didn’t change). It also checks that there are no Variable aliases between the inputs and outputs (no shared references), and returns the user output plus Variable updates. The wrapper function calls the transformed function, applies the Variable updates to the input Variables, and returns the user output. To support graphs, we simply convert objects to a tree representation before passing them to jax, and back to graphs before passing them to the user code.

## Backward Compatibility

When tree mode is on by default, code that relies on graphs, graph updates, and prefix filters will stop working. There are two ways to port existing code, the first is reverting the defaults config via `set_graph_mode` and `set_graph_updates` somewhere in the after the imports:

```py
from flax import nnx
...
nnx.set_graph_mode(True)
nnx.set_graph_updates(True)
```

The previous implementation of the transform APIs will also be accessible via the `nnx.compat` module. They are implemented as partials that set `graph=True` and `graph_updates=True`:

```py
nnx.compat.split = partial(nnx.split, graph=True)
...
nnx.compat.jit = partial(nnx.jit, graph=True, graph_updates=True)
...
```

   
The above shortcuts will make it such that porting existing code (if needed) is as simple as performing some rewrites:

`nnx.split` → `nnx.compat.split`  
`nnx.jit` → `nnx.compat.jit`  
…

## Breaking changes

### Prefix filters {#prefix-filters}

Code that relies on prefix filters such as StateAxes, StateSharding, and DiffState will require some restructuring as JAX has no equivalent mechanisms (these were added to make Linen migration easier). The solution is to use `split` and `merge` to create state groups, and pass each group through their corresponding tree prefix on the jax transform. For example:

```py
# previous code
state_axes = nnx.StateAxes({some_filter: 0, ...: None})

@nnx.vmap(in_axis=state_axes, graph=True, graph_updates=True)
def f(model):
  ...
```

This can be rewritten to `split` the model into two state groups using the previous filter, passing the groups as separate arguments, one vectorized and the other broadcasted, and using `merge` to reconstruct the model inside the transform.

```py
# new code
graphdef, vectorized, broadcasted = nnx.split(model, some_filter, ...)

@nnx.vmap(in_axis=(0, None))
def f(vectorized, broadcasted):
  model = nnx.merge(graphdef, vectorized, broadcasted)
  ...
```

This is roughly how prefix filters were implemented under the hood.

### nnx.grad

Code that uses `nnx.grad` will change in two ways:

1. The first argument will no longer be differentiated w.r.t. to `Param`s only, this is because `grad` used this prefix filter by default: `DiffState(0, Param)`.  
2. The gradients of NNX Pytree/Module types will no longer be `State` types. Now they just follow JAX and return the same input type.

Concretely it means that code like this:

```py
# previous code
def loss_fn(model: Foo):
  ...

# uses argnums=nnx.DiffState(0, nnx.Param)
grads = nnx.grad(loss_fn)(model)
```

Now has to explicitly use `split` and `merge` if to avoid calculating gradients for the non-differentiable state:

```py
# new code
def graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

def loss_fn(params, nondiff):
  model = nnx.merge(graphdef, params, nondiff)
  ...

# uses argnums=0
grads = nnx.grad(loss_fn)(params, nondiff)
```

If there is no non-differentiable the `model` can be passed in directly but the gradients will now be of the same type:

```py
# new code
def loss_fn(model: Foo):
  ...

# uses argnums=0
grads: Foo = nnx.grad(loss_fn)(model)
```

### nnx.custom_vjp

Previously `nnx.custom_vjp` did two particular things:

1. The backward function returned the gradients of the Variable updates (`m_updates_g`) along with the output gradient.  
2. The tangent for nnx.Pytree/Module objects were of type `nnx.State`.

For a `Foo` Module with `x: Param` and `y: Param` attributes, a simple example could look like this:

```py
# previous code
@nnx.custom_vjp
def f(m: Foo):
  return jnp.sin(m.x) * m.y

def f_fwd(m: Foo):
  return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)

def f_bwd(res, g):
  (m_updates_g,), out_g = g
  cos_x, sin_x, m = res
  m_g: nnx.State = nnx.clone(m_updates_g) # create copy
  m_g['x'][...] = cos_x * out_g * m.y
  m_g['y'][...] = sin_x * out_g
  return (m_g,)  # State gradient
```

In the new implementation gradients for Variable updates are not returned, and the tangent type is the same as the input type (`Foo`), this matches the behavior of `jax.custom_vjp`: 

```py
# new code
@nnx.custom_vjp
def f(m: Foo):
  return jnp.sin(m.x) * m.y

def f_fwd(m: Foo):
  return f(m), (jnp.cos(m.x), jnp.sin(m.x), m)

def f_bwd(res, g): # no gradients for updates
  cos_x, sin_x, m = res
  m_g: Foo = nnx.clone(m) # create copy
  m_g.x[...] = cos_x * g * m.y
  m_g.y[...] = sin_x * g
  return (m_g,) # Foo gradient
```

Note that to avoid losing information, now differentiable Variables are not allowed to be updated inside `nnx.custom_vjp`.

### transform\_metadata

Previously NNX transforms like `vmap` and `scan` had a `transform_metadata` metadata argument that allowed them to update the sharding metadata.

```py
# old code
@nnx.split_rngs(8)
@nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: 'din'})
class create_stack(rngs):  # 'din' added to out_sharding metadata
  return nnx.Variable(rngs.uniform((16,)), out_sharding=('dout',))

v_stack = create_stack(nnx.Rngs(0))
assert v_stack.shape == (8, 16)
assert v_stack.out_shardings == ('din', 'dout')
```

The new simplified NNX transform implementations don’t support this argument. However, to keep supporting the behavior, a new `nnx.transform_metadata` transform is introduced that can be inserted to get back the same results. TODO: mention it works on `jax.vmap`.

```py
# new code
@nnx.split_rngs(8)
@nnx.vmap(in_axes=0, out_axes=0)
@nnx.transform_metadata(in_axes=0, out_axes=0, partition='din')
class create_stack(rngs):  # 'din' added to out_sharding metadata
  return nnx.Variable(rngs.uniform((16,)), out_sharding=('dout',))

v_stack = create_stack(nnx.Rngs(0))
assert v_stack.shape == (8, 16)
assert v_stack.out_shardings == ('din', 'dout')
```

`transform_metada` accepts `in_axes` and `out_axes`, these should match the values passed to the corresponding transform.

### Module.sow

Previously, `Module.sow` used graph updates to capture intermediate values during computations and propagate them outside, it was used in conjunction with `nnx.pop` to log and extract intermediates:

```py
# old code
class Foo(nnx.Module):
  def __call__(self, x):
    self.sow(nnx.Intermediate, "y_mean", jnp.mean(x))
    return x

model = Foo()
result = model(x)
intermediates = nnx.pop(model, nnx.Intermediate) # extract intermediate values
```

To achieve the same without graph updates we’ve added a new `nnx.capture` API which allows for a similar workflow.

```py
# New Code
class Foo(nnx.Module):
  def __call__(self, x):
    self.sow(nnx.Intermediate, "y_mean", jnp.mean(x))
    return x

model = Foo()
result, intermediates = nnx.capture(model, nnx.Intermediate)(x)
```

In general, `nnx.capture` takes a function or Module to be transformed, a `nnx.Variable` subclass to collect, and an optional `init` argument to initialize the collected state, which will be stored within `nnx.Variable` objects. `nnx.capture` creates a `__captures__: tuple[Variable, ...]` attribute on each `Module` instance, each Variable in `__captures__` contains a dictionary which `sow` and `perturb` populate.

### Module.perturb

Similarly, `Module.perturb` was previously used to extract the gradients of intermediate values. This was done in two steps: initializing a perturbation state by running a module once, and then passing the perturbation state as a differentiable target to `grad`.

```py
class Model(nnx.Module):
  def __call__(self, x):
    x = self.perturb('grad_of_x', x)
    ...
    return y

# old code
@nnx.jit
def train_step(model, optimizer, x, y):
  model(x) # Initialize perturbation state
  def loss_fn(model):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)
  diff_state = nnx.DiffState(0, (nnx.Param, nnx.Perturbation))
  grads = nnx.grad(loss_fn, argnums=diff_state)(model)
  grads, interm_grads = nnx.state(grads, nnx.Param, nnx.Perturbation)
  optimizer.update(model, grads)
  nnx.pop(model, nnx.Perturbation) # clean up perturbations
  return interm_grads
```

Similar pattern can be used with  `nnx.capture` during both perturbation initialization and when running the forward pass to insert the differentiable perturbations state. In this version explicitly pass the `perturbs` state as a separate argument and use `argnums` to specify that both arguments are differentiable:

```py
# new code
@nnx.jit
def train_step(model, optimizer, x, y):
  _, perturbs = nnx.capture(model, nnx.Perturbation)(x) # init perturbations
  def loss_fn(model, perturbs):
    y_pred = nnx.capture(model, init=perturbs)(x)
    return jnp.mean((y_pred - y) ** 2)
  grads, interm_grads = nnx.grad(loss_fn, argnums=(0, 1))(model, perturbs)
  optimizer.update(model, grads)
  return interm_grads
```

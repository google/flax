---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Use Flax NNX and Linen together

This guide is for existing Flax users who want to make their codebase a mixture of Flax Linen and Flax NNX `Module`s, which is made possible thanks to the `flax.nnx.bridge` API.

This will be helpful if you:

* Want to migrate your codebase to NNX gradually, one module at a time;
* Have external dependency that already moved to NNX but you haven't, or is still in Linen while you've moved to NNX.

We hope this allows you to move and try out NNX at your own pace, and leverage the best of both worlds. We will also talk about how to resolve the caveats of interoperating the two APIs, on a few aspects that they are fundamentally different.

**Note**:

This guide is about glueing Linen and NNX modules. To migrate an existing Linen module to NNX, check out the [Migrate from Flax Linen to Flax NNX](https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html) guide.

And all built-in Linen layers should have equivalent NNX versions! Check out the list of [Built-in NNX layers](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/index.html).

```{code-cell} ipython3
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

from flax import nnx
from flax import linen as nn
from flax.nnx import bridge
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from typing import *
```

## Submodule is all you need

A Flax model is always a tree of modules - either old Linen modules (`flax.linen.Module`, usually written as `nn.Module`) or NNX modules (`nnx.Module`).

An `nnx.bridge` wrapper glues the two types together, in both ways:

* `nnx.bridge.ToNNX`: Convert a Linen module to NNX, so that it can be a submodule of another NNX module, or stand alone to be trained in NNX-style training loops.
* `nnx.bridge.ToLinen`: Vice versa, convert a NNX module to Linen.

This means you can move in either top-down or bottom-up behavior: convert the whole Linen module to NNX, then gradually move down, or convert all the lower level modules to NNX then move up.

+++

## The Basics

There are two fundamental difference between Linen and NNX modules:

* **Stateless vs. stateful**: Linen module instances are stateless: variables are returned from a purely functional `.init()` call and managed separately. NNX modules, however, owns its variables as instance attributes.

* **Lazy vs. eager**: Linen modules only allocate space to create variables when they actually see their input. Whereas NNX module instances create their variables the moment they are instantiated, without seeing a sample input.

With that in mind, let's look at how the `nnx.bridge` wrappers tackle the differences.

### Linen -> NNX

Since Linen modules may require an input to create variables, we semi-formally supported lazy initialization in the NNX modules converted from Linen. The Linen variables are created when you give it a sample input.

For you, it's calling `nnx.bridge.lazy_init()` where you call `module.init()` in Linen code.

(Note: you can call `nnx.display` upon any NNX module to inspect all its variables and state.)

```{code-cell} ipython3
class LinenDot(nn.Module):
  out_dim: int
  w_init: Callable[..., Any] = nn.initializers.lecun_normal()
  @nn.compact
  def __call__(self, x):
    # Linen might need the input shape to create the weight!
    w = self.param('w', self.w_init, (x.shape[-1], self.out_dim))
    return x @ w

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.ToNNX(LinenDot(64),
                     rngs=nnx.Rngs(0))  # => `model = LinenDot(64)` in Linen
bridge.lazy_init(model, x)              # => `var = model.init(key, x)` in Linen
y = model(x)                            # => `y = model.apply(var, x)` in Linen

nnx.display(model)

# In-place swap your weight array and the model still works!
model.w.value = jax.random.normal(jax.random.key(1), (32, 64))
assert not jnp.allclose(y, model(x))
```

`nnx.bridge.lazy_init` also works even if the top-level module is a pure-NNX one, so you can do sub-moduling as you wish:

```{code-cell} ipython3
class NNXOuter(nnx.Module):
  def __init__(self, out_dim: int, rngs: nnx.Rngs):
    self.dot = nnx.bridge.ToNNX(LinenDot(out_dim), rngs=rngs)
    self.b = nnx.Param(jax.random.uniform(rngs.params(), (1, out_dim,)))

  def __call__(self, x):
    return self.dot(x) + self.b

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.lazy_init(NNXOuter(64, rngs=nnx.Rngs(0)), x)  # Can fit into one line
nnx.display(model)
```

The Linen weight is already converted to a typical NNX variable, which is a thin wrapper of the actual JAX array value within. Here, `w` is an `nnx.Param` because it belongs to the `params` collection of `LinenDot` module.

We will talk more about different collections and types in the [NNX Variable <-> Linen Collections](#variable-types-vs-collections) section. Right now, just know that they are converted to NNX variables like native ones.

```{code-cell} ipython3
assert isinstance(model.dot.w, nnx.Param)
assert isinstance(model.dot.w.value, jax.Array)
```

If you create this model witout using `nnx.bridge.lazy_init`, the NNX variables defined outside will be initialized as usual, but the Linen part (wrapped inside `ToNNX`) will not.

```{code-cell} ipython3
partial_model = NNXOuter(64, rngs=nnx.Rngs(0))
nnx.display(partial_model)
```

```{code-cell} ipython3
full_model = bridge.lazy_init(partial_model, x)
nnx.display(full_model)
```

### NNX -> Linen

To convert an NNX module to Linen, you should forward your creation arguments to `bridge.ToLinen` and let it handle the actual creation process.

This is because NNX module instance initializes all its variables eagerly when it is created, which consumes memory and compute. On the other hand, Linen modules are stateless, and the typical `init` and `apply` process involves multiple creation of them. So `bridge.to_linen` will handle the actual module creation and make sure no memory is allocated twice.

```{code-cell} ipython3
class NNXDot(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
    self.w = nnx.Param(nnx.initializers.lecun_normal()(
      rngs.params(), (in_dim, out_dim)))
  def __call__(self, x: jax.Array):
    return x @ self.w

x = jax.random.normal(jax.random.key(42), (4, 32))
# Pass in the arguments, not an actual module
model = bridge.to_linen(NNXDot, 32, out_dim=64)
variables = model.init(jax.random.key(0), x)
y = model.apply(variables, x)

print(list(variables.keys()))
print(variables['params']['w'].shape)  # => (32, 64)
print(y.shape)                         # => (4, 64)
```

`bridge.to_linen` is actually a convenience wrapper around the Linen module `bridge.ToLinen`. Most likely you won't need to use `ToLinen` directly at all, unless you are using one of the built-in arguments of `ToLinen`. For example, if your NNX module doesn't want to be initialized with RNG handling:

```{code-cell} ipython3
class NNXAddConstant(nnx.Module):
  def __init__(self):
    self.constant = nnx.Variable(jnp.array(1))
  def __call__(self, x):
    return x + self.constant

# You have to use `skip_rng=True` because this module's `__init__` don't
# take `rng` as argument
model = bridge.ToLinen(NNXAddConstant, skip_rng=True)
y, var = model.init_with_output(jax.random.key(0), x)
```

Similar to `ToNNX`, you can use `ToLinen` to create a submodule of another Linen module.

```{code-cell} ipython3
class LinenOuter(nn.Module):
  out_dim: int
  @nn.compact
  def __call__(self, x):
    dot = bridge.to_linen(NNXDot, x.shape[-1], self.out_dim)
    b = self.param('b', nn.initializers.lecun_normal(), (1, self.out_dim))
    return dot(x) + b

x = jax.random.normal(jax.random.key(42), (4, 32))
model = LinenOuter(out_dim=64)
y, variables = model.init_with_output(jax.random.key(0), x)
w, b = variables['params']['ToLinen_0']['w'], variables['params']['b']
print(w.shape, b.shape, y.shape)
```

## Handling RNG keys

All Flax modules, Linen or NNX, automatically handle the RNG keys for variable creation and random layers like dropouts. However, the specific logics of RNG key splitting are different, so you cannot generate the same params between Linen and NNX modules, even if you pass in same keys.

Another difference is that NNX modules are stateful, so they can track and update the RNG keys within themselves.

### Linen to NNX

If you convert a Linen module to NNX, you enjoy the stateful benefit and don't need to pass in extra RNG keys on every module call. You can use always `nnx.reseed` to reset the RNG state within.

```{code-cell} ipython3
x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.ToNNX(nn.Dropout(rate=0.5, deterministic=False), rngs=nnx.Rngs(dropout=0))
# We don't really need to call lazy_init because no extra params were created here,
# but it's a good practice to always add this line.
bridge.lazy_init(model, x)
y1, y2 = model(x), model(x)
assert not jnp.allclose(y1, y2)  # Two runs yield different outputs!

# Reset the dropout RNG seed, so that next model run will be the same as the first.
nnx.reseed(model, dropout=0)
y1 = model(x)
nnx.reseed(model, dropout=0)
y2 = model(x)
assert jnp.allclose(y1, y2)  # Two runs yield the same output!
```

### NNX to Linen

`to_linen` will automatically take the `rngs` dict argument and create a `Rngs` object that is passed to the underlying NNX module via the `rngs` keyword argument. If the module holds internal `RngState`, `to_linen` will always call reseed using the `rngs` dict to reset the RNG state.

```{code-cell} ipython3
x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.to_linen(nnx.Dropout, rate=0.5)
variables = model.init({'dropout': jax.random.key(0)}, x)

# Just pass different RNG keys for every `apply()` call.
y1 = model.apply(variables, x, rngs={'dropout': jax.random.key(1)})
y2 = model.apply(variables, x, rngs={'dropout': jax.random.key(2)})
assert not jnp.allclose(y1, y2)  # Every call yields different output!
y3 = model.apply(variables, x, rngs={'dropout': jax.random.key(1)})
assert jnp.allclose(y1, y3)      # When you use same top-level RNG, outputs are same
```

## NNX variable types vs. Linen collections

When you want to group some variables as one category, in Linen you use different collections; in NNX, since all variables shall be top-level Python attributes, you use different variable types.

Therefore, when mixing Linen and NNX modules, Flax must know the 1-to-1 mapping between Linen collections and NNX variable types, so that `ToNNX` and `ToLinen` can do the conversion automatically.

Flax keeps a registry for this, and it already covers all Flax's built-in Linen collections. You can register extra mapping of NNX variable type and Linen collection names using `nnx.register_variable_name_type_pair`.

### Linen to NNX

For any collection of your Linen module, `ToNNX` will convert all its endpoint arrays (aka. leaves) to a subtype of `nnx.Variable`, either from registry or automatically created on-the-fly.

(However, we still keep the whole collection as one class attribute, because Linen modules may have duplicated names over different collections.)

```{code-cell} ipython3
class LinenMultiCollections(nn.Module):
  out_dim: int
  def setup(self):
    self.w = self.param('w', nn.initializers.lecun_normal(), (x.shape[-1], self.out_dim))
    self.b = self.param('b', nn.zeros_init(), (self.out_dim,))
    self.count = self.variable('counter', 'count', lambda: jnp.zeros((), jnp.int32))

  def __call__(self, x):
    if not self.is_initializing():
      self.count.value += 1
    y = x @ self.w + self.b
    self.sow('intermediates', 'dot_sum', jnp.sum(y))
    return y

x = jax.random.normal(jax.random.key(42), (2, 4))
model = bridge.lazy_init(bridge.ToNNX(LinenMultiCollections(3), rngs=nnx.Rngs(0)), x)
print(model.w)        # Of type `nnx.Param` - note this is still under attribute `params`
print(model.b)        # Of type `nnx.Param`
print(model.count)    # Of type `counter` - auto-created type from the collection name
print(type(model.count))

y = model(x, mutable=True)  # Linen's `sow()` needs `mutable=True` to trigger
print(model.dot_sum)        # Of type `nnx.Intermediates`
```

You can quickly separate different types of NNX variables apart using `nnx.split`.

This can be handy when you only want to set some variables as trainable.

```{code-cell} ipython3
# Separate variables of different types with nnx.split
CountType = type(model.count)
static, params, counter, the_rest = nnx.split(model, nnx.Param, CountType, ...)
print('All Params:', list(params.keys()))
print('All Counters:', list(counter.keys()))
print('All the rest (intermediates and RNG keys):', list(the_rest.keys()))

model = nnx.merge(static, params, counter, the_rest)  # You can merge them back at any time
y = model(x, mutable=True)  # still works!
```

    All Params: ['b', 'w']
    All Counters: ['count']
    All the rest (intermediates and RNG keys): ['dot_sum', 'rngs']

+++

### NNX to Linen

If you define custom NNX variable types, you should register their names with `nnx.register_variable_name` so that they go to the desired collections.

```{code-cell} ipython3
@nnx.register_variable_name('counts', overwrite=True)
class Count(nnx.Variable): pass


class NNXMultiCollections(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (din, dout)))
    self.lora = nnx.LoRA(din, 3, dout, rngs=rngs)
    self.count = Count(jnp.array(0))

  def __call__(self, x):
    self.count += 1
    return (x @ self.w.value) + self.lora(x)

xkey, pkey, dkey = jax.random.split(jax.random.key(0), 3)
x = jax.random.normal(xkey, (2, 4))
model = bridge.to_linen(NNXMultiCollections, 4, 3)
var = model.init({'params': pkey, 'dropout': dkey}, x)
print('All Linen collections:', list(var.keys()))
print(var['params'])
```

    All Linen collections: ['LoRAParam', 'params', 'counts']
    {'w': Array([[ 0.2916921 ,  0.22780475,  0.06553137],
           [ 0.17487915, -0.34043145,  0.24764155],
           [ 0.6420431 ,  0.6220095 , -0.44769976],
           [ 0.11161668,  0.83873135, -0.7446058 ]], dtype=float32)}

+++

## Partition metadata

Flax uses a metadata wrapper box over the raw JAX array to annotate how a variable should be sharded.

In Linen, this is an optional feature that triggered by using `nn.with_partitioning` on initializers (see more on [Linen partition metadata guide](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html)). In NNX, since all NNX variables are wrapped by `nnx.Variable` class anyway, that class will hold the sharding annotations too.

The `bridge.ToNNX` and `bridge.ToLinen` API will automatically convert the sharding annotations, if you use the built-in annotation methods (aka. `nn.with_partitioning` for Linen and `nnx.with_partitioning` for NNX).

### Linen to NNX

Even if you are not using any partition metadata in your Linen module, the variable JAX arrays will be converted to `nnx.Variable`s that wraps the true JAX array within.

If you use `nn.with_partitioning` to annotate your Linen module's variables, the annotation will be converted to a `.sharding` field in the corresponding `nnx.Variable`.

You can then use `nnx.with_sharding_constraint` to explicitly put the arrays into the annotated partitions within a `jax.jit`-compiled function, to initialize the whole model with every array at the right sharding.

```{code-cell} ipython3
class LinenDotWithPartitioning(nn.Module):
  out_dim: int
  @nn.compact
  def __call__(self, x):
    w = self.param('w', nn.with_partitioning(nn.initializers.lecun_normal(),
                                             ('in', 'out')),
                   (x.shape[-1], self.out_dim))
    return x @ w

@nnx.jit
def create_sharded_nnx_module(x):
  model = bridge.lazy_init(
    bridge.ToNNX(LinenDotWithPartitioning(64), rngs=nnx.Rngs(0)), x)
  state = nnx.state(model)
  sharded_state = nnx.with_sharding_constraint(state, nnx.get_partition_spec(state))
  nnx.update(model, sharded_state)
  return model


print(f'We have {len(jax.devices())} fake JAX devices now to partition this model...')
mesh = jax.sharding.Mesh(devices=mesh_utils.create_device_mesh((2, 4)),
                         axis_names=('in', 'out'))
x = jax.random.normal(jax.random.key(42), (4, 32))
with mesh:
  model = create_sharded_nnx_module(x)

print(type(model.w))           # `nnx.Param`
print(model.w.sharding)        # The partition annotation attached with `w`
print(model.w.value.sharding)  # The underlying JAX array is sharded across the 2x4 mesh
```

    We have 8 fake JAX devices now to partition this model...
    <class 'flax.nnx.variables.Param'>
    ('in', 'out')
    GSPMDSharding({devices=[2,4]<=[8]})

+++

### NNX to Linen

If you are not using any metadata feature of the `nnx.Variable` (i.e., no sharding annotation, no registered hooks), the converted Linen module will not add a metadata wrapper to your NNX variable, and you don't need to worry about it.

But if you did add sharding annotations to your NNX variables, `ToLinen` will convert them to a default Linen partition metadata class called `bridge.NNXMeta`, retaining all the metadata you put into the NNX variable.

Like with any Linen metadata wrappers, you can use `linen.unbox()` to get the raw JAX array tree.

```{code-cell} ipython3
class NNXDotWithParititioning(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
    init_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), ('in', 'out'))
    self.w = nnx.Param(init_fn(rngs.params(), (in_dim, out_dim)))
  def __call__(self, x: jax.Array):
    return x @ self.w

x = jax.random.normal(jax.random.key(42), (4, 32))

@jax.jit
def create_sharded_variables(key, x):
  model = bridge.to_linen(NNXDotWithParititioning, 32, 64)
  variables = model.init(key, x)
  # A `NNXMeta` wrapper of the underlying `nnx.Param`
  assert type(variables['params']['w']) == bridge.NNXMeta
  # The annotation coming from the `nnx.Param` => (in, out)
  assert variables['params']['w'].metadata['sharding'] == ('in', 'out')

  unboxed_variables = nn.unbox(variables)
  variable_pspecs = nn.get_partition_spec(variables)
  assert isinstance(unboxed_variables['params']['w'], jax.Array)
  assert variable_pspecs['params']['w'] == jax.sharding.PartitionSpec('in', 'out')

  sharded_vars = jax.tree.map(jax.lax.with_sharding_constraint,
                              nn.unbox(variables),
                              nn.get_partition_spec(variables))
  return sharded_vars

with mesh:
  variables = create_sharded_variables(jax.random.key(0), x)

# The underlying JAX array is sharded across the 2x4 mesh
print(variables['params']['w'].sharding)
```

    GSPMDSharding({devices=[2,4]<=[8]})

+++

## Lifted transforms

In general, if you want to apply Linen/NNX-style lifted transforms upon an `nnx.bridge`-converted module, just go ahead and do it in the usual Linen/NNX syntax.

For Linen-style transforms, note that `bridge.ToLinen` is the top level module class, so you may want to just use it as the first argument of your transforms (which needs to be a `linen.Module` class in most cases)

### Linen to NNX

NNX style lifted transforms are similar to JAX transforms, and they work on functions.

```{code-cell} ipython3
class NNXVmapped(nnx.Module):
  def __init__(self, out_dim: int, vmap_axis_size: int, rngs: nnx.Rngs):
    self.linen_dot = nnx.bridge.ToNNX(nn.Dense(out_dim, use_bias=False), rngs=rngs)
    self.vmap_axis_size = vmap_axis_size

  def __call__(self, x):

    @nnx.split_rngs(splits=self.vmap_axis_size)
    @nnx.vmap(in_axes=(0, 0), axis_size=self.vmap_axis_size)
    def vmap_fn(submodule, x):
      return submodule(x)

    return vmap_fn(self.linen_dot, x)

x = jax.random.normal(jax.random.key(0), (4, 32))
model = bridge.lazy_init(NNXVmapped(64, 4, rngs=nnx.Rngs(0)), x)

print(model.linen_dot.kernel.shape) # (4, 32, 64) - first axis with dim 4 got vmapped
y = model(x)
print(y.shape)
```

    (4, 32, 64)
    (4, 64)

+++

### NNX to Linen

Note that `bridge.ToLinen` is the top level module class, so you may want to just use it as the first argument of your transforms (which needs to be a `linen.Module` class in most cases).

`ToLien` can naturally be used with Linen transforms like `nn.vmap` or `nn.scan`.

```{code-cell} ipython3
class LinenVmapped(nn.Module):
  dout: int
  @nn.compact
  def __call__(self, x):
    inner = nn.vmap(bridge.ToLinen, variable_axes={'params': 0}, split_rngs={'params': True}
                    )(nnx.Linear, args=(x.shape[-1], self.dout))
    return inner(x)

x = jax.random.normal(jax.random.key(42), (4, 32))
model = LinenVmapped(64)
var = model.init(jax.random.key(0), x)
print(var['params']['VmapToLinen_0']['kernel'].shape)  # (4, 32, 64) - leading dim 4 got vmapped
y = model.apply(var, x)
print(y.shape)
```

    (4, 32, 64)
    (4, 64)

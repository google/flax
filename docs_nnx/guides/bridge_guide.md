# Use Flax NNX along with Flax Linen

This guide is for existing Flax users who want to make their codebase a mixture of Flax Linen and Flax NNX `Module`s, which is made possible thanks to the `flax.nnx.bridge` API.

This will be helpful if you:

* Want to migrate your codebase to NNX gradually, one module at a time;
* Have external dependency that already moved to NNX but you haven't, or is still in Linen while you've moved to NNX.

We hope this allows you to move and try out NNX at your own pace, and leverage the best of both worlds. We will also talk about how to resolve the caveats of interoperating the two APIs, on a few aspects that they are fundamentally different.

**Note**:

This guide is about glueing Linen and NNX modules. To migrate an existing Linen module to NNX, check out the [Migrate from Flax Linen to Flax NNX](https://flax.readthedocs.io/en/latest/nnx/haiku_linen_vs_nnx.html) guide.

And all built-in Linen layers should have equivalent NNX versions! Check out the list of [Built-in NNX layers](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/index.html).


```python
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


## The Basics

There are two fundamental difference between Linen and NNX modules:

* **Stateless vs. stateful**: Linen module instances are stateless: variables are returned from a purely functional `.init()` call and managed separately. NNX modules, however, owns its variables as instance attributes.

* **Lazy vs. eager**: Linen modules only allocate space to create variables when they actually see their input. Whereas NNX module instances create their variables the moment they are instantiated, without seeing a sample input.

With that in mind, let's look at how the `nnx.bridge` wrappers tackle the differences.

### Linen -> NNX

Since Linen modules may require an input to create variables, we semi-formally supported lazy initialization in the NNX modules converted from Linen. The Linen variables are created when you give it a sample input.

For you, it's calling `nnx.bridge.lazy_init()` where you call `module.init()` in Linen code.

(Note: you can call `nnx.display` upon any NNX module to inspect all its variables and state.)


```python
class LinenDot(nn.Module):
  out_dim: int
  w_init: Callable[..., Any] = nn.initializers.lecun_normal()
  @nn.compact
  def __call__(self, x):
    # Linen might need the input shape to create the weight!
    w = self.param('w', self.w_init, (x.shape[-1], self.out_dim))
    return x @ w

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.ToNNX(LinenDot(64), rngs=nnx.Rngs(0))  # => `model = LinenDot(64)` in Linen
bridge.lazy_init(model, x)                            # => `var = model.init(key, x)` in Linen
y = model(x)                                          # => `y = model.apply(var, x)` in Linen

nnx.display(model)

# In-place swap your weight array and the model still works!
model.params['w'].value = jax.random.normal(jax.random.key(1), (32, 64))
assert not jnp.allclose(y, model(x))
```

    ToNNX(
      module=LinenDot(
          # attributes
          out_dim = 64
          w_init = init
      ),
      rngs=Rngs(
        default=RngStream(
          key=RngKey(
            value=Array((), dtype=key<fry>) overlaying:
            [0 0],
            tag='default'
          ),
          count=RngCount(
            value=Array(1, dtype=uint32),
            tag='default'
          )
        )
      ),
      linen_collections=('params',),
      params={'w': Param(
        value=Array(shape=(32, 64), dtype=float32)
      )}
    )


`nnx.bridge.lazy_init` also works even if the top-level module is a pure-NNX one, so you can do sub-moduling as you wish:


```python
class NNXOuter(nnx.Module):
  def __init__(self, out_dim: int, rngs: nnx.Rngs):
    self.dot = nnx.bridge.ToNNX(LinenDot(out_dim), rngs=rngs)
    self.b = nnx.Param(jax.random.uniform(rngs.params(), (1, out_dim,)))

  def __call__(self, x):
    return self.dot(x) + self.b

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.lazy_init(NNXOuter(64, rngs=nnx.Rngs(0)), x)  # Can fit them into one line too
nnx.display(model)
```

    NNXOuter(
      dot=ToNNX(
        module=LinenDot(
            # attributes
            out_dim = 64
            w_init = init
        ),
        rngs=Rngs(
          default=RngStream(
            key=RngKey(
              value=Array((), dtype=key<fry>) overlaying:
              [0 0],
              tag='default'
            ),
            count=RngCount(
              value=Array(1, dtype=uint32),
              tag='default'
            )
          )
        ),
        linen_collections=('params',),
        params={'w': Param(
          value=Array(shape=(32, 64), dtype=float32)
        )}
      ),
      b=Param(
        value=Array(shape=(1, 64), dtype=float32)
      )
    )


The Linen weight is already converted to a typical NNX variable, which is a thin wrapper of the actual JAX array value within. Here, `w` is an `nnx.Param` because it belongs to the `params` collection of `LinenDot` module.

We will talk more about different collections and types in the [NNX Variable <-> Linen Collections](#variable-types-vs-collections) section. Right now, just know that they are converted to NNX variables like native ones.


```python
assert isinstance(model.dot.params['w'], nnx.Param)
assert isinstance(model.dot.params['w'].value, jax.Array)
```

If you create this model witout using `nnx.bridge.lazy_init`, the NNX variables defined outside will be initialized as usual, but the Linen part (wrapped inside `ToNNX`) will not.


```python
partial_model = NNXOuter(64, rngs=nnx.Rngs(0))
nnx.display(partial_model)
```

    NNXOuter(
      dot=ToNNX(
        module=LinenDot(
            # attributes
            out_dim = 64
            w_init = init
        ),
        rngs=Rngs(
          default=RngStream(
            key=RngKey(
              value=Array((), dtype=key<fry>) overlaying:
              [0 0],
              tag='default'
            ),
            count=RngCount(
              value=Array(1, dtype=uint32),
              tag='default'
            )
          )
        ),
        linen_collections=()
      ),
      b=Param(
        value=Array(shape=(1, 64), dtype=float32)
      )
    )



```python
full_model = bridge.lazy_init(partial_model, x)
nnx.display(full_model)
```

    NNXOuter(
      dot=ToNNX(
        module=LinenDot(
            # attributes
            out_dim = 64
            w_init = init
        ),
        rngs=Rngs(
          default=RngStream(
            key=RngKey(
              value=Array((), dtype=key<fry>) overlaying:
              [0 0],
              tag='default'
            ),
            count=RngCount(
              value=Array(1, dtype=uint32),
              tag='default'
            )
          )
        ),
        linen_collections=('params',),
        params={'w': Param(
          value=Array(shape=(32, 64), dtype=float32)
        )}
      ),
      b=Param(
        value=Array(shape=(1, 64), dtype=float32)
      )
    )


### NNX -> Linen

To convert an NNX module to Linen, you should forward your creation arguments to `bridge.ToLinen` and let it handle the actual creation process.

This is because NNX module instance initializes all its variables eagerly when it is created, which consumes memory and compute. On the other hand, Linen modules are stateless, and the typical `init` and `apply` process involves multiple creation of them. So `bridge.to_linen` will handle the actual module creation and make sure no memory is allocated twice.


```python
class NNXDot(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
    self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (in_dim, out_dim)))
  def __call__(self, x: jax.Array):
    return x @ self.w

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.to_linen(NNXDot, 32, out_dim=64)  # <- Pass in the arguments, not an actual module
variables = model.init(jax.random.key(0), x)
y = model.apply(variables, x)

print(list(variables.keys()))
print(variables['params']['w'].value.shape)  # => (32, 64)
print(y.shape)                               # => (4, 64)

```

    ['nnx', 'params']
    (32, 64)
    (4, 64)


Note that `ToLinen` modules need to track an extra variable collection - `nnx` - for the static metadata of the underlying NNX module.


```python
# This new field stores the static data that defines the underlying `NNXDot`
print(type(variables['nnx']['graphdef']))    # => `nnx.graph.NodeDef`
```

    <class 'flax.nnx.graph.NodeDef'>


`bridge.to_linen` is actually a convenience wrapper around the Linen module `bridge.ToLinen`. Most likely you won't need to use `ToLinen` directly at all, unless you are using one of the built-in arguments of `ToLinen`. For example, if your NNX module doesn't want to be initialized with RNG handling:


```python
class NNXAddConstant(nnx.Module):
  def __init__(self):
    self.constant = nnx.Variable(jnp.array(1))
  def __call__(self, x):
    return x + self.constant

# You have to use `skip_rng=True` because your module `__init__` don't take `rng` as argument
model = bridge.ToLinen(NNXAddConstant, skip_rng=True)
y, var = model.init_with_output(jax.random.key(0), x)
```

You may notice that you need to an additional `.value` to access this Flax `w` param. This is because all NNX variables will be wrapped with an `nnx.Variable` class, which will allow it to be annotated with various information, such as its partitioning. This was translated into an equivalent `nnx.bridge.NNXMeta` wrapper.

If you use [partition metadata in Linen](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html), you can learn more about how that works in NNX in [Partition Metadata Section](#partition-metadata) below.


```python
print(type(variables['params']['w']))         # => nnx.bridge.NNXMeta
print(type(variables['params']['w'].value))   # => jax.Array
```

    <class 'flax.nnx.bridge.variables.NNXMeta'>
    <class 'jaxlib.xla_extension.ArrayImpl'>


Similar to `ToNNX`, you can use `ToLinen` to create a submodule of another Linen module.


```python
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
w, b = variables['params']['ToLinen_0']['w'].value, variables['params']['b']
print(w.shape, b.shape, y.shape)
```

    (32, 64) (1, 64) (4, 64)


## Handling RNG keys

All Flax modules, Linen or NNX, automatically handle the RNG keys for variable creation and random layers like dropouts. However, the specific logics of RNG key splitting are different, so you cannot generate the same params between Linen and NNX modules, even if you pass in same keys.

Another difference is that NNX modules are stateful, so they can track and update the RNG keys within themselves.

### Linen to NNX

If you convert a Linen module to NNX, you enjoy the stateful benefit and don't need to pass in extra RNG keys on every module call. You can use always `nnx.reseed` to reset the RNG state within.


```python
x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.ToNNX(nn.Dropout(rate=0.5, deterministic=False), rngs=nnx.Rngs(dropout=0))
bridge.lazy_init(model, x)       # We don't really need this b/c no extra params were created here,
                                 # but it's a good practice to always add this line.
y1, y2 = model(x), model(x)
assert not jnp.allclose(y1, y2)  # Two runs yield different outputs!

# Reset the dropout RNG seed, so that next model run will be the same as the first.
nnx.reseed(model, dropout=0)
assert jnp.allclose(y1, model(x))
```

### NNX to Linen

If you convert an NNX module to Linen, the underlying NNX module's RNG states will still be part of the top-level `variables`. On the other hand, Linen `apply()` call accepts different RNG keys on each call, which resets the internal Linen environment and allow different random data to be generated.

Now, it really depends on whether your underlying NNX module generates new random data from its RNG state, or from the passed-in argument. Fortunately, `nnx.Dropout` supports both - using passed-in keys if there is any, and use its own RNG state if not.

And this leaves you with two style options of handling the RNG keys:

* The NNX style (recommended): Let the underlying NNX state manage the RNG keys, no need to pass in extra keys in `apply()`. This means a few more lines to mutate the `variables` for every apply call, but things will look easier once your whole model no longer needs `ToLinen`.

* The Linen style: Just pass different RNG keys for every `apply()` call.


```python
x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.to_linen(nnx.Dropout, rate=0.5)
variables = model.init({'dropout': jax.random.key(0)}, x)

# The NNX RNG state was stored inside `variables`
print('The RNG key in state:', variables['RngKey']['rngs']['dropout']['key'].value)
print('Number of key splits:', variables['RngCount']['rngs']['dropout']['count'].value)

# NNX style: Must set `RngCount` as mutable and update the variables after every `apply`
y1, updates = model.apply(variables, x, mutable=['RngCount'])
variables |= updates
y2, updates = model.apply(variables, x, mutable=['RngCount'])
variables |= updates
print('Number of key splits after y2:', variables['RngCount']['rngs']['dropout']['count'].value)
assert not jnp.allclose(y1, y2)  # Every call yields different output!

# Linen style: Just pass different RNG keys for every `apply()` call.
y3 = model.apply(variables, x, rngs={'dropout': jax.random.key(1)})
y4 = model.apply(variables, x, rngs={'dropout': jax.random.key(2)})
assert not jnp.allclose(y3, y4)  # Every call yields different output!
y5 = model.apply(variables, x, rngs={'dropout': jax.random.key(1)})
assert jnp.allclose(y3, y5)      # When you use same top-level RNG, outputs are same
```

    The RNG key in state: Array((), dtype=key<fry>) overlaying:
    [1428664606 3351135085]
    Number of key splits: 0
    Number of key splits after y2: 2


## NNX variable types vs. Linen collections

When you want to group some variables as one category, in Linen you use different collections; in NNX, since all variables shall be top-level Python attributes, you use different variable types.

Therefore, when mixing Linen and NNX modules, Flax must know the 1-to-1 mapping between Linen collections and NNX variable types, so that `ToNNX` and `ToLinen` can do the conversion automatically.

Flax keeps a registry for this, and it already covers all Flax's built-in Linen collections. You can register extra mapping of NNX variable type and Linen collection names using `nnx.register_variable_name_type_pair`.

### Linen to NNX

For any collection of your Linen module, `ToNNX` will convert all its endpoint arrays (aka. leaves) to a subtype of `nnx.Variable`, either from registry or automatically created on-the-fly.

(However, we still keep the whole collection as one class attribute, because Linen modules may have duplicated names over different collections.)


```python
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
print(model.params['w'])        # Of type `nnx.Param` - note this is still under attribute `params`
print(model.params['b'])        # Of type `nnx.Param`
print(model.counter['count'])   # Of type `counter` - an auto-created dummy type from the name "counter"
print(type(model.counter['count']))

y = model(x, mutable=True)              # Linen's `sow()` needs `mutable=True` to trigger
print(model.intermediates['dot_sum'])   # Of type `nnx.Intermediates`
```

    Param(
      value=Array([[ 0.35401407,  0.38010964, -0.20674096],
             [-0.7356256 ,  0.35613298, -0.5099556 ],
             [-0.4783049 ,  0.4310735 ,  0.30137998],
             [-0.6102254 , -0.2668519 , -1.053598  ]], dtype=float32)
    )
    Param(
      value=Array([0., 0., 0.], dtype=float32)
    )
    counter(
      value=Array(0, dtype=int32)
    )
    <class 'abc.counter'>
    (Intermediate(
      value=Array(6.932987, dtype=float32)
    ),)


You can quickly separate different types of NNX variables apart using `nnx.split`.

This can be handy when you only want to set some variables as trainable.


```python
# Separate variables of different types with nnx.split
CountType = type(model.counter['count'])
static, params, counter, the_rest = nnx.split(model, nnx.Param, CountType, ...)
print('All Params:', list(params['params'].keys()))
print('All Counters:', list(counter['counter'].keys()))
print('All the rest (intermediates and RNG keys):', list(the_rest.keys()))

model = nnx.merge(static, params, counter, the_rest)  # You can merge them back at any time
```

    All Params: ['b', 'w']
    All Counters: ['count']
    All the rest (intermediates and RNG keys): ['intermediates', 'rngs']


### NNX to Linen

If you define custom NNX variable types, you should register their names with `nnx.register_variable_name_type_pair` so that they go to the desired collections.


```python
class Count(nnx.Variable): pass
nnx.register_variable_name_type_pair('counts', Count, overwrite=True)

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

    All Linen collections: ['nnx', 'LoRAParam', 'params', 'counts']
    {'w': NNXMeta(var_type=<class 'flax.nnx.variables.Param'>, value=Array([[ 0.2916921 ,  0.22780475,  0.06553137],
           [ 0.17487915, -0.34043145,  0.24764155],
           [ 0.6420431 ,  0.6220095 , -0.44769976],
           [ 0.11161668,  0.83873135, -0.7446058 ]], dtype=float32), metadata={'get_value_hooks': (), 'set_value_hooks': (), 'create_value_hooks': (), 'add_axis_hooks': (), 'remove_axis_hooks': ()})}


## Partition metadata

Flax uses a metadata wrapper box over the raw JAX array to annotate how a variable should be sharded.

In Linen, this is an optional feature that triggered by using `nn.with_partitioning` on initializers (see more on [Linen partition metadata guide](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html)). In NNX, since all NNX variables are wrapped by `nnx.Variable` class anyway, that class will hold the sharding annotations too.

The `bridge.ToNNX` and `bridge.ToLinen` API will automatically convert the sharding annotations, if you use the built-in annotation methods (aka. `nn.with_partitioning` for Linen and `nnx.with_partitioning` for NNX).

### Linen to NNX

Even if you are not using any partition metadata in your Linen module, the variable JAX arrays will be converted to `nnx.Variable`s that wraps the true JAX array within.

If you use `nn.with_partitioning` to annotate your Linen module's variables, the annotation will be converted to a `.sharding` field in the corresponding `nnx.Variable`.

You can then use `nnx.with_sharding_constraint` to explicitly put the arrays into the annotated partitions within a `jax.jit`-compiled function, to initialize the whole model with every array at the right sharding.


```python
class LinenDotWithPartitioning(nn.Module):
  out_dim: int
  @nn.compact
  def __call__(self, x):
    w = self.param('w', nn.with_partitioning(nn.initializers.lecun_normal(), ('in', 'out')),
                   (x.shape[-1], self.out_dim))
    return x @ w

@nnx.jit
def create_sharded_nnx_module(x):
  model = bridge.lazy_init(bridge.ToNNX(LinenDotWithPartitioning(64), rngs=nnx.Rngs(0)), x)
  state = nnx.state(model)
  sharded_state = nnx.with_sharding_constraint(state, nnx.get_partition_spec(state))
  nnx.update(model, sharded_state)
  return model


print(f'We have {len(jax.devices())} fake JAX devices now to partition this model...')
mesh = jax.sharding.Mesh(devices=mesh_utils.create_device_mesh((2, 4)), axis_names=('in', 'out'))
x = jax.random.normal(jax.random.key(42), (4, 32))
with mesh:
  model = create_sharded_nnx_module(x)

print(type(model.params['w']))            # `nnx.Param`
print(model.params['w'].sharding)         # The partition annotation attached with the weight `w`
print(model.params['w'].value.sharding)   # The underlying JAX array is sharded across the 2x4 mesh
```

    We have 8 fake JAX devices now to partition this model...
    <class 'flax.nnx.variables.Param'>
    ('in', 'out')
    GSPMDSharding({devices=[2,4]<=[8]})


### NNX to Linen

Since all NNX variables are wrapped with `nnx.Variable` box, the converted Linen module will have all variables boxed too. We have a default Linen partition metadata class called `bridge.NNXMeta` to store these converted NNX variables.

`nnx.with_partitioning` will automatically shard the array with the annotation if it is called within a `jax.sharding.Mesh` context, so you don't need to do `with_sharding_constraint` yourself.

Like with any Linen metadata wrappers, you can use `linen.unbox()` to get the raw JAX array tree.


```python
class NNXDotWithParititioning(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
    init_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), ('in', 'out'))
    self.w = nnx.Param(init_fn(rngs.params(), (in_dim, out_dim)))
  def __call__(self, x: jax.Array):
    return x @ self.w

x = jax.random.normal(jax.random.key(42), (4, 32))
model = bridge.to_linen(NNXDotWithParititioning, 32, 64)

with mesh:
  variables = jax.jit(model.init)(jax.random.key(0), x)

print(type(variables['params']['w']))                 # A `NNXMeta` wrapper of the underlying `nnx.Param`
print(variables['params']['w'].metadata['sharding'])  # The annotation coming from the `nnx.Param`
print(variables['params']['w'].value.sharding)   # The underlying JAX array is sharded across the 2x4 mesh

unboxed = nn.unbox(variables)
print(type(unboxed['params']['w']))     # The raw jax.Array
```

    <class 'flax.nnx.bridge.variables.NNXMeta'>
    ('in', 'out')
    GSPMDSharding({devices=[2,4]<=[8]})
    <class 'jaxlib.xla_extension.ArrayImpl'>


## Lifted transforms

In general, if you want to apply Linen/NNX-style lifted transforms upon an `nnx.bridge`-converted module, just go ahead and do it in the usual Linen/NNX syntax.

For Linen-style transforms, note that `bridge.ToLinen` is the top level module class, so you may want to just use it as the first argument of your transforms (which needs to be a `linen.Module` class in most cases)

### Linen to NNX

NNX style lifted transforms are similar to JAX transforms, and they work on functions.


```python
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

print(model.linen_dot.params['kernel'].shape) # (4, 32, 64) - first axis with dim 4 got vmapped
y = model(x)
print(y.shape)
```

    (4, 32, 64)
    (4, 64)


### NNX to Linen

Note that `bridge.ToLinen` is the top level module class, so you may want to just use it as the first argument of your transforms (which needs to be a `linen.Module` class in most cases).

Also, since `bridge.ToLinen` introduced this extra `nnx` collection, you need to mark it when using the axis-changing transforms (`linen.vmap`, `linen.scan`, etc) to make sure they are passed inside.


```python
class LinenVmapped(nn.Module):
  dout: int
  @nn.compact
  def __call__(self, x):
    inner = nn.vmap(bridge.ToLinen, variable_axes={'params': 0, 'nnx': None}, split_rngs={'params': True}
                    )(nnx.Linear, args=(x.shape[-1], self.dout))
    return inner(x)

x = jax.random.normal(jax.random.key(42), (4, 32))
model = LinenVmapped(64)
var = model.init(jax.random.key(0), x)
print(var['params']['VmapToLinen_0']['kernel'].value.shape)  # (4, 32, 64) - leading dim 4 got vmapped
y = model.apply(var, x)
print(y.shape)
```

    (4, 32, 64)
    (4, 64)


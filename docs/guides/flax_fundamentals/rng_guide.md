---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Randomness and PRNGs in Flax

+++

In this guide, you will learn how Flax uses [JAX's explicit pseudorandom number generator (PRNG) keys](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng) to emulate randomness, and adds some additional features to make it easier for users to thread PRNG keys through different Flax `Module`s.

If you are new to JAX PRNG keys or need a refresher, check out:
- [JAX 101: PRNGs in JAX](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html)
- [🔪 JAX - The Sharp Bits 🔪: Random Numbers](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers).

+++

## Setup

+++

Install or upgrade Flax, and then import some necessary dependencies.

**Note:** This guide uses the `--xla_force_host_platform_device_count=8` flag to emulate multiple devices in a CPU environment in a Google Colab/Jupyter Notebook. You don’t need this if you are already using a multi-device Google Cloud TPU environment, for example, on Google Cloud or in a Kaggle VM with a TPU.

```{code-cell}
:outputId: bb13d0ba-f908-4746-e4d3-f5916e0670ff
:tags: [skip-execution]

!pip install -q flax
```

```{code-cell}
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```{code-cell}
import flax, flax.linen as nn
import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

import hashlib
```

```{code-cell}
:outputId: ab5e0d3b-2d51-44ee-b823-d152ad1a10b2

jax.devices()
```

Set the JAX config variable `jax_threefry_partitionable` to `True`. This will be the default value in the future and makes the PRNG more efficiently auto-parallelizable under `jax.jit`. Refer to [JAX discussion](https://github.com/google/jax/discussions/18480) for more details.

```{code-cell}
jax.config.update('jax_threefry_partitionable', True)
assert jax.config.jax_threefry_partitionable == True
assert jax.config.jax_default_prng_impl == 'threefry2x32'
```

## Receiving, manipulating and creating PRNG keys with `Module.make_rng`

+++

The primary method Flax uses to receive, manipulate and create PRNG keys is via the `Module` method [`self.make_rng`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.make_rng). It is a method that accepts a string name that represents an "RNG stream". Each RNG stream has an initial starting seed PRNG key, which the user passes in as a dictionary argument (i.e. into an [`.init`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.init) or [`.apply`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.apply) function), and the starting seed is used by `self.make_rng` to generate more PRNG keys for that stream.

Note that this method can only be called with bounded modules (see [The Flax Module lifecycle](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html#top-level-modules)).

```{code-cell}
:outputId: 8304fbc2-ab0a-42f0-9d4c-c88a74012e83

class RNGModule(nn.Module):
  @nn.compact
  def __call__(self):
    print(self.make_rng('rng_stream'))
    print(self.make_rng('rng_stream'))
    print(self.make_rng('rng_stream'))

rng_module = RNGModule()
variables = rng_module.init({'rng_stream': jax.random.key(0)})
```

Now if we use a different starting seed PRNG key, we will generate different values (as intended).

```{code-cell}
:outputId: 1cfd9632-11cf-43f4-b922-63038705a195

variables = rng_module.init({'rng_stream': jax.random.key(1)})
```

Calling `self.make_rng` for one stream will not affect the random values generated from another stream; i.e. the call order doesn't matter.

```{code-cell}
:outputId: 60dde8ff-08a9-4743-c7ec-c997aa0537b6

class RNGModuleTwoStreams(nn.Module):
  @nn.compact
  def __call__(self):
    # same value as first code snippet above
    print(f"rng_stream1: {self.make_rng('rng_stream1')}")
    # same value as second code snippet above
    print(f"rng_stream2: {self.make_rng('rng_stream2')}")
    # same value as first code snippet above
    print(f"rng_stream1: {self.make_rng('rng_stream1')}")
    # same value as second code snippet above
    print(f"rng_stream2: {self.make_rng('rng_stream2')}")
    # same value as first code snippet above
    print(f"rng_stream1: {self.make_rng('rng_stream1')}")
    # same value as second code snippet above
    print(f"rng_stream2: {self.make_rng('rng_stream2')}")

rng_module_two_streams = RNGModuleTwoStreams()
variables = rng_module_two_streams.init(
  {'rng_stream1': jax.random.key(0), 'rng_stream2': jax.random.key(1)}
)
```

Providing the same seed PRNG key will result in the same values being generated (provided that the same operations are used for those keys).

```{code-cell}
:outputId: 12e8c3c5-a32c-46ab-ba71-98cc90aaed70

variables = rng_module_two_streams.init(
  {'rng_stream1': jax.random.key(0), 'rng_stream2': jax.random.key(0)}
)
```

### How `self.make_rng` works under the hood

+++

This is what happens when `self.make_rng` (`flax.linen.Module.make_rng`) is called:
* The following data is collected:
  * The path of the `Module` as provided by `self.scope.path` (the top-level root module has an empty path `()`).
  * The `self.make_rng` call count. That is, the number of times `self.make_rng` has been called for this specific stream (including this call).
    * **Note:** Each sub-`Module` will have its own individual call count that's separate from other `Module`s. For example, a `Module` that has called `self.make_rng('params')` twice and contains a sub-`Module` that has called `self.make_rng('params')` once, will have a call count of 2 and 1 for each of the RNG stream `'params'`, respectively.
* The data is bundled into a tuple and fed into a hash function and produces an integer.
* The generated integer is folded into the RNG stream's starting seed PRNG key to generate a new, unique PRNG key.

+++

Below is a slightly simplified version of the hash function that Flax uses for `self.make_rng`:

```{code-cell}
def produce_hash(data):
  m = hashlib.sha1()
  for x in data:
    if isinstance(x, str):
      m.update(x.encode('utf-8'))
    elif isinstance(x, int):
      m.update(x.to_bytes((x.bit_length() + 7) // 8, byteorder='big'))
    else:
      raise ValueError(f'Expected int or string, got: {x}')
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return hash_int
```

And now you can manually reproduce the PRNG keys generated from `self.make_rng`:

```{code-cell}
:outputId: 6bf352c1-086a-4c7a-8ab9-92abde614270

stream_seed = jax.random.key(0)
for call_count in range(1, 4):
  hash_int = produce_hash(data=(call_count,))
  print(jax.random.fold_in(stream_seed, jnp.uint32(hash_int)))
```

```{code-cell}
:outputId: 3809fddc-7319-4d55-ccc5-7b7c78160ea9

variables = rng_module.init({'rng_stream': jax.random.key(0)})
```

### Sub-`Module`s and `self.make_rng`

+++

This section explores how `self.make_rng` (`flax.linen.Module.make_rng`) behaves with sub-`Module`s.

Consider the following example:

```{code-cell}
:outputId: cc88c770-80ff-4093-e37f-6dc10c24006f

class RNGSubSubModule(nn.Module):
  def __call__(self):
    print(f"{self.name}, count 1: {self.make_rng('rng_stream')}")
    print(f"{self.name}, count 2: {self.make_rng('rng_stream')}")

class RNGSubModule(nn.Module):
  @nn.compact
  def __call__(self):
    print(f"{self.name}, count 1: {self.make_rng('rng_stream')}")
    print(f"{self.name}, count 2: {self.make_rng('rng_stream')}")
    RNGSubSubModule()()

class RNGModule(nn.Module):
  @nn.compact
  def __call__(self):
    print(f"RNGModule, count 1: {self.make_rng('rng_stream')}")
    print(f"RNGModule, count 2: {self.make_rng('rng_stream')}")
    RNGSubModule()()

rng_module = RNGModule()
variables = rng_module.init({'rng_stream': jax.random.key(0)})
```

As previously discussed, the data that is fed into the Flax hash function consists of:

  * The path of the `Module`, provided by `self.scope.path` (the top-level root module has an empty path `()`); and
  * The call count for the specific RNG stream.

In addition, note that each Flax `Module` and sub-`Module` have their own individual call counts, even for the same RNG stream. The convention for sub-`Module` names is: `f'{module_name}_{module_number}'`. For example, the first `Dense` sub-`Module` will be called `Dense_0`, the second one will be called `Dense_1`, and so on.

Therefore, the following data will be fed into the hash function:

  * For `RNGModule`: The data is just the call count, such as `(1,)` and `(2,)`, since the root `Module` has an empty path.
  * For `RNGSubModule`: The data is `('RNGSubModule_0', 1)` and `('RNGSubModule_0', 2)`.
  * For `RNGSubSubModule`: The data is `('RNGSubModule_0', 'RNGSubSubModule_0', 1)` and `('RNGSubModule_0', 'RNGSubSubModule_0', 2)`.

With this data, you can manually reproduce the PRNG keys generated from the `Module` and sub-`Module`s using `self.make_rng`.

For example:

```{code-cell}
:outputId: d38896e6-8061-4f54-fa30-680b3f524071

stream_seed = jax.random.key(0)
for initial_data in ((), ('RNGSubModule_0',), ('RNGSubModule_0', 'RNGSubSubModule_0')):
  if initial_data:
    module_name = initial_data[-1]
  else:
    module_name = 'RNGModule'
  for call_count in (1, 2):
    hash_int = produce_hash(data=initial_data+(call_count,))
    rng_key = jax.random.fold_in(stream_seed, jnp.uint32(hash_int))
    print(f"{module_name}, count {call_count}: {rng_key}")
```

If the same sub-`Module` class is used multiple times, you can increment the suffix of the sub-`Module` name accordingly. For example: `RNGSubModule_0`, `RNGSubModule_1`, and so on.

```{code-cell}
:outputId: 556a79c6-0b38-4736-f5ad-79fd1976b191

class RNGSubModule(nn.Module):
  @nn.compact
  def __call__(self):
    print(f"{self.name}, count 1: {self.make_rng('rng_stream')}")
    print(f"{self.name}, count 2: {self.make_rng('rng_stream')}")

class RNGModule(nn.Module):
  @nn.compact
  def __call__(self):
    print(f"RNGModule, count 1: {self.make_rng('rng_stream')}")
    print(f"RNGModule, count 2: {self.make_rng('rng_stream')}")
    RNGSubModule()()
    RNGSubModule()()

rng_module = RNGModule()
variables = rng_module.init({'rng_stream': jax.random.key(0)})
```

```{code-cell}
:outputId: 433f5e48-d379-490f-e499-ef4c24032776

stream_seed = jax.random.key(0)
for initial_data in ((), ('RNGSubModule_0',), ('RNGSubModule_1',)):
  if initial_data:
    module_name = initial_data[-1]
  else:
    module_name = 'RNGModule'
  for call_count in (1, 2):
    hash_int = produce_hash(data=initial_data+(call_count,))
    rng_key = jax.random.fold_in(stream_seed, jnp.uint32(hash_int))
    print(f"{module_name}, count {call_count}: {rng_key}")
```

### Using `self.param` and `self.variable`

+++

Flax users have the option of creating additional parameters and variables in their modules by using the [`self.param`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.param) and [`self.variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.variable) `Module` methods. An `init_fn` argument must be passed to these methods so that it can generate the initial value of the parameter/variable. `self.make_rng` is commonly used implicitly or explicitly in this `init_fn`, since many initializer functions are stochastic in nature and require a PRNG key. See the full list of Flax initializers [here](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html).

There are a couple of differences between the two methods that the user should take note of:
* `self.param` always creates a parameter in the `'params'` [collection](https://flax.readthedocs.io/en/latest/glossary.html#term-Params-parameters), whereas `self.variable` creates a variable in any [collection](https://flax.readthedocs.io/en/latest/glossary.html#term-Variable-collections) the user specifies
* `self.param` will automatically call `self.make_rng('params')` and pass in the generated PRNG key implicitly to the `init_fn` of the parameter you instantiated (it will be passed in as the first argument), whereas users will have to manually specify what RNG stream to call `self.make_rng` on in the `init_fn` of `self.variable` (it could be `'params'` or something different).

Below is an example using both `self.param` and `self.variable`:

```{code-cell}
:outputId: fc7df4d6-5d6e-4c3c-98f2-d5b0e9a8a45e

class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    # kernel will use 'params' seed, initial data will include 'Dense_0', call count 1
    x = nn.Dense(2, kernel_init=jax.random.normal, use_bias=False)(x)
    # model_param will use 'params' seed, call count 1
    model_param = self.param('model_param', jax.random.normal, x.shape)
    # model_variable1 will use 'params' seed, call count 2
    model_variable1 = self.variable(
      'other_collection',
      'model_variable1',
      lambda: jax.random.normal(self.make_rng('params'), x.shape),
    )
    # model_variable2 will use 'other' seed, call count 1
    model_variable2 = self.variable(
      'other_collection',
      'model_variable2',
      lambda: jax.random.normal(self.make_rng('other'), x.shape),
    )
    # kernel will use 'params' seed, initial data will include 'Dense_1', call count 1
    # bias will use 'params' seed, initial data will include 'Dense_1', call count 2
    x = nn.Dense(2, kernel_init=jax.random.normal, bias_init=jax.random.normal)(
      x
    )
    return x

model = Model()
variables = model.init(
  {'params': jax.random.key(0), 'other': jax.random.key(1)}, jnp.ones((2, 2))
)
print(variables['params']['Dense_0']['kernel'])
print(variables['params']['model_param'])
print(variables['other_collection']['model_variable1'])
print(variables['other_collection']['model_variable2'])
print(variables['params']['Dense_1']['kernel'])
print(variables['params']['Dense_1']['bias'])
```

Remember:
* there is a separate count for each RNG stream; this is why the count for `self.make_rng('other')` starts at 1 even though there were earlier calls of `self.make_rng('params')`
* each submodule has their own separate count for each rng stream; this is why each `Dense` layer has their own separate count for `self.make_rng('params')` and why `model_param` and `model_variable1` share the same count (since they are defined within the same top-level parent module)

```{code-cell}
:outputId: 40745795-8bbc-4c31-811a-f3658e5459d7

params_seed = jax.random.key(0)
other_seed = jax.random.key(1)
for initial_data, count, seed, shape in (
  (('Dense_0',), 1, params_seed, (2, 2)),
  ((), 1, params_seed, (2, 2)),
  ((), 2, params_seed, (2, 2)),
  ((), 1, other_seed, (2, 2)),
  (('Dense_1',), 1, params_seed, (2, 2)),
  (('Dense_1',), 2, params_seed, (1, 2)),
):
  hash_int = produce_hash(data=(*initial_data, count))
  rng_key = jax.random.fold_in(seed, jnp.uint32(hash_int))
  print(jax.random.normal(rng_key, shape))
```

### Managing RNG streams inside a training loop

+++

Below is an example of managing RNG streams from `self.make_rng`, `self.param`, `self.variable` and `nn.Dropout` in a training loop (note: `nn.Dropout` requires a seed PRNG key to be passed in the `'dropout'` RNG stream, since it implicitly calls `self.make_rng('dropout')`):

```{code-cell}
class SubModule(nn.Module):
  @nn.compact
  def __call__(self, x, train):
    # variables created using `self.param` will use `self.make_rng('params')`
    kernel = self.param('submodule_kernel', jax.random.normal, x.shape)
    x = x + kernel
    # `nn.Dropout` will use self.make_rng('dropout')
    x = nn.Dropout(0.2)(x, deterministic=not train)
    # `nn.Dense` will use self.make_rng('params')
    x = nn.Dense(3)(x)
    return x

class Model(nn.Module):
  @nn.compact
  def __call__(self, x, train):
    # make kernel use `self.make_rng('other')`
    kernel = self.variable(
      'other_collection',
      'module_kernel',
      lambda: jax.random.normal(self.make_rng('other'), x.shape),
    )
    x = (
      x + kernel.value
    )  # `.value` will extract the underlying value of the variable
    x = SubModule()(x, train)
    # `nn.Dropout` will use self.make_rng('dropout')
    x = nn.Dropout(0.2)(x, deterministic=not train)
    # `nn.Dense` will use self.make_rng('params')
    x = nn.Dense(2)(x)
    return x

params_rng, other_rng, train_rng = jax.random.split(jax.random.key(0), 3)
init_rngs = {'params': params_rng, 'other': other_rng}

x = jnp.ones((1, 3))
y = jnp.ones((1, 2))

module = Model()
variables = module.init(init_rngs, x, train=False)
```

```{code-cell}
:outputId: 9e689bfd-ba73-4b78-c3c0-89693caada2d

def update(variables, rng):
  # we don't need to provide a 'params' or 'other' rng, as only 'dropout' rng will be used during training
  # split the rng to get a dropout_rng to be used for this training iteration,
  # and to get another rng key to be used for the next training iteration
  dropout_rng, next_rng = jax.random.split(rng)
  def loss(params):
    out = module.apply(
      {'params': params, 'other_collection': variables['other_collection']},
      x,
      train=True,
      rngs={'dropout': dropout_rng},
    )
    return jnp.mean((y - out) ** 2)
  grads = jax.grad(loss)(variables['params'])
  params = jax.tree_map(lambda p, g: p - 1e-3 * g, variables['params'], grads)
  return {
    'params': params,
    'other_collection': variables['other_collection'],
  }, next_rng

for _ in range(10):
  variables, train_rng = update(variables, train_rng)
  out = module.apply(variables, x, train=False)
  print(jnp.mean((y - out)**2))
```

### 🔪 Sharp edge 🔪 - unintentionally generating the same values

There is an edge case where the same value can be unintentionally generated.
See the [Flax issue](https://github.com/google/flax/issues/2157) for more details.

```{code-cell}
:outputId: 80e45c76-bb4d-48ab-8fac-8b14126428d1

class Leaf(nn.Module):
  def __call__(self, x):
    return x + jax.random.randint(self.make_rng("rng"), (), 0, 100)

class Node(nn.Module):
  leaf_name: str
  @nn.compact
  def __call__(self, x):
    return Leaf(name=self.leaf_name)(x)

class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    return (Node(name="ab", leaf_name="cdef")(x),
            Node(name="abc", leaf_name="def")(x),
    )

out1, out2 = Model().apply({}, 0, rngs={"rng": jax.random.key(33)})
out1 == out2 # same output, despite having different submodule names
```

This occurs because the hash function [concatenates strings together](https://docs.python.org/3/library/hashlib.html#hashlib.hash.update), so the data `('AB', 'C')` is equivalent to data `('A', 'BC')` when fed into the hash function, therefore producing the same hash int.

```{code-cell}
:outputId: 07fc8349-41a3-4c74-d25c-0ffc4aa1be0a

print(produce_hash(data=('A', 'B', 'C', 1)))
print(produce_hash(data=('AB', 'C', 1)))
print(produce_hash(data=('A', 'BC', 1)))
print(produce_hash(data=('ABC', 1)))
```

To avoid this edge case, users can flip the `flax_fix_rng_separator` [configuration flag](https://flax.readthedocs.io/en/latest/api_reference/flax.config.html#flax.configurations.Config.flax_fix_rng_separator) to `True`.

```{code-cell}
:outputId: 4ba0c3cb-e903-40af-ab3f-687d83c257c9

flax.config.update('flax_fix_rng_separator', True)
out1, out2 = Model().apply({}, 0, rngs={"rng": jax.random.key(33)})
out1 == out2 # different output
```

## Managing RNG's on multiple devices

+++

This section will show examples on how to use `jit` and `shard_map` to use RNG's in multi-device settings.

+++

### Using `jax.jit`

+++

When using [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), we can use RNG's as we did before, but we now include `in_shardings` and `out_shardings` arguments to shard specify how to shard input and output data.

For more details on training on multiple devices in Flax using `jax.jit`, see our [Scale up Flax Modules on multiple devices guide](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#) and [lm1b example](https://github.com/google/flax/tree/main/examples/lm1b).

```{code-cell}
:outputId: 9deab9d8-3e15-4be0-8d91-279f6984ee99

# Create a mesh and annotate the axis with a name.
device_mesh = mesh_utils.create_device_mesh((8,))
print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data',))
print(mesh)

data_sharding = NamedSharding(mesh, PartitionSpec('data',))
print(data_sharding)
```

```{code-cell}
:outputId: 9b47eef2-612e-4d9f-ffb2-1e868cb52d86

class Model(nn.Module):
  @nn.compact
  def __call__(self, x, add_noise):
    x = nn.Dense(1)(x)
    # use jnp.where for control flow; for more details see: https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
    return jnp.where(
      add_noise, x + jax.random.normal(self.make_rng('params'), x.shape), x
    )

module = Model()
init_rng, apply_rng = jax.random.split(jax.random.key(0))
x = jnp.ones((8, 1))
variables = module.init(init_rng, x, False)

# create custom forward function, since jit does not support kwargs when in_shardings is specified
def forward(variables, x, add_noise, rng):
  return module.apply(variables, x, add_noise, rngs={'params': rng})

# shard the inputs x across devices
# replicate the variables, add_noise boolean and rng key across devices
# shard the output across devices
jit_forward = jax.jit(
  forward,
  in_shardings=(None, data_sharding, None, None),
  out_shardings=data_sharding,
)
out = jit_forward(variables, x, True, apply_rng)
out
```

The output is different given the same input, meaning the RNG key was used to add noise to the output.

We can also confirm that the output is sharded across devices:

```{code-cell}
:outputId: f6596bdc-89b4-46bf-ba99-84b1009d8156

out.addressable_shards
```

Another way to visualize the output sharding:

```{code-cell}
:outputId: 7df87ccd-0979-4b85-fbac-1d31aac53276

jax.debug.visualize_array_sharding(out)
```

If we choose not to add noise, then the output is the same across all batches (as expected, since the input is the same for all batches):

```{code-cell}
:outputId: cb7eb7d0-3fa7-4f72-a129-18901dc61bb1

out = jit_forward(variables, x, False, apply_rng)
print(out)
```

We can confirm the un-jitted function produces the same values, albeit unsharded (note there may be small numerical differences due to compiler optimizations from jitting):

```{code-cell}
:outputId: b88834c1-ebb2-422a-a665-1732015a3974

out = forward(variables, x, True, apply_rng)
out
```

```{code-cell}
:outputId: 5b06f58b-7f1f-4cd6-c29f-3da5c0ea189f

out = forward(variables, x, False, apply_rng)
out
```

### Using `shard_map`

+++

When using [`jax.experimental.shard_map.shard_map`](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html), the important parts to remember are to:
* split your PRNG key to produce a different key for each device
* the PRNG keys will be sharded automatically to each device (provided you use the correct partition specification), but the [**rank of the original batched PRNG key array will not be reduced**](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html#rank-reducing-vs-rank-preserving-maps-over-array-axes); e.g.
with a batch of 8 PRNG keys and 8 devices, each device will see a PRNG key batch of size 1 within the `shard_map`-ed function
  * therefore to access the PRNG key itself, we need to index slice into it (see the example below)

```{code-cell}
:outputId: 29b2911f-28a7-4a25-fcf7-4c3544dafada

def forward(variables, x, add_noise, rng_key_batch):
  # rng_key_batch is a batch of size 1 containing 1 PRNG key
  # index slice into the rng_key_batch to access the PRNG key
  return module.apply(
    variables, x, add_noise, rngs={'params': rng_key_batch[0]}
  )

# define partition specifications
data_pspec = PartitionSpec('data')
no_pspec = PartitionSpec()

# shard the inputs x and rng keys across devices
# replicate the variables and add_noise boolean across devices
# shard the output across devices
shmap_forward = shard_map(
  forward,
  mesh=mesh,
  in_specs=(no_pspec, data_pspec, no_pspec, data_pspec),
  out_specs=data_pspec,
)
# get 8 different rng's that will be used by the 8 devices when doing forward inference
apply_rngs = jax.random.split(apply_rng, 8)
out = shmap_forward(variables, x, True, apply_rngs)
out
```

Confirm that the output is sharded across devices:

```{code-cell}
:outputId: 2d74dff9-3a7d-47df-afbc-3bda4b8edf51

out.addressable_shards
```

```{code-cell}
:outputId: 28bfe59b-ded1-455a-c894-f1c457ec28bf

jax.debug.visualize_array_sharding(out)
```

## Lifted transforms

+++

[Flax lifted transforms](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/transformations.html) allow you to use [JAX transforms](https://github.com/google/jax#transformations) with `Module` arguments. This section will show you how to control how PRNG keys are split in Flax lifted transforms.

Refer to [Lifted transformations](https://flax.readthedocs.io/en/latest/developer_notes/lift.html) for more detail.

+++

### `nn.vmap`

+++

We can use [`nn.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.vmap.html) to create a batched `Dense` layer:

```{code-cell}
:outputId: 7d7c4c3c-2ed2-40dd-f8d3-bb1ef91d93eb

x = jnp.ones((3, 2))

BatchDense = nn.vmap(
    nn.Dense,
    in_axes=0, out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False})

BatchDense(2).init(jax.random.key(0), x)
```

By denoting `variable_axes={'params': 0}'`, we vectorize the `params` Arrays on the first axis. However the parameter values generated are all identical to each other:

```{code-cell}
:outputId: 93152c8f-ada9-4cf5-c0e2-560a284b3981

BatchDense = nn.vmap(
    nn.Dense,
    in_axes=0, out_axes=0,
    variable_axes={'params': 0},
    split_rngs={'params': False})

BatchDense(2).init(jax.random.key(0), x)
```

If we also make `split_rngs={'params': True}`, then the PRNG key we provide is split across the variable axis (in this case, the batch axis 0), and we can generate different parameters for each batch input:

```{code-cell}
:outputId: dfbf4e49-55dc-4a6f-bece-396cb48d875e

BatchDense = nn.vmap(
    nn.Dense,
    in_axes=0, out_axes=0,
    variable_axes={'params': 0},
    split_rngs={'params': True})

BatchDense(2).init(jax.random.key(0), x)
```

Adding a variable via `self.variable` is straightforward:

```{code-cell}
:outputId: 20fa39a3-9e7a-4a58-8ad7-48a0119ec466

class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2)(x)
    kernel = self.variable(
      'other_collection',
      'kernel',
      lambda: jax.random.normal(self.make_rng('other'), x.shape),
    )
    return x + kernel.value

BatchModel = nn.vmap(
  Model,
  in_axes=0,
  out_axes=0,
  variable_axes={'params': 0, 'other_collection': 0},
  split_rngs={'params': True, 'other': True},
)

BatchModel().init({'params': jax.random.key(0), 'other': jax.random.key(1)}, x)
```

We can control which RNG stream to split, for example, if we only wanted to split the `'params'` RNG stream, then the variables generated from `self.variable` will be the same for each batch input:

```{code-cell}
:outputId: 69d52ec5-9018-43c8-9d77-65e9bf00b79f

BatchModel = nn.vmap(
    Model,
    in_axes=0, out_axes=0,
    variable_axes={'params': 0, 'other_collection': 0},
    split_rngs={'params': True, 'other': False})

BatchModel().init({'params': jax.random.key(0), 'other': jax.random.key(1)}, x)
```

We can also control which parameters / variables should be generated for each batch input, for example, if we only wanted `'params'` to generate separate parameters for each batch input:

```{code-cell}
:outputId: b01187f9-aded-41bc-d17f-e2965355675a

BatchModel = nn.vmap(
    Model,
    in_axes=0, out_axes=0,
    variable_axes={'params': 0, 'other_collection': None},
    split_rngs={'params': True, 'other': False})

BatchModel().init({'params': jax.random.key(0), 'other': jax.random.key(1)}, x)
```

### `nn.scan`

+++

We can use [`nn.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.scan.html) to create a scanned `Module` layer (this is useful for simplifying repetitively stacked submodules):

```{code-cell}
:outputId: fda568b4-3c1d-4a5e-96f0-058c6fc5b49a

x = jnp.ones((3, 2))

class ResidualMLPBlock(nn.Module):
  @nn.compact
  def __call__(self, x, _):
    h = nn.Dense(features=2)(x)
    h = nn.relu(h)
    return x + h, None # return an empty carry

ScanMLP = nn.scan(
      ResidualMLPBlock, variable_axes={'params': 0},
      variable_broadcast=False, split_rngs={'params': True},
      length=3)

ScanMLP().init(jax.random.key(0), x, None) # pass in an empty carry
```

Similar to before, we can control whether to split the RNG stream or not, for example, if we wanted all the stacked modules to be initialized to the same parameter values, we can pass in `split_rngs={'params': False}`:

```{code-cell}
:outputId: 620aff62-e20d-4794-b3af-5a5058c2471d

ScanMLP = nn.scan(
      ResidualMLPBlock, variable_axes={'params': 0},
      variable_broadcast=False, split_rngs={'params': False},
      length=3)

ScanMLP().init(jax.random.key(0), x, None)
```

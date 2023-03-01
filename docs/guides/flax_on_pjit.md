---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "2a9f78765c0c"}

# Scale up Flax Modules on multiple devices with `pjit`

This guide shows how to scale up [Flax Modules](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html) on multiple devices and hosts using JAX's [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html#module-jax.experimental.pjit) and [`flax.linen.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.spmd).

## Flax and `pjit`

[`jax.experimental.pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) provides a way to automatically compile and scale up JAX computations. `pjit` has the following benefits:

* `pjit` has the similar interface of [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) and works as a decorator on a function that needs to be compiled.
* When using `pjit`, you can write code as if it runs on a single device, and `pjit` will automatically compile and run it on multiple devices using the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm. 
* With `pjit` you can state how the input and output of your code is partitioned across devices, and the compiler will figure out how to: 1) partition everything inside; and 2) compile inter-device communications.

To learn more, refer to [JAX-101 pjit tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html) and [JAX in multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).

Flax provides several functionalities that can help you use `pjit` on [Flax Modules](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html), including:

1. An interface to specify partitions of your data when defining [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module).
2. Utility functions to generate the partition information that `pjit` requires to run.
3. An interface to customize your axis names called "logical axis annotations" to decouple both your Module code and partition plan to experiment with different partition layouts more easily.

+++ {"id": "0fa8ccbf573a"}

## Setup

Install Flax from HEAD:

```{code-cell} ipython3
:id: 867203db3bef
:tags: [skip-execution]

# Once Flax v0.6.4 is released, use `pip3 install flax`.
! pip3 install -qq "git+https://github.com/google/flax.git@main#egg=flax"
```

+++ {"id": "a9601432b448"}

## Imports

Import some necessary dependencies.

**Note:** This guide uses the `--xla_force_host_platform_device_count=8` flag to emulate multiple devices in a CPU environment in a Google Colab/Jupyter Notebook. Check out the [JAX-101 pjit tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html#setup) to learn more about emulating a multi-device TPU environment (in which case you should ignore running `os.environ`).

```{code-cell} ipython3
:id: f8f42d1174e5

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```{code-cell} ipython3
:id: b8da40732f0b

import functools
import numpy as np
import jax

from jax import lax, random, numpy as jnp

import flax
from flax import struct, traverse_util, linen as nn
from flax.linen import spmd # Flax Linen SPMD.
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax # Optax for common losses and optimizers. 
```

+++ {"id": "c0d280def897"}

Next, import all the `pjit`-related libraries.

> **Note:** [`jax.experimental.pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) is still in the experimental package of JAX, so there may be changes in the API in future.

1. Start a 2x4 device mesh (8 devices)—this is the same as the layout of [TPU v3-8](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#single_tpu_board).
2. Annotate each axis with a name. A typical way to annotate axis names is `('data', 'model')`, where:
  * `'data'`: the mesh dimension used for data-parallel sharding of the batch dimension of inputs and activations.
  * `'model'`: the mesh dimension used for sharding parameters of the model across devices.

```{code-cell} ipython3
:id: 684fe9fe13a0

from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

# Start a device mesh.
device_mesh = mesh_utils.create_device_mesh((2, 4))
print(device_mesh)
# Annotate each axis with a name.
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
mesh
```

+++ {"id": "307d39db6d94"}

## Define a layer

Before defining a model, create an example layer called `DotReluDot` (by subclassing `flax.linen.Module`), which creates two parameters `W1` and `W2` for dot product multiplication, and uses the `jax.nn.relu` (ReLU) activation function in-between.

To use this layer in `pjit` efficiently, apply the following APIs to annotate the parameters and intermediate variables correctly:

1. Use [`flax.linen.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_partitioning.html#flax.linen.with_partitioning) to decorate the initializer function when creating parameters `W1` and `W2`.

2. Apply [`pjit.with_sharding_constraint`](https://github.com/google/jax/blob/main/jax/_src/pjit.py#L1516) to annotate intermediate variables like `y` and `z` to force a particular sharding pattern under `pjit` when the ideal constraint is known.

  * This step is optional, but can sometimes help auto-SPMD to partition efficiently. In the example below, the call is not required, because `pjit` will figure out the same sharding layout for `y` and `z` regardless.

```{code-cell} ipython3
:id: b74c049968dc

class DotReluDot(nn.Module):
  depth: int
  @nn.compact
  def __call__(self, x):
    W1 = self.param(
        'W1', 
        nn.with_partitioning(nn.initializers.xavier_normal(), (None, 'model')),
        (x.shape[-1], self.depth))

    y = jax.nn.relu(jnp.dot(x, W1))
    # Force a local sharding annotation.
    y = with_sharding_constraint(y, PartitionSpec('data', 'model'))

    W2 = self.param(
        'W2', 
        nn.with_partitioning(nn.initializers.xavier_normal(), ('model', None)),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # Force a local sharding annotation.
    z = with_sharding_constraint(z, PartitionSpec('data', None))

    # Return a tuple to conform with the API `flax.linen.scan` as shown in the cell below.
    return z, None
```

+++ {"id": "cbac5321c08e"}

Note that device axis names like `'data'`, `'model'` or `None` are passed into both `flax.linen.with_partitioning` and `pjit_with_sharding_constraint` API calls. This refers to how each dimension of this data should be sharded — either across one of the device mesh dimensions, or not sharded at all.

For example:

* When you define `W1` with shape `(x.shape[-1], self.depth)` and annotate as `(None, 'model')`:

  * The first dimension (of length `x.shape[-1]`) will be replicated across all devices.
  * The second dimension (of length `self.depth`) will be sharded over the `'model'` axis of the device mesh. This means `W1` will be sharded 4-way on devices `(0, 4)`, `(1, 5)`, `(2, 6)` and `(3, 7)`, on this dimension.

* When you annotate the output `z` as `('data', None)`:

  * The first dimension — the batch dimension — will be sharded over the `'data'` axis. This means half of the batch will be processed on devices `0-3` (first four devices), and another half on devices `4-7` (the remaining four devices).
  * The second dimension — the data depth dimension — will be replicated across all devices.

+++ {"id": "b8389c11af79"}

## Define a model with `flax.linen.scan` lifted transformation

This guide uses `flax.linen.scan` to demonstrate how [Flax lifted transforms](https://flax.readthedocs.io/en/latest/developer_notes/lift.html#supported-transformations), such as `scan`, can work together with [JAX `pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html).

Having created `DotReluDot`, define the `MLP` model (by subclassing `flax.linen.Module`) as multiple layers of `DotReluDot`.

To replicate identical layers, you can either use `flax.linen.scan`, or a for-loop:

* `flax.linen.scan` can offer faster compilation times.
* The for-loop can be faster on runtime.

The code below shows how to apply both methods.

**Note:** `flax.linen.scan` has another dimension for the parameters (the dimension over which `scan` is applied). You need to use the [`metadata_params`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.scan.html#flax.linen.scan) argument to annotate the partition of this dimension. Since the parameters inside your `DotReluDot` (a sub-`Module`) are already sharded along the `model` axis, you don't need to partition multiple layers across the `model` dimension here, and therefore you should denote it as `None`.

```{code-cell} ipython3
:id: a0ea0dcccbc3

class MLP(nn.Module):
  num_layers: int
  depth: int
  use_scan: bool
  @nn.compact
  def __call__(self, x):
    if self.use_scan:
      x, _ = nn.scan(DotReluDot, length=self.num_layers, 
                     variable_axes={"params": 0},
                     split_rngs={"params": True},
                     metadata_params={nn.PARTITION_NAME: None}
                     )(self.depth)(x)
    else:
      for i in range(self.num_layers):
        x, _ = DotReluDot(self.depth)(x)
    return x
```

+++ {"id": "5b3abfef359d"}

## Specify sharding (includes initialization and `TrainState` creation)

Next, generate the [`jax.sharding.PartitionSpec`](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html?#more-information-on-partitionspec) that `pjit` should receive as annotations of _input_ and _output_ data. `PartitionSpec` is a tuple of 2 axes (in a 2x4 mesh). To learn more, refer to [JAX-101: Introduction to `pjit`](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html).

### Specify the input

For data parallelism, you can shard the batched _input_ `x` across the `data` axis by denoting the batch axis as `data`:

```{code-cell} ipython3
:id: 4b8472d462f2

x_spec = PartitionSpec('data', None)  # dimensions: (batch, length)
x_spec
```

+++ {"id": "06d134795ae1"}

### Generate a `PartitionSpec` for the output

Next, generate a [`PartitionSpec`](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html?#more-information-on-partitionspec) for the _output_, you need to use some actual output as a reference.

1. Instantiate a model.
2. Evaluate `model.init` abstractly using [`jax.eval_shape`](https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html).
3. Use [`flax.linen.get_partition_spec`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.get_partition_spec.html) to automatically generate the `PartitionSpec`.

The code below shows how to get the output spec if you use `flax.training.train_state` to carry out your initialization and training steps, in which case your `pjit`ted function will output a `TrainState`. 

(In a simpler case, people might choose the variable dict as in `variables = model.init(k, x)` as their `pjit`ted function's output. That works too.)

```{code-cell} ipython3
:id: 8b913a2e57d3

# MLP hyperparameters.
BATCH, LAYERS, DEPTH, USE_SCAN = 8, 4, 1024, True
# Create fake inputs.
x = jnp.ones((BATCH, DEPTH))
# Initialize a PRNG key.
k = random.PRNGKey(0)

# Create an Optax optimizer.
optimizer = optax.adam(learning_rate=0.001)
# Instantiate the model.
model = MLP(LAYERS, DEPTH, USE_SCAN)

# A functional way of model initialization.
def init_fn(k, x, model, optimizer):
  variables = model.init(k, x) # Initialize the model.
  state = train_state.TrainState.create( # Create a `TrainState`.
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer)
  return state

with mesh:
  # Create an abstract closure to wrap the function before feeding it in
  # because `jax.eval_shape` only takes pytrees as arguments`.
  abstract_variables = jax.eval_shape(
      functools.partial(init_fn, model=model, optimizer=optimizer), k, x)
# This `state_spec` has the same pytree structure as the output
# of the `init_fn`.
state_spec = nn.get_partition_spec(abstract_variables)
state_spec
```

+++ {"id": "2ec24614050b"}

## Apply `pjit` to compile the code

Now you can apply JAX [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html#module-jax.experimental.pjit) to your `init_fn` in a similar fashion as [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) but with two extra arguments: `in_axis_resources` and `out_axis_resources`.

You need to add a `with mesh:` context when running a `pjit`ted function, so that it can refer to `mesh` (an instance of `jax.sharding.Mesh`) to allocate data on devices correctly.

```{code-cell} ipython3
:id: a298c5d03c0d

pjit_init_fn = pjit(init_fn,
                    static_argnums=(2, 3),
                    in_axis_resources=(PartitionSpec(None), x_spec),  # PRNG key and x
                    out_axis_resources=state_spec
                    )
with mesh:
  initialized_state = pjit_init_fn(k, x, model, optimizer)
jax.tree_map(jnp.shape, initialized_state)
```

+++ {"id": "8f74b009f11f"}

## Inspect the Module output

Note that in the output of `initialized_state`, the `params` `W1` and `W2` are of type [`flax.linen.Partitioned`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Partitioned.html). This is a wrapper around the actual `jax.Array` that allows Flax to record metadata associated with it. You can access the raw `jax.Array` by adding `.value` or running `.unbox()`.

You can also check the underlying [`jax.sharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html) of the JAX array, which gives a hint on the way it is partitioned.

```{code-cell} ipython3
:id: 19243982c892

print(type(initialized_state.params['ScanDotReluDot_0']['W1']))
print(type(initialized_state.params['ScanDotReluDot_0']['W1'].value))
print(initialized_state.params['ScanDotReluDot_0']['W1'].value.shape)
```

```{code-cell} ipython3
:id: 2067c419a826

print(initialized_state.params['ScanDotReluDot_0']['W1'].value.sharding)
```

+++ {"id": "273547d3ab89"}

You can use [`jax.tree_map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html) to perform mass computation on a dict of boxed params, in the same way as on a dict of JAX arrays.

```{code-cell} ipython3
:id: 29b3dae156a2

diff = jax.tree_map(
    lambda a, b: a - b, 
    initialized_state.params['ScanDotReluDot_0'], initialized_state.params['ScanDotReluDot_0'])
print(jax.tree_map(jnp.shape, diff))
diff_array = diff['W1'].unbox()
print(type(diff_array))
print(diff_array.shape)
```

+++ {"id": "f7e1ccb14c6b"}

## Apply `pjit` to the train step and inference 

Now, you create a `pjit`ted training step:

```{code-cell} ipython3
:id: 4e3cc300cfee

def train_step(state, x):
  # A fake loss function.
  def loss_unrolled(params):
    y = model.apply({'params': params}, x)
    return y.sum()
  grad_fn = jax.grad(loss_unrolled)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

pjit_step_fn = pjit(train_step,
                    in_axis_resources=(state_spec, x_spec),  # input annotations
                    out_axis_resources=state_spec,           # output annotations
                    )
with mesh:
  new_state = pjit_step_fn(initialized_state, x)
```

+++ {"id": "2bae79e2e71b"}

Apply `pjit` to inference. Note that, similar to `jax.jit`, you can use a decorator like `@functools.partial(pjit, ...)` to directly compile your function.

```{code-cell} ipython3
:id: c9264a48b9ee

@functools.partial(pjit, in_axis_resources=(state_spec, x_spec), out_axis_resources=x_spec)
def pjit_apply_fn(state, x):
  return state.apply_fn({'params': state.params}, x)

with mesh:
  y = pjit_apply_fn(new_state, x)
print(type(y))
print(y.dtype)
print(y.shape)
```

+++ {"id": "7daa9e6e6eb4"}

## Profiling

If you are running on a TPU pod or a pod slice, you can use a custom `block_all` utility function, as defined below, to measure the performance:

```{code-cell} ipython3
:id: a68d7cb2eb89

%%timeit

def block_all(xs):
  jax.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

with mesh:
  new_state = block_all(pjit_step_fn(initialized_state, x))
```

+++ {"id": "51420b514d53"}

## Logical axis annotation

JAX auto SPMD encourages users to explore different sharding layouts to find the optimal one. To this end, in Flax you actually can annotate the dimensions of any data with more descriptive axis names (not just device mesh axis names like `'data'` and `'model'`). 

The `LogicalDotReluDot` and `LogicalMLP` Module definition below are similar to the Modules you created earlier, except for the following:

1. All axes are annotated with more concrete, meaningful names, such as `'embed'`, `'hidden'`, `'batch'` and `'layer'`. These names are referred to as _logical axis names_ in Flax. They make the dimensional changes inside model definitions more readable.

2. `flax.linen.spmd.with_logical_partitioning` replaces `flax.linen.with_partitioning`; and `flax.linen.spmd.with_logical_constraint` replaces `pjit.with_sharding_constraint`, to recognize the logical axis names.

```{code-cell} ipython3
:id: a26f85a9e772

class LogicalDotReluDot(nn.Module):
  depth: int
  @nn.compact
  def __call__(self, x):
    W1 = self.param(
        'W1', 
        spmd.with_logical_partitioning(nn.initializers.xavier_normal(), ('embed', 'hidden')),
        (x.shape[-1], self.depth)) 

    y = jax.nn.relu(jnp.dot(x, W1))
    # Force a local sharding annotation.
    y = spmd.with_logical_constraint(y, ('batch', 'hidden'))

    W2 = self.param(
        'W2', 
        spmd.with_logical_partitioning(nn.initializers.xavier_normal(), ('hidden', 'embed')),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # Force a local sharding annotation.
    z = spmd.with_logical_constraint(z, ('batch', 'embed'))
    return z, None

class LogicalMLP(nn.Module):
  num_layers: int
  depth: int
  use_scan: bool
  @nn.compact
  def __call__(self, x):
    if self.use_scan:
      x, _ = nn.scan(LogicalDotReluDot, length=self.num_layers, 
                    variable_axes={"params": 0},
                    split_rngs={"params": True},
                    metadata_params={nn.PARTITION_NAME: 'layer'}
                    )(self.depth)(x)
    else:
      for i in range(self.num_layers):
        x, _ = DotReluDot(self.depth)(x)
    return x
```

+++ {"id": "0de93ec6cbd6"}

The `LogicalMLP` model definition generates a set of `PartitionSpec` with logical axis names.

Repeat the steps from earlier: instantiate a model, evaluate the `init_fn` abstractly, and use `flax.linen.get_partition_spec` to automatically generate the `PartitionSpec`:

```{code-cell} ipython3
:id: 14db7a1e30fd

logical_model = LogicalMLP(LAYERS, DEPTH, USE_SCAN)
logical_abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=logical_model, optimizer=optimizer), k, x)
logical_output_spec = nn.get_partition_spec(logical_abstract_variables)
logical_output_spec
```

+++ {"id": "d1c9b74e50b9"}

To allow the device mesh to take your model correctly, you need to decide which of these logical axis names are mapped to the device axis `'data'` or `'model'`. This rule is a list of (`logical_axis_name`, `device_axis_name`) tuples, and `jax.linen.spmd.logical_to_mesh` will convert them to the spec that `pjit` accepts.

This allows you to change the rules and try out new partition layouts without modifying the model definition.

```{code-cell} ipython3
:id: 711cb4bde093

# Unspecified rule means unsharded by default, so no need to specify `('embed', None)` and `('layer', None)`.
rules = (('batch', 'data'),
         ('hidden', 'model'))

logical_state_spec = spmd.logical_to_mesh(logical_output_spec, rules)
logical_state_spec
```

+++ {"id": "58475fffb2de"}

You can verify that the `logical_state_spec` here has the same content as `state_spec` in the previous ("non-logical") example. This will be the `out_axis_resources` you specify when creating `pjit`ted functions.

```{code-cell} ipython3
:id: 589ff774bb4c

state_spec.params['ScanDotReluDot_0'] == logical_state_spec.params['ScanLogicalDotReluDot_0']
```

```{code-cell} ipython3
:id: 77e07a0ab309

logical_pjit_init_fn = pjit(init_fn,
                            static_argnums=(2, 3),
                            in_axis_resources=(PartitionSpec(None), x_spec),  # RNG key and x
                            out_axis_resources=logical_state_spec
                            )
with mesh:
  logical_initialized_state = logical_pjit_init_fn(k, x, logical_model, optimizer)
jax.tree_map(jnp.shape, logical_initialized_state)
```

+++ {"id": "ae1754a3031d"}

## When to use device axis / logical axis

Choosing when to use a device or logical axis depends on how much you want to control the partitioning of your model.

If you want a very simple model, or you are very confident of your way of partitioning, defining it with __device mesh axis__ can potentially save you a few extra lines of code of converting the logical naming back to the device naming.

On the other hand, the __logical naming__ helpers are useful for exploring different sharding layouts. Use this if you want to experiment around and find the most optimal partition layout for your model.

In really advanced use cases, you may have more complicated sharding patterns that require annotating *activation* dimension names differently from *parameter* dimension names. When people wish to have more fine-grained control on manual mesh assignments, directly using __device axis names__ could be more helpful.

+++ {"id": "576bdd5cd782"}

## Save the data

You can use [`flax.training.checkpoints`](https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html) to save the cross-device array, as shown in the [Save and load checkpoints guide - Multi-host/multi-process checkpointing](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#multi-host-multi-process-checkpointing). This is especially required if you are running on a multi-host environment (for example, a TPU pod).

Keep in mind that to restore the arrays to the desired partition, you need to provide a sample `target` pytree that has the same structure and has the desired `PartitionSpec` in place for each JAX array. The `PartitionSpec` you use to restore the array doesn't necessarily need to be the same as the ones you used to store the array.

---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Scale up Flax modules on multiple devices

This guide shows how to scale up Flax Modules on multiple devices and hosts using JAX's [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html#module-jax.experimental.pjit) and [`flax.linen.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.spmd).

## Flax and `pjit`

[`jax.experimental.pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) provides a way to automatically compile and scale up JAX computations. It provides the following benefits:

* `pjit` has the similar interface of [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) and works as a decorator on a function that needs to be compiled.

* When using `pjit`, you can write code as if it runs on a single device, and `pjit` will automatically compile and run it on multiple devices using the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm. 

* With `pjit` you can state how the input and output of your code is partitioned across devices, and the compiler will figure out how to partition everything inside, and how to compile inter-device communications.

* For more information, refer to [JAX-101 pjit tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html) and [JAX in multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).

Flax provides a few API to help you use `pjit` on Flax Modules, including:

* An interface to specify partitions of your data when defining [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module).

* Utility function to generate the partition information that `pjit` requires to run.

* An interface to customize your axis names called "logical axis annotations" to decouple both your Module code and partition plan to experiment with different partition layouts more easily.



# Setup

Install Flax from HEAD.

```python
# Once Flax v0.6.4 is out, replace this with `pip3 install flax`.
!pip3 install -qq "git+https://github.com/google/flax.git@main#egg=flax"
```

## Imports

Import some necessary dependencies.

**Note:** This guide uses `--xla_force_host_platform_device_count=8` to emulate multiple devices on a CPU environment in a Google Colab/Jupyter Notebook. You can also follow the [JAX-101 pjit tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html#setup) to learn how to emulate a multi-device TPU environment (in which case you should ignore the `os.environ` cell).

```python
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```python
import functools
import numpy as np
import jax

from jax import lax, random, numpy as jnp

import flax
from flax import struct, traverse_util, linen as nn
from flax.linen import spmd
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax
```

In the next step, import all the `pjit`-related libraries.

> **Note:** [`jax.experimental.pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) is still in the experimental package of JAX, so there may be changes in the API in future.

Start a device mesh using the 8 devices available—in this example, set them as a 2x4 device mesh, which is the same as the layout of [TPU v3-8](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#single_tpu_board).

Annotate each axis with a name. A typical way to annotate axis names is `('data', 'model')`, where:
  * `'data'`: the mesh dimension used for data-parallel sharding of the batch dimension of inputs and activations.
  * `'model'`: the mesh dimension used for sharding parameters of the model across devices.


```python
from jax.experimental.pjit import pjit, with_sharding_constraint, PartitionSpec
from jax.experimental.maps import Mesh
from jax.experimental import mesh_utils

device_mesh = mesh_utils.create_device_mesh((2, 4))
print(device_mesh)
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
mesh
```

# Define a model

Define an example layer `DotReluDot` (by subclassing `flax.linen.Module`), which creates two parameters `W1` and `W2` for dot products, and uses the `jax.nn.relu` activation function in-between. 

To use this layer in `pjit` efficiently, apply the following APIs to annotate the parameters and intermediate variables correctly:

* Use [`flax.linen.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_partitioning.html#flax.linen.with_partitioning) to decorate the initializer function when creating parameters `W1` and `W2`.

* Apply [`pjit.with_sharding_constraint`](https://github.com/google/jax/blob/main/jax/_src/pjit.py#L1516) to annotate intermediate variables like `y` and `z` to force a particular sharding pattern under `pjit` when the ideal constraint is known.

  * This step is optional, but can sometimes help auto-SPMD to partition efficiently. In the example below, the call is not required, because `pjit` will figure out the same sharding layout for `y` and `z` regardless.

```python
class DotReluDot(nn.Module):
  depth: int
  @nn.compact
  def __call__(self, x):
    W1 = self.param(
        'W1', 
        nn.with_partitioning(nn.initializers.xavier_normal(), (None, 'model')),
        (x.shape[-1], self.depth))

    y = jax.nn.relu(jnp.dot(x, W1))
    # force a local sharding annotation.
    y = with_sharding_constraint(y, PartitionSpec('data', 'model'))

    W2 = self.param(
        'W2', 
        nn.with_partitioning(nn.initializers.xavier_normal(), ('model', None)),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # force a local sharding annotation.
    z = with_sharding_constraint(z, PartitionSpec('data', None))

    # Return a tuple to conform with the API `nn.scan` as shown in the cell below.
    return z, None

```

Note that values like `'data'`, `'model'` or `None` are passed into these API calls. This refers to how each dimension of this data should be sharded - either across one of the device mesh dimensions, or not sharded at all.

For example:

* When we define `W1` with shape `(x.shape[-1], self.depth)` and annotate as `(None, 'model')`):

  * The first dimension (of length `x.shape[-1]`) will be replicated across all devices.
 
  * The second dimension (of length `self.depth`) will be sharded over the `model` axis of the device mesh. This means `W1` will be sharded 4-way on devices `(0, 4)`, `(1, 5)`, `(2, 6)` and `(3, 7)`, on this dimension.
  
* When we annotate the output `z` as `('data', None)`:

  * The first dimension (aka. the batch dimension) will be sharded over the `data` axis, which  means half of the batch will be processed on devices `0-3` and another half on devices `4-7`.
  
  * The second dimension (aka. the data depth dimension) will be replicated across all devices.


## Lifted transforms

Now define the `MLP` model as multiple layers of `DotReluDot`.

To replicate identical layers, you can either use `flax.linen.scan` or a for-loop: `flax.linen.scan` can offer faster compilation times, whereas the for-loop can be faster on runtime. This guide uses `flax.linen.scan` simply to show that [Flax lifted transforms](https://flax.readthedocs.io/en/latest/advanced_topics/lift.html#supported-transformations) work together with [JAX `pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html).

**Note:** `flax.linen.scan` will introduce another dimension for the params (the dimension over which `scan` is applied), and you need to use the `metadata_params` argument to annotate the partition of this dimension. Since the parameters inside your `DotReluDot` (a sub-`Module`) are already sharded along the `model` axis, you don't need to partition multiple layers across the `model` dimension here, and therefore you should denote it as `None`.

```python
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

# Sharding specification

Next, you need to generate the [`jax.experimental.pjit.PartitionSpec`](html) that `pjit` should receive as annotations of _input_ and _output_ data.

For data parallelism, you can shard the batched _input_ `x` across the `data` axis, by denoting the batch axis as `data`:

```python
x_spec = PartitionSpec('data', None)  # dimensions: (batch, length)
x_spec
```

However, to generate `PartitionSpec` for the _output_, you need to use some actual output as reference. One solution is to create a model, and then evaluate `model.init` abstractly using `jax.eval_shape`, and then use `nn.get_partition_spec` to automatically generate the `PartitionSpec`.

The code below shows how to get the output spec if you use `flax.training.train_state` to carry out your initialization and training steps, in which case your `pjit`ted function will output a `TrainState`. 

(In a simpler case, people might choose the variable dict as in `variables = model.init(k, x)` as their `pjit`ted function's output. That works too.)

> A side note: Here we define our `init_fn` as purely functional and takes `model` and `optimizer` as arguments. This is not necessary - you can simply define with `def init_fn(k, x):` and all will work fine here.
> 
> This guide doesn't do it because later we will show you another way to define your model and will run `init_fn` with another model instance. However, this is problematic because `jax.eval_shape` only takes pytrees as arguments, so we have to create an abstract closure before feeding the function in.

```python
# MLP hyperparams
BATCH, LAYERS, DEPTH, USE_SCAN = 8, 4, 1024, True
# fake inputs
x = jnp.ones((BATCH, DEPTH))
k = random.PRNGKey(0)

optimizer = optax.adam(learning_rate=0.001)
model = MLP(LAYERS, DEPTH, USE_SCAN)

def init_fn(k, x, model, optimizer):
  variables = model.init(k, x)
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer)
  return state

with mesh:
  abstract_variables = jax.eval_shape(
      functools.partial(init_fn, model=model, optimizer=optimizer), k, x)
# we need a rule on every leaf of the input variables, in this case a nested dict.
state_spec = nn.get_partition_spec(abstract_variables)
state_spec
```

# `pjit` the initialization and train step

Now you can `pjit` the `init_fn` in a similar fashion as `jit`, with two extra args `in_axis_resources` and `out_axis_resources`.

You need to add `with mesh:` context when running a `pjit`ted function, so that it can refer to `mesh` to allocate data on devices correctly.

```python
pjit_init_fn = pjit(init_fn,
                    static_argnums=(2, 3),
                    in_axis_resources=(PartitionSpec(None), x_spec),  # RNG key and x
                    out_axis_resources=state_spec
                    )
with mesh:
  initialized_state = pjit_init_fn(k, x, model, optimizer)
jax.tree_map(jnp.shape, initialized_state)
```

## Looking into the module output

Note that in the output `initialized_state`, the params `W1` and `W2` are of type [`Partitioned`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Partitioned.html). This is a wrapper around the actual JAX array that allows Flax to record metadata associated with it. You can access the raw JAX array by adding `.value` or running `.unbox()`.

You can also check the underlying `.sharding` of the JAX array, which gives a hint on the way it is partitioned.

```python
print(type(initialized_state.params['ScanDotReluDot_0']['W1']))
print(type(initialized_state.params['ScanDotReluDot_0']['W1'].value))
print(initialized_state.params['ScanDotReluDot_0']['W1'].value.shape)
```

```python
print(initialized_state.params['ScanDotReluDot_0']['W1'].value.sharding)
```

You can use `jax.tree_map` to perform mass computation on a dict of boxed params, in the same way as on a dict of JAX arrays.

```python
diff = jax.tree_map(
    lambda a, b: a - b, 
    initialized_state.params['ScanDotReluDot_0'], initialized_state.params['ScanDotReluDot_0'])
print(jax.tree_map(jnp.shape, diff))
diff_array = diff['W1'].unbox()
print(type(diff_array))
print(diff_array.shape)
```

## `pjit` the train step and inference 

Now we can do the same and make a `pjit`ted training step. 

```python
def train_step(state, x):
  # fake loss fn
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

Now you can do the same for inference. Note that similar to `jax.jit`, you can use a decorator like `@functools.partial(pjit, ...)` to directly compile your function.

```python
@functools.partial(pjit, in_axis_resources=(state_spec, x_spec), out_axis_resources=x_spec)
def pjit_apply_fn(state, x):
  return state.apply_fn({'params': state.params}, x)

with mesh:
  y = pjit_apply_fn(new_state, x)
print(type(y))
print(y.dtype)
print(y.shape)
```

## Profiling

If you are running on a TPU pod or pod slice, you can use the `block_all` utility function to observe a performance increase.

```python
%%timeit

# check this out by profiling, your model is made model parallel automatically by XLA:SPMD!
def block_all(xs):
  jax.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

with mesh:
  new_state = block_all(pjit_step_fn(initialized_state, x))
```

# Logical axis annotation

JAX auto SPMD encourages users to explore different sharding layouts to find the optimal one. To this end, in Flax you actually can annotate the dimensions of any array with more descriptive axis names, instead of only the device mesh axis names (i.e., `data` and `model`). 

Check out the `Logical-` model definitions below. It's exactly the same with the model above, except for two differences:

1. All axes are annotated with more concrete, meaningful names - like `embed`, `hidden`, `batch` and `layer`. These names are referred as "logical axis names" in Flax. They make the dimensional changes inside model definitions more readable.

2. `spmd.with_logical_partitioning` replaces `nn.with_partitioning` and `spmd.with_logical_constraint` replaces `pjit.with_sharding_constraint`, to recognize the logical axis names.

```python
class LogicalDotReluDot(nn.Module):
  depth: int
  @nn.compact
  def __call__(self, x):
    W1 = self.param(
        'W1', 
        spmd.with_logical_partitioning(nn.initializers.xavier_normal(), ('embed', 'hidden')),
        (x.shape[-1], self.depth)) 

    y = jax.nn.relu(jnp.dot(x, W1))
    # force a local sharding annotation, fairly redundant in this case.
    y = spmd.with_logical_constraint(y, ('batch', 'hidden'))

    W2 = self.param(
        'W2', 
        spmd.with_logical_partitioning(nn.initializers.xavier_normal(), ('hidden', 'embed')),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # force a local sharding annotation, fairly redundant in this case.
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

This model definition, of course, generates a set of `PartitionSpec` with the logical axis names.

```python
logical_model = LogicalMLP(LAYERS, DEPTH, USE_SCAN)
init_fn_logical_abstract = functools.partial(init_fn, model=logical_model, optimizer=optimizer)
logical_abstract_variables = jax.eval_shape(init_fn_logical_abstract, k, x)
logical_output_spec = nn.get_partition_spec(logical_abstract_variables)
logical_output_spec
```

To allow the device mesh to take your model correctly, you should decide which of these logical axis names should be mapped to the device axis `data` or `model`. This rule is a list of (`logical_axis_name`, `device_axis_name`) tuples, and `spmd.logical_to_mesh` will convert them to the spec that `pjit` accepts.

This allows you to change the rules and try out new partition layouts without modifying the model definition.

```python
# Unspecified rule means unsharded by default, so no need to specify ('embed', None) and ('layer', None).
rules = (('batch', 'data'),
         ('hidden', 'model'))

logical_state_spec = spmd.logical_to_mesh(logical_output_spec, rules)
logical_state_spec
```

You can verify that the `logical_state_spec` here has the same content as `state_spec` on our previous example. This will be the `out_axis_resources` you specify when creating `pjit`ted fucntions.

```python
state_spec.params['ScanDotReluDot_0'] == logical_state_spec.params['ScanLogicalDotReluDot_0']
```

```python
logical_pjit_init_fn = pjit(init_fn,
                            static_argnums=(2, 3),
                            in_axis_resources=(PartitionSpec(None), x_spec),  # RNG key and x
                            out_axis_resources=logical_state_spec
                            )
with mesh:
  logical_initialized_state = logical_pjit_init_fn(k, x, logical_model, optimizer)
jax.tree_map(jnp.shape, logical_initialized_state)
```

## When to use device axis, and when logical axis?

It all depends on how much you want to control the partitioning of your model.

If you want a very simple model or you are very confident of your way of partitioning, defining it with __device mesh axis__ can save you a few extra lines of code of converting the logical naming back to the device naming.

On the other hand, the __logical naming__ helpers are useful for exploring different sharding layouts. Use this if you want to expriment around and find the most optimal partition layout for your model.

In really advanced use cases, you may have more complicated sharding patterns that require annotating *activation* dimension names differently from *parameter* dimension names. When people wish to have more fine-grained control on manual mesh assignments, directly using __device axis names__ could be more helpful.


# Save the params

You can use `flax.training.checkpoints` to save the cross-device array, as shown in our [checkpointing guide for multi-process arrays](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#multi-host-multi-process-checkpointing). This is especially required if you are running on a multi-host environment (e.g. a TPU pod).

Keep in mind that in order to restore the arrays to the desired partition, you need to provide a sample `target` pytree that has the same structure and has the desired `PartitionSpec` in place for each JAX array. The `PartitionSpec` you use to restore the array doesn't necessarily need to be the same as the ones you used to store the array.

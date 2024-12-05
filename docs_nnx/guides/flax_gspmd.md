---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Scale up on multiple devices

This guide demonstrates how to scale up [Flax NNX `Module`s](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module) on [multiple devices and hosts](Multi-host and multi-process environments) - such as GPUs, Google TPUs, and CPUs - using the [JAX just-in-time compilation machinery (`jax.jit`)](https://jax.readthedocs.io/en/latest/jit-compilation.html) and [`flax.nnx.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html).

+++

## Overview

Flax relies on [JAX](https://jax.readthedocs.io) for numeric computations and scaling the computations up across multiple devices, such as GPU and Google TPUs. At the core of scaling up is the [JAX just-in-time (`jax.jit`) compiler `jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html). Throughout this guide, you will be using Flax’s own [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) transform, which wraps around [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) and works more conveniently with Flax NNX `Module`s.

> **Note:** To learn more about Flax’s transformations, such as `nnx.jit` and `nnx.vmap`, go to [Why Flax NNX? - Transforms](https://flax.readthedocs.io/en/latest/why.html#transforms), [Transformations](https://flax.readthedocs.io/en/latest/guides/transforms.html), and [Flax NNX vs JAX Transformations](https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html).

JAX compilation follows the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm. This means you write Python code as if it runs only on one device, and [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) will [automatically compile](https://jax.readthedocs.io/en/latest/jit-compilation.html#jit-compilation) and [run it](https://jax.readthedocs.io/en/latest/sharded-computation.html) on [multiple devices](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

To ensure the compilation performance, you often need to instruct JAX how your model's variables need to be sharded across devices. This is where Flax NNX's Sharding Metadata API - [`flax.nnx.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html) - comes in. It helps you annotate your model variables with this information.

> **Note to Flax Linen users**: The [`flax.nnx.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html) API is similar to what is described in [the Linen Flax on `(p)jit` guide](https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html) on the model definition level. However, the top-level code in Flax NNX is simpler due to the benefits brought by Flax NNX, and some text explanations will be more updated and clearer.

If you are new parallelization in JAX, you can learn more about its APIs for scaling up in the following tutorials:

- [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html): A 101 level tutorial covering the basics of automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), semi-automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) and [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html), and manual sharding with [`shard_map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.shard_map.shard_map.html#jax.experimental.shard_map.shard_map).
- [JAX in multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).
- [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): A more detailed tutorial about parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) and [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html). Study it after the [101](https://jax.readthedocs.io/en/latest/sharded-computation.html).
- [Manual parallelism with `shard_map`](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html): Another more in-depth doc that follows the [101](https://jax.readthedocs.io/en/latest/sharded-computation.html).

+++

### Setup

Import some necessary dependencies.

**Note:** This guide uses the `--xla_force_host_platform_device_count=8` flag to emulate multiple devices in a CPU environment in a Google Colab/Jupyter Notebook. You don't need this if you are already using a multi-device TPU environment.

```{code-cell} ipython3
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```{code-cell} ipython3
from typing import *

import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from flax import nnx

import optax # Optax for common losses and optimizers.
```

```{code-cell} ipython3
print(f'You have 8 “fake” JAX devices now: {jax.devices()}')
```

The code below shows how to import and set up the JAX-level device API, following JAX's [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) guide:

1. Start a 2x4 device `mesh` (8 devices) using the JAX [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh). This layout is the same as on a [TPU v3-8](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#single_tpu_board) (also 8 devices).

2. Annotate each axis with a name using the `axis_names` parameter. A typical way to annotate axis names is `axis_name=('data', 'model')`, where:

  * `'data'`: the mesh dimension used for data-parallel sharding of the batch dimension of inputs and activations.
  * `'model'`: the mesh dimension used for sharding parameters of the model across devices.

```{code-cell} ipython3
# Create a mesh of two dimensions and annotate each axis with a name.
mesh = Mesh(devices=np.array(jax.devices()).reshape(2, 4),
            axis_names=('data', 'model'))
print(mesh)
```

## Define a model with specified sharding

Next, create an example layer called `DotReluDot` that subclasses Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module).
- This layer carries out two dot product multiplications upon the input `x`, and uses the `jax.nn.relu` (ReLU) activation function in-between.
- To annotate a model variable with their ideal sharding, you can use [`flax.nnx.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.with_partitioning) to wrap over its initializer function. Essentially, this calls [`flax.nnx.with_metadata`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.with_metadata) which adds a `.sharding` attribute field to the corresponding [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable).

> **Note:** This annotation will be [preserved and adjusted accordingly across lifted transformations in Flax NNX](https://flax.readthedocs.io/en/latest/guides/transforms.html#axes-metadata). This means if you use sharding annotations along with any transform that modifies axes (like [`nnx.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html), [`nnx.scan`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)), you need to provide sharding of that additional axis via the `transform_metadata` arg. Check out the [Flax NNX transformations (transforms) guide](https://flax.readthedocs.io/en/latest/guides/transforms.html) to learn more.

```{code-cell} ipython3
class DotReluDot(nnx.Module):
  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()

    # Initialize a sublayer `self.dot1` and annotate its kernel with.
    # `sharding (None, 'model')`.
    self.dot1 = nnx.Linear(
      depth, depth,
      kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
      use_bias=False,  # or use `bias_init` to give it annotation too
      rngs=rngs)

    # Initialize a weight param `w2` and annotate with sharding ('model', None).
    # Note that this is simply adding `.sharding` to the variable as metadata!
    self.w2 = nnx.Param(
      init_fn(rngs.params(), (depth, depth)),  # RNG key and shape for W2 creation
      sharding=('model', None),
    )

  def __call__(self, x: jax.Array):
    y = self.dot1(x)
    y = jax.nn.relu(y)
    # In data parallelism, input / intermediate value's first dimension (batch)
    # will be sharded on `data` axis
    y = jax.lax.with_sharding_constraint(y, PartitionSpec('data', 'model'))
    z = jnp.dot(y, self.w2.value)
    return z
```

### Understand sharding names

The so-called "sharding annotations" are essentially tuples of device axis names like `'data'`, `'model'` or `None`. This describes how each dimension of this JAX array should be sharded — either across one of the device mesh dimensions, or not sharded at all.

So, when you define `W1` with shape `(depth, depth)` and annotate as `(None, 'model')`:

* The first dimension will be replicated across all devices.
* The second dimension will be sharded over the `'model'` axis of the device mesh. This means `W1` will be sharded 4-way on devices `(0, 4)`, `(1, 5)`, `(2, 6)` and `(3, 7)`, in this dimension.

JAX's [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) guide offers more examples and explanations.

+++

## Initialize a sharded model

Now, you have annotations attached to the Flax [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable), but the actual weights have not been sharded yet. If you just go ahead and create this model, all [`jax.Array`s](https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array) are still stuck in device `0`. In practice, you'd want to avoid this, because a large model will "OOM" (will cause the device to run out of memory) in this situation, while all the other devices are not utilized.

```{code-cell} ipython3
unsharded_model = DotReluDot(1024, rngs=nnx.Rngs(0))

# You have annotations stuck there, yay!
print(unsharded_model.dot1.kernel.sharding)     # (None, 'model')
print(unsharded_model.w2.sharding)              # ('model', None)

# But the actual arrays are not sharded?
print(unsharded_model.dot1.kernel.value.sharding)  # SingleDeviceSharding
print(unsharded_model.w2.value.sharding)           # SingleDeviceSharding
```

Here, you should leverage JAX's compilation mechanism via Flax’s [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) to create the sharded model. The key is to initialize a model and assign shardings upon the model state within a `jit`ted function:

1. Use [`nnx.get_partition_spec`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.get_partition_spec) to strip out the `.sharding` annotations attached upon model variables.

1. Call [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) to bind the model state with the sharding annotations. This API tells the top-level `jit` how to shard a variable!

1. Throw away the unsharded state and return the model based upon the sharded state.

1. Compile the whole function with `nnx.jit`, which allows the output to be a stateful Flax NNX `Module`.

1. Run it under a device mesh context so that JAX knows which devices to shard it to.

The entire compiled `create_sharded_model()` function will directly generate a model with sharded JAX arrays, and no single-device "OOM" will happen!

```{code-cell} ipython3
@nnx.jit
def create_sharded_model():
  model = DotReluDot(1024, rngs=nnx.Rngs(0)) # Unsharded at this moment.
  state = nnx.state(model)                   # The model's state, a pure pytree.
  pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(model, sharded_state)           # The model is sharded now!
  return model

with mesh:
  sharded_model = create_sharded_model()

# They are some `GSPMDSharding` now - not a single device!
print(sharded_model.dot1.kernel.value.sharding)
print(sharded_model.w2.value.sharding)

# Check out their equivalency with some easier-to-read sharding descriptions
assert sharded_model.dot1.kernel.value.sharding.is_equivalent_to(
  NamedSharding(mesh, PartitionSpec(None, 'model')), ndim=2
)
assert sharded_model.w2.value.sharding.is_equivalent_to(
  NamedSharding(mesh, PartitionSpec('model', None)), ndim=2
)
```

You can view the sharding of any 1-D or 2-D array with [`jax.debug.visualize_array_sharding`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html):

```{code-cell} ipython3
print("sharded_model.dot1.kernel (None, 'model') :")
jax.debug.visualize_array_sharding(sharded_model.dot1.kernel.value)
print("sharded_model.w2 ('model', None) :")
jax.debug.visualize_array_sharding(sharded_model.w2.value)
```

### On `jax.lax.with_sharding_constraint` (semi-automatic parallelization)

The key to shard a JAX array is to call [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) inside a `jax.jit`ted function. Note that it will throw an error if not under a JAX device mesh context.

> **Note:** Both [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html) and [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) in the JAX documentation cover automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), and semi-automatic parallelization with `jax.jit` and [`jax.lax.with_sharding_constraint](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) in greater detail.

You may have noticed you also used [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) once in the model definition to constraint the sharding of an intermediate value. This is just to show that you can always use it orthogonally with the Flax NNX API if you want to explicitly shard values that are not model variables.

This brings a question: Why use the Flax NNX Annotation API then? Why not just add JAX sharding constraints inside the model definition? The most important reason is that you still need the explicit annotations to load a sharded model from an on-disk checkpoint. This is described in the next section.

+++

## Load a sharded model from a checkpoint

Now you learned how to initialize a sharded model without OOM, but what about loading it from a checkpoint on disk? JAX checkpointing libraries, such as [Orbax](https://orbax.readthedocs.io/en/latest/), usually support loading a model sharded if a sharding pytree is provided.

You can generate such a sharding pytree with Flax’s [`nnx.get_named_sharding`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.get_named_sharding). To avoid any real memory allocation, use the [`nnx.eval_shape`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.eval_shape) transform to generate a model of abstract JAX arrays, and only use its `.sharding` annotations to obtain the sharding tree.

Below is an example that demonstrates using Orbax's `StandardCheckpointer` API. (Go to the [Orbax documentation site](https://orbax.readthedocs.io/en/latest/) to learn about their latest and most recommended APIs.)

```{code-cell} ipython3
import orbax.checkpoint as ocp

# Save the sharded state.
sharded_state = nnx.state(sharded_model)
path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path / 'checkpoint_name', sharded_state)

# Load a sharded state from checkpoint, without `sharded_model` or `sharded_state`.
abs_model = nnx.eval_shape(lambda: DotReluDot(1024, rngs=nnx.Rngs(0)))
abs_state = nnx.state(abs_model)
# Orbax API expects a tree of abstract `jax.ShapeDtypeStruct`
# that contains both sharding and the shape/dtype of the arrays.
abs_state = jax.tree.map(
  lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
  abs_state, nnx.get_named_sharding(abs_state, mesh)
)
loaded_sharded = checkpointer.restore(path / 'checkpoint_name',
                                      target=abs_state)
jax.debug.visualize_array_sharding(loaded_sharded['dot1']['kernel'].value)
jax.debug.visualize_array_sharding(loaded_sharded['w2'].value)
```

## Compile the training loop

Now, after either initialization or loading the checkpoint, you have a sharded model. To carry out the compiled scaled up training, you need to shard the inputs as well.

- In the data parallelism example, the training data has its batch dimension sharded across the `data` device axis, so you should put your data in sharding `('data', None)`. You can use [`jax.device_put`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html#jax.device_put) for this.
- Note that with the correct sharding for all inputs, the output will be sharded in the most natural way even without `jit` compilation. 
- In the example below, even without [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) on the output `y`, it was still sharded as `('data', None)`.

> If you are interested in why: The second matmul of `DotReluDot.__call__` has two inputs of sharding `('data', 'model')` and `('model', None)`, in which both inputs' contraction axis are `model`. So a reduce-scatter matmul happened and will naturally shard the output as `('data', None)`. Check out the [JAX shard map collective guide](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-2-psum-scatter-the-result) and its examples if you want to learn mathematically how it happens at a low-level.

```{code-cell} ipython3
# In data parallelism, the first dimension (batch) will be sharded on the `data` axis.
data_sharding = NamedSharding(mesh, PartitionSpec('data', None))
input = jax.device_put(jnp.ones((8, 1024)), data_sharding)

with mesh:
  output = sharded_model(input)
print(output.shape)
jax.debug.visualize_array_sharding(output)  # Also sharded as `('data', None)`.
```

Now the rest of the training loop is pretty conventional - it is almost the same as the example in [Flax NNX Basics](https://flax.readthedocs.io/en/latest/nnx_basics.html#transforms):
- Except that the inputs and labels are also explicitly sharded.
- [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.jit) will adjust and automatically choose the best layout based on how its inputs are already sharded, so try out different shardings for your own model and inputs.

```{code-cell} ipython3
optimizer = nnx.Optimizer(sharded_model, optax.adam(1e-3))  # reference sharing

@nnx.jit
def train_step(model, optimizer, x, y):
  def loss_fn(model: DotReluDot):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)

  return loss

input = jax.device_put(jax.random.normal(jax.random.key(1), (8, 1024)), data_sharding)
label = jax.device_put(jax.random.normal(jax.random.key(2), (8, 1024)), data_sharding)

with mesh:
  for i in range(5):
    loss = train_step(sharded_model, optimizer, input, label)
    print(loss)    # Model (over-)fitting to the labels quickly.
```

## Profiling

If you are using a Google TPU pod or a pod slice, you can create a custom `block_all()` utility function, as defined below, to measure the performance:

```{code-cell} ipython3
%%timeit

def block_all(xs):
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

with mesh:
  new_state = block_all(train_step(sharded_model, optimizer, input, label))
```

## Logical axis annotation

JAX's [automatic](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) [SPMD]((https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD)) encourages users to explore different sharding layouts to find the optimal one. To this end, in Flax you have the option to annotate with more descriptive axis names (not just device mesh axis names like `'data'` and `'model'`), as long as you provide a mapping from your alias to the device mesh axes.

You can provide the mapping along with the annotation as another metadata of the corresponding [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable), or overwrite it at top-level. Check out the `LogicalDotReluDot()` example below.

```{code-cell} ipython3
# The mapping from alias annotation to the device mesh.
sharding_rules = (('batch', 'data'), ('hidden', 'model'), ('embed', None))

class LogicalDotReluDot(nnx.Module):
  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()

    # Initialize a sublayer `self.dot1`.
    self.dot1 = nnx.Linear(
      depth, depth,
      kernel_init=nnx.with_metadata(
        # Provide the sharding rules here.
        init_fn, sharding=('embed', 'hidden'), sharding_rules=sharding_rules),
      use_bias=False,
      rngs=rngs)

    # Initialize a weight param `w2`.
    self.w2 = nnx.Param(
      # Didn't provide the sharding rules here to show you how to overwrite it later.
      nnx.with_metadata(init_fn, sharding=('hidden', 'embed'))(
        rngs.params(), (depth, depth))
    )

  def __call__(self, x: jax.Array):
    y = self.dot1(x)
    y = jax.nn.relu(y)
    # Unfortunately the logical aliasing doesn't work on lower-level JAX calls.
    y = jax.lax.with_sharding_constraint(y, PartitionSpec('data', None))
    z = jnp.dot(y, self.w2.value)
    return z
```

If you didn't provide all `sharding_rule` annotations in the model definition, you can write a few lines to add it to Flax’s [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) of the model, before the call of [`nnx.get_partition_spec`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.get_partition_spec) or [`nnx.get_named_sharding`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.get_named_sharding).

```{code-cell} ipython3
def add_sharding_rule(vs: nnx.VariableState) -> nnx.VariableState:
  vs.sharding_rules = sharding_rules
  return vs

@nnx.jit
def create_sharded_logical_model():
  model = LogicalDotReluDot(1024, rngs=nnx.Rngs(0))
  state = nnx.state(model)
  state = jax.tree.map(add_sharding_rule, state,
                       is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(model, sharded_state)
  return model

with mesh:
  sharded_logical_model = create_sharded_logical_model()

jax.debug.visualize_array_sharding(sharded_logical_model.dot1.kernel.value)
jax.debug.visualize_array_sharding(sharded_logical_model.w2.value)

# Check out their equivalency with some easier-to-read sharding descriptions.
assert sharded_logical_model.dot1.kernel.value.sharding.is_equivalent_to(
  NamedSharding(mesh, PartitionSpec(None, 'model')), ndim=2
)
assert sharded_logical_model.w2.value.sharding.is_equivalent_to(
  NamedSharding(mesh, PartitionSpec('model', None)), ndim=2
)

with mesh:
  logical_output = sharded_logical_model(input)
  assert logical_output.sharding.is_equivalent_to(
    NamedSharding(mesh, PartitionSpec('data', None)), ndim=2
  )
```

### When to use device axis / logical axis

Choosing when to use a device or logical axis depends on how much you want to control the partitioning of your model:

* **Device mesh axis**:

  * For a simpler model, this can save you a few extra lines of code of converting the logical naming back to the device naming.

  * Shardings of intermediate *activation* values can only be done via [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) and device mesh axis. Therefore, if you want super fine-grained control over your model's sharding, directly using device mesh axis names everywhere might be less confusing.

* **Logical naming**: This is helpful if you want to experiment around and find the most optimal partition layout for your *model weights*.

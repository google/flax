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

This guide demonstrates how to scale up a Flax NNX model on multiple accelerators (GPUs or Google TPUs) using JAX's parallel programming APIs.

[Introduction to Parallel Programming](https://docs.jax.dev/en/latest/sharded-computation.html) is a fantastic guide to learn about the distributed programming essentials of JAX. It describes three parallelism APIs - automatic, explicit and manual - for different levels of control.

This guide will primarily cover the automatic scenario, which use the [`jax.jit`](https://jax.readthedocs.io/en/latest/jit-compilation.html) to compile your single-device code as multi-device. You will use [`flax.nnx.spmd`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html) APIs to annotate your model variables with how it should be sharded.

If you want to follow explicit sharding style, follow [JAX Explicit Sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html) guide and use JAX's relevant APIs. No API on Flax side is needed.

+++

### Setup

```{code-cell} ipython3
from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import optax
import flax
from flax import nnx

# Ignore this if you are already running on a TPU or GPU
if not jax._src.xla_bridge.backends_are_initialized():
  jax.config.update('jax_num_cpu_devices', 8)
print(f'You have 8 “fake” JAX devices now: {jax.devices()}')
```

Set up a `2x4` device mesh as the [JAX data sharding tutorial](https://docs.jax.dev/en/latest/sharded-computation.html#key-concept-data-sharding) instructs.

In this guide we use a standard FSDP layout and shard our devices on two axes - `data` and `model`, for doing batch data parallelism and tensor parallelism.

```{code-cell} ipython3
# Create an auto-mode mesh of two dimensions and annotate each axis with a name.
auto_mesh = jax.make_mesh((2, 4), ('data', 'model'))
```

> Compatibility Note: This guide covers the [eager sharding feature](https://flax.readthedocs.io/en/latest/flip/4844-var-eager-sharding.html) that greatly simplifies creating sharded model. If your project already used Flax GSPMD API on version `flax<0.12`, you might have turned the feature off to keep your code working. Users can toggle this feature using the `nnx.use_eager_sharding` function.

```{code-cell} ipython3
nnx.use_eager_sharding(True)
assert nnx.using_eager_sharding()
```

The `nnx.use_eager_sharding` function can also be used as a context manager to toggle the eager sharding feature within a specific scope.

```{code-cell} ipython3
with nnx.use_eager_sharding(False):
  assert not nnx.using_eager_sharding()
```

You can also enable eager sharding on a per-variable basis by passing `eager_sharding=False` during variable initialization. The mesh can also be passed this way.

```{code-cell} ipython3
nnx.Param(jnp.ones(4,4), sharding_names=(None, 'model'), eager_sharding=True, mesh=auto_mesh)
```

## Shard a single-array model

Let's begin by sharding the simplest component possible - a Flax variable.

When you define a Flax variable, you can pass in a metadata field called `sharding_names`, to specify how the underlying JAX array should be sharded. This field should be a tuple of names, each of which refer to how an axis of the array should be sharded.

**You must have an existing device mesh** and create a sharding-annotated `nnx.Variable` within its scope. This allows the result variable to be sharded accordingly on those devices. The device mesh can be your actual accelerator mesh, or a dummy fake CPU mesh like in this notebook.

```{code-cell} ipython3
rngs = nnx.Rngs(0)

with jax.set_mesh(auto_mesh):
  w = nnx.Param(
    rngs.lecun_normal()((4, 8)),
    sharding_names=(None, 'model')
  )
  print(w.sharding.spec)
  jax.debug.visualize_array_sharding(w)  # already sharded!
```

### Initialize with style

When using existing modules, you can apply [`flax.nnx.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/spmd.html#flax.nnx.with_partitioning) on initializers to achieve the same effect. Here we create a sharded `nnx.Linear` module with only the kernel weight.

Also, you should use `jax.jit` for the whole initialization for maximum performance. This is because without `jax.jit`, a single-device variable must be created first before we apply sharding constraints and then make it sharded, which is wasteful. `jax.jit` will automatically optimize this out.

```{code-cell} ipython3
@jax.jit
def init_sharded_linear(key):
  init_fn = nnx.nn.linear.default_kernel_init
  # Shard your parameter along `model` dimension, as in model/tensor parallelism
  return nnx.Linear(4, 8, use_bias=False, rngs=nnx.Rngs(key),
                    kernel_init=nnx.with_partitioning(init_fn, (None, 'model')))

with jax.set_mesh(auto_mesh):
  key= rngs()
  linear = init_sharded_linear(key)
  assert linear.kernel.sharding.spec == P(None, 'model') # already sharded!
```

### Run the model

If you also shard your input correctly, JAX would be able to carry out the most natural and optimized computation and produce your output as sharded.

You should still make sure to `jax.jit` for maximum performance, and also to explicitly control how each array is sharded when you want to. We will give an example of that control in the next section.

> Note: You need to `jax.jit` a pure function that takes the model as an argument, instead of jitting the callable model directly.

```{code-cell} ipython3
# For simple computations, you can get correctly-sharded output without jitting
# In this case, ('data', None) @ (None, 'model') = ('data', 'model')
with jax.set_mesh(auto_mesh):
  # Create your input data, sharded along `data` dimension, as in data parallelism
  x = jax.device_put(jnp.ones((16, 4)), P('data', None))

  # Run the model forward function, jitted
  y = jax.jit(lambda m, x: m(x))(linear, x)
  print(y.sharding.spec)                       # sharded: ('data', 'model')
  jax.debug.visualize_array_sharding(y)
```

## Shard a wholesome model

Now we construct a more wholesome model to show a few advanced tricks. Check out this simple `DotReluDot` module that does two matmuls, and the `MultiDotReluDot` module that creates an arbitrary stack of `DotReluDot` sublayers.

Make note of the following:

* **Additional axis annotation**: Transforms like `vmap` and `scan` will add additional dimensions to the JAX arrays. Unfortunately, in auto sharding mode you will need to use `nnx.vmap` and `nnx.scan` instead of raw JAX transforms, so that both JAX and Flax knows how to shard this dimension. You won't need this in [explicit sharding mode](#explicit-sharding).

* [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#constraining-shardings-of-intermediates-in-jitted-code): They can help you to enforce specific shardings on intermediate activations. Only works under an auto mode mesh context.

```{code-cell} ipython3
class DotReluDot(nnx.Module):
  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()
    self.dot1 = nnx.Linear(
      depth, depth,
      kernel_init=nnx.with_partitioning(init_fn, (None, 'model')),
      use_bias=False,  # or use `bias_init` to give it annotation too
      rngs=rngs)
    self.w2 = nnx.Param(
      init_fn(rngs.params(), (depth, depth)),  # RNG key and shape for W2 creation
      sharding=('model', None),
    )

  def __call__(self, x: jax.Array):
    y = self.dot1(x)
    y = jax.nn.relu(y)
    y = jax.lax.with_sharding_constraint(y, P('data', 'model'))
    z = jnp.dot(y, self.w2[...])
    return z

class MultiDotReluDot(nnx.Module):
  def __init__(self, depth: int, num_layers: int, rngs: nnx.Rngs):
    # Annotate the additional axis with sharding=None, meaning it will be
    # replicated across all devices.
    @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None})
    def create_sublayers(r):
      return DotReluDot(depth, r)
    self.layers = create_sublayers(rngs.fork(split=num_layers))

  def __call__(self, x):
    def scan_over_layers(x, layer):
      return layer(x), None
    x, _ = jax.lax.scan(scan_over_layers, x, self.layers)
    return x
```

Now a sample training loop, using `jax.jit`.

```{code-cell} ipython3
@jax.jit
def train_step(model, optimizer, x, y):
  def loss_fn(model: DotReluDot):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = jax.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)
  return model, loss


with jax.set_mesh(auto_mesh):
  # Training data
  input = jax.device_put(rngs.normal((8, 1024)), P('data', None))
  label = jax.device_put(rngs.normal((8, 1024)), P('data', None))
  # Model and optimizer
  model = MultiDotReluDot(1024, 2, rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

  # The loop
  for i in range(5):
    model, loss = train_step(model, optimizer, input, label)
    print(loss)    # Model (over-)fitting to the labels quickly.
```

## Profiling

If you are using a Google TPU pod or a pod slice, you can create a custom `block_all()` utility function, as defined below, to measure the performance:

```{code-cell} ipython3
%%timeit

def block_all(xs):
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

with jax.set_mesh(auto_mesh):
  new_state = block_all(train_step(model, optimizer, input, label))
```

## Load a sharded model from a checkpoint

Now you learned how to initialize a sharded model without OOM, but what about saving and loading it from a checkpoint on disk? JAX checkpointing libraries, such as [Orbax](https://orbax.readthedocs.io/en/latest/), support loading a model distributedly if a sharding pytree is provided. Below is an example that uses Orbax's `StandardCheckpointer` API.

Make sure you save a model's state, especially if your model shares some variables across modules. Given a You can generate an identical abstract pytree with shardings using Flax’s `nnx.get_abstract_model`.

```{code-cell} ipython3
import orbax.checkpoint as ocp

# Save the sharded state.
sharded_state = nnx.state(model)
path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path / 'checkpoint_name', sharded_state)

# Load a sharded state from the checkpoint.
graphdef, abs_state = nnx.get_abstract_model(
  lambda: MultiDotReluDot(1024, 2, rngs=nnx.Rngs(0)), auto_mesh)
restored_state = checkpointer.restore(path / 'checkpoint_name',
                                      target=abs_state)
restored_model = nnx.merge(graphdef, abs_state)
print(restored_model.layers.dot1.kernel.sharding.spec)
print(restored_model.layers.dot1.kernel.shape)
```

## Logical axis annotation

JAX's [automatic](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) [SPMD](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) encourages users to explore different sharding layouts to find the optimal one. To this end, in Flax you have the option to annotate with more descriptive axis names (not just device mesh axis names like `'data'` and `'model'`), as long as you provide a mapping from your alias to the device mesh axes.

You can provide the mapping along with the annotation as another metadata of the corresponding [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable), or overwrite it at top-level. Check out the `LogicalDotReluDot` example below.

```{code-cell} ipython3
# The mapping from alias annotation to the device mesh.
sharding_rules = (('batch', 'data'), ('hidden', 'model'), ('embed', None))

class LogicalDotReluDot(nnx.Module):
  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()
    self.dot1 = nnx.Linear(
      depth, depth,
      kernel_init=nnx.with_partitioning(init_fn, ('embed', 'hidden')),
      use_bias=False,  # or use `bias_init` to give it annotation too
      rngs=rngs)
    self.w2 = nnx.Param(
      init_fn(rngs.params(), (depth, depth)),  # RNG key and shape for W2 creation
      sharding=('hidden', 'embed'),
    )

  def __call__(self, x: jax.Array):
    y = self.dot1(x)
    y = jax.nn.relu(y)
    # Unfortunately the logical aliasing doesn't work on lower-level JAX calls.
    y = jax.lax.with_sharding_constraint(y, P('data', None))
    z = jnp.dot(y, self.w2[...])
    return z

class LogicalMultiDotReluDot(nnx.Module):
  def __init__(self, depth: int, num_layers: int, rngs: nnx.Rngs):
    @nnx.vmap(transform_metadata={nnx.PARTITION_NAME: None})
    def create_sublayers(r):
      return LogicalDotReluDot(depth, r)
    self.layers = create_sublayers(rngs.fork(split=num_layers))

  def __call__(self, x):
    def scan_over_layers(x, layer):
      return layer(x), None
    x, _ = jax.lax.scan(scan_over_layers, x, self.layers)
    return x
```

If you didn't provide all `sharding_rule` annotations in the model definition, you can apply them at top level by put them into the context via `nnx.logical_axis_rules`.

```{code-cell} ipython3
with jax.set_mesh(auto_mesh), nnx.logical_axis_rules(sharding_rules):
  # Model and optimizer
  logical_model = LogicalMultiDotReluDot(1024, 2, rngs=nnx.Rngs(0))
  logical_output = logical_model(input)

# Check out their equivalency with some easier-to-read sharding descriptions.
assert logical_model.layers.dot1.kernel.sharding.is_equivalent_to(
  NamedSharding(auto_mesh, P(None, None, 'model')), ndim=3
)
assert logical_model.layers.w2.sharding.is_equivalent_to(
  NamedSharding(auto_mesh, P(None, 'model', None)), ndim=3
)
assert logical_output.sharding.is_equivalent_to(
  NamedSharding(auto_mesh, P('data', None)), ndim=2
)
```

### When to use device axis / logical axis

Choosing when to use a device or logical axis depends on how much you want to control the partitioning of your model:

* **Device mesh axis**:

  * For a simpler model, this can save you a few extra lines of code of converting the logical naming back to the device naming.

  * Shardings of intermediate *activation* values can only be done via [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html) and device mesh axis. Therefore, if you want super fine-grained control over your model's sharding, directly using device mesh axis names everywhere might be less confusing.

* **Logical naming**: This is helpful if you want to experiment around and find the most optimal partition layout for your *model weights*.

+++

## Explicit sharding

[Explicit sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html), also called "sharding-in-types", is a new JAX sharding feature that allows every sharding of every array to be deterministic and explicit. Instead of letting XLA compiler figure out the shardings, you as user would explicitly state the shardings via JAX APIs.

For education purposes, we provide a simple Flax model example using explicit sharding. Note how you specify shardings for this model:

* Parameters: `out_sharding` argument passed into JAX initializers.

* Ambigious computations like `jnp.dot`: provide `out_sharding` argument to specify the output sharding.

* Additional dimension from transforms: use `jax.vmap`'s argument `spmd_axis_name`, instead of Flax lifted transforms.

```{code-cell} ipython3
# Explicit axis mesh
explicit_mesh = jax.make_mesh((2, 4), ('data', 'model'),
                              axis_types=(AxisType.Explicit, AxisType.Explicit))

class ExplicitDotReluDot(nnx.Module):
  def __init__(self, depth: int, rngs: nnx.Rngs):
    init_fn = nnx.initializers.lecun_normal()
    self.dot1 = nnx.Linear(
      depth, depth,
      kernel_init=partial(init_fn, out_sharding=P(None, 'model')),
      use_bias=False,
      rngs=rngs)
    self.w2 = nnx.Param(
      init_fn(rngs.params(), (depth, depth), out_sharding=P('model', None)),
    )
    self.b2 = nnx.Param(jnp.zeros((depth, ), out_sharding=P(None,)))

  def __call__(self, x: jax.Array):
    y = self.dot1(x)
    y = jax.nn.relu(y)
    z = jnp.dot(y, self.w2[...], out_sharding=P('data', None))
    z = z + self.b2
    return z


class ExplicitMultiDotReluDot(nnx.Module):
  def __init__(self, depth: int, num_layers: int, rngs: nnx.Rngs):
    # Annotate the additional axis with sharding=None, meaning it will be
    # replicated across all devices.
    @partial(jax.vmap, spmd_axis_name=None)
    def create_sublayers(r):
      return ExplicitDotReluDot(depth, r)
    self.layers = create_sublayers(rngs.fork(split=num_layers))

  def __call__(self, x):
    def scan_over_layers(x, layer):
      return layer(x), None
    x, _ = jax.lax.scan(scan_over_layers, x, self.layers)
    return x


with jax.set_mesh(explicit_mesh):
  model = ExplicitMultiDotReluDot(1024, 2, rngs=nnx.Rngs(0))
  x = jax.device_put(rngs.normal((8, 1024)),
                     NamedSharding(explicit_mesh, P('data', None)))
  y = model(x)

print(model.layers.dot1.kernel.sharding.spec)
print(model.layers.w2.sharding.spec)
assert x.sharding.is_equivalent_to(y.sharding, ndim=2)
```

One thing easier in explicit mode is that you can obtain the abstract array tree with shardings via `jax.eval_shape`, instead of calling `nnx.get_abstract_sharding`. This is not possible in auto mode.

```{code-cell} ipython3
# Get the sharding tree to load checkpoint with
with jax.set_mesh(explicit_mesh):
  abs_model = jax.eval_shape(
    lambda: ExplicitMultiDotReluDot(1024, 2, rngs=nnx.Rngs(0)))
  print(abs_model.layers.dot1.kernel.sharding.spec)
  print(abs_model.layers.w2.sharding.spec)
```

## Further readings

JAX has abundant documentation on scaled computing.

- [Introduction to parallel programming](https://jax.readthedocs.io/en/latest/sharded-computation.html): A 101 level tutorial covering the basics of automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), semi-automatic parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) and [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html), and manual sharding with [`shard_map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.shard_map.shard_map.html#jax.experimental.shard_map.shard_map).
- [JAX in multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).
- [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): A more detailed tutorial about parallelization with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) and [`jax.lax.with_sharding_constraint`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).

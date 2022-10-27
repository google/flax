---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Checkpointing with `flax.training.checkpoints`

In this guide, you will learn about [`flax.training.checkpoints`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#module-flax.training.checkpoints)—a simplistic and generic checkpointing library built into Flax. With Flax Checkpoints, you can save and load model parameters, metadata, and a variety of Python data. In addition, it provides basic features for versioning, automatic bookkeeping of past checkpoints, and asynchronous saving to reduce training wait time.

This guide covers the following:

* Basic saving and loading of checkpoints with [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint).
* More flexible and sustainable ways to load checkpoints ([`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint)).
* How to save and load checkpoints when you run in multi-host scenarios with
[`flax.training.checkpoints.save_checkpoint_multiprocess`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess).

## Setup

Install/upgrade Flax, JAX, [Optax](https://optax.readthedocs.io/) and [TensorStore](https://google.github.io/tensorstore/). For JAX installation with GPU/TPU support, visit [this section on GitHub](https://github.com/google/jax#installation).

```{code-cell} ipython3
!pip install -U -q flax jax jaxlib optax tensorstore
```

Note: Before running `import jax`, create eight fake devices to mimic multi-host checkpointing in this notebook. Note that the order of imports is important here. The `os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'` command works only with the CPU backend. This means it won't work with GPU/TPU acceleration on if you're running this notebook in Google Colab. If you are already running the code on multiple devices (for example, in a 4x2 TPU environment), you can skip running the next cell.

```{code-cell} ipython3
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```{code-cell} ipython3
from typing import Optional, Any
import shutil

import numpy as np
import jax
from jax import random, numpy as jnp
from jax.experimental import maps, PartitionSpec, pjit
from jax.experimental.gda_serialization.serialization import GlobalAsyncCheckpointManager

import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
from flax import struct, serialization
import optax
```

## Save checkpoints

In Flax, you save and load any given JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) using the `flax.training.checkpoints` package. This includes not only typical Python and NumPy containers, but also customized classes extended from [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass). That means you can store almost any data generated—not only your model parameters, but any arrays/dictionaries, metadata/configs, and so on.

Create a pytree with many data structures and containers, and play with it:

```{code-cell} ipython3
# A simple model with one linear layer.
key1, key2 = random.split(random.PRNGKey(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.1)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)

# Some arbitrary nested pytree with a dictionary, a string, and a NumPy array.
config = {'dimensions': np.array([5, 3]), 'name': 'dense'}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}
ckpt
```

Now save the checkpoint. You can add annotations like step number, prefix, and so on to your checkpoint.

When saving a checkpoint, Flax will bookkeep the existing checkpoints based on your arguments. For example, by setting `overwrite=False` in [`flax.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint), Flax will not automatically save your checkpoint if there is already a step that is equal to the current one or newer is present in the checkpoint directory. By setting `keep=2`, Flax will keep a maximum of 2 checkpoints in the directory. Learn more in the [API reference](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#module-flax.training.checkpoints).

```{code-cell} ipython3
# Import Flax Checkpoints.
from flax.training import checkpoints

from jax.experimental.gda_serialization.serialization import GlobalAsyncCheckpointManager

ckpt_dir = 'tmp/flax-checkpointing'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                            target=ckpt,
                            step=0,
                            overwrite=False,
                            keep=2)
```

## Restore checkpoints

To restore a checkpoint, use [`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint) and pass in the checkpoint directory. Flax will automatically select the latest checkpoint in the directory. You can also choose to specify a step number or the path of the checkpoint file. You can always restore a pytree out of your checkpoints by setting `target=None`.

```{code-cell} ipython3
raw_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
raw_restored
```

However, when using `target=None`, the restored `raw_restored` will be different from the original `ckpt` in the following ways:

1. There is no TrainState now, and only some raw weights and Optax state numbers remain.
1. `metadata.dimentions` and `data` should be arrays, but restored as dictionaries with integers as keys.
1. Previously, `data[0]` was a JAX NumPy array (`jnp.array`) —now it's a NumPy array (`numpy.array`).

While (3) would not affect future work because JAX will automatically convert NumPy arrays to JAX arrays once the computation starts, (1) and (2) may lead to confusions.

To resolve this, you should pass an example `target` in `flax.training.checkpoints.restore_checkpoint` to let Flax know exactly what structure it should restore to. The `target` should introduce any custom Flax dataclasses explicitly, and have the same structure as the saved checkpoint.

It's often recommended to refactor out the process of initializing a checkpoint's structure (for example, a `TrainState`), so that saving/loading is easier and less error-prone. This is because complicated objects like `apply_fn` and `tx` are not stored in the checkpoint file and must be initiated by code.

```{code-cell} ipython3
empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=np.zeros_like(variables['params']),  # values of the tree leaf doesn't matter
    tx=tx,
)
target = {'model': empty_state, 'config': None, 'data': [jnp.zeros_like(x1)]}
state_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=0)
state_restored
```

### Backward/forward dataclass compatibility

The flexibility of using *Flax dataclasses*—[`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass)—means that changes in Flax dataclass fields may break your existing checkpoints. For example, if you decide to add a field `batch_stats` to your `TrainState`, old checkpoints without this field may not be successfully restored. Same goes for removing a field in your dataclass.

Note: Flax supports [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass), not Python's built-in `dataclasses.dataclass`.

```{code-cell} ipython3
class CustomTrainState(train_state.TrainState):
    batch_stats: Any = None

custom_state = CustomTrainState.create(
    apply_fn=state.apply_fn,
    params=state.params,
    tx=state.tx,
    batch_stats=np.arange(10),
)

# Use a custom state to read the old `TrainState` checkpoint.
custom_target = {'model': custom_state, 'config': None, 'data': [jnp.zeros_like(x1)]}
try:
    checkpoints.restore_checkpoint(ckpt_dir, target=custom_target, step=0)
except ValueError as e:
    print('ValueError when target state has an unmentioned field:')
    print(e)
    print('')


# Use the old `TrainState` to read the custom state checkpoint.
custom_ckpt = {'model': custom_state, 'config': config, 'data': [x1]}
checkpoints.save_checkpoint(ckpt_dir, custom_ckpt, step=1, overwrite=True, keep=2)
try:
    checkpoints.restore_checkpoint(ckpt_dir, target=target, step=1)
except ValueError as e:
    print('ValueError when target state misses a recorded field:')
    print(e)
    
```

It is recommended to keep your checkpoints up to date with your pytree dataclass definitions. You can keep a copy of your code along with your checkpoints.

But if you must restore checkpoints and Flax dataclasses with incompatible fields, you can manually add/remove corresponding fields before passing in the correct target structure:

```{code-cell} ipython3
# Pass no target to get a raw state dictionary first.
raw_state_dict = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=0)
# Add/remove fields as needed.
raw_state_dict['model']['batch_stats'] = np.arange(10)
# Restore the classes with correct target now
serialization.from_state_dict(custom_target, raw_state_dict)
```

## Asynchronized checkpointing

Checkpointing is I/O heavy, and if you have a large amount of data to save, it may be worthwhile to put it into a background thread, while continuing with your training.

You can do this by creating an `async_manager` (as demonstrated in the code cell below) and let it track your save thread.

`async_manager` is a parameter in [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint) with the default setting at `None`.

Note that you should use the same `async_manager` to handle all your async saves across your training steps, so that it can make sure that a previous async save is done before the next one begins. This allows bookkeeping like `keep` and `overwrite` to be consistent across steps.

Whenever you want to explicitly wait until an async save is done, you can call `async_manager.wait_previous_save()`.

```{code-cell} ipython3
am = checkpoints.AsyncManager()

# Mimic a training loop here:
for step in range(2, 3):
    checkpoints.save_checkpoint(ckpt_dir, ckpt, step=2, overwrite=True, keep=3, async_manager=am)
    # ... Continue with your work...

# ... Until a time when you want to wait until the save completes:
am.wait_previous_save()  # Block until the checkpoint saving is completed.
checkpoints.restore_checkpoint(ckpt_dir, target=None, step=2)
```

## Multi-host/multi-process checkpointing

JAX provides a few ways to scale up your code on multiple hosts at the same time. This usually happens when the number of devices (CPU/GPU/TPU) is so large that different devices are managed by different hosts (CPU). To get started on JAX in multi-process settings, check out [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html).

In the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm with JAX [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html), a large multi-process array can have its data sharded across different devices (check out the `pjit` [JAX-101 tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)). This kind of data array needs a special experimental JAX API—[`GlobalAsyncCheckpointManager`](https://github.com/google/jax/blob/c7dcd0913bc8b4878a7e45184553c331254b801a/jax/experimental/gda_serialization/serialization.py#L452)—to save and restore checkpoints. This API lets each host dump its data shards to a single shared storage, such as a Google Cloud bucket.

Flax provides an easy interface for users to pass in a `GlobalAsyncCheckpointManager` and store pytrees with multi-process arrays in the same fashion as single-process pytrees. Just use [`flax.training.checkpoints.save_checkpoint_multiprocess`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess) with the same arguments.

Unfortunately, Python Jupyter notebooks are single-host only and cannot activate the multi-host mode. As a workaround, use the following code as a sample to run your multi-host checkpointing.

```{code-cell} ipython3
# Set up a checkpoint with a multi-process array.

# In reality, you should set this with multiple num_processes.
# Refer to https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster
jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)
```

```{code-cell} ipython3
# Create a multi-process array.
mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x', 'y'))

f = pjit.pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', 'y'))

with maps.Mesh(mesh.devices, mesh.axis_names):
    mp_array = f(np.arange(8 * 2).reshape(8, 2))

# Make it a pytree as usual.
mp_ckpt = {'model': mp_array}
```

### Example: Save a checkpoint in a multi-process setting with `save_checkpoint_multiprocess`

The arguments in [`flax.training.checkpoints.save_checkpoint_multiprocess`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess) are the same as in [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint), except for the additional `gda_manager` argument.

If your checkpoint is too large, you can specify `timeout_secs` in the manager and give it more time to finish writing.

```{code-cell} ipython3
gacm = GlobalAsyncCheckpointManager(timeout_secs=50)
checkpoints.save_checkpoint_multiprocess(ckpt_dir, mp_ckpt, step=3, overwrite=True, 
                                         keep=4, gda_manager=gacm)
```

### Example: Restoring a checkpoint with `flax.training.checkpoints.restore_checkpoint`

Note that, when using [`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint), you need to pass a `target` with valid multi-process arrays at the correct structural location. Flax only uses the `target` arrays' meshes and mesh axes to restore the checkpoint. This means that the multi-process array in the `target` arg doesn't have to be as large as your checkpoint's size (the shape of the multi-process array doesn't need to have the same shape as the actual array in your checkpoint).

```{code-cell} ipython3
with maps.Mesh(mesh.devices, mesh.axis_names):
    mp_smaller_array = f(np.zeros(8).reshape(4, 2))

mp_target = {'model': mp_smaller_array}
mp_restored = checkpoints.restore_checkpoint(ckpt_dir, target=mp_target, 
                                             step=3, gda_manager=gacm)
mp_restored
```

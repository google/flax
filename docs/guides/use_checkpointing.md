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

# Save and load checkpoints

In this guide, you will learn about saving and loading checkpoints with Flax and [Orbax](https://github.com/google/orbax). With Flax, you can save and load model parameters, metadata, and a variety of Python data using Orbax. 

Orbax provides a customizable and flexible API for various array types and storage formats. In addition, Flax provides basic features for versioning, automatic bookkeeping of past checkpoints, and asynchronous saving to reduce training wait time.

> **_Ongoing migration:_** In the foreseeable future, Flax's checkpointing functionality will gradually be migrated to Orbax from `flax.training.checkpoints`. All existing features in the Flax API will continue to be supported, but the API will change. You are encouraged to try out the new API by creating an [`orbax.Checkpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/checkpointer.py) and pass it in your Flax API calls as an argument `orbax_checkpointer`, as demonstrated later in this guide. This guide provides the most up-to-date code examples for using Orbax and Flax for checkpointing.

This guide covers the following:

* Basic saving and loading of checkpoints with [`orbax.Checkpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/checkpointer.py) and [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint).
* More flexible and sustainable ways to load checkpoints ([`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint)).
* How to save and load checkpoints when you run in multi-host scenarios with
[`flax.training.checkpoints.save_checkpoint_multiprocess`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess).


## Setup

Install/upgrade Flax and [Orbax](https://github.com/google/orbax). For JAX installation with GPU/TPU support, visit [this section on GitHub](https://github.com/google/jax#installation).

```python
!pip install -U -qq flax orbax

# Orbax needs to enable asyncio in a Colab environment.
!pip install -qq nest_asyncio
```

Note: Before running `import jax`, create eight fake devices to mimic [multi-host environment](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html?#aside-hosts-and-devices-in-jax) this notebook. Note that the order of imports is important here. The `os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'` command works only with the CPU backend. This means it won't work with GPU/TPU acceleration on if you're running this notebook in Google Colab. If you are already running the code on multiple devices (for example, in a 4x2 TPU environment), you can skip running the next cell.

```python
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```python
from typing import Optional, Any
import shutil

import numpy as np
import jax
from jax import random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint as orbax

import optax
import nest_asyncio
nest_asyncio.apply()
```

## Save checkpoints

In Flax, you save and load any given JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) using the `flax.training.checkpoints` package. This includes not only typical Python and NumPy containers, but also customized classes extended from [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass). That means you can store almost any data generated—not only your model parameters, but any arrays/dictionaries, metadata/configs, and so on.

Create a pytree with many data structures and containers, and play with it:

```python
# A simple model with one linear layer.
key1, key2 = random.split(random.PRNGKey(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)
# Perform a simple gradient update similar to the one during a normal training workflow.
state = state.apply_gradients(grads=jax.tree_map(jnp.ones_like, state.params))

# Some arbitrary nested pytree with a dictionary, a string, and a NumPy array.
config = {'dimensions': np.array([5, 3]), 'name': 'dense'}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}
ckpt
```

Now save the checkpoint with Flax and Orbax. You can add annotations like step number, prefix, and so on to your checkpoint.

When saving a checkpoint, Flax will bookkeep the existing checkpoints based on your arguments. For example, by setting `overwrite=False` in [`flax.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint), Flax will not automatically save your checkpoint if there is already a step that is equal to or newer than the current one presently in the checkpoint directory. By setting `keep=2`, Flax will keep a maximum of 2 checkpoints in the directory. Learn more in the [API reference](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#module-flax.training.checkpoints).

You can start to use Orbax to handle the underlying save by creating an [`orbax.Checkpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/checkpointer.py), and pass it into the `flax.checkpoints.save_checkpoint` call.

```python
# Import Flax Checkpoints.
from flax.training import checkpoints

ckpt_dir = 'tmp/flax-checkpointing'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                            target=ckpt,
                            step=0,
                            overwrite=False,
                            keep=2,
                            orbax_checkpointer=orbax_checkpointer)
```

## Restore checkpoints

To restore a checkpoint, use [`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint) and pass in the checkpoint directory. Flax will automatically select the latest checkpoint in the directory. You can also choose to specify a step number or the path of the checkpoint file.

With the migration to Orbax in progress, `restore_checkpoint` can automatically identify whether a checkpoint is saved in the legacy (Flax) or Orbax version, and restore the pytree correctly.

You can always restore a pytree out of your checkpoints by setting `target=None`.

```python
raw_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
raw_restored
```

However, when using `target=None`, the restored `raw_restored` will be different from the original `ckpt` in the following ways:

1. There is no TrainState now, and only some raw weights and Optax state numbers remain.
1. `metadata.dimensions` and `data` should be arrays, but restored as dictionaries with integers as keys.
1. Previously, `data[0]` was a JAX NumPy array (`jnp.array`) —now it's a NumPy array (`numpy.array`).

While (3) would not affect future work because JAX will [automatically convert](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html) NumPy arrays to JAX arrays once the computation starts, (1) and (2) may lead to confusions.

To resolve this, you should pass an example `target` in `flax.training.checkpoints.restore_checkpoint` to let Flax know exactly what structure it should restore to. The `target` should introduce any custom Flax dataclasses explicitly, and have the same structure as the saved checkpoint.

It's often recommended to refactor out the process of initializing a checkpoint's structure (for example, a [`TrainState`](https://flax.readthedocs.io/en/latest/flip/1009-optimizer-api.html?#train-state)), so that saving/loading is easier and less error-prone. This is because complicated objects like `apply_fn` and `tx` (optimizer) are not stored in the checkpoint file and must be initiated by code.

```python
empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=jax.tree_map(np.zeros_like, variables['params']),  # values of the tree leaf doesn't matter
    tx=tx,
)
empty_config = {'dimensions': np.array([0, 0]), 'name': ''}
target = {'model': empty_state, 'config': empty_config, 'data': [jnp.zeros_like(x1)]}
state_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=0)
state_restored
```

### Backward/forward dataclass compatibility

The flexibility of using *Flax dataclasses*—[`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass)—means that changes in Flax dataclass fields may break your existing checkpoints. For example, if you decide to add a field `batch_stats` to your `TrainState` (like when using [batch normalization](https://flax.readthedocs.io/en/latest/guides/batch_norm.html)), old checkpoints without this field may not be successfully restored. Same goes for removing a field in your dataclass.

Note: Flax supports [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass), not Python's built-in `dataclasses.dataclass`.

```python
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
checkpoints.save_checkpoint(ckpt_dir, custom_ckpt, step=1, overwrite=True, 
                            keep=2, orbax_checkpointer=orbax_checkpointer)
try:
    checkpoints.restore_checkpoint(ckpt_dir, target=target, step=1)
except ValueError as e:
    print('ValueError when target state misses a recorded field:')
    print(e)
    
```

It is recommended to keep your checkpoints up to date with your pytree dataclass definitions. You can keep a copy of your code along with your checkpoints.

But if you must restore checkpoints and Flax dataclasses with incompatible fields, you can manually add/remove corresponding fields before passing in the correct target structure:

```python
# Pass no target to get a raw state dictionary first.
raw_state_dict = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=0)
# Add/remove fields as needed.
raw_state_dict['model']['batch_stats'] = np.arange(10)
# Restore the classes with correct target now
serialization.from_state_dict(custom_target, raw_state_dict)
```

## Asynchronized checkpointing

Checkpointing is I/O heavy, and if you have a large amount of data to save, it may be worthwhile to put it into a background thread, while continuing with your training.

You can do this by creating an [`orbax.AsyncCheckpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/async_checkpointer.py) (as demonstrated in the code cell below) and let it track your save thread.

Note: You should use the same `async_checkpointer` to handle all your async saves across your training steps, so that it can make sure that a previous async save is done before the next one begins. This enables bookkeeping, such as `keep` (the number of checkpoints) and `overwrite` to be consistent across steps.

Whenever you want to explicitly wait until an async save is done, you can call `async_checkpointer.wait_until_finished()`. Alternatively, you can pass in `orbax_checkpointer=async_checkpointer` when running `restore_checkpoint` and Flax will automatically wait and restore safely.

```python
# `orbax.AsyncCheckpointer` needs some multi-process initialization, because it was
# originally designed for multi-process large model checkpointing.
# For Python notebooks or other single-process setting, just set up with `num_processes=1`.
# Refer to https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster
# for how to set it up in multi-process scenarios.
jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)

async_checkpointer = orbax.AsyncCheckpointer(orbax.PyTreeCheckpointHandler(), timeout_secs=50)

# Mimic a training loop here:
for step in range(2, 3):
    checkpoints.save_checkpoint(ckpt_dir, ckpt, step=2, overwrite=True, keep=3,
                                orbax_checkpointer=async_checkpointer)
    # ... Continue with your work...

# ... Until a time when you want to wait until the save completes:
async_checkpointer.wait_until_finished()  # Blocks until the checkpoint saving is completed.
checkpoints.restore_checkpoint(ckpt_dir, target=None, step=2)
```

## Multi-host/multi-process checkpointing

JAX provides a few ways to scale up your code on multiple hosts at the same time. This usually happens when the number of devices (CPU/GPU/TPU) is so large that different devices are managed by different hosts (CPU). To get started on JAX in multi-process settings, check out [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html) and the [distributed array guide](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

In the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm with JAX [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html), a large multi-process array can have its data sharded across different devices (check out the `pjit` [JAX-101 tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)). When a multi-process array is serialized, each host dumps its data shards to a single shared storage, such as a Google Cloud bucket.

Orbax supports saving and loading pytrees with multi-process arrays in the same fashion as single-process pytrees. However, it's recommended to use the asynchronized [`orbax.AsyncCheckpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/async_checkpointer.py) to save large multi-process arrays on another thread, so that you can perform computation alongside the saves.

To save multi-process arrays, use [`flax.training.checkpoints.save_checkpoint_multiprocess()`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess) in place of `save_checkpoint()` and with the same arguments.

Unfortunately, Python Jupyter notebooks are single-host only and cannot activate the multi-host mode. You can treat the following code as an example for running your multi-host checkpointing:

```python
# Multi-host related imports.
from jax.experimental import maps, PartitionSpec, pjit
```

```python
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

The arguments in [`flax.training.checkpoints.save_checkpoint_multiprocess`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess) are the same as in [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint).

If your checkpoint is too large, you can specify `timeout_secs` in the manager and give it more time to finish writing.

```python
async_checkpointer = orbax.AsyncCheckpointer(orbax.PyTreeCheckpointHandler(), timeout_secs=50)
checkpoints.save_checkpoint_multiprocess(ckpt_dir, 
                                         mp_ckpt, 
                                         step=3, 
                                         overwrite=True, 
                                         keep=4, 
                                         orbax_checkpointer=async_checkpointer)
```

### Example: Restoring a checkpoint with `flax.training.checkpoints.restore_checkpoint`

Note that, when using [`flax.training.checkpoints.restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint), you need to pass a `target` with valid multi-process arrays at the correct structural location. Flax only uses the `target` arrays' meshes and mesh axes to restore the checkpoint. This means that the multi-process array in the `target` arg doesn't have to be as large as your checkpoint's size (the shape of the multi-process array doesn't need to have the same shape as the actual array in your checkpoint).

```python
with maps.Mesh(mesh.devices, mesh.axis_names):
    mp_smaller_array = f(np.zeros(8).reshape(4, 2))

mp_target = {'model': mp_smaller_array}
mp_restored = checkpoints.restore_checkpoint(ckpt_dir, 
                                             target=mp_target, 
                                             step=3,
                                             orbax_checkpointer=async_checkpointer)
mp_restored
```

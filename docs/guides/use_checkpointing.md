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

# Use Checkpointing

`flax.checkpoints` library is a simplistic, generic library to save and load model parameters, metadata and a variety of Python data. It also provides basic feature for versioning, automatic bookkeeping of past checkpoints, and async saving to reduce training wait time.

In this example you will find:

* Basic save/load of checkpoints.
* More flexible and sustainable ways to load checkpoints.
* How to save/load checkpoints when you run in multi-host scenarios.

+++

## Setup

```{code-cell} ipython3
# Create 8 fake devices to mimic multihost checkpointing
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

from typing import Optional, Any
import shutil
import time

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

## Saving Checkpoints

Flax checkpointing can save and load any given [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html). This includes not only typical Python and Numpy containers, but also customized classes extended from `flax.struct.dataclass`. That means you can store almost any data generated - not only your model params, but any arrays/dicts, metadata/configs, etc.

Let's create a pytree with many data structures and containers and play with it:

```{code-cell} ipython3
# Some simple model
key1, key2 = random.split(random.PRNGKey(0))
x1 = random.normal(key1, (5,))      # some JAX array
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.1)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)

# Some arbitary nested pytree with dict, string and numpy array
config = {'dimensions': np.array([5, 3]), 'name': 'dense'}

# Bundle them together!
ckpt = {'model': state, 'config': config, 'data': [x1]}
ckpt
```

Now save the checkpoint. You can add annotations like step number, prefix, etc to your checkpoint.

When saving the checkpoint, Flax will bookkeep existed checkpoints based on your args. For example, with `overwrite=False`, Flax will not automatically save your checkpoint if a step equal or newer is present in the checkpoint directory. With `keep=2` Flax will keep a maximum of 2 checkpoints in the directory. Explore [API reference](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#module-flax.training.checkpoints) for more options.

```{code-cell} ipython3
ckpt_dir = 'tmp/flax-checkpointing'
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existed checkpoints from last notebook run
checkpoints.save_checkpoint(ckpt_dir, ckpt, step=0, overwrite=False, keep=2)
```

## Restoring Checkpoints

To restore the checkpoint, pass in the checkpoint directory. Flax will automatically select the latest checkpoint in the directory. You can also choose to specify a step number or the path of the checkpoint file.

You could always restore a Pytree out of your checkpoints with `target=None`:

```{code-cell} ipython3
raw_restored = checkpoints.restore_checkpoint(ckpt_dir, target=None)
raw_restored
```

However, when using `target=None`, the restored `raw_restored` will be different from the original `ckpt` in a couple of ways:

1. There is no TrainState now, and only some raw weights and Optax state numbers remained;
1. `metadata.dimentions` and `data` should be arrays, but restored as dict with integers as keys;
1. `data[0]` used to be a `jnp.array` but now is a `numpy.array`. 

While (3) would not affect future work because JAX will automatically convert Numpy arrays to JAX arrays once computation starts, (1) and (2) may lead to confusions.

In order to solve this, you should pass an example `target` to let Flax know exactly what structure it should restore to. `target` should introduce any custom dataclasses explicitly, and have the same structure as the saved checkpoint.

It's often recommended to refactor out the process of initializing a checkpoint's structure (e.g., a TrainState), so that saving/loading is easier and less error-prone. This is because complicated objects like `apply_fn` and `tx` are not stored in the checkpoint file and must be initiated by code.

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

### Back/Forward Dataclass Compatibility

The flexibility of using dataclasses means that changes in dataclass fields could break your existed checkpoints. For example, if you decide to add a field `batch_stats` to your `TrainState`, old checkpoints without this field could not be successfully restored. Same goes for removing a field in your dataclass.

Note that Flax supports `flax.struct.dataclass`, not Python's built-in `dataclasses.dataclass`.

```{code-cell} ipython3
class CustomTrainState(train_state.TrainState):
    batch_stats: Any = None

custom_state = CustomTrainState.create(
    apply_fn=state.apply_fn,
    params=state.params,
    tx=state.tx,
    batch_stats=np.arange(10),
)

# Use custom state to read old TrainState checkpoint
custom_target = {'model': custom_state, 'config': None, 'data': [jnp.zeros_like(x1)]}
try:
    checkpoints.restore_checkpoint(ckpt_dir, target=custom_target, step=0)
except ValueError as e:
    print('ValueError when target state has an unmentioned field:')
    print(e)
    print('')


# Use old TrainState to read the custom state checkpoint
custom_ckpt = {'model': custom_state, 'config': config, 'data': [x1]}
checkpoints.save_checkpoint(ckpt_dir, custom_ckpt, step=1, overwrite=True, keep=2)
try:
    checkpoints.restore_checkpoint(ckpt_dir, target=target, step=1)
except ValueError as e:
    print('ValueError when target state misses a recorded field:')
    print(e)
    
```

It is recommended to keep your checkpoints up to date with your pytree dataclass definitions. You could keep a copy of your code along with your checkpoints.

But if you must restore checkpoints and dataclasses with incompatible fields, you could manually add/remove corresponding fields before passing in the correct target structure:

```{code-cell} ipython3
# Pass no target to get a raw state dictionary first
raw_state_dict = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=0)
# Add/remove fields as needed.
raw_state_dict['model']['batch_stats'] = np.arange(10)
# Restore the classes with correct target now
serialization.from_state_dict(custom_target, raw_state_dict)
```

## Asynchronized Checkpointing

Checkpointing is I/O heavy and if you have large amount of data to save, it may be worthwhile to put it into a background thread and continue with your training meanwhile. You could do that by creating an `async_manager` and let it track your save thread.

Note that you should use the same `async_manager` to handle all your async saves across your training steps, so that it can make sure that a previous async save is done before the next one begins. This allows bookkeeping like `keep` and `overwrite` to be consistent across steps.

Whenever you want to explicitly wait until an async save is done, call `wait_previous_save()`.

```{code-cell} ipython3
am = checkpoints.AsyncManager()

# Mimic a training loop here
for step in range(2, 3):
    checkpoints.save_checkpoint(ckpt_dir, ckpt, step=2, overwrite=True, keep=3, async_manager=am)
    # ... Continue with your work...

# ... Until a time, when you want to wait until the save completes:
am.wait_previous_save()  # Block until save completes
checkpoints.restore_checkpoint(ckpt_dir, target=None, step=2)
```

## Multihost/multiprocess Checkpointing

JAX provided a few ways to scale up your code on multiple hosts at the same time. This usually happens when the number of devices (CPU/GPU/TPU) is so large that different devices are managed by different hosts (CPU). To get started on JAX in multi-process settings, read [here](https://jax.readthedocs.io/en/latest/multi_process.html).

In SPMD paradigm with `jax.pjit` ([tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)), a large, multiprocess array could have its data shards on different devices. These data arrays needs a special JAX API `GlobalAsyncCheckpointManager` to save and restore. This API lets each host to dump its data shards to a single shared storage, e.g. a Google Cloud bucket. 

Flax provide easy interface for users to pass in a `GlobalAsyncCheckpointManager` and store pytrees with multi-process arrays in the same fashion as single-process pytrees. Just use `checkpoints.save_checkpoint_multiprocess()` with the same arguments.

Unfortunately Python notebooks are single-host only and cannot activate the multi-host mode. Use the following code as a sample to run your multi-host checkpointing.

```{code-cell} ipython3
# Setup a checkpoint with a multiprocess array

# In reality you should set this with multiple num_processes
# See https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster
jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)

# Create a multiprocess array
jax.config.update('jax_array', True)
mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x', 'y'))
f = pjit.pjit(
  lambda x: x,
  in_axis_resources=None,
  out_axis_resources=PartitionSpec('x', 'y'))
with maps.Mesh(mesh.devices, mesh.axis_names):
    mp_array = f(np.arange(8 * 2).reshape(8, 2))

# Make it a pytree as normal
mp_ckpt = {'model': mp_array}
```

Sample call to save the checkpoint. Note that all the arguments are the same as `save_checkpoint`, except for an additional `gda_manager` argument. 

If your checkpoint is too large, you could add `timeout_secs` to the manager and gives it more time to finish writing.

```{code-cell} ipython3
gacm = GlobalAsyncCheckpointManager(timeout_secs=50)
checkpoints.save_checkpoint_multiprocess(ckpt_dir, mp_ckpt, step=3, overwrite=True, 
                                         keep=4, gda_manager=gacm)
```

Sample code to restore the checkpoint. 

Note that you need to pass a target with valid multiprocess arrays at the correct structual location. Flax only uses the target arrays' meshes and mesh axes to restore the checkpoint, so the array itself need not to be as large as your checkpoint's.

```{code-cell} ipython3
with maps.Mesh(mesh.devices, mesh.axis_names):
    mp_smaller_array = f(np.zeros(8).reshape(4, 2))

mp_target = {'model': mp_smaller_array}
mp_restored = checkpoints.restore_checkpoint(ckpt_dir, target=mp_target, 
                                             step=3, gda_manager=gacm)
mp_restored
```

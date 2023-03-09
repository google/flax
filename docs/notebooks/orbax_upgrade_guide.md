---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Migrate checkpointing to Orbax

This guide shows how to convert Flax's checkpoint saving and restoring calls — [`flax.training.checkpoints.save_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint) and [`restore_checkpoint`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.restore_checkpoint) — to the equivalent [Orbax](https://github.com/google/orbax) methods. Orbax provides a flexible and customizable API for managing checkpoints for various objects. Note that as Flax's checkpointing is being migrated to Orbax from `flax.training.checkpoints`, all existing features in the Flax API will continue to be supported, but the API will change.

You will learn how to migrate to Orbax through the following scenarios:

* The most common use case: Saving/loading and managing checkpoints
* A "lightweight" use case: "Pure" saving/loading without the top-level variable
* Restoring checkpoints without a target pytree
* Async checkpointing
* Saving/loading a single JAX or NumPy Array

To learn more about Orbax, check out the [quick start introductory Colab notebook](http://colab.research.google.com/github/google/orbax/blob/main/orbax//checkpoint/orbax_checkpoint.ipynb) and [the official Orbax documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md).


## Setup

Install/upgrade [JAX](https://github.com/google/jax#installation), Flax and [Orbax](https://github.com/google/orbax). In addition, install [`nest_asyncio`](https://github.com/erdewit/nest_asyncio).

```python tags=["skip-execution"]
!pip3 install -qq -U jaxlib jax flax orbax

# This is only required for Orbax in Colab/notebook scenarios.
!pip3 install -qq nest_asyncio
```

```python
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import flax
from flax.training import checkpoints, orbax_utils
import orbax.checkpoint
import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as orbax_checkpoint
# Orbax needs to have asyncio enabled in the Colab environment.
import nest_asyncio
nest_asyncio.apply()
```

```python
# Create some dummy variables for this example.
MAX_STEPS = 5
CKPT_PYTREE = [12, {'foo': 'str', 'bar': np.array((2, 3))}, [1, 4, 10]]
TARGET_PYTREE = [0, {'foo': '', 'bar': np.array((0))}, [0, 0, 0]]


# For removing any existing checkpoints from the last notebook run:
import shutil
if os.path.exists('./tmp'):
    shutil.rmtree('./tmp')
```

## Most common use case: Saving/loading and managing checkpoints

This section covers the following scenario:

*  Your original Flax `save_checkpoint()` or `save_checkpoint_multiprocess()` call contains the following arguments: `prefix`, `keep`, `keep_every_n_steps`; or
*  You want to use some automatic management logic for your checkpoints (for example, for deleting old data, deleting data based on metrics/loss, and so on).

In this case, you need to use `orbax.CheckpointManager`. This allows you to not only save and load your model, but also manage your checkpoints and delete outdated checkpoints *automatically*.

To upgrade your code:

1. Create and keep an `orbax.CheckpointManager` instance at the top level, customized with `orbax.CheckpointManagerOptions`.
2. At runtime, call `orbax.CheckpointManager.save()` to save your data.
3. Then, call `orbax.CheckpointManager.restore()` to restore your data.
4. And, if your checkpoint includes some multi-host/multi-process array, pass the correct `mesh` into `flax.training.orbax_utils.restore_args_from_target()` to generate the correct `restore_args` before restoring.

Below are code examples for before and after the migration to Orbax:

```python
CKPT_DIR = './tmp/'
flax.config.update('flax_use_orbax_checkpointing', False)

# Before: Using the Flax API

# Inside a training loop
for step in range(MAX_STEPS):
   # do your training
   checkpoints.save_checkpoint(CKPT_DIR, CKPT_PYTREE, step=step,
                               prefix='test_', keep=3, keep_every_n_steps=2)


checkpoints.restore_checkpoint(CKPT_DIR, target=TARGET_PYTREE, step=4, prefix='test_')
```

```python
CKPT_DIR = './tmp/orbax'

# After: Using the Orbax API

# At the top level
mgr_options = orbax_checkpoint.CheckpointManagerOptions(
    create=True, max_to_keep=3, keep_period=2, step_prefix='test_')
ckpt_mgr = orbax.checkpoint.CheckpointManager(
    CKPT_DIR,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

# Inside a training loop
for step in range(MAX_STEPS):
   # do your training
   save_args = orbax_utils.save_args_from_target(CKPT_PYTREE)
   ckpt_mgr.save(step, CKPT_PYTREE, save_kwargs={'save_args': save_args})


restore_args = orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None)
ckpt_mgr.restore(4, items=TARGET_PYTREE, restore_kwargs={'restore_args': restore_args})
```

## A "lightweight" case: "Pure" saving/loading without the top-level variable

If you prefer to not maintain a top-level checkpoint manager, you can still save and restore any individual checkpoint with an `orbax.checkpoint.Checkpointer`. Note that this means you cannot use all the Orbax management features.

To migrate to Orbax code, instead of using the `overwrite` argument in `flax.save_checkpoint()` use the `force` argument in `orbax.checkpoint.Checkpointer.save()`.

For example:

```python
PURE_CKPT_DIR = './tmp/pure'
flax.config.update('flax_use_orbax_checkpointing', False)

# Before: Using the Flax API
checkpoints.save_checkpoint(PURE_CKPT_DIR, CKPT_PYTREE, step=0, overwrite=True)
checkpoints.restore_checkpoint(PURE_CKPT_DIR, target=TARGET_PYTREE)
```

```python
PURE_CKPT_DIR = './tmp/pure/orbax'

# After: Using the Orbax API
ckptr = orbax_checkpoint.Checkpointer(orbax_checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
ckptr.save(PURE_CKPT_DIR, CKPT_PYTREE,
           save_args=orbax_utils.save_args_from_target(CKPT_PYTREE), force=True)
ckptr.restore(PURE_CKPT_DIR, item=TARGET_PYTREE,
              restore_args=orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None))
```

## Restoring checkpoints without a target pytree

If you need to restore your checkpoints without a target pytree, pass `item=None` to `orbax.checkpoint.Checkpointer` or pass `items=None` to `orbax.CheckpointManager`'s `.restore()` method, which should trigger the restoration.

For example:

```python
NOTARGET_CKPT_DIR = './tmp/no_target'
flax.config.update('flax_use_orbax_checkpointing', False)

# Before: Using the Flax API
checkpoints.save_checkpoint(NOTARGET_CKPT_DIR, CKPT_PYTREE, step=0)
checkpoints.restore_checkpoint(NOTARGET_CKPT_DIR, target=None)
```

```python
NOTARGET_CKPT_DIR = './tmp/no_target/orbax'

# After: Using the Orbax API
ckptr = orbax_checkpoint.Checkpointer(orbax_checkpoint.PyTreeCheckpointHandler())
ckptr.save(NOTARGET_CKPT_DIR, CKPT_PYTREE,
           save_args=orbax_utils.save_args_from_target(CKPT_PYTREE))
ckptr.restore(NOTARGET_CKPT_DIR, item=None)
```

## Async checkpointing

To make your checkpoint-saving asynchronous, substitute `orbax.checkpoint.Checkpointer` with `orbax.checkpoint.AsyncCheckpointer`.

Then, you can call `orbax.checkpoint.AsyncCheckpointer.wait_until_finished()` or Orbax's `CheckpointerManager.wait_until_finished()` to wait for the save the complete.

For more details, read the [checkpoint guide](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#asynchronized-checkpointing).

## Saving/loading a single JAX or NumPy Array

The `orbax.checkpoint.PyTreeCheckpointHandler` class, as the name suggests, can only be used for pytrees. Therefore, if you need to save/restore a single pytree leaf (for example, an array), use `orbax.checkpoint.ArrayCheckpointHandler` instead.

For example:

```python
ARR_CKPT_DIR = './tmp/singleton'
flax.config.update('flax_use_orbax_checkpointing', False)

# Before: Using the Flax API
checkpoints.save_checkpoint(ARR_CKPT_DIR, jnp.arange(10), step=0)
checkpoints.restore_checkpoint(ARR_CKPT_DIR, target=None)
```

```python
ARR_CKPT_DIR = './tmp/singleton/orbax'

# After: Using the Orbax API
ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler())  # stateless object, can be created on-fly
ckptr.save(ARR_CKPT_DIR, jnp.arange(10))
ckptr.restore(ARR_CKPT_DIR, item=None)
```

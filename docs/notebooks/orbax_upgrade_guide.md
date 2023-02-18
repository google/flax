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

# Migrate Flax Checkpointing to Orbax

This guide shows you how to convert a `flax.training.checkpoints.save_checkpoint` or `restore_checkpoint` call to the equivalent in `Orbax` (go/orbax).

See also Orbax's quick start [colab introduction](http://colab.research.google.com/github/google/orbax/blob/main/orbax//checkpoint/orbax_checkpoint.ipynb) and [official documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md).


## Setup

```python tags=["skip-execution"]
!pip3 install -qq -U jaxlib jax flax orbax

# This is required for Orbax only for notebook scenarios.
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
# Orbax needs to enable asyncio in colab environment.
import nest_asyncio
nest_asyncio.apply()
```

```python
# Some dummy variables to showcase
MAX_STEPS = 5
CKPT_PYTREE = [12, {'foo': 'str', 'bar': np.array((2, 3))}, [1, 4, 10]]
TARGET_PYTREE = [0, {'foo': '', 'bar': np.array((0))}, [0, 0, 0]]


# Remove any existing checkpoints from the last notebook run.
import shutil
if os.path.exists('./tmp'):
    shutil.rmtree('./tmp')
```

## Most Common Case: Save/Load + Management

```python
CKPT_DIR = './tmp/'

# Before

# Inside a training loop
for step in range(MAX_STEPS):
   # ... do your training ...
   checkpoints.save_checkpoint(CKPT_DIR, CKPT_PYTREE, step=step,
                               prefix='test_', keep=3, keep_every_n_steps=2)


checkpoints.restore_checkpoint(CKPT_DIR, target=TARGET_PYTREE, step=4, prefix='test_')
```

```python
CKPT_DIR = './tmp/'

# After

# At top level
mgr_options = orbax_checkpoint.CheckpointManagerOptions(
    max_to_keep=3, keep_period=2, step_prefix='test_')
ckpt_mgr = orbax.checkpoint.CheckpointManager(
    CKPT_DIR,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

# Inside a training loop
for step in range(MAX_STEPS):
   # ... do your training ...
   save_args = orbax_utils.save_args_from_target(CKPT_PYTREE)
   ckpt_mgr.save(step, CKPT_PYTREE, save_kwargs={'save_args': save_args})


restore_args = orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None)
ckpt_mgr.restore(4, items=TARGET_PYTREE, restore_kwargs={'restore_args': restore_args})
```

## Lightweight Case: Pure Save/Load without Top-level Variable

```python
PURE_CKPT_DIR = './tmp/pure'

# Before
checkpoints.save_checkpoint(PURE_CKPT_DIR, CKPT_PYTREE, step=0, overwrite=True)
checkpoints.restore_checkpoint(PURE_CKPT_DIR, target=TARGET_PYTREE)
```

```python
PURE_CKPT_DIR_ORBAX = './tmp/pure/orbax'

# After
ckptr = orbax_checkpoint.Checkpointer(orbax_checkpoint.PyTreeCheckpointHandler())  # stateless object, can be created on-fly
ckptr.save(PURE_CKPT_DIR_ORBAX, CKPT_PYTREE,
           save_args=orbax_utils.save_args_from_target(CKPT_PYTREE), force=True)
ckptr.restore(PURE_CKPT_DIR_ORBAX, item=TARGET_PYTREE,
              restore_args=orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None))
```

## Restore without a target pytree

```python
NOTARGET_CKPT_DIR = './tmp/no_target'

# Before
checkpoints.save_checkpoint(NOTARGET_CKPT_DIR, CKPT_PYTREE, step=0)
checkpoints.restore_checkpoint(NOTARGET_CKPT_DIR, target=None)
```

```python
NOTARGET_CKPT_DIR = './tmp/no_target/orbax'

# After
ckptr = orbax_checkpoint.Checkpointer(orbax_checkpoint.PyTreeCheckpointHandler())
ckptr.save(NOTARGET_CKPT_DIR, CKPT_PYTREE,
           save_args=orbax_utils.save_args_from_target(CKPT_PYTREE))
ckptr.restore(NOTARGET_CKPT_DIR, item=None)
```

## Save/Load a single JAX or Numpy Array

```python
ARR_CKPT_DIR = './tmp/singleton'

# Before
checkpoints.save_checkpoint(ARR_CKPT_DIR, jnp.arange(10), step=0)
checkpoints.restore_checkpoint(ARR_CKPT_DIR, target=None)
```

```python
ARR_CKPT_DIR = './tmp/singleton/orbax'

# After
ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler())  # stateless object, can be created on-fly
ckptr.save(ARR_CKPT_DIR, jnp.arange(10))
ckptr.restore(ARR_CKPT_DIR, item=None)
```

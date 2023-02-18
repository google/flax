.. image:: https://colab.research.google.com/assets/colab-badge.svg
:target: https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/orbax_upgrade_guide.ipynb

Upgrading my codebase to Orbax
==============================

This guide shows you how to convert a ``flax.training.checkpoints`` call to the equivalent in `Orbax <https://github.com/google/orbax>`_.

See also Orbax's quick start `colab introduction <http://colab.research.google.com/github/google/orbax/blob/main/orbax//checkpoint/orbax_checkpoint.ipynb>`_ and `official documentation <https://github.com/google/orbax/blob/main/docs/checkpoint.md>`_.

Alternatively to this page, you can click the "Open in Colab" link above to run the following code in Colab environment.

.. testsetup::

  import flax
  from flax.training import checkpoints, orbax_utils
  import orbax
  import jax
  import jax.numpy as jnp
  import numpy as np

  # Orbax needs to enable asyncio in colab environment.
  import nest_asyncio
  nest_asyncio.apply()

  # Set up the directory.
  import os
  import shutil
  if os.path.exists('./tmp/'):
    shutil.rmtree('./tmp/')
  os.makedirs('./tmp/')


Setup
---------------------------------------

.. testcode::

  # Some pytrees to showcase
  MAX_STEPS = 5
  CKPT_PYTREE = [12, {'foo': 'str', 'bar': np.array((2, 3))}, [1, 4, 10]]
  TARGET_PYTREE = [0, {'foo': '', 'bar': np.array((0))}, [0, 0, 0]]

Most Common Case: Save/Load + Management
---------------------------------------

Follow this if:

*  Your original Flax ``save_checkpoint()`` or ``save_checkpoint_multiprocess()`` call contains these args: ``prefix``, ``keep``, ``keep_every_n_steps``.

*  You want to use some automatic management logic for your checkpoints (e.g., delete old data, delete based on metrics/loss, etc).

Then you should switch to using an ``orbax.CheckpointManager``. This allows you to not only save and load your model, but also manage your checkpoints and delete outdated checkpoints automatically.

Modify your code to:

1. Create and keep an ``orbax.CheckpointManager`` instance at the top level, customized with ``orbax.CheckpointManagerOptions``

2. In runtime, call ``CheckpointManager.save()`` to save your data.

3. Call ``CheckpointManager.restore()`` to restore your data.

4. If your checkpoint includes some multihost/multiprocess array, you need to pass the correct ``mesh`` into a ``restore_args_from_target()`` to generate the correct ``restore_args`` before restoring.

See below for code examples for before and after migration.

.. codediff::
  :title_left: flax.checkpoints
  :title_right: orbax.checkpoint
  :sync:

  CKPT_DIR = './tmp/'

  # Inside a training loop
  for step in range(MAX_STEPS):
    # ... do your training ...
    checkpoints.save_checkpoint(CKPT_DIR, CKPT_PYTREE, step=step,
                                prefix='test_', keep=3, keep_every_n_steps=2)


  checkpoints.restore_checkpoint(CKPT_DIR, target=TARGET_PYTREE, step=4, prefix='test_')

  ---

  CKPT_DIR = './tmp/'

  # At top level
  mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=3, keep_period=2, step_prefix='test_')
  ckpt_mgr = orbax.checkpoint.CheckpointManager(
    CKPT_DIR,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

  # Inside a training loop
  for step in range(MAX_STEPS):
    # ... do your training ...
    save_args = flax.training.orbax_utils.save_args_from_target(CKPT_PYTREE)
    ckpt_mgr.save(step, CKPT_PYTREE, save_kwargs={'save_args': save_args})


  restore_args = flax.training.orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None)
  ckpt_mgr.restore(4, items=TARGET_PYTREE, restore_kwargs={'restore_args': restore_args})


Lightweight Case: Pure Save/Load without Setup
-----------------------------------

If you prefer to not maintain a top-level checkpoint manager, you can still save and restore any individual checkpoint with an ``orbax.checkpoint.Checkpointer``. Note that this means you cannot use all the management features.

For argument ``overwrite`` in ``flax.save_checkpoint()``, use argument ``force`` in ``Checkpointer.save()`` instead.

.. codediff::
  :title_left: flax.checkpoints
  :title_right: orbax.checkpoint
  :sync:

  PURE_CKPT_DIR = './tmp/pure'

  checkpoints.save_checkpoint(PURE_CKPT_DIR, CKPT_PYTREE, step=0, overwrite=True)
  checkpoints.restore_checkpoint(PURE_CKPT_DIR, target=TARGET_PYTREE)

  ---

  PURE_CKPT_DIR = './tmp/pure'

  ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())  # stateless object, can be created on-fly
  ckptr.save(PURE_CKPT_DIR, CKPT_PYTREE,
             save_args=flax.training.orbax_utils.save_args_from_target(CKPT_PYTREE), force=True)
  ckptr.restore(PURE_CKPT_DIR, item=TARGET_PYTREE,
                restore_args=flax.training.orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None))



Restore without a target pytree
-----------------------------------

Pass ``item=None`` to Orbax ``Checkpointer`` or ``items=None`` to ``CheckpointManager``'s ``.restore()`` should trigger restoration.

.. codediff::
  :title_left: flax.checkpoints
  :title_right: orbax.checkpoint
  :sync:

  NOTARGET_CKPT_DIR = './tmp/no_target'

  checkpoints.save_checkpoint(NOTARGET_CKPT_DIR, CKPT_PYTREE, step=0)
  checkpoints.restore_checkpoint(NOTARGET_CKPT_DIR, target=None)

  ---

  NOTARGET_CKPT_DIR = './tmp/no_target'

  # stateless object, can be created on-fly
  ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
  ckptr.save(NOTARGET_CKPT_DIR, CKPT_PYTREE,
             save_args=flax.training.orbax_utils.save_args_from_target(CKPT_PYTREE))
  ckptr.restore(NOTARGET_CKPT_DIR, item=None)


Async Checkpointing
-----------------------------------

Substitute ``orbax.checkpoint.Checkpointer`` with ``orbax.checkpoint.AsyncCheckpointer`` makes all saves async.

You can later call ``AsyncCheckpointer.wait_until_finished()`` or ``CheckpointerManager.wait_until_finished()`` to wait for the save the complete.

See more details on the `checkpoint guide <https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#asynchronized-checkpointing>`_.


Save/Load a single JAX or Numpy Array
-----------------------------------

``orbax.checkpoint.PyTreeCheckpointHandler``, as the name suggests, is only for pytrees. If you want to save/restore a single Pytree leaf (e.g., an array), use ``orbax.checkpoint.ArrayCheckpointHandler`` instead.

.. codediff::
  :title_left: flax.checkpoints
  :title_right: orbax.checkpoint
  :sync:

  ARR_CKPT_DIR = './tmp/singleton'

  checkpoints.save_checkpoint(ARR_CKPT_DIR, jnp.arange(10), step=0)
  checkpoints.restore_checkpoint(ARR_CKPT_DIR, target=None)

  ---

  ARR_CKPT_DIR = './tmp/singleton'

  ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler())
  ckptr.save(ARR_CKPT_DIR, jnp.arange(10))
  ckptr.restore(ARR_CKPT_DIR, item=None)



Final Words
-----------

This guide only shows you how to migrate an existed Flax checkpointing call to Orbax. Orbax as a tool provides much more functionalities and is actively developing new features. Please stay tuned with their `official github repository <https://github.com/google/orbax>`_ for more!

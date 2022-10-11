
flax.training package
=====================

Checkpoints
------------------------

.. currentmodule:: flax.training.checkpoints

.. automodule:: flax.training.checkpoints

.. autofunction:: save_checkpoint

.. autofunction:: save_checkpoint_multiprocess

.. autofunction:: latest_checkpoint

.. autofunction:: restore_checkpoint

.. autofunction:: convert_pre_linen

Learning rate schedules
------------------------

.. currentmodule:: flax.training.lr_schedule

.. automodule:: flax.training.lr_schedule

.. autofunction:: create_constant_learning_rate_schedule

.. autofunction:: create_stepped_learning_rate_schedule

.. autofunction:: create_cosine_learning_rate_schedule

Train state
------------------------

.. currentmodule:: flax.training.train_state

.. autoclass:: TrainState
    :members: apply_gradients, create

Early Stopping
------------------------

.. currentmodule:: flax.training.early_stopping

.. autoclass:: EarlyStopping
    :members: reset, update

Common Utilities
------------------------

.. currentmodule:: flax.training.common_utils

.. autofunction:: shard

.. autofunction:: shard_prng_key

.. autofunction:: stack_forest

.. autofunction:: get_metrics

.. autofunction:: onehot

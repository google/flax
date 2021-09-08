Early Stopping
=============================
Early stopping is used while building model to stop training loop when specific conditions are met. Some such conditions are:

* Train loss not decreasing at all.
* Train loss is decreasing but the decrease in loss is is insignificant.

How to use
-----------------------------

Make the below changes in annotated MNIST example (https://flax.readthedocs.io/en/latest/notebooks/annotated_mnist.html) to achieve early stopping.

Use :code:`early_stopping.EarlyStopping` class imported from :code:`flax.training` to create an early stopping object :code:`es`.

.. testcode::

  from flax.training import early_stopping


:code:`min_delta` is the minimum expected decrease in loss metric. 
:code:`es.should_stop` becomes true if decrease in the loss metric is greater than :code:`min_delta` for consecutive :code:`patience` number of epochs. 
Otherwise, it is :code:`false` indicating that we must continue training.


A sample training loop looks like this


.. codediff:: 
  :title_left: With early stopping
  :title_right: Without early stopping

  # instantiate
  es = early_stopping.EarlyStopping(min_delta=0.2, patience=2)
  
  for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    # the global `es` is overwritten with the lates
    state, es = train_epoch(state, train_ds, batch_size, epoch, input_rng, es)
    
    if es.should_stop:
      # stop the training loop
      break 
  ---
  for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state = train_epoch(state, train_ds, batch_size, epoch, input_rng)


And :code:`train_epoch` function looks like this after changes in annotated MNIST example


.. codediff:: 
  :title_left: With early stopping
  :title_right: Without early stopping

def train_epoch(state, train_ds, batch_size, epoch, rng, es):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  #  early stopping check after very epoch
  # note: `did_improve` means decrease in metric score. Not increase in metric score.
  did_improve, es = es.update(epoch_metrics_np['loss'])

  return state, es
  ---
  for epoch in range(1, num_epochs + 1):
def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state
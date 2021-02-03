Managing Parameters and State
=============================

We will show you how to...

* define a learning rate schedule
* train a simple model using that schedule

.. code-block:: python
  
  def create_triangular_schedule(lr_min, lr_max, steps_per_cycle):
  """Return fn from epoch to LR, linearly interpolating between `lr_min`->`lr_max`->`lr_min`."""
    top = (steps_per_cycle + 1) // 2
    def learning_rate_fn(step):
      cycle_step = step % steps_per_cycle
      if cycle_step < top:
        lr = lr_min + cycle_step/top * (lr_max - lr_min)
      else:
        lr = lr_max - ((cycle_step - top)/top) * (lr_max - lr_min)
      return lr
    return learning_rate_fn

This example shows how to create a simple cyclical learning rate scheduler, the triangular scheduler.

To use the schedule one must simply create a learning rate function by passing the hyperparameters to the 
`create_triangular_schedule` function and then use that function to compute the learning rate for your updates.
For example using this schedule on MNIST would require changing the train_step function
.. code-block:: python
  
  def train_step(optimizer, batch, learning_rate_fn):  
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    step = optimizer.state.step
    lr = learning_rate_fn(step)
    optimizer = optimizer.apply_gradient(grad, {"learning_rate":lr})
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics

And the train_epoch function:
def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  #If you want 4 cycles per epoch
  learning_rate_fn = create_triangular_schedule(3e-3, 3e-2, steps_per_epoch//4)
  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, batch, learning_rate_fn)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
               epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

  return optimizer, epoch_metrics_np
Early stopping
=============================
Early stopping is an iteration control mechanism to avoid overfitting the network.
The idea is to break from the training loop if a particular criteria is met.

In this example we'll see 

* How ``flax.training.early_stopping.EarlyStopping`` helps in early stopping.
* How we can change a regular training loop to add this functionality.

Consider the basic ``train_epoch`` function that trains a model for a single epoch. Inspired from
the MNIST example `here <https://github.com/google/flax/blob/master/examples/mnist/train.py#L105>`_.


To use it, we will call the function inside a loop ``num_epochs`` times. However with early
stopping, we also want it to stop training once some criteria is met. The criteria here will 
be if the difference between losses recorded in the current epoch and previous epoch is less than
``min_delta`` consecutively for ``patience`` times.

.. codediff::
  :title_left: Training without early stopping
  :title_right: Training with early stopping

  
  for epoch in range(1, num_epochs+1):
    rng, input_rng = jax.random.split(rng)
    optimizer, train_metrics = train_epoch(
        optimizer, train_ds, config.batch_size, epoch, input_rng)




  ---
  early_stop = EarlyStopping(min_delta=1e-3, patience=2) #!
  for epoch in range(1, num_epochs+1):
    rng, input_rng = jax.random.split(rng)
    optimizer, train_metrics = train_epoch(
        optimizer, train_ds, config.batch_size, epoch, input_rng)
    _, early_stop = early_stop.update(train_metrics['loss']) #!
    if early_stop.should_stop: #!
      print('Met early stopping criteria, breaking...') #!
      break #!

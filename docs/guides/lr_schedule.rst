Learning rate scheduling
=============================

The learning rate is considered one of the most important hyperparameters for
training deep neural networks, but choosing it can be quite hard.
Rather than simply using a fixed learning rate, it is common to use a learning rate scheduler.
In this example, we will use the *cosine scheduler*.
Before the cosine scheduler comes into play, we start with a so-called *warmup* period in which the
learning rate increases linearly for ``warmup_epochs`` epochs.
For more information about the cosine scheduler, check out the paper
`"SGDR: Stochastic Gradient Descent with Warm Restarts" <https://arxiv.org/abs/1608.03983>`_.

We will show you how to...

* define a learning rate schedule
* train a simple model using that schedule


.. testsetup::

  import jax
  import jax.numpy as jnp
  import flax.linen as nn
  from flax.training import train_state
  import optax
  import numpy as np
  import tensorflow_datasets as tfds
  import functools
  import ml_collections
  from absl import logging


  class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      return x

  def get_dummy_data(ds_size):
    image = np.random.rand(ds_size, 28, 28, 1)
    label = np.random.randint(low=0, high=10, size=(ds_size,))
    return {'image': image, 'label': label}

  def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.001
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10
    config.warmup_epochs = 2
    config.train_ds_size = 128
    return config

  def compute_metrics(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


.. testcode::

  def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn

To use the schedule, we must create a learning rate function by passing the hyperparameters to the
``create_learning_rate_fn`` function and then pass the function to your |Optax|_ optimizer.
For example using this schedule on MNIST would require changing the ``train_step`` function:

.. |Optax| replace:: ``Optax``
.. _Optax: https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules

.. codediff::
  :title_left: Default learning rate
  :title_right: Learning rate schedule
  :sync:

  @jax.jit
  def train_step(state, batch):
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      one_hot = jax.nn.one_hot(batch['label'], 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])


    return new_state, metrics
  ---
  @functools.partial(jax.jit, static_argnums=2) #!
  def train_step(state, batch, learning_rate_fn): #!
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      one_hot = jax.nn.one_hot(batch['label'], 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    lr = learning_rate_fn(state.step) #!
    metrics['learning_rate'] = lr #!
    return new_state, metrics

And the ``train_epoch`` function:

.. codediff::
  :title_left: Default learning rate
  :title_right: Learning rate schedule
  :sync:

  def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Trains for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
      batch = {k: v[perm, ...] for k, v in train_ds.items()}
      state, metrics = train_step(state, batch)
      batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]}

    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                 epoch_metrics['loss'], epoch_metrics['accuracy'] * 100)

    return state, epoch_metrics
  ---
  def train_epoch(state, train_ds, batch_size, epoch, learning_rate_fn, rng): #!
    """Trains for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
      batch = {k: v[perm, ...] for k, v in train_ds.items()}
      state, metrics = train_step(state, batch, learning_rate_fn) #!
      batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]}

    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                 epoch_metrics['loss'], epoch_metrics['accuracy'] * 100)

    return state, epoch_metrics


And the ``create_train_state`` function:


.. codediff::
  :title_left: Default learning rate
  :title_right: Learning rate schedule
  :sync:

  def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)
  ---
  def create_train_state(rng, config, learning_rate_fn): #!
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate_fn, config.momentum) #!
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)


.. testcleanup::

  config = get_config()

  train_ds_size = config.train_ds_size
  steps_per_epoch = train_ds_size // config.batch_size
  learning_rate_fn = create_learning_rate_fn(config, config.learning_rate, steps_per_epoch)

  rng = jax.random.PRNGKey(0)
  state = create_train_state(rng, config, learning_rate_fn)

  train_ds = get_dummy_data(config.train_ds_size)
  rng, _ = jax.random.split(rng)
  state, epoch_metrics = train_epoch(state, train_ds, config.batch_size, 0, learning_rate_fn, rng)

  assert 'accuracy' in epoch_metrics and 'learning_rate' in epoch_metrics




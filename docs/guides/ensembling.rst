Ensembling on multiple devices
==============================

We show how to train an ensemble of CNNs on the MNIST dataset, where the size of
the ensemble is equal to the number of available devices. In short, this change
be described as:

* make a number of functions parallel using |jax.pmap()|_,
* split the random seed to obtain different parameter initialization,
* replicate the inputs and unreplicate the outputs where necessary,
* average probabilities across devices to compute the predictions.

In this HOWTO we omit some of the code such as imports, the CNN module, and
metrics computation, but they can be found in the `MNIST example`_.

.. testsetup::

  import functools
  from flax import jax_utils

  # Copied from examples/mnist/train.py
  from absl import logging
  from flax import linen as nn
  from flax.training import train_state
  import jax
  import jax.numpy as jnp
  import numpy as np
  import optax

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

  # Fake data for faster execution.
  def get_datasets():
    train_ds = test_ds = {
      'image': jnp.zeros([64, 28, 28, 1]),
      'label': jnp.zeros([64], jnp.int32),
    }
    return train_ds, test_ds

  # Modified from examples/mnist/configs.default.py
  learning_rate = 0.1
  momentum = 0.9
  batch_size = 32
  num_epochs = 1


Parallel functions
------------------

We start by creating a parallel version of ``create_train_state()``, which
retrieves the initial parameters of the models. We do this using |jax.pmap()|_.
The effect of "pmapping" a function is that it will compile the function with
XLA (similar to |jax.jit()|_), but execute it in parallel on XLA devices (e.g.,
GPUs/TPUs).

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble
  :sync:

  #!
  def create_train_state(rng, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)
  ---
  @functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))  #!
  def create_train_state(rng, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

Note that for the single-model code above, we use |jax.jit()|_ to lazily
initialize the model (see `Module.init`_'s documentation for more details).
For the ensembling case, |jax.pmap()|_ will map over the first axis of the
provided argument ``rng`` by default, so we should make sure that we provide
a different value for each device when we call this function later on.

Note also how we specify that ``learning_rate`` and ``momentum`` are static
arguments, which means the concrete values of these arguments will be used,
rather than abstract shapes. This is necessary because the provided arguments
will be scalar values. For more details see `JIT mechanics: tracing and static
variables`_.

Next we simply do the same for the functions ``apply_model()`` and
``update_model()``. To compute the predictions from the ensemble, we take the
average of the individual probabilities. We use |jax.lax.pmean()|_ to compute
the average *across devices*. This also requires us to specify the
``axis_name`` to both |jax.pmap()|_ and |jax.lax.pmean()|_.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble
  :sync:

  @jax.jit  #!
  def apply_model(state, images, labels):
    def loss_fn(params):
      logits = CNN().apply({'params': params}, images)
      one_hot = jax.nn.one_hot(labels, 10)
      loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
      return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    #!
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)  #!
    return grads, loss, accuracy

  @jax.jit  #!
  def update_model(state, grads):
    return state.apply_gradients(grads=grads)
  ---
  @functools.partial(jax.pmap, axis_name='ensemble')  #!
  def apply_model(state, images, labels):
    def loss_fn(params):
      logits = CNN().apply({'params': params}, images)
      one_hot = jax.nn.one_hot(labels, 10)
      loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
      return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    probs = jax.lax.pmean(jax.nn.softmax(logits), axis_name='ensemble')  #!
    accuracy = jnp.mean(jnp.argmax(probs, -1) == labels)  #!
    return grads, loss, accuracy

  @jax.pmap  #!
  def update_model(state, grads):
    return state.apply_gradients(grads=grads)

Training the Ensemble
---------------------

Next we transform the ``train_epoch()`` function. When calling the pmapped
functions from above, we mainly need to take care of duplicating the arguments
for all devices where necessary, and de-duplicating the return values.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble
  :sync:

  def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
      batch_images = train_ds['image'][perm, ...]  #!
      batch_labels = train_ds['label'][perm, ...]  #!
      grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
      state = update_model(state, grads)
      epoch_loss.append(loss)  #!
      epoch_accuracy.append(accuracy)  #!
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy
  ---
  def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
      batch_images = jax_utils.replicate(train_ds['image'][perm, ...])  #!
      batch_labels = jax_utils.replicate(train_ds['label'][perm, ...])  #!
      grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
      state = update_model(state, grads)
      epoch_loss.append(jax_utils.unreplicate(loss))  #!
      epoch_accuracy.append(jax_utils.unreplicate(accuracy))  #!
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

As can be seen, we do not have to make any changes to the logic around the
``state``. This is because, as we will see below in our training code,
the train state is replicated already, so when we pass it to ``train_step()``,
things will just work fine since ``train_step()`` is pmapped. However,
the train dataset is not yet replicated, so we do that here. Since replicating
the entire train dataset is too memory intensive we do it at the batch level.

We can now rewrite the actual training logic. This consists of two simple
changes: making sure the RNGs are replicated when we pass them to
``create_train_state()``, and replicating the test dataset, which is much
smaller than the train dataset so we can do this for the entire dataset
directly.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble
  :sync:

  train_ds, test_ds = get_datasets()
  #!
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, learning_rate, momentum)  #!
  #!

  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, batch_size, input_rng)

    _, test_loss, test_accuracy = apply_model(  #!
        state, test_ds['image'], test_ds['label'])  #!

    logging.info(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, '
        'test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))
  ---
  train_ds, test_ds = get_datasets()
  test_ds = jax_utils.replicate(test_ds)  #!
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(jax.random.split(init_rng, jax.device_count()), #!
                             learning_rate, momentum)  #!

  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, batch_size, input_rng)

    _, test_loss, test_accuracy = jax_utils.unreplicate(  #!
        apply_model(state, test_ds['image'], test_ds['label']))  #!

    logging.info(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, '
        'test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))


.. |jax.jit()| replace:: ``jax.jit()``
.. _jax.jit(): https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#To-JIT-or-not-to-JIT
.. |jax.pmap()| replace:: ``jax.pmap()``
.. _jax.pmap(): https://jax.readthedocs.io/en/latest/jax.html#jax.pmap
.. |jax.lax.pmean()| replace:: ``jax.lax.pmean()``
.. _jax.lax.pmean(): https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.pmean.html
.. _Module.init: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init
.. _`JIT mechanics: tracing and static variables`: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#JIT-mechanics:-tracing-and-static-variables
.. _`MNIST example`: https://github.com/google/flax/blob/main/examples/mnist/train.py

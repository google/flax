Ensembling on multiple devices
=============================

We show how to train an ensemble of CNNs on the MNIST dataset, where the size of
the ensemble is equal to the number of available devices. In short, this change
be described as: 

* make a number of functions parallel using ``jax.pmap``, 
* replicate the inputs carefully,
* make sure the parallel and non-parallel logic interacts correctly.

In this HOWTO we omit some of the code such as imports, the CNN module, and
metrics computation, but they can be found in the `MNIST example`_.

.. testsetup::

  # Since this HOWTO's code is part of our tests (which are often ran locally on
  # CPU), we use a very small CNN, we only run for 1 epoch, and we make sure we
  # are using mock data.

  from absl import logging
  from flax import jax_utils
  from flax import linen as nn
  from flax import optim
  from flax.metrics import tensorboard
  import jax
  import jax.numpy as jnp
  from jax import random
  import ml_collections
  import numpy as np
  import tensorflow_datasets as tfds
  import functools

  num_epochs = 1

  class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=1, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=1, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=1)(x)
      x = nn.relu(x)
      x = nn.Dense(features=1)(x)
      x = nn.log_softmax(x)
      return x

  def get_datasets():
    """Load fake MNIST data."""
    # Converts dataset from list of dicts to dict of lists.
    to_dict = lambda x: {k: np.array([d[k] for d in x]) for k in ['image', 'label']}
    with tfds.testing.mock_data(num_examples=100):
      train_ds = to_dict(tfds.as_numpy(tfds.load('mnist', split='train')))
      test_ds = to_dict(tfds.as_numpy(tfds.load('mnist', split='test')))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


  def onehot(labels, num_classes=10):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


  def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


  def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

Parallel functions
--------------------------------

We start by creating a parallel version of ``get_initial_params``, which
retrieves the initial parameters of the models. We do this using `jax.pmap`_.
The effect of "pmapping" a function is that it will compile the function with
XLA (similar to `jax.jit`_), but execute it in parallel on XLA devices (e.g., 
GPUs/TPUs).

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble

  @jax.jit #!
  def get_initial_params(key):
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_val)['params']
    return initial_params

  ---
  @jax.pmap #!
  def get_initial_params(key):
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_val)['params']
    return initial_params

Note that for the single-model code above, we use `jax.jit`_ to lazily model 
(see `Module.init`_'s documentation for more details). For the ensembling
case, `jax.pmap`_ will map over the first axis of the provided argument ``key``
by default, so we should make sure that we provide one key for each device when
we call this function later on.

Next we simply do the same for the functions ``create_optimizer``, 
``train_step``, and ``eval_step``. We also make a minor change to 
``eval_model``, which ensures the metrics are used correctly in the parallel
setting.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble

  # #!
  def create_optimizer(params, learning_rate=0.1, beta=0.9):
    optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                   beta=beta)
    optimizer = optimizer_def.create(params)
    return optimizer

  @jax.jit #!
  def train_step(optimizer, batch):
    """Train for a single step."""
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics

  @jax.jit #!
  def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits, batch['label'])

  def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics) #!
    return summary['loss'], summary['accuracy']
  ---
  @functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2)) #!
  def create_optimizer(params, learning_rate=0.1, beta=0.9):
    optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                   beta=beta)
    optimizer = optimizer_def.create(params)
    return optimizer
  
  @jax.pmap #!
  def train_step(optimizer, batch):
    """Train for a single step."""
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics

  @jax.pmap #!
  def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits, batch['label'])

  def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = metrics #!
    return summary['loss'], summary['accuracy']

Note that for ``create_optimizer`` we also specify that ``learning_rate``
and ``beta`` are static arguments, which means the concrete values of these 
arguments will be used, rather than abstract shapes. This is necessary because
the provided arguments will be scalar values. For more details see 
`JIT mechanics: tracing and static variables`_.

Training the Ensemble
--------------------------------

Next we transform the ``train_epoch`` function.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble

  def train_epoch(optimizer, train_ds, rng, batch_size=10):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
      batch = {k: v[perm, ...] for k, v in train_ds.items()}

      optimizer, metrics = train_step(optimizer, batch)
      batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    
    
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) #!
        for k in batch_metrics_np[0]} #!

    return optimizer, epoch_metrics_np
  ---
  def train_epoch(optimizer, train_ds, rng, batch_size=10):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
      batch = {k: v[perm, ...] for k, v in train_ds.items()}
      batch = jax_utils.replicate(batch) #!
      optimizer, metrics = train_step(optimizer, batch)
      batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    batch_metrics_np = jax.tree_multimap(lambda *xs: np.array(xs), *batch_metrics_np) #!
    epoch_metrics_np = {
           k: np.mean(batch_metrics_np[k], axis=0) #!
           for k in batch_metrics_np} #!

    return optimizer, epoch_metrics_np

As can be seen, we do not have to make any changes to the logic around the
``optimizer``. This is because, as we will see below in our training code,
the optimizer is replicated already, so when we pass it to ``train_step``,
things will just work fine since ``train_step`` is pmapped. However, 
the train dataset is not yet replicated, so we do that here. Since replicating 
the entire train dataset is too memory intensive we do it at the batch level.

The rest of the changes relate to making sure the batch metrics are stored
correctly for all devices. We use ``jax.tree_multimap`` to stack all of the
metrics from each device into numpy arrays, such that e.g.,
``batch_metrics_np['loss']`` has shape ``(jax.device_count(), )``.

We can now rewrite the actual training logic. This consists of two simple
changes: making sure the RNGs are replicate when we pass them to
``get_initial_params``, and replicating the test dataset, which is much smaller
than the train dataset so we can do this for the entire dataset directly.

.. codediff::
  :title_left: Single-model
  :title_right: Ensemble

  train_ds, test_ds = get_datasets()


  rng, init_rng = random.split(random.PRNGKey(0))
  params = get_initial_params(init_rng) #!
  optimizer = create_optimizer(params, learning_rate=0.1, momentum=0.9) #!

  for epoch in range(num_epochs):
    rng, input_rng = random.split(rng)
    optimizer, _ = train_epoch(optimizer, train_ds, input_rng)
    loss, accuracy = eval_model(optimizer.target, test_ds)

    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f', #!
                epoch, loss, accuracy * 100)
  ---
  train_ds, test_ds = get_datasets()
  test_ds = jax_utils.replicate(test_ds) #!
  
  rng, init_rng = random.split(random.PRNGKey(0))
  params = get_initial_params(random.split(rng, jax.device_count())) #!
  optimizer = create_optimizer(params, 0.1, 0.9) #!

  for epoch in range(num_epochs):
    rng, input_rng = random.split(rng)
    optimizer, _ = train_epoch(optimizer, train_ds, input_rng)
    loss, accuracy = eval_model(optimizer.target, test_ds)

    logging.info('eval epoch: %d, loss: %s, accuracy: %s', #!
                epoch, loss, accuracy * 100)

Note that ``create_optimizer`` is using positional arguments in the ensembling
case. This is because we defined those arguments as static broadcasted
arguments, and those should be positional rather then keyword arguments.

.. _jax.jit: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#To-JIT-or-not-to-JIT
.. _jax.pmap: https://jax.readthedocs.io/en/latest/jax.html#jax.pmap
.. _Module.init: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init
.. _`JIT mechanics: tracing and static variables`: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#JIT-mechanics:-tracing-and-static-variables
.. _`MNIST example`: https://github.com/google/flax/blob/master/examples/mnist/train.py
# The annotated MNIST image classification example with Flax

This tutorial demonstrates how to construct a simple convolutional neural network (CNN) using the Flax Linen API and train the network for image classification on the MNIST dataset.

1. Import JAX, [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html), Flax, ordinary NumPy, and TensorFlow Datasets (TFDS). Flax can use any data-loading pipeline and this example demonstrates how to utilize TFDS.

```python
import jax
import jax.numpy as jnp            # JAX NumPy

from flax import linen as nn        # The Linen API
from flax import optim              # Optimizers

import numpy as np                 # Ordinary NumPy
import tensorflow_datasets as tfds  # TFDS for MNIST
```

2. Create a convolutional neural network with the Linen API by subclassing [`Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#core-module-abstraction). Because the architecture in this example is relatively simple—you're just stacking layers—you can define the inlined submodules directly within the `__call__` method and wrap it with the [`@compact`](https://flax.readthedocs.io/en/latest/flax.linen.html#compact-methods) decorator.

```python
class CNN(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1)) # Flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
    x = nn.log_softmax(x)
    return x
```

3. Define a cross-entropy loss function using just [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html) that takes the model's logits and label vectors and returns a scalar loss. The labels can be one-hot encoded with [`jax.nn.one_hot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.one_hot.html), as demonstrated below.

```python
def cross_entropy_loss(logits, labels):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))
```

4. Define a function for your optimizer:

  - Choose `Momentum` from the [`flax.optim`](https://flax.readthedocs.io/en/latest/flax.optim.html) package; and
  - Wrap the model parameters (`params`) with the [`flax.optim.OptimizerDef.create`](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.OptimizerDef.create) method and initialize with parameter dicts.

```python
def create_optimizer(params, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(params)
  return optimizer
```

5. Create a function for parameter initialization:

  - Set the initial shape of the kernel (note that JAX and Flax are [row-based](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html#Model-parameters-&-initialization)); and
  - Initialize the module parameters of your network (`CNN`) with the [`init`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init) method using the PRNGKey, which returns parameters (note that the parameters are explicitly tracked separately from the model defintion).

```python
def get_initial_params(key):
  init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_shape)['params']
  return initial_params
```

6. For loss and accuracy metrics, create a separate function:

```python
def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics
```

7. Define a function that loads and prepares the MNIST dataset and converts the samples to floating-point numbers.

```python
def get_datasets():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  # Split into training/test sets
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  # Convert to floating-points
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  return train_ds, test_ds
```

8. Write a training step function that:

  - Evaluates the neural network given the parameters and a batch of input images with the [`Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply) method.
  - Computes the `cross_entropy_loss` loss function.
  - Evaluates the loss function and its gradient using [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad).
  - Applies a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions) of gradients ([`flax.optim.Optimizer.apply_gradient`](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.Optimizer.apply_gradient)) to the optimizer to update the model's parameters.
  - Computes the metrics using `compute_metrics` (defined earlier).

  Use JAX's [`@jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) decorator to trace the entire `train_step` function and just-in-time([`jit`]-compile with [XLA](https://www.tensorflow.org/xla) into fused device operations that run faster and more efficiently on hardware accelerators.

```python
@jax.jit
def train_step(optimizer, batch):
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics
```

9. Create a function that evaluates your model on the test set with [`Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply):

```python
# JIT compile
@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'])
```

10. Define a training function that:

  - Shuffles the training data before each epoch using [`jax.random.permutation`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.permutation.html) that takes a PRNGKey as a parameter (check the [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG)).
  - Runs an optimization step for each batch.
  - Retrieves the training metrics from the device with `jax.device_get` and computes their mean across each batch in an epoch.
  - Returns the optimizer with updated parameters and the training loss and accuracy metrics.

```python
def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  batch_metrics = []

  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, batch)
    batch_metrics.append(metrics)

  training_batch_metrics = jax.device_get(batch_metrics)
  training_epoch_metrics = {
      k: np.mean([metrics[k] for metrics in training_batch_metrics])
      for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

  return optimizer, training_epoch_metrics
```

11. Create a model evaluation function that:

  - Retrieves the evaluation metrics from the device with `jax.device_get`.
  - Copies the metrics [data stored](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) in a JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions).

```python
def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)    # Evalue the model on the test set
  metrics = jax.device_get(metrics)
  eval_summary = jax.tree_map(lambda x: x.item(), metrics)
  return eval_summary['loss'], eval_summary['accuracy']
```

12. Download the dataset and preprocess it:

```python
train_ds, test_ds = get_datasets()
```

13. Initialize the parameters and instantiate the optimizer with [PRNGs](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG):

  - Get one [PRNGKey](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax.random.PRNGKey) and [split](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split) it to get a second key that you'll use for parameter initialization. (Learn more about [PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) and [JAX PRNG design](https://github.com/google/jax/blob/master/design_notes/prng.md).)

```python
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
```

14. Initialize the model's parameters:

```python
params = get_initial_params(init_rng)
```

15. Initiate the variables and instantiate the optimizer:

```python
learning_rate = 0.1
beta = 0.9
num_epochs = 10
batch_size = 32

optimizer = create_optimizer(params, learning_rate=learning_rate, beta=beta)
```

16. Train the network and evaluate it:

```python
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  optimizer, train_metrics = train_epoch(optimizer, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch 
  test_loss, test_accuracy = eval_model(optimizer.target, test_ds)
  print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
```

    Training - epoch: 1, loss: 0.1326, accuracy: 95.94
    Testing - epoch: 1, loss: 0.05, accuracy: 98.24
    Training - epoch: 2, loss: 0.0468, accuracy: 98.58
    Testing - epoch: 2, loss: 0.04, accuracy: 98.62
    Training - epoch: 3, loss: 0.0325, accuracy: 98.99
    Testing - epoch: 3, loss: 0.03, accuracy: 99.09
    Training - epoch: 4, loss: 0.0228, accuracy: 99.25
    Testing - epoch: 4, loss: 0.03, accuracy: 99.03
    Training - epoch: 5, loss: 0.0178, accuracy: 99.46
    Testing - epoch: 5, loss: 0.03, accuracy: 99.17
    Training - epoch: 6, loss: 0.0188, accuracy: 99.39
    Testing - epoch: 6, loss: 0.03, accuracy: 99.04
    Training - epoch: 7, loss: 0.0133, accuracy: 99.55
    Testing - epoch: 7, loss: 0.05, accuracy: 98.98
    Training - epoch: 8, loss: 0.0136, accuracy: 99.58
    Testing - epoch: 8, loss: 0.04, accuracy: 99.07
    Training - epoch: 9, loss: 0.0090, accuracy: 99.73
    Testing - epoch: 9, loss: 0.03, accuracy: 99.18
    Training - epoch: 10, loss: 0.0073, accuracy: 99.78
    Testing - epoch: 10, loss: 0.03, accuracy: 99.20

Once the training and testing is done after 10 epochs, the output should show that your model was able to achieve approximately 99% accuracy.
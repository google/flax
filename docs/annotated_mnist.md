# The annotated MNIST image classification example with Flax

This tutorial uses Flax—a high-performance deep learning library for JAX designed for flexibility—to show you how to construct a simple convolutional neural network (CNN) using the Linen API and train the network for image classification on the MNIST dataset.

If you're new to JAX, check out [JAX quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) and [JAX for the impatient](https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html). To learn more about Flax and its Linen API, refer to [Flax basics](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html), [Flax patterns: Managing state and parameters](https://flax.readthedocs.io/en/latest/patterns/state_params.html), [Linen design principles](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html), and the [Linen introduction](https://github.com/google/flax/blob/master/docs/notebooks/linen_intro.ipynb) notebook.

This tutorial has the following workflow:

- Perform a quick setup.
- Create a neural network model with the Linen API that classifies images.
- Define a loss function, an optimizer, a parameter initializer, and a metrics function.
- Create a dataset function, then load the dataset.
- Initialize the parameters and the optimizer.
- And, finally, train the network and evaluate it.

If you're using [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) (Colab), enable the GPU acceleration (**Runtime** > **Change runtime type** > **Hardware accelerator**:**GPU**).

## Setup

1. Download and install JAX, Jaxlib, Flax, and TensorFlow Datasets (TFDS) (to download MNIST). Check the Flax [docs](https://flax.readthedocs.io/en/latest/installation.html) and the JAX [README](https://github.com/google/jax/blob/master/README.md) for additional information.


```python
!pip install --upgrade -q pip jax jaxlib flax tensorflow-datasets
```

    [K     |████████████████████████████████| 1.5MB 15.7MB/s 
    [K     |████████████████████████████████| 163kB 53.2MB/s 
    [K     |████████████████████████████████| 3.6MB 55.6MB/s 
    [?25h

2. Import JAX, [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html) (which lets you run code on GPUs and TPUs), Flax, ordinary NumPy, and TFDS. Flax can use any data-loading pipeline and this example demonstrates how to utilize TFDS.


```python
import jax
import jax.numpy as jnp            # JAX NumPy

from flax import linen as nn        # The Linen API
from flax import optim              # Optimizers

import numpy as np                 # Ordinary NumPy
import tensorflow_datasets as tfds  # TFDS for MNIST
```

## Build the Linen model

Build a convolutional neural network with the Flax Linen API by subclassing [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#core-module-abstraction). Because the architecture in this example is relatively simple—you're just stacking layers—you can define the inlined submodules directly within the `__call__` method and wrap it with the `@compact` decorator ([`flax.linen.compact`](https://flax.readthedocs.io/en/latest/flax.linen.html#compact-methods)).


```python
class CNN(nn.Module):

  @nn.compact
  # Provide a constructor to register a new parameter 
  # and return its initial value
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

## Create a loss function, an optimizer, a parameter initializer, and a metrics function

1. Next, create a loss function—such as [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy)—using just [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html) that takes the model's logits and label vectors and returns a scalar loss. The labels need to be one-hot encoded, so define a separate function for that task, as demonstrated below.


```python
def onehot(labels, num_classes=10): # There are 10 classes in MNIST
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32) # Convert to float32 data type (JAX's default `dtype`)
```


```python
def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))
```

2. Define a function for your optimizer that takes the model parameters, the learning rate, and a beta argument (for the [Momentum optimizer](http://www.columbia.edu/~nq6/publications/momentum.pdf)):

  - Choose `Momentum` from the [`flax.optim`](https://flax.readthedocs.io/en/latest/flax.optim.html) package; and
  - Wrap the model parameters (`params`) with the [`flax.optim.OptimizerDef.create`](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.OptimizerDef.create) method and initialize with parameter dicts.


```python
def create_optimizer(params, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(params)
  return optimizer
```

3. Create a function for parameter initialization:

  - Set the initial shape of the kernel (note that JAX and Flax are [row-based](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html#Model-parameters-&-initialization)); and
  - Initialize the module parameters of your network (`CNN`) with the [`flax.linen.Module.init`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init) method using the PRNGKey and parameters (note that the parameters aren't stored with the models).


```python
def get_initial_params(key):
  init_shape = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_shape)['params']
  return initial_params
```

4. For loss and accuracy metrics, create a separate function:


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

## The dataset

Define a function that:
  - Uses TFDS to load and prepare the MNIST dataset; and
  - Converts the samples to floating-point numbers.


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

## Training and evaluation functions

1. Write a training step function that:

  - Evaluates the neural network given the parameters and a batch of input images with the [`flax.linen.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply) method.
  - Computes the `cross_entropy_loss` loss function.
  - Evaluates the loss function and its gradient using [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad) (check the [JAX autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Evaluate-a-function-and-its-gradient-using-value_and_grad) to learn more).
  - Applies a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions) of gradients ([`flax.optim.Optimizer.apply_gradient`](https://flax.readthedocs.io/en/latest/flax.optim.html#flax.optim.Optimizer.apply_gradient)) to the optimizer to update the model's parameters.
  - Computes the metrics using `compute_metrics` (defined earlier).

  Use JAX's [`@jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) decorator to trace the entire `train_step` function` and just-in-time([`jit`]-compile with [XLA](https://www.tensorflow.org/xla) into fused device operations that run faster and more efficiently on hardware accelerators.


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

2. Create a [`jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit)-compiled function that evaluates the model on the test set using [`flax.linen.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply):


```python
@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'])
```

3. Define a training function for one epoch that:

  - Shuffles the training data before each epoch using [`jax.random.permutation`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.permutation.html) that takes a PRNGKey as a parameter (discussed in more detail later in this tutorial and in [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG)).
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

4. Create a model evaluation function that:

  - Evalues the model on the test set.
  - Retrieves the evaluation metrics from the device with `jax.device_get`.
  - Copies the metrics [data stored](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) in a JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions).
  - Returns the test loss and accuracy.


```python
def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
  metrics = jax.device_get(metrics)
  eval_summary = jax.tree_map(lambda x: x.item(), metrics)
  return eval_summary['loss'], eval_summary['accuracy']
```

## Load the dataset

Download the dataset and preprocess it with `get_datasets` you defined earlier:


```python
train_ds, test_ds = get_datasets()
```


## Initialize the parameters and instantiate the optimizer

1. Before you start training the model, you need to randomly initialize the parameters.

  In NumPy, you would usually use the stateful pseudorandom number generators (PRNG). JAX, however, uses an explicit PRNG (refer to [JAX - the sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG) for details):

  - Start by getting two unsigned 32-bit integers with [`jax.random.PRNGKey`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax.random.PRNGKey) as one key (`rng`).
  - Then, split the PRNG to obtain a usable subkey for parameter initialization (`init_rng` below) using [`jax.random.split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split). 

  Note that in JAX and Flax you can have [separate PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables) (with different names, such as `rng` and `init_rng` below) inside `Module`s for different applications.


```python
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
```

2. You can now use the PRNG subkey—`init_rng`—to initialize the model's parameters by calling the predefined `get_initial_params` function:


```python
params = get_initial_params(init_rng)
```

3. Next, set the default learning rate and beta arguments for the Momentum optimizer and instantiate it:


```python
learning_rate = 0.1
beta = 0.9

optimizer = create_optimizer(params, learning_rate=learning_rate, beta=beta)
```

## Train the network and evaluate it

1. Set the default number of epochs and the size of each batch:


```python
num_epochs = 10
batch_size = 32
```

2. Finally, begin training and evaluating the model:

  - Similar to the parameter initialization stage, [split](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG) the PRNG key to get a new subkey—`input_rng`—with [`jax.random.split`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax.random.split). `input_rng` will be used to permute image data during the shuffling stage when training.
  - Run an optimization step over a training batch (`train_epoch`).
  - Evaluate on the test set after each training epoch (`eval_model`).
  - Retrieve the metrics from the device and print them.


```python
for epoch in range(1, num_epochs + 1):
  rng, input_rng = jax.random.split(rng)
  optimizer, train_metrics = train_epoch(optimizer, train_ds, batch_size, epoch, input_rng)
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

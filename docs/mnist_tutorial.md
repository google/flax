---
jupytext:
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/mnist_tutorial.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/mnist_tutorial.ipynb)

# MNIST Tutorial

Welcome to NNX! This tutorial will guide you through building and training a simple convolutional 
neural network (CNN) on the MNIST dataset using the NNX API. NNX is a Python neural network library
built upon [JAX](https://github.com/google/jax) and currently offered as an experimental module within 
[Flax](https://github.com/google/flax).

+++

## 1. Install NNX

Since NNX is under active development, we recommend using the latest version from the Flax GitHub repository:

```{code-cell}
:tags: [skip-execution]

# !pip install git+https://github.com/google/flax.git
```

## 2. Load the MNIST Dataset

First, the MNIST dataset is loaded and prepared for training and testing using 
Tensorflow Datasets. Image values are normalized, the data is shuffled and divided 
into batches, and samples are prefetched to enhance performance.

```{code-cell}
import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf  # TensorFlow operations

tf.random.set_seed(0)  # set random seed for reproducibility

num_epochs = 10
batch_size = 32

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize test set

# create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
train_ds = train_ds.repeat(num_epochs).shuffle(1024)
# group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
# create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
test_ds = test_ds.shuffle(1024)
# group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
```

## 3. Define the Network with NNX

Create a convolutional neural network with NNX by subclassing `nnx.Module`.

```{code-cell}
from flax.experimental import nnx  # NNX API
from functools import partial

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x

model = CNN(rngs=nnx.Rngs(0))
nnx.display(model)
```

### Run model

Let's put our model to the test!  We'll perform a forward pass with arbitrary data and print the results.

```{code-cell}
:outputId: 2c580f41-bf5d-40ec-f1cf-ab7f319a84da

import jax.numpy as jnp  # JAX NumPy

y = model(jnp.ones((1, 28, 28, 1)))
nnx.display(y)
```

## 4. Create Optimizer and Metrics

In NNX, we create an `Optimizer` object to manage the model's parameters and apply gradients during training. `Optimizer` receives the model parameters and an `optax` optimizer that will define the update rules. Additionally, we'll define a `MultiMetric` object to keep track of the `Accuracy` and the `Average` loss.

```{code-cell}
import optax

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(), 
  loss=nnx.metrics.Average('loss'),
)

nnx.display(optimizer)
```

## 5. Training step

We define a loss function using cross entropy loss (see more details in [`optax.softmax_cross_entropy_with_integer_labels()`](https://optax.readthedocs.io/en/latest/api/losses.html#optax.softmax_cross_entropy_with_integer_labels)) that our model will optimize over. In addition to the loss, the logits are also outputted since they will be used to calculate the accuracy metric during training and testing.

```{code-cell}
def loss_fn(model: CNN, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits
```

Next, we create the training step function. This function takes the `model` and a data `batch` and does the following:

* Computes the loss, logits and gradients with respect to the loss function using `nnx.value_and_grad`.
* Updates training accuracy using the loss, logits, and batch labels.
* Updates model parameters via the optimizer by applying the gradient updates.

```{code-cell}
@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])
  optimizer.update(grads)
```

The [`nnx.jit`](https://flax.readthedocs.io/en/latest/api_reference/flax.experimental.nnx/transforms.html#flax.experimental.nnx.jit) decorator traces the `train_step` function for just-in-time compilation with 
[XLA](https://www.tensorflow.org/xla), optimizing performance on 
hardware accelerators. `nnx.jit` is similar to [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit),
except it can transforms functions that contain NNX objects as inputs and outputs.

## 6. Evaluation step

Create a separate function to calculate loss and accuracy metrics for the test batch, since this will be outside the `train_step` function. Loss is determined using the `optax.softmax_cross_entropy_with_integer_labels` function, since we're reusing the loss function defined earlier.

```{code-cell}
@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])
```

## 7. Seed randomness

For reproducible dataset shuffling (using `tf.data.Dataset.shuffle`), set the TF random seed.

```{code-cell}
tf.random.set_seed(0)
```

## 8. Train and Evaluate

Now we train a model using batches of data for 10 epochs, evaluate its performance 
on the test set after each epoch, and log the training and testing metrics (loss and
accuracy) throughout the process. Typically this leads to a model with around 99% accuracy.

```{code-cell}
:outputId: 258a2c76-2c8f-4a9e-d48b-dde57c342a87

num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
  # Run the optimization for one step and make a stateful update to the following:
  # - the train state's model parameters
  # - the optimizer state
  # - the training loss and accuracy batch metrics
  train_step(model, optimizer, metrics, batch)

  if (step + 1) % num_steps_per_epoch == 0:  # one training epoch has passed
    # Log training metrics
    for metric, value in metrics.compute().items():  # compute metrics
      metrics_history[f'train_{metric}'].append(value)  # record metrics
    metrics.reset()  # reset metrics for test set

    # Compute metrics on the test set after each training epoch
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, test_batch)

    # Log test metrics
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # reset metrics for next training epoch

    print(
      f"train epoch: {(step+1) // num_steps_per_epoch}, "
      f"loss: {metrics_history['train_loss'][-1]}, "
      f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
    )
    print(
      f"test epoch: {(step+1) // num_steps_per_epoch}, "
      f"loss: {metrics_history['test_loss'][-1]}, "
      f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
    )
```

## 9. Visualize Metrics

Use Matplotlib to create plots for loss and accuracy.

```{code-cell}
:outputId: 431a2fcd-44fa-4202-f55a-906555f060ac

import matplotlib.pyplot as plt  # Visualization

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
  ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
```

## 10. Perform inference on test set

Define a jitted inference function, `pred_step`, to generate predictions on the test set using the learned model parameters. This will enable you to visualize test images alongside their predicted labels for a qualitative assessment of model performance.

```{code-cell}
@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)
```

```{code-cell}
:outputId: 1db5a01c-9d70-4f7d-8c0d-0a3ad8252d3e

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
  ax.set_title(f'label={pred[i]}')
  ax.axis('off')
```

Congratulations! You made it to the end of the annotated MNIST example.

---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.10.4 64-bit
  language: python
  name: python3
---

+++ {"id": "P9WN5UZzRqkE"}

# Dropout in image classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/tutorials/dropout_tutorial.ipynb)

This tutorial provides an end-to-end example of a simple image classification model with a dropout layer in Flax. The [dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) stochastic regularization technique randomly removes hidden and visible units in a network. The randomness in Flax's [`Dropout`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Dropout.html#flax.linen.Dropout) layer is handled internally with [`flax.linen.Module.make_rng`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.make_rng). This is covered in more detail in [🔪 Flax - The Sharp Bits 🔪 `flax.linen.Dropout` layer and randomness](https://flax.readthedocs.io/en/latest/notebooks/flax_sharp_bits.html#flax-linen-dropout-layer-and-randomness).

The example below uses a lot of fundamental concepts covered in [Getting started](https://flax.readthedocs.io/en/latest/getting_started.html). If you're new to Flax, start there.

+++ {"id": "NpnTy21qSWYV"}

## Setup

- Install/upgrade Flax, which will also set up [Optax](https://optax.readthedocs.io/) (for common optimizers and loss functions), and JAX.
- Install [TensorFlow Datasets](https://www.tensorflow.org/datasets) to load a dataset for this tutorial.
- Import the necessary libraries.

```{code-cell}
:id: 4Q1sL4cQTbAt

!pip install --upgrade -q flax tensorflow_datasets 
```

```{code-cell}
:id: 1kYKJYxxXUiL

import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Flax Linen API
from flax.training import train_state  # A Flax dataclass to keep the train state

import numpy as np                     # Ordinary NumPy
import optax                           # The Optax library
import tensorflow_datasets as tfds     # TFDS for the dataset
```

+++ {"id": "xTf7YtdgWWNY"}

Use a JAX PRNG key and split it to get one key for parameter initialization, and another one for dropout randomness:

```{code-cell}
:id: G9lWcsK1WUzL

seed = 0
root_key = jax.random.PRNGKey(seed=seed)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
```

+++ {"id": "P9pSTAIOSU1X"}

Create a simple Flax model, subclassed from [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics). Note that:

- To add a dropout layer in Flax, use [`flax.linen.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Dropout.html#flax.linen.Dropout).
- In `flax.linen.Dropout`, the `deterministic` argument is `None` by default. If `false`, the inputs are scaled by `1 / (1 - dropout_rate)` and masked. When it's `true`, no mask is applied (the dropout is turned off).

```{code-cell}
:id: yNWL1sDOZpTj

# A simple convolutional network with a dropout layer.
class CNN(nn.Module):
  training: bool
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # Flatten
    # Set the dropout layer with a rate of 50% .
    # When the `deterministic` flag is `True`, dropout is turned off.
    x = nn.Dropout(rate=0.5, deterministic=not self.training)(x)
    x = nn.Dense(features=10)(x)
    return x
```

+++ {"id": "wk-E94kkT51x"}

Define the loss function using [Optax](https://optax.readthedocs.io/):

```{code-cell}
:id: HXUHeCptgRXF

def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
```

+++ {"id": "UDOYyP4uT-Wt"}

Create a function for the loss and accuracy metrics:

```{code-cell}
:id: -IkOdtakLok5

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics
```

+++ {"id": "pL8F_kN5UAg1"}

Write a function for loading your dataset with [TensorFlow Datasets](https://www.tensorflow.org/datasets):

```{code-cell}
:id: t9sGGsSzLx13

def get_datasets():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds
```

+++ {"id": "TEiC2gBXUHlS"}

Create another function for creating the Flax [`TrainState`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state) with an [Optax](https://optax.readthedocs.io/) optimizer. Remember that:

- When initializing the variables, use the `params_key` PRNG key (the `params_key` is equivalent to a dictionary of PRNGs).
- The model constructor is `training=False` before you start training.

```{code-cell}
:id: aae3rz125R6z

def create_train_state(rng, learning_rate):
  # Instantiate the model with `training=False`.
  cnn = CNN(training=False)
  # Initialize the `params`. Use the `params_key` PRNG key.
  # (Here, you are providing only one PRNG key.) 
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  # Use the Adam optimizer with a learning rate.
  tx = optax.adam(learning_rate)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)
```

+++ {"id": "MRDfpGlkV5m5"}

Define the training step function. Note that:

- During the forward pass with [`flax.linen.apply()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#init-apply), use the `'dropout'` key (`dropout_key`) for the `rngs` argument.
- The model constructor argument should be set to `training=True`.

```{code-cell}
:id: CPsXnjKVH0hV

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    # Perform the forward pass with `flax.linen.apply()`.
    # Use the `dropout_key` for the `rngs` argument.
    logits = CNN(training=True).apply({'params': params}, batch['image'], rngs={'dropout': dropout_key})
    # Calculatet the loss,
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  # Compute the gradients
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics
```

+++ {"id": "BN5u62E6V8kJ"}

Write the evaluation step function. Remember to set the model constructor argument to `training=false`.

```{code-cell}
:id: IKODKHNNH_1k

@jax.jit
def eval_step(params, batch):
  logits = CNN(training=False).apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])
```

+++ {"id": "2dio74TEWBjt"}

Create a function for training the model for one epoch:

```{code-cell}
:id: 6dY_lKUz5SuG

def train_epoch(state, train_ds, batch_size, epoch, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch.
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # Compute the mean of metrics across each batch in an epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('Train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state
```

+++ {"id": "fuX4Ewz3hMZM"}

Create a model evaluation function:

```{code-cell}
:id: Sq5z385KPTuu

def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']
```

+++ {"id": "h-Ij6G7jw8xc"}

Download the dataset and split it into training and test sets:

```{code-cell}
:id: P5JcAhIs5Szz

train_ds, test_ds = get_datasets()
```

+++ {"id": "RDfng16c5S31"}

Initialize the Flax `TrainState`:

```{code-cell}
:id: qrLNat6c5S7Z

learning_rate = 0.1

state = create_train_state(rng=params_key, learning_rate=learning_rate)
```

+++ {"id": "b66WaNi8xBHz"}

Train the model over 10 epochs, and evaluate it:

```{code-cell}
:id: BpSx0ZBV5S-o

num_epochs = 10
batch_size = 32

for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling.
  main_key, input_rng = jax.random.split(key=root_key)
  # Run an optimization step over a training batch.
  state = train_epoch(state=state,
                      train_ds=train_ds,
                      batch_size=batch_size,
                      epoch=epoch,
                      rng=input_rng)
  # Evaluate on the test set after each training epoch.
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print('  Test epoch: %d, loss: %.2f, accuracy: %.2f' % (
      epoch, test_loss, test_accuracy * 100))
```

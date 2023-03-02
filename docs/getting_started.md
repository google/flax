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

+++ {"id": "6eea21b3"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/getting_started.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/getting_started.ipynb)

# Quickstart

Welcome to Flax!

Flax is an open source Python neural network library built on top of [JAX](https://github.com/google/jax). This tutorial demonstrates how to construct a simple convolutional neural
network (CNN) using the [Flax](https://flax.readthedocs.io) Linen API and train
the network for image classification on the MNIST dataset.

+++ {"id": "nwJWKIhdwxDo"}

## 1. Install Flax

```{code-cell}
:id: bb81587e
:tags: [skip-execution]

!pip install -q flax
```

+++ {"id": "b529fbef"}

## 2. Loading data

Flax can use any
data-loading pipeline and this example demonstrates how to utilize TFDS. Define a function that loads and prepares the MNIST dataset and converts the
samples to floating-point numbers.

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1673483483044
id: bRlrHqZVXZvk
---
import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf             # TensorFlow operations

def get_datasets(num_epochs, batch_size):
  """Load MNIST train and test datasets into memory."""
  train_ds = tfds.load('mnist', split='train')
  test_ds = tfds.load('mnist', split='test')

  train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                           tf.float32) / 255.,
                                          'label': sample['label']}) # normalize train set
  test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                         tf.float32) / 255.,
                                        'label': sample['label']}) # normalize test set

  train_ds = train_ds.repeat(num_epochs).shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
  test_ds = test_ds.shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency

  return train_ds, test_ds
```

+++ {"id": "7057395a"}

## 3. Define network

Create a convolutional neural network with the Linen API by subclassing
[Flax Module](https://flax.readthedocs.io/en/latest/flax.linen.html#core-module-abstraction).
Because the architecture in this example is relatively simple—you're just
stacking layers—you can define the inlined submodules directly within the
`__call__` method and wrap it with the
[`@compact`](https://flax.readthedocs.io/en/latest/flax.linen.html#compact-methods)
decorator. To learn more about the Flax Linen `@compact` decorator, refer to the [`setup` vs `compact`](https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html) guide.

```{code-cell}
---
executionInfo:
  elapsed: 53
  status: ok
  timestamp: 1673483483208
id: cbc079cd
---
from flax import linen as nn  # Linen API

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
```

+++ {"id": "hy7iRu7_zlx-"}

### View model layers

Create an instance of the Flax Module and use the [`Module.tabulate`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.tabulate) method to visualize a table of the model layers by passing an RNG key and template image input.

```{code-cell}
---
executionInfo:
  elapsed: 103
  status: ok
  timestamp: 1673483483427
id: lDHfog81zLQa
outputId: 2c580f41-bf5d-40ec-f1cf-ab7f319a84da
---
import jax
import jax.numpy as jnp  # JAX NumPy

cnn = CNN()
print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1))))
```

+++ {"id": "4b5ac16e"}

## 4. Create a `TrainState`

A common pattern in Flax is to create a single dataclass that represents the
entire training state, including step number, parameters, and optimizer state.

Because this is such a common pattern, Flax provides the class
[`flax.training.train_state.TrainState`](https://flax.readthedocs.io/en/latest/flax.training.html#train-state)
that serves most basic usecases.

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1673483483631
id: qXr7JDpIxGNZ
outputId: 1249b7fb-6787-41eb-b34c-61d736300844
---
!pip install -q clu
```

```{code-cell}
---
executionInfo:
  elapsed: 1
  status: ok
  timestamp: 1673483483754
id: CJDaJNijyOji
---
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
```

+++ {"id": "8b86b5f1"}

We will be using the `clu` library for computing metrics. For more information on `clu`, refer to the [repo](https://github.com/google/CommonLoopUtils) and [notebook](https://colab.research.google.com/github/google/CommonLoopUtils/blob/master/clu_synopsis.ipynb#scrollTo=ueom-uBWLbeQ).

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1673483483958
id: 7W0qf7FC9uG5
---
@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')
```

+++ {"id": "f3ce5e4c"}

You can then subclass `train_state.TrainState` so that it also contains metrics. This has the advantage that we only need
to pass around a single argument to functions like `train_step()` (see below) to calculate the loss, update the parameters and compute the metrics all at once.

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1673483484125
id: e0102447
---
class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())
```

+++ {"id": "a15de484"}

## 5. Training step

A function that:

- Evaluates the neural network given the parameters and a batch of input images
  with [`TrainState.apply_fn`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState) (which contains the [`Module.apply`](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply)
  method (forward pass)).
- Computes the cross entropy loss, using the predefined [`optax.softmax_cross_entropy_with_integer_labels()`](https://optax.readthedocs.io/en/latest/api.html#optax.softmax_cross_entropy_with_integer_labels). Note that this function expects integer labels, so there is no need to convert labels to onehot encoding.
- Evaluates the gradient of the loss function using
  [`jax.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad).
- Applies a
  [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions)
  of gradients to the optimizer to update the model's parameters.

Use JAX's [@jit](https://jax.readthedocs.io/en/latest/jax.html#jax.jit)
decorator to trace the entire `train_step` function and just-in-time compile
it with [XLA](https://www.tensorflow.org/xla) into fused device operations
that run faster and more efficiently on hardware accelerators.

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1673483484293
id: 9b0af486
---
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state
```

+++ {"id": "0ff5145f"}

## 6. Metric computation

Create a separate function for loss and accuracy metrics. Loss is calculated using the `optax.softmax_cross_entropy_with_integer_labels` function, while accuracy is calculated using `clu.metrics`.

```{code-cell}
---
executionInfo:
  elapsed: 53
  status: ok
  timestamp: 1673483484460
id: 961bf70b
---
@jax.jit
def compute_metrics(*, state, batch):
  logits = state.apply_fn({'params': state.params}, batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=batch['label'], loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state
```

+++ {"id": "497241c3"}

## 7. Download data

```{code-cell}
---
executionInfo:
  elapsed: 515
  status: ok
  timestamp: 1673483485090
id: bff5393e
---
num_epochs = 10
batch_size = 32

train_ds, test_ds = get_datasets(num_epochs, batch_size)
```

+++ {"id": "809ae1a0"}

## 8. Seed randomness

- Set the TF random seed to ensure dataset shuffling (with `tf.data.Dataset.shuffle`) is reproducible.
- Get one
  [PRNGKey](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax.random.PRNGKey)
  and use it for parameter initialization. (Learn
  more about
  [JAX PRNG design](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html)
  and [PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables).)

```{code-cell}
---
executionInfo:
  elapsed: 59
  status: ok
  timestamp: 1673483485268
id: xC4MFyBsfT-U
---
tf.random.set_seed(0)
```

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1673483485436
id: e4f6f4d3
---
init_rng = jax.random.PRNGKey(0)
```

+++ {"id": "80fbb60b"}

## 9. Initialize the `TrainState`

Remember that the function `create_train_state` initializes the model parameters, optimizer and metrics
and puts them into the training state dataclass that is returned.

```{code-cell}
---
executionInfo:
  elapsed: 56
  status: ok
  timestamp: 1673483485606
id: 445fcab0
---
learning_rate = 0.01
momentum = 0.9
```

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1673483485777
id: 5221eafd
---
state = create_train_state(cnn, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.
```

+++ {"id": "b1c00230"}

## 10. Train and evaluate

Create a "shuffled" dataset by:
- Repeating the dataset equal to the number of training epochs
- Allocating a buffer of size 1024 (containing the first 1024 samples in the dataset) of which to randomly sample batches from
  - Everytime a sample is randomly drawn from the buffer, the next sample in the dataset is loaded into the buffer

Define a training loop that:
- Randomly samples batches from the dataset.
- Runs an optimization step for each training batch.
- Computes the mean training metrics across each batch in an epoch.
- Computes the metrics for the test set using the updated parameters.
- Records the train and test metrics for visualization.

Once the training and testing is done after 10 epochs, the output should show that your model was able to achieve approximately 99% accuracy.

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1673483485947
id: '74295360'
---
# since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
```

```{code-cell}
---
executionInfo:
  elapsed: 1
  status: ok
  timestamp: 1673483486076
id: cRtnMZuQFlKl
---
metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}
```

```{code-cell}
---
executionInfo:
  elapsed: 17908
  status: ok
  timestamp: 1673483504133
id: 2c40ce90
outputId: 258a2c76-2c8f-4a9e-d48b-dde57c342a87
---
for step,batch in enumerate(train_ds.as_numpy_iterator()):

  # Run optimization steps over training batches and compute batch metrics
  state = train_step(state, batch) # get updated train state (which contains the updated parameters)
  state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

  if (step+1) % num_steps_per_epoch == 0: # one training epoch has passed
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

    # Compute metrics on the test set after each training epoch
    test_state = state
    for test_batch in test_ds.as_numpy_iterator():
      test_state = compute_metrics(state=test_state, batch=test_batch)

    for metric,value in test_state.metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)

    print(f"train epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss: {metrics_history['train_loss'][-1]}, "
          f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
    print(f"test epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss: {metrics_history['test_loss'][-1]}, "
          f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
```

+++ {"id": "gfsecJzvzgCT"}

## 11. Visualize metrics

```{code-cell}
---
executionInfo:
  elapsed: 358
  status: ok
  timestamp: 1673483504621
id: Zs5atiqIG9Kz
outputId: 431a2fcd-44fa-4202-f55a-906555f060ac
---
import matplotlib.pyplot as plt  # Visualization

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train','test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
  ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()
```

+++ {"id": "qQbKS0tV3sZ1"}

## 12. Perform inference on test set

Define a jitted inference function `pred_step`. Use the learned parameters to do model inference on the test set and visualize the images and their corresponding predicted labels.

```{code-cell}
---
executionInfo:
  elapsed: 580
  status: ok
  timestamp: 1673483505350
id: DFwxgBQf44ks
---
@jax.jit
def pred_step(state, batch):
  logits = state.apply_fn({'params': state.params}, test_batch['image'])
  return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(state, test_batch)
```

```{code-cell}
---
executionInfo:
  elapsed: 1250
  status: ok
  timestamp: 1673483506723
id: 5d5nF3u44JFI
outputId: 1db5a01c-9d70-4f7d-8c0d-0a3ad8252d3e
---
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
    ax.set_title(f"label={pred[i]}")
    ax.axis('off')
```

+++ {"id": "edb528b6"}

Congratulations! You made it to the end of the annotated MNIST example. You can revisit
the same example, but structured differently as a couple of Python modules, test
modules, config files, another Colab, and documentation in Flax's Git repo:

[https://github.com/google/flax/tree/main/examples/mnist](https://github.com/google/flax/tree/main/examples/mnist)

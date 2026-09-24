---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "JUM4j43iqagk"}

# Instructions
* Open a Colab project that you want to import code snippets to
* Go to ```Tools``` --> ```Settings```
* Go to ```Site``` --> ```Custom snippet notebook URL```
  * paste https://colab.research.google.com/github/google/flax/blob/flax_docs/docs/code_snippets.ipynb
  * click ```Save```
* Go to ```Help``` --> ```Search code snippets```
* Enter a header title from this Colab project (e.g. 'Imports', 'CNN', etc.) in the ```Filter code snippets``` field and import the code snippet into your Colab project

+++ {"id": "OV71CcWhjFc6"}

# Imports

```{code-cell}
:id: nDItEhLPi9hW

!pip install -q flax
```

```{code-cell}
:id: UZcqhiQLjGXa

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state, checkpoints

import numpy as np

import optax

import tensorflow_datasets as tfds

import os
import shutil
```

+++ {"id": "pB8nIPudp29d"}

# TensorFlow Datasets (TFDS)

```{code-cell}
:id: RzPnPpzQ2f_Z

# get a list of TensorFlow datasets that contain keyword
keyword = 'image'
[dataset for dataset in tfds.list_builders() if keyword in dataset]
```

```{code-cell}
:id: A8BytP1jqGB_

# load MNIST data
def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds
```

+++ {"id": "tgc4drZskjvr"}

# Module Template

```{code-cell}
:id: qcqUiO09kktp

from jax import random
import jax.numpy as jnp
from flax import linen as nn

class Foo(nn.Module):
  @nn.compact
  def __call__(self, x):
    pass

x = jnp.zeros((3, 4))
m = Foo()
variables = m.init(random.PRNGKey(0), x)
output = m.apply(variables, x)
```

+++ {"id": "ScrmK70F8wrk"}

# MLP

```{code-cell}
:id: oiaNt_Twgu9Z

class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # providing a name is optional though!
      # the default autonames would be "Dense_0", "Dense_1", ...
    return x
```

+++ {"id": "sVRaJzHhqVKr"}

# CNN

```{code-cell}
:id: X6EKVZhv8yuU

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

+++ {"id": "vsQH8K1qqXr3"}

# ResNet

```{code-cell}
:id: fUKMfiU2DbJH

# https://github.com/google/flax/tree/main/examples/imagenet/models.py
import os
from google.colab import files
if 'flax' not in os.listdir():
  !git clone https://github.com/google/flax.git
files.view('flax/examples/imagenet/models.py')
```

+++ {"id": "JgS8EtvMqc6q"}

# LSTM

```{code-cell}
---
colab:
  background_save: true
id: JjlmRAwsp0hf
---
# https://github.com/google/flax/tree/main/examples/seq2seq/models.py
import os
from google.colab import files
if 'flax' not in os.listdir():
  !git clone https://github.com/google/flax.git
files.view('flax/examples/seq2seq/models.py')
```

+++ {"id": "D8szTfNzqxZi"}

# Transformer

```{code-cell}
:id: kb8iG_edqynn

# https://github.com/google/flax/tree/main/examples/nlp_seq/models.py
import os
from google.colab import files
if 'flax' not in os.listdir():
  !git clone https://github.com/google/flax.git
files.view('flax/examples/nlp_seq/models.py')
```

+++ {"id": "L5RoBf_0HEFf"}

# Loss functions

```{code-cell}
:id: -xaQR5b5HDkg

def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
```

+++ {"id": "EXGFyIYCXJqZ"}

# Training loop

```{code-cell}
:id: CVpbzELFXN7S

def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)
```

```{code-cell}
:id: KKykybJgXPEo

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics
```

```{code-cell}
:id: mfXsJswGXQWv

@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])
```

```{code-cell}
:id: tfH2ykrK4ycS

# training loop
for epoch in range(1, n_epochs+1):
  state, metrics = train_step(state, train_ds)
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print(f"epoch: {epoch}, train loss: {metrics['loss']}, " + 
        f"train accuracy: {metrics['accuracy']*100}%, " + 
        f"test loss: {test_loss}, test accuracy: {test_accuracy*100}%")
```

+++ {"id": "LMqXB1EWXgcd"}

# Metrics

```{code-cell}
:id: c4L1uXtXXf7k

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics
```

```{code-cell}
:id: 0YE7lEPIXYa8

def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']
```

+++ {"id": "qVSkMgSRh1oJ"}

# Checkpoints

```{code-cell}
:id: KZeNFetnipWp

# https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#
```

```{code-cell}
:id: 8JPIL02eiBDP

# A simple model with one linear layer.
key1, key2 = random.split(random.PRNGKey(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.1)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)

# Some arbitrary nested pytree with a dictionary, a string, and a NumPy array.
config = {'dimensions': np.array([5, 3]), 'name': 'dense'}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}
```

```{code-cell}
---
colab:
  height: 35
executionInfo:
  elapsed: 59
  status: ok
  timestamp: 1668641917298
  user:
    displayName: Marcus Chiam
    userId: '17531616275590396120'
  user_tz: 480
id: lF0hLLtwiX1l
outputId: e3d3819a-a1ac-40fa-cbfb-f4d95614baa9
---
# Import Flax Checkpoints.
ckpt_dir = 'tmp/flax-checkpointing'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                            target=ckpt,
                            step=0,
                            overwrite=False,
                            keep=2)
```

```{code-cell}
:id: HCnpcW4pzOdD

empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=np.zeros_like(variables['params']),  # values of the tree leaf doesn't matter
    tx=tx,
)
target = {'model': empty_state, 'config': None, 'data': [jnp.zeros_like(x1)]}
state_restored = checkpoints.restore_checkpoint(ckpt_dir, target=target, step=0)
```

+++ {"id": "S56_oDtpTTDB"}

# Batch Norm

```{code-cell}
:id: 5LrS-QGtiw9E

# https://flax.readthedocs.io/en/latest/guides/batch_norm.html
```

```{code-cell}
:id: WDJMysBhTUK9

# Defining the model
class MLP(nn.Module):
  @nn.compact
  def __call__(self, x, train: bool):
    x = nn.Dense(features=4)(x)
    x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x
```

```{code-cell}
:id: yMm3pT5FUFzK

# batch_stats collection
mlp = MLP()
x = jnp.ones((1, 3))
variables = mlp.init(jax.random.PRNGKey(0), x, train=False)
params = variables['params']
batch_stats = variables['batch_stats']

jax.tree_util.tree_map(jnp.shape, variables)
```

```{code-cell}
:id: CFqGBk5lUDBg

# apply
y, updates = mlp.apply(
  {'params': params, 'batch_stats': batch_stats},
  x,
  train=True, mutable=['batch_stats']
)
batch_stats = updates['batch_stats']
```

```{code-cell}
:id: 4unQ1MJVURKm

# training and evaluation
class TrainState(train_state.TrainState):
  batch_stats: Any

state = TrainState.create(
  apply_fn=mlp.apply,
  params=params,
  batch_stats=batch_stats,
  tx=optax.adam(1e-3),
)
```

```{code-cell}
:id: LNlwMBZ0UWRW

@jax.jit
def train_step(state: TrainState, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits, updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      x=batch['image'], train=True, mutable=['batch_stats'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label'])
    return loss, (logits, updates)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits, updates)), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=updates['batch_stats'])
  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```

```{code-cell}
:id: WIPvVAI3UYjJ

@jax.jit
def eval_step(state: TrainState, batch):
  """Train for a single step."""
  logits = state.apply_fn(
    {'params': params, 'batch_stats': state.batch_stats},
    x=batch['image'], train=False)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label'])
  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```

+++ {"id": "L0ibKRrokXBy"}

# MNIST

```{code-cell}
:id: T-1MCdKxoO1e

# https://colab.sandbox.google.com/github/google/flax/blob/main/docs/getting_started.ipynb
import os
from google.colab import files
if 'flax' not in os.listdir():
  !git clone https://github.com/google/flax.git
files.view('flax/docs/getting_started.md')
```

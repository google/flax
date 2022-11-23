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

+++ {"id": "P9WN5UZzRqkE"}

# Variational autoencoder with a learning rate schedule

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/tutorials/vae_tutorial.ipynb)

This tutorial demonstrates how to train a simple variational autoencoder (VAE) end-to-end with learning rate scheduling using Flax and [Optax](https://optax.readthedocs.io/).

For the optimizer schedule, you will use the linear warmup followed by cosine decay ([`optax.warmup_cosine_decay_schedule`](https://optax.readthedocs.io/en/latest/api.html#optax.warmup_cosine_decay_schedule)).

The tutorial uses a lot of fundamental concepts covered in [Getting started](https://flax.readthedocs.io/en/latest/getting_started.html). If you're new to Flax, start there.

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
import tensorflow as tf                # TensorFlow for preprocessing operations (`tf.cast`, `tf.reshape`)
import tensorflow_datasets as tfds     # TFDS for the dataset
```

+++ {"id": "P9pSTAIOSU1X"}

Create a simple [variational autoencoder](https://arxiv.org/abs/1312.6114) model, subclassed from [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics).

```{code-cell}
:id: 24sGnMGXYBaU

class Encoder(nn.Module):
  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x
```

```{code-cell}
:id: rPnXF1LRYqan

class Decoder(nn.Module):

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(784, name='fc2')(z)
    return z
```

```{code-cell}
:id: -IahsXnifllL

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = jax.random.normal(rng, logvar.shape)
  return mean + eps * std
```

```{code-cell}
:id: 9pOD0G49e593

class VAE(nn.Module):
  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))
```

```{code-cell}
:id: ewLjblgq5mQC

def model():
  return VAE(latents=latents)
```

+++ {"id": "wk-E94kkT51x"}

Define the binary cross-entropy loss function.

In addition to optimizers, Optax provides a number of common loss functions, including [`optax.sigmoid_binary_cross_entropy`](https://optax.readthedocs.io/en/latest/api.html#optax.sigmoid_binary_cross_entropy).

```{code-cell}
:id: HXUHeCptgRXF

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
 logits = nn.log_sigmoid(logits)
 return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))
```

+++ {"id": "arucT1uPf00j"}

Define the KL divergence:

```{code-cell}
:id: g4F9g-Frf352

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
```

+++ {"id": "UDOYyP4uT-Wt"}

Create a function for the loss and accuracy metrics:

```{code-cell}
:id: -IkOdtakLok5

def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {
      'bce': bce_loss,
      'kld': kld_loss,
      'loss': bce_loss + kld_loss
  }
```

+++ {"id": "MRDfpGlkV5m5"}

Define the training step function. Note that:

- During the forward pass with [`flax.linen.apply()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#init-apply).
- The model constructor argument should be set to `training=True`.

```{code-cell}
:id: CPsXnjKVH0hV

@jax.jit
def train_step(state, batch, z_rng):
  """Train for a single step."""
  def loss_fn(params):
    # Perform the forward pass with `flax.linen.apply()`.
    recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)
    # Calculate the binary cross-entropy loss.
    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    # Calculate the KL divergence loss.
    kld_loss = kl_divergence(mean, logvar).mean()
    # Calculate the total loss.
    loss = bce_loss + kld_loss
    return loss
  # Compute the gradients
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)
```

+++ {"id": "BN5u62E6V8kJ"}

Write the evaluation step function. Remember to set the model constructor argument to `training=false`.

```{code-cell}
:id: IKODKHNNH_1k

@jax.jit
def eval_step(params, images, z, z_rng):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images, z_rng)
    comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                  recon_images[:8].reshape(-1, 28, 28, 1)])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    return metrics, comparison, generate_images

  return nn.apply(eval_model, model())({'params': params})
```

+++ {"id": "h-Ij6G7jw8xc"}

Download the dataset and split it into training and test sets:

```{code-cell}
:id: 421NeRsFue4K

def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x

# Write a function for loading your dataset with [TensorFlow Datasets](https://www.tensorflow.org/datasets):
def get_datasets():
  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()
  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(batch_size)
  train_ds = iter(tfds.as_numpy(train_ds))

  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = np.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)

  return train_ds, test_ds
```

```{code-cell}
:id: t65bqn3E7xPz

batch_size = 128

train_ds, test_ds = get_datasets()
```

+++ {"id": "nrjHKpNfj4bw"}

Use a JAX PRNG key and split it to get one key for parameter initialization:

```{code-cell}
:id: itqf1Acoh5HO

seed = 0
rng = jax.random.PRNGKey(seed=seed)
rng, key = jax.random.split(key=rng)
```

```{code-cell}
:id: ig8yKN7p1TuC

latents = 20

rng, z_key, eval_rng = jax.random.split(rng, 3)
z = jax.random.normal(z_key, (64, latents))
```

+++ {"id": "RDfng16c5S31"}

Create and initialize the Flax [`TrainState`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state) with an [Optax](https://optax.readthedocs.io/) optimizer. Remember that:

- When initializing the variables, use the `params_key` PRNG key (the `params_key` is equivalent to a dictionary of PRNGs).
- The model constructor is `training=False` before you start training.

This example uses the Yogi optimizer ([`optax.adabelief`](https://optax.readthedocs.io/en/latest/api.html#yogi)).

```{code-cell}
:id: qrLNat6c5S7Z

def create_train_state(rng, learning_rate_fn):
  # Instantiate the model with `training=False`.
  init_data = jnp.ones((batch_size, 784), jnp.float32)
  params = model().init(key, init_data, rng)['params']
  # Use an Optax optimizer.
  # The `learning_rate_fn` is an Optax learning rate schedule (defined further below).
  tx = optax.yogi(learning_rate_fn)
  return train_state.TrainState.create(
      apply_fn=model().apply, params=params, tx=tx)
```

+++ {"id": "VSffwrxFKZso"}

Define a learning rate schedule that uses [`optax.linear_schedule`](https://optax.readthedocs.io/en/latest/api.html#optax.linear_schedule) and [`optax.cosine_decay_schedule`](https://optax.readthedocs.io/en/latest/api.html#optax.cosine_decay_schedule) (cosine learning rate decay).

Note: You can learn more about Optax, its optimizers, loss functions and schedules in the [Optax tutorial](https://optax.readthedocs.io/en/latest/optax-101.html).

```{code-cell}
:id: LWM0mJYR7jsJ

def create_learning_rate_fn(base_learning_rate, warmup_steps, steps_per_epoch):
  warmup_fn = optax.linear_schedule(
      init_value=0.0,
      end_value=base_learning_rate,
      transition_steps=warmup_steps * steps_per_epoch,
  )
  cosine_epochs = max(num_epochs - warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch
  )
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_steps, steps_per_epoch],
  )

  return schedule_fn
```

+++ {"id": "vWxcjUaDKzOj"}

Instantiate the learning rate schedule:

```{code-cell}
:id: r-QJVoan9vNw

num_epochs = 10 # For simplicity, train for 10 epochs.
base_learning_rate = 0.0001
warmup_steps = 1.0
steps_per_epoch = 10 

learning_rate_fn = create_learning_rate_fn(
    base_learning_rate=base_learning_rate,
    warmup_steps=warmup_steps,
    steps_per_epoch=steps_per_epoch
    )
```

+++ {"id": "ROGRDeMdK58v"}

Initialize the Flax `TrainState`, passing in the learning rate schedule for the optimizer:

```{code-cell}
:id: wa8dsLoJ95na

state = create_train_state(rng=key, learning_rate_fn=learning_rate_fn)
```

+++ {"id": "b66WaNi8xBHz"}

Train the model for 10 epochs:

```{code-cell}
:id: BpSx0ZBV5S-o

for epoch in range(num_epochs):
  for _ in range(steps_per_epoch):
    batch = next(train_ds)  
    # Use a separate PRNG key to permute image data during shuffling.
    rng, key = jax.random.split(key=key)
      # Run an optimization step over a training batch.
    state = train_step(state=state, batch=batch, z_rng=key)

  metrics, comparison, sample = eval_step(params=state.params, images=test_ds, z=z, z_rng=eval_rng)
  print('Eval epoch: %d, loss: %.2f, binary cross-entropy: %.2f, KL divergence: %.2f' % (
      epoch+1, metrics['loss'], metrics['bce'], metrics['kld']))
```

# Flax: A neural network library for JAX designed for flexibility

**NOTE**: This is pre-release software. If you want to use it, please get in touch
with us at flax-dev@google.com.

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning
with the normal NumPy API
(+ additional APIs for special accelerator ops when needed)

JAX has some super powerful primitives, which you can compose arbitrarily:

* Autodiff (`jax.grad`): Efficient any-order gradients w.r.t any variables
* JIT compilation (`jax.jit`): Trace any function ⟶ fused accelerator ops
* Vectorization (`jax.vmap`): Automatically batch code written for individual samples
* Parallelization (`jax.pmap`): Automatically parallelize code across multiple accelerators (including across hosts, e.g. for TPU pods)

## What is Flax?

Flax is a neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to the framework.

Flax comes with:

* Common layers (`flax.nn`): Dense, Conv, BatchNorm, Attention, ...

* Optimizers (`flax.optim`): SGD, Momentum, Adam, LARS

* ...with replication (`optimizer.replicate()`): Multi-device training with any
  optimizer

* A ResNet ImageNet example, ready to be forked for your research.

* ...more examples in the works

### Flax Modules

In its core, Flax is built around parameterised functions called Modules.
These Modules override `apply` and can be used just like normal functions.

TODO: Clarify the nuances in the statement above.

For example you can define a learned linear transformation as follows:

```
from flax import nn
import jax.numpy as jnp

class Linear(nn.Module):
  def apply(self, x, num_features, kernel_init_fn):
    input_features = x.shape[-1]
    W = self.param('W', (input_features, num_features), kernel_init_fn)
    return jnp.dot(x, W)
```

## CPU-only Installation

You will need Python 3.5 or later.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```

## GPU accelerated installation

First install `jaxlib`; please follow the instructions in the
[Jax readme](https://github.com/google/jax/blob/master/README.md).
If they are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```


## Full end-to-end MNIST example

**NOTE**: See [docs/annotated_mnist.md](docs/annotated_mnist.md) for a version
with detailed annotations for each code block.

```py
import jax
import flax
import numpy as onp
import jax.numpy as jnp
import tensorflow_datasets as tfds

class CNN(flax.nn.Module):
  def apply(self, x):
    x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
    x = jax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
    x = flax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, features=256)
    x = flax.nn.relu(x)
    x = flax.nn.Dense(x, features=10)
    x = flax.nn.log_softmax(x)
    return x

@jax.vmap
def cross_entropy_loss(logits, label):
  return -logits[label]

def compute_metrics(logits, labels):
  loss = jnp.mean(cross_entropy_loss(logits, labels))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def train_step(optimizer, batch):
  def loss_fn(model):
    logits = model(batch['image'])
    loss = jnp.mean(cross_entropy_loss(
        logits, batch['label']))
    return loss, logits
  optimizer, _, _ = optimizer.optimize(loss_fn)
  return optimizer

@jax.jit
def eval(model, eval_ds):
  logits = model(eval_ds['image'] / 255.0)
  return compute_metrics(logits, eval_ds['label'])

def train():
  train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
  train_ds = train_ds.cache().shuffle(1000).batch(128)
  test_ds = tfds.as_numpy(tfds.load(
      'mnist', split=tfds.Split.TEST, batch_size=-1))

  _, model = CNN.create_by_shape(
      jax.random.PRNGKey(0),
      [((1, 28, 28, 1), jnp.float32)])

  optimizer = flax.optim.Momentum(
      learning_rate=0.1, beta=0.9).create(model)

  for epoch in range(10):
    for batch in tfds.as_numpy(train_ds):
      batch['image'] = batch['image'] / 255.0
      optimizer = train_step(optimizer, batch)

    metrics = eval(optimizer.target, test_ds)
    print('eval epoch: %d, loss: %.4f, accuracy: %.2f'
         % (epoch+1,
          metrics['loss'], metrics['accuracy'] * 100))
```

## More end-to-end examples

**NOTE**: We are still testing these examples across all supported hardware configurations.

* [ResNet on ImageNet](examples/imagenet)

* [Language Modeling on LM1b](examples/lm1b)

## HOWTOs

HOWTOs are sample diffs showing how to change various things in your training
code.

Here are a few examples.



### Polyak averaging

This diff shows how to modify the MNIST example above to evaluate with
an exponential moving average of parameters over the course of training.

Note that no special framework support was needed.

```py
--- a/mnist.py
+++ b/mnist-polyak.py
@@ -29,14 +29,17 @@ def compute_metrics(logits, labels):
   return {'loss': loss, 'accuracy': accuracy}

 @jax.jit
-def train_step(optimizer, batch):
+def train_step(optimizer, params_ema, batch):
   def loss_fn(model):
     logits = model(batch['image'])
     loss = jnp.mean(cross_entropy_loss(
         logits, batch['label']))
     return loss, logits
   optimizer, _, _ = optimizer.optimize(loss_fn)
-  return optimizer
+  params_ema = jax.tree_multimap(
+    lambda p_ema, p: p_ema * 0.99 + p * 0.01,
+    params_ema, optimizer.target.params)
+  return optimizer, params_ema

 @jax.jit
 def eval(model, eval_ds):
@@ -59,9 +62,9 @@ def train():
   for epoch in range(10):
     for batch in tfds.as_numpy(train_ds):
       batch['image'] = batch['image'] / 255.0
-      optimizer = train_step(optimizer, batch)
+      optimizer, params_ema = train_step(optimizer, params_ema, batch)

-    metrics = eval(optimizer.target, test_ds)
+    metrics = eval(optimizer.target.replace(params=params_ema), test_ds)
     print('eval epoch: %d, loss: %.4f, accuracy: %.2f'
          % (epoch+1,
           metrics['loss'], metrics['accuracy'] * 100))
```

## Getting involved

**Have questions? Want to learn more? Reach out to us at flax-dev@google.com**

### Want to help?

We're happy to work together, either remotely or in Amsterdam.

In addition to general improvements
to the framework, here are some specific things that would be great to have:

#### Help build more HOWTOs

- Batch Norm
- Checkpointing
- ...and many more

#### Help build new end-to-end examples

- Translation
- CIFAR10 (in-progress)
- Semantic Segmentation
- GAN
- VAE
- ...and your proposal!

# Note

This is not an official Google product.

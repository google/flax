# Flax: A neural network library for JAX designed for flexibility

The FLAX documentation can be found here: https://flax.readthedocs.io/en/latest/

**NOTE**: This is pre-release software and not yet ready for general use. If you want to use it, please get in touch with us at flax-dev@google.com.

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

**NOTE**: See [docs/annotated_mnist.md](docs/annotated_mnist.md) for an MNIST
example with detailed annotations for each code block.

### Flax Modules

In its core, Flax is built around parameterised functions called Modules.
These Modules override `apply` and can be used just like normal functions.

TODO: Clarify the nuances in the statement above.

For example you can define a learned linear transformation as follows:

```py
from flax import nn
import jax.numpy as jnp

class Linear(nn.Module):
  def apply(self, x, num_features, kernel_init_fn):
    input_features = x.shape[-1]
    W = self.param('W', (input_features, num_features), kernel_init_fn)
    return jnp.dot(x, W)
```

You can also use `nn.module` as a function decorator to create a new module, as
long as you don't need access to `self` for creating parameters directly:

```py
@nn.module
def DenseLayer(x, features):
  x = flax.nn.Dense(x, features)
  x = flax.nn.relu(x)
  return x
```

## CPU-only Installation

You will need Python 3.5 or later.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```

## GPU accelerated installation

First install `jaxlib`; please follow the instructions in the
[JAX readme](https://github.com/google/jax/blob/master/README.md).
If they are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Now install `flax` from Github:

```
> pip install git+https://github.com/google-research/flax.git@prerelease
```


## Full end-to-end MNIST example

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
  train_ds = train_ds.map(lambda x: {'image':tf.cast(x['image'], tf.float32),
                                     'label':tf.cast(x['label'], tf.int32)})
  train_ds = train_ds.cache().shuffle(1000).batch(128)
  test_ds = tfds.as_numpy(tfds.load(
      'mnist', split=tfds.Split.TEST, batch_size=-1))
  test_ds = {'image': test_ds['image'].astype(jnp.float32),
             'label': test_ds['label'].astype(jnp.int32)}

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

* [Language Modeling on LM1b](examples/lm1b) with a Transformer architecture

## HOWTOs

HOWTOs are sample diffs showing how to change various things in your training
code.

Here is an example.


### Ensemble learning

This howto changes the MNIST example from training a single CNNs to training an
ensemble of CNNs, such that each CNN is trained on its own device. Each CNN
reports the accuracy and loss. 

```py
diff --git a/examples/mnist/train.py b/examples/mnist/train.py
index e93321f..7ac4fe2 100644
--- a/examples/mnist/train.py
+++ b/examples/mnist/train.py
@@ -85,9 +86,11 @@ def create_model(key):
   return model
 
-def create_optimizer(model, learning_rate, beta):
-  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
-  optimizer = optimizer_def.create(model)
+@jax.pmap
+def create_optimizers(rng):
+  optimizer_def = optim.Momentum(
+      learning_rate=FLAGS.learning_rate, beta=FLAGS.momentum)
+  optimizer = optimizer_def.create(create_model(rng))
   return optimizer
 
@@ -110,7 +113,7 @@ def compute_metrics(logits, labels):
   return metrics
 
-@jax.jit
+@functools.partial(jax.pmap)
 def train_step(optimizer, batch):
   """Train for a single step."""
   def loss_fn(model):
@@ -122,13 +125,17 @@ def train_step(optimizer, batch):
   return optimizer, metrics
 
-@jax.jit
+@jax.pmap
 def eval_step(model, batch):
   logits = model(batch['image'])
   return compute_metrics(logits, batch['label'])
 
-def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
+def replicate(tree_obj, num_replicas):
+  return jax.tree_map(lambda x: onp.array([x] * num_replicas), tree_obj)
+
+
+def train_epoch(optimizers, train_ds, batch_size, epoch, rng, num_models):
   """Train for a single epoch."""
   train_ds_size = len(train_ds['image'])
   steps_per_epoch = train_ds_size // batch_size
@@ -139,25 +146,27 @@ def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
   batch_metrics = []
   for perm in perms:
     batch = {k: v[perm] for k, v in train_ds.items()}
-    optimizer, metrics = train_step(optimizer, batch)
+    batch = replicate(batch, num_models)
+    optimizers, metrics = train_step(optimizers, batch)
     batch_metrics.append(metrics)
 
   # compute mean of metrics across each batch in epoch.
   batch_metrics_np = jax.device_get(batch_metrics)
+  batch_metrics_np = jax.tree_multimap(lambda *xs: onp.array(xs),
+                                       *batch_metrics_np)
   epoch_metrics_np = {
-      k: onp.mean([metrics[k] for metrics in batch_metrics_np])
-      for k in batch_metrics_np[0]}
-
-  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
+      k: onp.mean(batch_metrics_np[k], axis=0) for k in batch_metrics_np
+  }
+  logging.info('train epoch: %d, loss: %s, accuracy: %s', epoch,
                epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)
 
-  return optimizer, epoch_metrics_np
+  return optimizers, epoch_metrics_np
 
 
-def eval_model(model, test_ds):
-  metrics = eval_step(model, test_ds)
+def eval_model(models, test_ds):
+  metrics = eval_step(models, test_ds)
   metrics = jax.device_get(metrics)
-  summary = jax.tree_map(lambda x: x.item(), metrics)
+  summary = metrics
   return summary['loss'], summary['accuracy']
 
 
@@ -174,19 +183,20 @@ def train(train_ds, test_ds):
 
   batch_size = FLAGS.batch_size
   num_epochs = FLAGS.num_epochs
+  num_models = jax.device_count()
 
-  model = create_model(rng)
-  optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)
+  optimizers = create_optimizers(random.split(rng, num_models))
 
   input_rng = onp.random.RandomState(0)
+  test_ds = replicate(test_ds, num_models)
 
   for epoch in range(1, num_epochs + 1):
-    optimizer, _ = train_epoch(
-        optimizer, train_ds, batch_size, epoch, input_rng)
-    loss, accuracy = eval_model(optimizer.target, test_ds)
-    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
-                 epoch, loss, accuracy * 100)
-  return optimizer
+    optimizers, _ = train_epoch(optimizers, train_ds, batch_size, epoch,
+                                input_rng, num_models)
+    loss, accuracy = eval_model(optimizers.target, test_ds)
+    logging.info('eval epoch: %d, loss: %s, accuracy: %s', epoch, loss,
+                 accuracy * 100)
+  return optimizers
```

## Getting involved

**Have questions? Want to learn more? Reach out to us at flax-dev@google.com**

### Want to help?

We're happy to work together, either remotely or in Amsterdam.

In addition to general improvements
to the framework, here are some specific things that would be great to have:

#### Help build more HOWTOs

(TODO: clarify list)

#### Help build new end-to-end examples

- Semantic Segmentation
- GAN
- VAE
- ...and your proposal!

# Note

This is not an official Google product.

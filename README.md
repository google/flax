# Flax: A neural network library for JAX designed for flexibility

**NOTE**: This is alpha software, but we encourage trying it out.
Changes will come to the API, but we'll use deprecation warnings when we can, and
keep track of them our [Changelog](CHANGELOG.md).

A growing community of researchers at Google are happily using
Flax daily for their research, and now we'd like to extend that support to the
open source community. GitHub issues are encouraged for open converation, but
in case you need to reach us directly, we're at flax-dev@google.com.

## Quickstart

**⟶ [Full documentation and API reference](https://flax.readthedocs.io/)**

**⟶ [Annotated full end-to-end MNIST example](docs/annotated_mnist.md)**

**⟶ [The Flax Guide](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html)** -- a guided walkthrough of the parts of Flax

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning
with the normal NumPy API
(+ additional APIs for special accelerator ops when needed)

JAX comes with powerful primitives, which you can compose arbitrarily:

* Autodiff (`jax.grad`): Efficient any-order gradients w.r.t any variables
* JIT compilation (`jax.jit`): Trace any function ⟶ fused accelerator ops
* Vectorization (`jax.vmap`): Automatically batch code written for individual samples
* Parallelization (`jax.pmap`): Automatically parallelize code across multiple accelerators (including across hosts, e.g. for large TPUs)

## What is Flax?

Flax is a high-performance neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to the framework.

Flax comes with everything you need to start your research, including:

* A module abstraction (`flax.nn.Module`) for parameterized functions such as neural network layers.

* Common layers (`flax.nn`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* Optimizers (`flax.optim`): SGD, Momentum, Adam, LARS

* Utilities and patterns: replicated training, serialization and checkpointing, metrics, prefetching on device

* Educational examples that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* HOWTO guides -- diffs that add functionality to educational base exampless

* Fast, tuned large-scale end-to-end examples: CIFAR10, ResNet ImageNet, Transformer LM1b

### An annotated MNIST example

See [docs/annotated_mnist.md](docs/annotated_mnist.md) for an MNIST
example with detailed annotations for each code block.

### Flax Modules

The core of Flax is the Module abstraction. Modules allow you to write parameterized functions just as if you were writing a normal numpy function with JAX. The Module api allows you to declare parameters and use them directly with the JAX api’s.

Modules are the one part of Flax with "magic" -- the magic is constrained, and enables a very ergonomic style,
where modules are defined in a single function with minimal boilerplate.

A few things to know about Modules:

1. Create a new module by subclassing `flax.nn.Module` and implementing the `apply` method.

2. Within `apply`, call `self.param(name, shape, init_func)` to register a new parameter and returns its initial value.

3. Apply submodules by calling `MySubModule(...args...)` within `MyModule.apply`. Parameters of `MySubModule` are stored
as a dictionary under the parameters `MyModule`. **NOTE:** this returns the *output* of `MySubModule`, not an instance. To get an access to an instance of `MySubModule` for re-use, use [`Module.partial`](https://flax.readthedocs.io/en/latest/flax.nn.html#flax.nn.Module.partial) or [`Module.shared`](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Parameter-sharing)

4. `MyModule.init(rng, ...)` is a pure function that calls `apply` in "init mode" and returnes a nested Python dict of initialized parameter values

5. `MyModule.call(params, ...)` is a pure function that calls `apply` in "call mode" and returnes the output of the module.

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

Read more about Flax Modules and the other parts of the Flax API in the [Flax Guide](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Flax-Modules)

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
    x = flax.nn.relu(x)
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
    return loss
  grad = jax.grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
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

  _, initial_params = CNN.init_by_shape(
      jax.random.PRNGKey(0),
      [((1, 28, 28, 1), jnp.float32)])
  model = nn.Model(CNN, initial_params)

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

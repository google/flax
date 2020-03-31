# Flax: A neural network library for JAX designed for flexibility

**NOTE**: Flax is being actively improved and has a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. Flax is in "early release stage" -- if that's your style,
now could be a good time to start using it.
We want to smooth out any rough edges so please report
any issues, questions or concerns as
[GitHub issues](https://github.com/google/flax/issues). Expect changes to the
API, but we'll use deprecation warnings when we can, and keep
track of them in our [Changelog](CHANGELOG.md).

In case you need to reach us directly, we're at flax-dev@google.com.

## Quickstart

**⟶ [Full documentation and API reference](https://flax.readthedocs.io/)**

**⟶ [Annotated full end-to-end MNIST example](https://flax.readthedocs.io/en/latest/annotated_mnist.html)**

**⟶ [The Flax Guide](https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html)** -- a guided walkthrough of the parts of Flax

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning
with the normal NumPy API
(+ additional APIs for special accelerator ops when needed)

JAX comes with powerful primitives, which you can compose arbitrarily:

* Autodiff (`jax.grad`): Efficient any-order gradients w.r.t any variables
* JIT compilation (`jax.jit`): Trace any function ⟶ fused accelerator ops
* Vectorization (`jax.vmap`): Automatically batch code written for individual samples
* Parallelization (`jax.pmap`): Automatically parallelize code across multiple accelerators (including across hosts, e.g. for TPU pods)

## What is Flax?

Flax is a high-performance neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to a framework.

Flax is being developed in close collaboration with the JAX team and 
comes with everything you need to start your research, including:

* **Common layers** (`flax.nn`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* **Optimizers** (`flax.optim`): SGD, Momentum, Adam, LARS

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **HOWTO guides** -- diffs that add functionality to educational base exampless

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Try Flax now by forking one of our starter examples

### Image Classification
⟶ [MNIST](examples/mnist) (also see [annotated version](https://flax.readthedocs.io/en/latest/annotated_mnist.html))

⟶ [CIFAR-10](examples/cifar10) (Wide ResNet w/ and w/o Shake-Shake, PyramidNet w/ShakeDrop)

⟶ [ResNet50 on ImageNet](examples/imagenet)

### Transformer Models
⟶ [Sequence tagging on Universal Dependencies](examples/nlp_seq)

⟶ [LM1b language modeling](examples/lm1b) **([try on a TPU in Colab](https://colab.research.google.com/github/google/flax/blob/master/examples/lm1b/Colab_Language_Model.ipynb))**

⟶ (work-in-progress) [WMT translation](https://github.com/google/flax/pull/133)

### RNNs
⟶ [LSTM seq2seq on number addition](examples/seq2seq)


### Generative Models
⟶ [Basic VAE](examples/vae)

### Graph Neural Networks
⟶ [Semi-supervised node classification on Zachary's karate club](examples/graph)

## The Flax Module abstraction in a nutshell

The core of Flax is the Module abstraction. Modules allow you to write parameterized functions just as if you were writing a normal numpy function with JAX. The Module api allows you to declare parameters and use them directly with the JAX api’s.

Modules are the one part of Flax with "magic" -- the magic is constrained, and enables a very ergonomic model construction style, where modules are defined in a single function with minimal boilerplate.

A few things to know about Modules:

1. Create a new module by subclassing `flax.nn.Module` and implementing the `apply` method.

2. Within `apply`, call `self.param(name, shape, init_func)` to register a new parameter and returns its initial value.

3. Apply submodules with `MySubModule(name=..., ...)` within `MyModule.apply`. Parameters of `MySubModule` are stored
as a dictionary under the parameters `MyModule` and accessible via `self.get_param(name=...)`. This applies `MySubmodule` once --
to re-use parameters, use [`Module.shared`](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Parameter-sharing)

4. `MyModule.init(rng, ...)` is a pure function that calls `apply` in "init mode" and returns a nested Python dict of initialized parameter values

5. `MyModule.call(params, ...)` is a pure function that calls `apply` in "call mode" and returns the output of the module.

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

**⟶ Read more about Modules in the [Flax Guide](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Flax-Modules)**

## A full ResNet implementation

(from [examples/imagenet/models.py](examples/imagenet/models.py))

```py
class ResidualBlock(nn.Module):
  def apply(self, x, filters, strides=(1, 1), train=True, dtype=jnp.float32):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=dtype)
    conv = nn.Conv.partial(bias=False, dtype=dtype)

    residual = x
    if needs_projection:
      residual = conv(residual, filters * 4, (1, 1), strides, name='proj_conv')
      residual = batch_norm(residual, name='proj_bn')

    y = conv(x, filters, (1, 1), name='conv1')
    y = batch_norm(y, name='bn1')
    y = nn.relu(y)
    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = batch_norm(y, name='bn2')
    y = nn.relu(y)
    y = conv(y, filters * 4, (1, 1), name='conv3')

    y = batch_norm(y, name='bn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  def apply(self, x, num_classes, num_filters=64, num_layers=50,
            train=True, dtype=jnp.float32):
    if num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[num_layers]
    x = nn.Conv(
        x, num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
        bias=False, dtype=dtype, name='init_conv')
    x = nn.BatchNorm(
        x, use_running_average=not train, momentum=0.9,
        epsilon=1e-5, dtype=dtype, name='init_bn')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(
            x, num_filters * 2 ** i, strides=strides,
            train=train, dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes)
    x = nn.log_softmax(x)
    return x
```

## Installation

You will need Python 3.6 or later.

For GPU support, first install `jaxlib`; please follow the
instructions in the [JAX
readme](https://github.com/google/jax/blob/master/README.md).  If they
are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Then install `flax` from PyPi:
```
> pip install flax
```

## TPU support

We currently have a [LM1b/Wikitext-2 language model with a Transformer architecture](https://colab.research.google.com/github/google/flax/blob/master/examples/lm1b/Colab_Language_Model.ipynb)
that's been tuned. You can run it directly via Colab.

At present, Cloud TPUs are network-attached, and Flax users typically feed in data from one or more additional VMs

When working with large-scale input data, it is important to create large enough VMs with sufficient network bandwidth to avoid having the TPUs bottlenecked waiting for input

TODO: Add an example for running on Google Cloud.

## Note

This is not an official Google product.

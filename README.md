# Flax: A neural network library for JAX designed for flexibility

**NOTE**: Flax is being actively improved and has a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. Flax is "early release" but now is a good time to
start using it. We want to smooth out any rough edges so please report
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

**⟶ Read more about Modules in the [Flax Guide](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Flax-Modules)**

## CPU-only Installation

You will need Python 3.5 or later.

Now install `flax` from PyPi:

```
> pip install flax
```

## GPU accelerated installation

First install `jaxlib`; please follow the instructions in the
[JAX readme](https://github.com/google/jax/blob/master/README.md).
If they are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Now install `flax` from PyPi:

```
> pip install flax
```



## List of end-to-end examples

**NOTE**: We are still testing these examples across all supported hardware configurations.

* [ResNet on ImageNet](examples/imagenet)

* [Language Modeling on LM1b](examples/lm1b) with a Transformer architecture

* WIP: [WMT translation](https://github.com/google/flax/pull/61) with a Transformer architecture and on-device beam decoding

# Note

This is not an official Google product.

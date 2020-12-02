# Flax: A neural network ecosystem for JAX designed for flexibility

[**Overview**](#overview)
| [**Quick install**](#quick-install)
| [**What does Flax look like?**](#what-does-flax-look-like)
| [**Documentation**](https://flax.readthedocs.io/)

[![coverage](https://badgen.net/codecov/c/github/google/flax)](https://codecov.io/github/google/flax)

Please check our [full documentation](https://flax.readthedocs.io/) website to learn everything you need to know about Flax.

**NOTE**: Flax is in use by a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. The new Flax ["Linen" module API](https://github.com/google/flax/tree/master/flax/linen) is now stable and we recommend it for all new projects. The old `flax.nn` API will be deprecated. Please report
any feature requests, issues, questions or concerns in our 
[discussion forum](https://github.com/google/flax/discussions), or just let us know 
what you're working on!

Expect changes to the
API, but we'll use deprecation warnings when we can, and keep
track of them in our [Changelog](CHANGELOG.md).

In case you need to reach us directly, we're at flax-dev@google.com.

## Overview

Flax is a high-performance neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to a framework.

Flax is being developed in close collaboration with the JAX team and 
comes with everything you need to start your research, including:

* **Neural network API** (`flax.linen`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* **Optimizers** (`flax.optim`): SGD, Momentum, Adam, LARS, Adagrad, LAMB, RMSprop

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Quick install

You will need Python 3.6 or later and a working [JAX](https://github.com/google/jax/blob/master/README.md)
installation (with or without GPU support, see instructions there). For a
CPU-only version:

```
> pip install --upgrade pip # To support manylinux2010 wheels.
> pip install --upgrade jax jaxlib # CPU-only
```

Then install Flax from PyPi:

```
> pip install flax
```

To upgrade to the latest version of Flax, you can use:

```
> pip install --upgrade git+https://github.com/google/flax.git
```

## What does Flax look like?

We provide here two examples using the Flax API: a simple multi-layer perceptron and a CNN. To learn more about the `Module` abstraction, please check our [docs](https://flax.readthedocs.io/).

```py
class SimpleMLP(nn.Module):
  """ A MLP model """
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat)(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x
```

```py
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
    x = nn.log_softmax(x)
    return x
```

## Note

This is not an official Google product.

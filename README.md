# Flax: A neural network library and ecosystem for JAX designed for flexibility

[**Overview**](#overview)
| [**What does Flax look like?**](#what-does-flax-look-like)
| [**Documentation**](https://flax.readthedocs.io/)

[![coverage](https://badgen.net/codecov/c/github/google/flax)](https://codecov.io/github/google/flax)

Please check our [full documentation](https://flax.readthedocs.io/) website to learn everything you need to know about Flax.

**NOTE**: Flax is in use by a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. The new Flax ["Linen" module API](https://github.com/google/flax/tree/master/flax/linen/README.md) is now stable and we recommend it for all new projects. The old `flax.nn` API will be deprecated. Please report
any feature requests, issues, questions or concerns in our 
[discussion forum](https://github.com/google/flax/discussions), or just let us know 
what you're working on!

We expect to add some improvements to Flax, but we only expect minor API changes to the core API. We will use [Changelog](CHANGELOG.md) entries and deprecation warnings when possible.

In case you want to reach us directly, we're at flax-dev@google.com.

## Overview

Flax is a high-performance neural network library and ecosystem for
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

## What does Flax look like?

We provide three examples using the Flax API: a simple multi-layer perceptron, a CNN and an auto-encoder. 

To learn more about the `Module` abstraction, please check our [docs](https://flax.readthedocs.io/), or visit our
[patterns](https://flax.readthedocs.io/en/latest/patterns/flax_patterns.html) page for additional concrete demonstrations of best practices.

```py
class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(Dense(feat)(x))
    x = Dense(self.features[-1])(x)
    return x
```

```py
class CNN(nn.Module):
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

```py
class AutoEncoder(Module):
  encoder_widths: Iterable
  decoder_widths: Iterable
  input_shape: Tuple = None

  def setup(self):
    self.encoder = MLP(self.encoder_widths)
    self.decoder = MLP(self.decoder_widths + (jnp.prod(self.input_shape, ))

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    assert x.shape[-len(self.input_shape):] == self.input_shape
    return self.encoder(jnp.reshape(x, (x.shape[0], -1)))

  def decode(self, z):
    z = self.decoder(z)
    x = nn.sigmoid(z)
    x = jnp.reshape(x, (x.shape[0],) + self.input_shape)
    return x
```

## Note

This is not an official Google product.

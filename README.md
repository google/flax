# Flax: A neural network ecosystem for JAX designed for flexibility

[**Overview**](#overview)
| [**Quickstart**](#quickstart)
| [**Trying Flax**](#trying-flax)
| [**Installation**](#installation)
| [**Full documentation**](https://flax.readthedocs.io/)


[![coverage](https://badgen.net/codecov/c/github/google/flax)](https://codecov.io/github/google/flax)

**NOTE**: Flax is in use by a growing community
of researchers and engineers at Google who happily use Flax for their
daily research. The new Flax ["Linen" module API](https://github.com/google/flax/tree/master/flax/linen) is now stable and we recommend it for all new projects. The old `flax.nn` API will be deprecated. Please report
any issues, questions or concerns in our 
[discussion forum](https://github.com/google/flax/discussions), or just let us know 
what you're working on!

Expect changes to the
API, but we'll use deprecation warnings when we can, and keep
track of them in our [Changelog](CHANGELOG.md).

In case you need to reach us directly, we're at flax-dev@google.com.

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

## Trying Flax

We keep here a limited list of canonical examples maintained by the Flax team that you can fork to get started. If you are looking for more examples, or others built by the community, please check the [linen_examples folder](linen_examples/).

### Image classification

* [ImageNet](linen_examples/imagenet/)
* [MNIST](linen_examples/mnist/)
* [PixelCNN](linen_examples/pixelcnn/)

### Reinforcement Learning

* [Proximal Policy Optimization](linen_examples/ppo/)

### Natural language processing

* [Sequence to sequence for number addition](linen_examples/seq2seq/)
* [Transformer model on WMT](linen_examples/wmt/)

### Generative models

* [Varational auto-encoder](linen_examples/vae/)


## What does Flax look like?

We provide here two examples using the Flax API: a simple multi-layer perceptron and a CNN. To learn more about the `Module` abstraction, please check our docs.

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

## Getting involved

Currently, you need to install Python 3.6 for developing Flax, and `svn` for running the `run_all_tests.sh` script. After installing these prerequisites, you can clone the repository, set up your local environment, and run all tests with the following commands:

```
git clone https://github.com/google/flax
cd flax
python3.6 -m virtualenv env
. env/bin/activate
pip install -e . .[testing]
./tests/run_all_tests.sh
```

Alternatively, you can also develop inside a Docker container : See [`dev/README.md`](dev/README.md).

We welcome pull requests, in particular for those issues [marked as PR-ready](https://github.com/google/flax/issues?q=is%3Aopen+is%3Aissue+label%3A%22Status%3A+pull+requests+welcome%22). For other proposals, we ask that you first open an Issue to discuss your planned contribution.

## Note

This is not an official Google product.

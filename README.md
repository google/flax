<div align="center">
<img src="https://raw.githubusercontent.com/google/flax/main/images/flax_logo_250px.png" alt="logo"></img>
</div>

# Flax: A neural network library and ecosystem for JAX designed for flexibility

![Build](https://github.com/google/flax/workflows/Build/badge.svg?branch=main) [![coverage](https://badgen.net/codecov/c/github/google/flax)](https://codecov.io/github/google/flax)


[**Overview**](#overview)
| [**Quick install**](#quick-install)
| [**What does Flax look like?**](#what-does-flax-look-like)
| [**Documentation**](https://flax.readthedocs.io/)

This README is a very short intro. **To learn everything you need to know about Flax, refer to our [full documentation](https://flax.readthedocs.io/).**

Flax was originally started by engineers and researchers within the Brain Team in Google Research (in close collaboration with the JAX team), and is now developed jointly with the open source community.

Flax is being used by a growing
community of hundreds of folks in various Alphabet research departments
for their daily work, as well as a [growing community
of open source
projects](https://github.com/google/flax/network/dependents?dependent_type=REPOSITORY).

The Flax team's mission is to serve the growing JAX neural network
research ecosystem -- both within Alphabet and with the broader community,
and to explore the use-cases where JAX shines. We use GitHub for almost
all of our coordination and planning, as well as where we discuss
upcoming design changes. We welcome feedback on any of our discussion,
issue and pull request threads. We are in the process of moving some
remaining internal design docs and conversation threads to GitHub
discussions, issues and pull requests. We hope to increasingly engage
with the needs and clarifications of the broader ecosystem. Please let
us know how we can help!

Please report any feature requests,
issues, questions or concerns in our [discussion
forum](https://github.com/google/flax/discussions), or just let us
know what you're working on!

We expect to improve Flax, but we don't anticipate significant
breaking changes to the core API. We use [Changelog](https://github.com/google/flax/tree/main/CHANGELOG.md)
entries and deprecation warnings when possible.

In case you want to reach us directly, we're at flax-dev@google.com.

## Overview

Flax is a high-performance neural network library and ecosystem for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to a framework.

Flax is being developed in close collaboration with the JAX team and
comes with everything you need to start your research, including:

* **Neural network API** (`flax.linen`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Quick install

You will need Python 3.6 or later, and a working [JAX](https://github.com/google/jax/blob/main/README.md)
installation (with or without GPU support - refer to [the instructions](https://github.com/google/jax/blob/main/README.md)).
For a CPU-only version of JAX:

```
pip install --upgrade pip # To support manylinux2010 wheels.
pip install --upgrade jax jaxlib # CPU-only
```

Then, install Flax from PyPi:

```
pip install flax
```

To upgrade to the latest version of Flax, you can use:

```
pip install --upgrade git+https://github.com/google/flax.git
```

## What does Flax look like?

We provide three examples using the Flax API: a simple multi-layer perceptron, a CNN and an auto-encoder.

To learn more about the `Module` abstraction, check out our [docs](https://flax.readthedocs.io/), our [broad intro to the Module abstraction](https://github.com/google/flax/blob/main/docs/notebooks/linen_intro.ipynb). For additional concrete demonstrations of best practices, refer to our
[guides](https://flax.readthedocs.io/en/latest/guides/index.html) and
[developer notes](https://flax.readthedocs.io/en/latest/developer_notes/index.html).

```py
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
output = model.apply(variables, batch)
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

model = CNN()
batch = jnp.ones((32, 64, 64, 10))  # (N, H, W, C) format
variables = model.init(jax.random.PRNGKey(0), batch)
output = model.apply(variables, batch)
```

```py
class AutoEncoder(nn.Module):
  encoder_widths: Sequence[int]
  decoder_widths: Sequence[int]
  input_shape: Sequence[int]

  def setup(self):
    input_dim = np.prod(self.input_shape)
    self.encoder = MLP(self.encoder_widths)
    self.decoder = MLP(self.decoder_widths + (input_dim,))

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    assert x.shape[1:] == self.input_shape
    return self.encoder(jnp.reshape(x, (x.shape[0], -1)))

  def decode(self, z):
    z = self.decoder(z)
    x = nn.sigmoid(z)
    x = jnp.reshape(x, (x.shape[0],) + self.input_shape)
    return x

model = AutoEncoder(encoder_widths=[20, 10, 5],
                    decoder_widths=[5, 10, 20],
                    input_shape=(12,))
batch = jnp.ones((16, 12))
variables = model.init(jax.random.PRNGKey(0), batch)
encoded = model.apply(variables, batch, method=model.encode)
decoded = model.apply(variables, encoded, method=model.decode)
```

## 🤗 Hugging Face

In-detail examples to train and evaluate a variety of Flax models for
Natural Language Processing, Computer Vision, and Speech Recognition are
actively maintained in the [🤗 Transformers repository](https://github.com/huggingface/transformers/tree/master/examples/flax).

As of October 2021, the [19 most-used Transformer architectures](https://huggingface.co/transformers/#supported-frameworks) are supported in Flax
and over 5000 pretrained checkpoints in Flax have been uploaded to the [🤗 Hub](https://huggingface.co/models?library=jax&sort=downloads).

## Citing Flax

To cite this repository:

```
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.6.5},
  year = {2023},
}
```

In the above bibtex entry, names are in alphabetical order, the version number
is intended to be that from [flax/version.py](https://github.com/google/flax/blob/main/flax/version.py), and the year corresponds to the project's open-source release.

## Note

Flax is an open source project maintained by a dedicated team in Google Research, but is not an official Google product.

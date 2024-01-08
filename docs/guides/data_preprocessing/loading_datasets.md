---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Loading datasets

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/guides/data_preprocessing/loading_datasets.ipynb)

A neural net written in Jax+Flax expects its input data as `jax.numpy` array instances. Therefore, loading a dataset from any source is as simple as converting it to `jax.numpy` types and reshaping it to the appropriate dimensions for your network.

As an example, this guide demonstrates how to import [MNIST](http://yann.lecun.com/exdb/mnist/) using the APIs from Torchvision, Tensorflow, and Hugging Face. We'll load the whole dataset into memory. For datasets that don't fit into memory the process is analogous but should be done in a batchwise fashion.

The MNIST dataset consists of greyscale images of 28x28 pixels of handwritten digits, and has a designated 60k/10k train/test split. The task is to predict the correct class (digit 0, ..., 9) of each image.

Assuming a CNN-based classifier, the input data should have shape `(B, 28, 28, 1)`, where the trailing singleton dimension denotes the greyscale image channel.

The labels are simply the integer denoting the digit corresponding to the image. Labels should therefore have shape `(B,)`, to enable loss computation with [`optax.softmax_cross_entropy_with_integer_labels`](https://optax.readthedocs.io/en/latest/api.html#optax.softmax_cross_entropy_with_integer_labels).

```{code-cell} ipython3
:tags: [skip-execution]

import numpy as np
import jax.numpy as jnp
```

## Loading from `torchvision.datasets`

```{code-cell} ipython3
:tags: [skip-execution]

import torchvision
```

```{code-cell} ipython3
:tags: [skip-execution]

def get_dataset_torch():
    mnist = {
        'train': torchvision.datasets.MNIST('./data', train=True, download=True),
        'test': torchvision.datasets.MNIST('./data', train=False, download=True)
    }

    ds = {}

    for split in ['train', 'test']:
        ds[split] = {
            'image': mnist[split].data.numpy(),
            'label': mnist[split].targets.numpy()
        }

        # cast from np to jnp and rescale the pixel values from [0,255] to [0,1]
        ds[split]['image'] = jnp.float32(ds[split]['image']) / 255
        ds[split]['label'] = jnp.int16(ds[split]['label'])

        # torchvision returns shape (B, 28, 28).
        # hence, append the trailing channel dimension.
        ds[split]['image'] = jnp.expand_dims(ds[split]['image'], 3)

    return ds['train'], ds['test']
```

```{code-cell} ipython3
:outputId: be39b756-d13e-4380-b99e-a5cbf61458cc
:tags: [skip-execution]

train, test = get_dataset_torch()
print(train['image'].shape, train['image'].dtype)
print(train['label'].shape, train['label'].dtype)
print(test['image'].shape, test['image'].dtype)
print(test['label'].shape, test['label'].dtype)
```

## Loading from `tensorflow_datasets`

```{code-cell} ipython3
:tags: [skip-execution]

import tensorflow_datasets as tfds
```

```{code-cell} ipython3
:tags: [skip-execution]

def get_dataset_tf():
    mnist = tfds.builder('mnist')
    mnist.download_and_prepare()

    ds = {}

    for split in ['train', 'test']:
        ds[split] = tfds.as_numpy(mnist.as_dataset(split=split, batch_size=-1))

        # cast to jnp and rescale pixel values
        ds[split]['image'] = jnp.float32(ds[split]['image']) / 255
        ds[split]['label'] = jnp.int16(ds[split]['label'])

    return ds['train'], ds['test']
```

```{code-cell} ipython3
:outputId: 25d2c468-cbc8-4971-a738-1295ce8c6f16
:tags: [skip-execution]

train, test = get_dataset_tf()
print(train['image'].shape, train['image'].dtype)
print(train['label'].shape, train['label'].dtype)
print(test['image'].shape, test['image'].dtype)
print(test['label'].shape, test['label'].dtype)
```

## Loading from 🤗 Hugging Face `datasets`

```{code-cell} ipython3
:tags: [skip-execution]

#!pip install datasets # datasets isn't preinstalled on Colab; uncomment to install
from datasets import load_dataset
```

```{code-cell} ipython3
:tags: [skip-execution]

def get_dataset_hf():
    mnist = load_dataset("mnist")

    ds = {}

    for split in ['train', 'test']:
        ds[split] = {
            'image': np.array([np.array(im) for im in mnist[split]['image']]),
            'label': np.array(mnist[split]['label'])
        }

        # cast to jnp and rescale pixel values
        ds[split]['image'] = jnp.float32(ds[split]['image']) / 255
        ds[split]['label'] = jnp.int16(ds[split]['label'])

        # append trailing channel dimension
        ds[split]['image'] = jnp.expand_dims(ds[split]['image'], 3)

    return ds['train'], ds['test']
```

```{code-cell} ipython3
:outputId: b026b33f-3bdd-4d26-867c-49400fff1c96
:tags: [skip-execution]

train, test = get_dataset_hf()
print(train['image'].shape, train['image'].dtype)
print(train['label'].shape, train['label'].dtype)
print(test['image'].shape, test['image'].dtype)
print(test['label'].shape, test['label'].dtype)
```

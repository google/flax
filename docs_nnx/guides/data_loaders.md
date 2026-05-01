---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Introduction to Data Loaders with JAX


This tutorial explores different data loading strategies for using **JAX**. While JAX doesn't include a built-in data loader, it seamlessly integrates with popular data loading libraries, including:

- [**PyTorch DataLoader**](https://github.com/pytorch/data)
- [**TensorFlow Datasets (TFDS)**](https://github.com/tensorflow/datasets)
- [**Grain**](https://github.com/google/grain)
- [**Hugging Face**](https://huggingface.co/docs/datasets/en/use_with_jax#data-loading)

In this tutorial, you'll learn how to efficiently load data using these libraries for a simple image classification task based on the MNIST dataset.

You should be familiar with how to write a training loop from the [MNIST Example](https://flax.readthedocs.io/en/stable/mnist_tutorial.html). For this tutorial, we'll use a dummy training step that takes in 4-D image arrays and 1-D label vectors. Our goal in data loading will be to create these tensors, implementing the `get_batches` generator below.

```python outputId="c51838df-69ad-4d81-e577-5bbe95f8f6e7"
!pip install jaxtyping
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int, Array
from flax import nnx

batch_size = 32

def train(model: nnx.Module, images:  Float[Array, "batch channels height width"], labels: Int[Array, "batch"]):
  pass

def train_loop(model):
  for images, labels in get_batches(train_ds):
    train(model, images, labels)
```

## Loading Hugging Face Datasets
In the previous [MNIST Example](https://flax.readthedocs.io/en/stable/mnist_tutorial.html), we saw how to use Hugging Face's `datasets` library with jax. Specifically, we downloaded the 'mnist' dataset and used different subsets of the data from the 'train' and 'test' splits. The `splits` object we get from `load_dataset` is just a dict mapping subset names to `Dataset` objects. Each `Dataset` is cached to an [Arrow](https://arrow.apache.org/) file for fast, efficient loading.

```python outputId="5fcff9ba-9886-4563-9bde-6e608f3c7d21"
from datasets import load_dataset

splits = load_dataset('mnist')
train_ds = splits['train'].shuffle(seed=0)
test_ds = splits['test']
isinstance(splits, dict)
```

When you take slices of these `Dataset` objects, you get dictionaries mapping feature names to lists of observations.

```python outputId="19cbc230-1e95-4280-d31f-1e931f5df789"
jax.tree.map(get_list_type, train_ds[1:32], is_leaf=lambda x: type(x) is list)
```

To convert images to jax Arrays, we can use `jnp.array`. This will materialize the array on the default device (which will be a GPU if you have one available).

```python outputId="d51ad1b5-a760-41ba-f1ec-de7dc02bb2f5"
img_array = jnp.array(train_ds[1]['image'], dtype=jnp.float32)
img_array.shape, img_array.max()
```

We can see that these arrays don't yet have a channel dimension, and that the values are between 0 and 255. We need to add a channel dimension and rescale them before giving them to the training loop. This gives us the batch iterator we saw in the MNIST tutorial.

```python

def get_hf_batches(ds):
  """Yield batches of normalized (image, label) numpy arrays."""
  for i in range(0, len(ds), batch_size):
    batch = ds[i : i + batch_size]
    if len(batch['label']) < batch_size:  # drop incomplete final batch
      break
    images = jnp.stack([
      jnp.array(img, dtype=jnp.float32)[None] / 255.0
      for img in batch['image']
    ])
    yield [images, jnp.array(batch['label'])]
```

## Loading Data with PyTorch DataLoaders

If you're coming to Jax from PyTorch, you might want to use PyTorch's data utilities instead. The process is pretty similar! This time, the "image to normalized array" transformation is already written for is: it's called `ToTensor`.

```python
# !pip install torch torchvision
from torch.utils import data
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
```

```python
mnist_dataset = MNIST("data", download=True, transform=ToTensor())
```

Pytorch's dataset doesn't come pre-split into train and test sets, so we'll have to do the splitting ourselves.

```python
train_ds, test_ds = data.random_split(mnist_dataset, [0.8, 0.2])
```

To package each dataset into batches, we'll use a `DataLoader`. Setting `num_workers > 0` enables multi-process data loading, which can accelerate data loading for larger datasets or intensive preprocessing tasks. Experiment with different values to find the optimal setting for your hardware and workload.

Note: When setting `num_workers > 0`, you may see the following `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.` This warning can be safely ignored since data loaders do not use JAX within the forked processes.

```python
train_dataloader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
```

Iterating over a `DataLoader` yields batches of Pytorch tensors. We'll need to convert them to Jax arrays before passing them to the training step.

```python outputId="63f25f72-d104-46b2-da88-1afa1f405f3b"
jax.tree.map(lambda a: a.shape, next(iter(train_dataloader)))
```

```python
def get_pt_batches(ds):
  for image, label in train_dataloader:
    yield jnp.array(image, dype=jnp.float32), jnp.array(label)
```

## Loading Data with TensorFlow Datasets (TFDS)

This section demonstrates how to load the MNIST dataset using TFDS. Currently, while TFDS does not require TensorFlow to *load* datasets, it does require Tensorflow to *download* datasets. By default, TensorFlow will try to hog the GPU when it loads, preventing Jax from allocating arrays. To stop this, we have to explicitly tell TensorFlow to knock it off.

Once you've downloaded the datasets with an initial call to `tfds.data_source`, you no longer need TensorFlow. The exposed API looks almost identical to Hugging Face's. Specifically, TFDS gives us a dictionary mapping from split names to datasets.

```python outputId="087b9812-2d40-45b2-db14-b29059585a25"
import tensorflow_datasets as tfds
import tensorflow as tf
from itertools import batched

# Ensuring CPU-Only Execution, disable any GPU usage(if applicable) for TF
tf.config.set_visible_devices([], device_type='GPU')

splits = tfds.data_source('mnist')
splits
```

Indexing each split gives you a dictionary with separate keys for each feature (in this case, 'image' and 'label'). For now, we'll normalize and aggregate these into batches with pure python, but in the next section we'll see how the `grain` data loader can make this process faster.

```python
def get_tfds_batches():
  for batch in batched(splits['train'], batch_size):
    images = jnp.array([a['image'] for a in batch], dtype=jnp.float32) / 255
    labels = jnp.array([a['label'] for a in batch])
    yield images, labels
```

## Loading Data with Grain

In the Hugging Face and TFDS examples above, we've done our data processing (datatype conversion, batching and normalization) in pure Python. Due to the GIL, this means that these processing steps are always performed sequentially. The `grain` library allows you to do this loading and processing in parallel. You can use `grain` to accelerate Hugging Face datasets or TFDS, but it also works fine with raw ArrayRecord or Parquet files.

To start, we need to tell `grain` what order to iterate over the dataset using an `IndexSampler`.

```python
!pip install grain
import grain

sampler = grain.samplers.IndexSampler(
    num_records=len(splits['train']),
    num_epochs=2,
    shuffle=True,
    seed=0)
```

We describe our data transformations by subclassing the `grain.transforms.Map` class.

```python
class ScalePixelVals(grain.transforms.Map):
  def map(self, x: int) -> int:
    x['image'] = x['image'].astype(jnp.float32) / 255
    return x
```

Finally, we package the results together with a `grain.DataLoader`.

```python
data_loader = grain.DataLoader(
    data_source=splits['train'],
    operations=[
        ScalePixelVals(),
        grain.transforms.Batch(batch_size)],
    sampler=sampler,
    worker_count=0)
```

```python
def get_grain_batches():
  for elt in data_loader:
    yield elt['image'], elt['label']
```

## Summary

This notebook has introduced efficient strategies for data loading on a CPU with JAX, demonstrating how to integrate popular libraries like PyTorch DataLoader, TensorFlow Datasets, Grain, and Hugging Face Datasets. Each library offers distinct advantages, enabling you to streamline the data loading process for machine learning tasks. By understanding the strengths of these methods, you can select the approach that best suits your project's specific requirements.

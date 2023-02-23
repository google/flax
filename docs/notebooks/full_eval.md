---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "SMNC51ldX-Nq"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/full_eval.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/notebooks/full_eval.ipynb)

This notebook only contains executable code cells for the examples mentioned in
https://flax.readthedocs.io/en/latest/guides/full_eval.html

Please refer to above link for a an explanation of the problem and the proposed solutions.

+++ {"id": "Um6ZK_o1W-Vu"}

### setup

```{code-cell} ipython3
:id: 62DTHYCYHWp1
:outputId: b38d096f-58db-4d61-effa-eafa4c732826
:tags: [skip-execution]

!pip install -q chex einops
# tfds.split_for_jax_process() was added in 4.5.1
!pip install -q tensorflow_datasets -U
# flax.jax_utils.pad_shard_unpad() is only available at HEAD
!pip install -q git+https://github.com/google/flax
```

```{code-cell} ipython3
:id: NdzAaRwVExA9

import collections

import chex
import einops
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import flax.jax_utils
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

chex.set_n_cpu_devices(8)
```

```{code-cell} ipython3
:id: E30SS9gCIvrV

per_device_batch_size = 512
dataset_name = 'mnist'
```

```{code-cell} ipython3
:id: bZ-HWxKZHf6I
:outputId: 639262cb-b617-4561-c31f-60b33156a15f

class FakeModel(nn.Module):
  num_classes: int
  @nn.compact
  def __call__(self, x):
    return jax.nn.one_hot(jnp.zeros([len(x)], jnp.int32), self.num_classes)

model = FakeModel(num_classes=10)
variables = {}
inputs = jnp.zeros([2, 28, 28, 1])
model.apply(variables, inputs)
```

+++ {"id": "sx65cZiiW_cq"}

### The problem

```{code-cell} ipython3
:id: yfGNjMBFWEUk
:outputId: 09f0c28b-d28e-4a7a-8afe-8797da44ad6d

# last batch has different shape
collections.Counter(
    tuple(batch['image'].shape)
    for batch in tfds.load('mnist', split='test').batch(per_device_batch_size)
)
```

```{code-cell} ipython3
:id: eFPK-Oysl1YS
:outputId: 293bd0e4-011e-41b4-de48-a53b9cfd0958

# need to drop remainder when using multiple batch levels in a dataparallel
# setup
sum(
    np.prod(batch['label'].shape)
    for batch in tfds.load('mnist', split='test')
        .batch(per_device_batch_size, drop_remainder=True)
        .batch(jax.local_device_count())
)
```

```{code-cell} ipython3
:id: DlAJwgYDmoxe
:outputId: 8bb353f3-98db-4645-e627-3c3683e36ea9

# having different number of examples for different hosts will result in SPMD
# violation when all examples are to be processed
process_count = 6
[
    len(tfds.load(dataset_name, split=tfds.split_for_jax_process(
        'test', process_index=process_index, process_count=process_count)))
    for process_index in range(process_count)
]
```

```{code-cell} ipython3
:id: oUb7QrR2Iwk9
:outputId: 19234b4a-9f9c-47c4-cbcc-5f7fb2573746

# baseline: simple batching, keep reminder
# => leads to recompilation & only works on single device

@jax.jit
def get_preds(variables, inputs):
  print('retrigger compilation', inputs.shape)
  return model.apply(variables, inputs)

ds = tfds.load(dataset_name, split='test')
ds = ds.batch(per_device_batch_size, drop_remainder=False)

correct = total = 0
for batch in ds.as_numpy_iterator():
  preds = get_preds(variables, batch['image'])
  total += len(batch['label'])
  correct += (batch['label'] == preds.argmax(axis=1)).sum()

correc = correct.item()
correct, total, correct / total
```

```{code-cell} ipython3
:id: dlJuEBcLKY94
:outputId: e94cf79c-a033-4bc3-a086-75ecd8bd21f0

# when the remainder is dropped, we can use multiple devices and avoid
# recompilations
# => but results are incorrect

@jax.pmap
def get_preds(variables, inputs):
  print('retrigger compilation', inputs.shape)
  return model.apply(variables, inputs)

ds = tfds.load(dataset_name, split=tfds.split_for_jax_process('test'))
# This `drop_remainder=True` is required so we can do a second batch level.
ds = ds.batch(per_device_batch_size, drop_remainder=True)
# This `drop_remainder=True` is required so we can avoid a recompilation.
ds = ds.batch(jax.local_device_count(), drop_remainder=True)

correct = total = 0
for batch in ds.as_numpy_iterator():
  preds = get_preds(variables, batch['image'])
  total += len(batch['label'].flatten())
  correct += (batch['label'] == preds.argmax(axis=-1)).sum()

correc = correct.item()
correct, total, correct / total
```

+++ {"id": "vfu54P0pJwEH"}

### The solution: padding

+++ {"id": "LIkNUHsfXKCp"}

#### Manual implementation

```{code-cell} ipython3
:id: I1hg8paaasXj
:outputId: 2e6c611d-357e-4e51-99d8-47c24d785b11

# manually padding
# => precise & allows for data parallelism

@jax.pmap
def get_preds(variables, inputs):
  print('retrigger compilation', inputs.shape)
  return model.apply(variables, inputs)

ds = tfds.load(dataset_name, split=tfds.split_for_jax_process('test'))
per_host_batch_size = per_device_batch_size * jax.local_device_count()
ds = ds.batch(per_host_batch_size, drop_remainder=False)

shard = lambda x: einops.rearrange(
    x, '(d b) ... -> d b ...', d=jax.local_device_count())
unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

correct = total = 0
for batch in ds.as_numpy_iterator():
  images = batch['image']
  n = len(images)
  padding = np.zeros([per_host_batch_size - n, *images.shape[1:]], images.dtype)
  padded_images = np.concatenate([images, padding])
  preds = unshard(get_preds(variables, shard(padded_images)))[:n]
  total += n
  correct += (batch['label'] == preds.argmax(axis=-1)).sum()

correct = correct.item()
correct, total, correct / total
```

+++ {"id": "Wh6CymyjXQ-a"}

#### Using `pad_shard_unpad()`

```{code-cell} ipython3
:id: pQX__5DfEX9g
:outputId: 71017214-c4ce-4da0-8db5-9300dba79c3a

# same as before, but using @pad_shard_unshard decorator

# manually padding
# => precise & allows for data parallelism

@jax.pmap
def get_preds(variables, inputs):
  print('retrigger compilation', inputs.shape)
  return model.apply(variables, inputs)

ds = tfds.load(dataset_name, split=tfds.split_for_jax_process('test'))
per_host_batch_size = per_device_batch_size * jax.local_device_count()
ds = ds.batch(per_host_batch_size, drop_remainder=False)

correct = total = 0
for batch in ds.as_numpy_iterator():
  preds = flax.jax_utils.pad_shard_unpad(get_preds)(
      variables, batch['image'], min_device_batch=per_device_batch_size)
  total += len(batch['image'])
  correct += (batch['label'] == preds.argmax(axis=-1)).sum()

correct = correct.item()
correct, total, correct / total
```

#### Computing metrics in `eval_step`

```{code-cell} ipython3
# moving the metrics computation into `eval_step()` and using `static_return`

# this pattern is often used with more complicated `clu.metrics`

def eval_step(metrics, variables, batch):
  print('retrigger compilation', {k: v.shape for k, v in batch.items()})
  preds = model.apply(variables, batch['image'])
  correct = (batch['mask'] & (batch['label'] == preds.argmax(axis=-1))).sum()
  total = batch['mask'].sum()
  return dict(
      correct=metrics['correct'] + jax.lax.psum(correct, axis_name='batch'),
      total=metrics['total'] + jax.lax.psum(total, axis_name='batch'),
  )

eval_step = jax.pmap(eval_step, axis_name='batch')
eval_step = flax.jax_utils.pad_shard_unpad(
    eval_step, static_argnums=(0, 1), static_return=True)

ds = tfds.load(dataset_name, split=tfds.split_for_jax_process('test'))
per_host_batch_size = per_device_batch_size * jax.local_device_count()
ds = ds.batch(per_host_batch_size, drop_remainder=False)

metrics = flax.jax_utils.replicate(dict(
    correct=jnp.array(0, jnp.int32),
    total=jnp.array(0, jnp.int32),)
)
for batch in ds.as_numpy_iterator():
  batch['mask'] = np.ones_like(batch['label'])
  metrics = eval_step(
      metrics, variables, batch,
      min_device_batch=per_device_batch_size)

correct, total = metrics['correct'][0].item(), metrics['total'][0].item()
correct, total, correct / total
```

+++ {"id": "ptn8NQeAXbeL"}

#### Multi-host complications

```{code-cell} ipython3
:id: MjtmUUjWPV1X
:outputId: 70ee173a-dcdf-4136-a3e0-6685c09f8198

# infinite zero padding

def with_infinite_padding(dataset):
  """Adds "infinite padding" to the dataset."""
  filler_element = tf.nest.map_structure(
      lambda spec: tf.zeros(spec.shape, spec.dtype)[None], dataset.element_spec)
  filler_element['mask'] = [False]
  filler_dataset = tf.data.Dataset.from_tensor_slices(filler_element)
  dataset = dataset.map(
      lambda features: dict(mask=True, **features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.concatenate(filler_dataset.repeat(None))

@jax.pmap
def get_preds(variables, inputs):
  print('retrigger compilation', inputs.shape)
  return model.apply(variables, inputs)

count_p = jax.pmap(
    lambda mask: jax.lax.psum(mask.sum(), axis_name='batch'),
    axis_name='batch',
)
count_correct_p = jax.pmap(
    lambda labels, preds, mask:
        jax.lax.psum((mask & (labels == preds)).sum(), axis_name='batch'),
    axis_name='batch',
)

ds = tfds.load(dataset_name, split=tfds.split_for_jax_process('test'))
ds = with_infinite_padding(ds).batch(per_device_batch_size).batch(jax.local_device_count())

correct = total = 0
for batch in ds.as_numpy_iterator():
  n = count_p(batch['mask'])[0].item()  # adds sync barrier
  if not n: break

  preds = get_preds(variables, batch['image']).argmax(axis=-1)
  total += n
  correct += count_correct_p(batch['label'], preds, batch['mask'])[0]

correct = correct.item()
correct, total, correct / total
```

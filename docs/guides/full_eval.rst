.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/full_eval.ipynb

Processing the entire Dataset
=============================

For efficiency reasons, we form batches that contain multiple examples and
process them in parallel. Especially when evaluating a model, it is important
that we process all examples and **avoid losing the remainder** of examples that
does not form a complete batch at the end.


The problem
-----------

When evaluating on a single device, one can either drop the last incomplete
batch, or one can form a last batch with a shape different from the preceding
batches. Doing the latter has the disadvantage that this will trigger a
**recompilation** of the ``eval_step()`` because XLA is not shape polymorphic.

.. code-block:: python

  collections.Counter(
      tuple(batch['image'].shape)
      for batch in tfds.load('mnist', split='test').batch(per_device_batch_size)
  )
  # output:
  # Counter({(272, 28, 28, 1): 1, (512, 28, 28, 1): 19})

The problem is accentuated when using multiple devices for data parallelism.  If
the batch size is not **divisible by the number devices**, then that last step
must be executed on a single device (or a subset of devices). Usually one would
drop the last batch, but this will lead to incorrect results.


.. code-block:: python

  sum(
      np.prod(batch['label'].shape)
      for batch in tfds.load('mnist', split='test')
          .batch(per_device_batch_size, drop_remainder=True)
          .batch(jax.local_device_count())
  )
  # output:
  # 9728

Using multiple hosts further complicates the situation because JAX uses the SPMD
paradigm and every host must execute the same program. We would usually form
non-overlapping splits for different hosts with |tfds.split_for_jax_process()|_,
but this can lead to **different numbers for different hosts**, resulting in
different JAX programs when all examples are to be processed.

.. code-block:: python

  process_count = 6
  [
      len(tfds.load(dataset_name, split=tfds.split_for_jax_process(
          'test', process_index=process_index, process_count=process_count)))
      for process_index in range(process_count)
  ]
  # output:
  # [1667, 1667, 1667, 1667, 1666, 1666]



.. |tfds.split_for_jax_process()| replace:: ``tfds.split_for_jax_process()``
.. _tfds.split_for_jax_process(): https://www.tensorflow.org/datasets/api_docs/python/tfds/split_for_jax_process


The solution: padding
---------------------

Even though it's possible to solve this problem by cleverly adjusting the number
of batches executed by different devices on different hosts, such a solution
quickly becomes complicated and makes the main eval loop hard to read with a lot
of cumbersome logic.

The more straightforward solution to this problem is to use padding at the end
of the dataset to make sure that the last batch has the same size as the
preceding batches.


Manual implementation
~~~~~~~~~~~~~~~~~~~~~

The last batch is manually padded to contain the same number of examples as in
the preceding batches. The predictions for the padded examples are discarded
from the computation.

.. code-block:: python

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


Using ``pad_shard_unpad()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above pattern, namely the pad→shard→predict→unshard→unpad sequence, can be
extracted into a utility wrapper ``pad_shard_unpad()``, which greatly simplifies
above evaluation loop.

.. code-block:: python

  correct = total = 0
  for batch in ds.as_numpy_iterator():
    preds = flax.jax_utils.pad_shard_unpad(get_preds)(
        vs, batch['image'], min_device_batch=per_device_batch_size)
    total += len(batch['image'])
    correct += (batch['label'] == preds.argmax(axis=-1)).sum()


Computing metrics in ``eval_step()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of returning the predictions and computing the metrics in the main
evaluation loop, we would often want to make the metric computation part of the
evaluation step, especially when using libraries like |jax_metrics|_, or
|clu.metrics|_.

In that case we would want to pass the metrics as a ``static_argnums`` (i.e. do
not shard/pad it), and treat the return value as ``static_return`` too (i.e. no
un-sharding or un-padding):

.. code-block:: python

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

.. |jax_metrics| replace:: ``clu.metrics``
.. _jax_metrics: https://github.com/cgarciae/jax_metrics


.. |clu.metrics| replace:: ``clu.metrics``
.. _clu.metrics: https://github.com/google/CommonLoopUtils/blob/main/clu/metrics.py


Adding "infinite padding"
~~~~~~~~~~~~~~~~~~~~~~~~~

The above solution works in most cases, but it has some limitations:

1. In the rare case where even splitting of the dataset on multiple hosts leads
   to a different number of batches. Imagine having a dataset of ``n=4097``
   examples, and evaluating this on ``h=8``, each having ``d=8`` local devices,
   and forming on-device batch sizes of ``b=128``. With even dataset splitting,
   the first host would get ``4096/8+1==513`` examples, and all other hosts
   would get ``4096/8==512`` examples. Forming per-host batches of ``d*b==512``
   this would lead to two batches on the first host, and a single batch on all
   other hosts, violating SPMD principles and hanging the multi-host setup in
   the last ``psum()`` directive (which would only be executed by the first
   host, but not the others).

2. When dropping examples dynamically by using ``ds.filter()``.

In these more complicated cases we could add "infinite padding" to the dataset,
on each of the hosts independently, and continuing processing examples until
*all* hosts run out of unpadded examples.

.. code-block:: python

  correct = total = 0
  for batch in ds.as_numpy_iterator():
    n = count_p(batch['mask'])[0].item()  # adds sync barrier
    if not n: break

    preds = get_preds(vs, batch['image']).argmax(axis=-1)
    total += n
    correct += count_correct_p(batch['label'], preds, batch['mask'])[0]

As for the other examples in this HOWTO, the complete executable code can be
found in the Colab:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/full_eval.ipynb

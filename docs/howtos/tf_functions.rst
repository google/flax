Tensorflow Functions in Flax
==============================

Below is a list of Tensorflow functions and their Flax counterpart.

Please feel free to file a Pull Request adding more functions!


.. list-table:: TF vs. Flax functions
   :widths: 25 25
   :header-rows: 1

   * - TF function
     - Flax function
   * - `tf.identity`_
     - ``lambda x: x``
   * - `tf.name_scope`_
     - "Create submodule" (1)
   * - `tf.print`_
     - `jax.experimental.host_callback.id_print`_

Notes:

1. Putting anything inside a new Module would have that effect. You can also 
   always give submodules a name explicitly with the ``name=... kwarg`` to the 
   submodule constuctor.

.. _tf.identity: https://www.tensorflow.org/api_docs/python/tf/identity
.. _tf.name_scope: https://www.tensorflow.org/api_docs/python/tf/name_scope
.. _tf.print: https://www.tensorflow.org/api_docs/python/tf/print
.. _jax.experimental.host_callback.id_print: https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html#jax.experimental.host_callback.id_print
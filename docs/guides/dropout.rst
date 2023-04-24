Dropout
=======

This guide provides an overview of how to apply
`dropout <https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf>`__
using :class:`flax.linen.Dropout`. Dropout is a stochastic regularization technique that randomly masks out certain activations
of a network.

Throughout the guide, you will be able to compare code examples with and without
Flax ``Dropout``.

.. testsetup::

  import flax.linen as nn
  import jax.numpy as jnp
  import jax
  import optax

PRNGs in JAX
******************

Since dropout is a random operation, it requires a pseudorandom number generator
(PRNG). In JAX PRNG state is explicit and passed around to every
function that requires randomness. To avoid repeating the same random numbers
a different PRNG state must be passed each time, for this JAX provides the
``jax.random.split`` and ``jax.random.fold_in`` functions:

.. testcode::

  root_key = jax.random.PRNGKey(seed=0)
  main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3) #!

To learn more, please refer to
`Pseudorandom numbers in JAX tutorial <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__
and `🔪 JAX - The Sharp Bits 🔪 Randomness and PRNG keys <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng>`__.


.. **Note:** In Flax, you provide *PRNG streams* with *names*, so that you can use them later
.. in your :meth:`flax.linen.Module`. For example, you pass the stream ``'params'``
.. for initializing parameters, and ``'dropout'`` for applying
.. :meth:`flax.linen.Dropout`.

Defining a model with ``Dropout``
**********************************

``Dropout`` is a ``Module`` exhibits a different behavior between training and
inference, the mode in which its operating is explicitly determined by the
``deterministic`` argument.

A common pattern is to accept a ``train`` or ``training`` argument in the parent
Flax ``Module``, and use it to define ``Dropouts``'s ``deterministic`` argument.

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  class MyModel(nn.Module):
    num_neurons: int

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(self.num_neurons)(x)



      return x
  ---
  class MyModel(nn.Module):
    num_neurons: int

    @nn.compact
    def __call__(self, x, training: bool): #!
      x = nn.Dense(self.num_neurons)(x)
      # Set the dropout layer with a `rate` of 50%. #!
      # When the `deterministic` flag is `True`, dropout is turned off. #!
      x = nn.Dropout(rate=0.5, deterministic=not training)(x) #!
      return x

**Note**: In other machine learning frameworks, like PyTorch or TensorFlow (Keras), the difference in runtime behavior
is specified via a mutable state or top-level flags, for example
PyTorch's `Module.eval <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval>`__
or Keras' ``Model.__call__``
`training <https://www.tensorflow.org/api_docs/python/tf/keras/Model#call>`__ flag.

Initialize the model
********************

After defining the model it can be instantiated so that it can be initialized
by calling :meth:`flax.linen.Module.init`, which returns a `variable dictionary <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.core.variables>`__
containing the model's parameters.

When using ``Dropout``, the main difference is that the ``training`` argument must be
provided. During initialization its better to disable dropout by setting
``training=False`` so an additional PRNG key is not required:

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  my_model = MyModel(num_neurons=3)
  x = jnp.empty((3, 4, 4))

  variables = my_model.init(params_key, x)
  params = variables['params']
  ---
  my_model = MyModel(num_neurons=3)
  x = jnp.empty((3, 4, 4))

  variables = my_model.init(params_key, x, training=False) #!
  params = variables['params']

Performing the forward pass during training
****************************************

When using :meth:`flax.linen.apply()` to run your model during training, two
additional things must be done:

* The ``training=True`` argument has to be passed to the model.
* A PRNG key must be provided for the ``rngs`` dictionary under the ``'dropout'`` key.

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  y = my_model.apply({'params': params}, x)
  ---
  y = my_model.apply({'params': params}, x, training=True, rngs={'dropout': dropout_key}) #!

During evaluation, set ``training=False`` and don't pass the ``rngs`` argument.

``TrainState`` and the training step
************************************

This section explains how to amend your code inside the training step function if
you have dropout enabled.

A common pattern is Flax is to create a :class:`~flax.training.train_state.TrainState`
object that contains all the state needed for training, including parameters and the optimizer
state. To train with dropout, create a ``TrainState`` subclass that contains a
``key`` field, and pass the ``key`` value to the :meth:`~flax.training.train_state.TrainState.create`
constructor:

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  from flax.training import train_state




  state = train_state.TrainState.create(
    apply_fn=my_model.apply,
    params=params,

    tx=optax.adam(1e-3)
  )
  ---
  from flax.training import train_state

  class TrainState(train_state.TrainState): #!
    key: jax.random.KeyArray #!

  state = TrainState.create( #!
    apply_fn=my_model.apply,
    params=params,
    key=dropout_key, #!
    tx=optax.adam(1e-3)
  )

Now, inside the ``train_sttep`` function generate a new PRNG
key from ``state.key`` so it can be used by ``Dropout`` at each step. Instead of repeatedly applying
`jax.random.split() <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__
and updating the ``state.key`` field, it is recommended to use
`jax.random.fold_in() <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html>`__
to get a unique PRNG key based on ``state.step``.

This key can be passed to the ``rngs`` dictionary under the ``'dropout'`` key in
the model's ``apply`` method:

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  @jax.jit
  def train_step(state: TrainState, batch):

    def loss_fn(params):
      logits = state.apply_fn(
        {'params': params},
        x=batch['image'],


        )
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

  ---
  @jax.jit
  def train_step(state: TrainState, batch, dropout_key): #!
    step_key = jax.random.fold_in(key=state.key, data=state.step) #!
    def loss_fn(params):
      logits = state.apply_fn(
        {'params': params},
        x=batch['image'],
        training=True, #!
        rngs={'dropout': step_key} #!
        )
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

Evaluation
**********
The ``eval_step`` is much simpler. Because ``dropout`` should be deterministic in the majority
of cases, no PRNG keys are needed, just make sure to set ``training=False``:

.. codediff::
  :title_left: No BatchNorm
  :title_right: With BatchNorm
  :sync:

  @jax.jit
  def eval_step(state: TrainState, batch):
    logits = state.apply_fn(
      {'params': params},
      x=batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label'])
    metrics = {
      'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics
  ---
  @jax.jit
  def eval_step(state: TrainState, batch):
    logits = state.apply_fn(
      {'params': params},
      x=batch['image'], training=False) #!
    loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label'])
    metrics = {
      'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics

Flax examples with dropout
**************************

* A `Transformer-based model <https://github.com/google/flax/blob/main/examples/wmt/models.py>`__
  trained on the WMT Machine Translation dataset. This example uses dropout and attention dropout.

* Applying word dropout to a batch of input IDs in a
  `text classification <https://github.com/google/flax/blob/main/examples/sst2/models.py>`__
  context. This example uses a custom :meth:`flax.linen.Dropout` layer.

More Flax examples that use Module ``make_rng()``
*************************************************

* Defining a prediction token in a decoder of a
  `sequence-to-sequence model <https://github.com/google/flax/blob/main/examples/seq2seq/models.py>`__.
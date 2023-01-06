Dropout
=======

This guide provides an overview of how to apply
`dropout <https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf>`__
using :meth:`flax.linen.Dropout`.

Dropout is a stochastic regularization technique that randomly removes hidden
and visible units in a network.

Throughout the guide, you will be able to compare code examples with and without
Flax ``Dropout``.

.. testsetup::

  import flax.linen as nn
  import jax.numpy as jnp
  import jax
  import optax

Split the PRNG key
******************

Since dropout is a random operation, it requires a pseudorandom number generator
(PRNG) state. Flax uses JAX's (splittable) PRNG keys, which have a number of
desirable properties for neutral networks. To learn more, refer to the
`Pseudorandom numbers in JAX tutorial <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.

**Note:** Recall that JAX has an explicit way of giving you PRNG keys:
you can fork the main PRNG state (such as ``key = jax.random.PRNGKey(seed=0)``)
into multiple new PRNG keys with ``key, subkey = jax.random.split(key)``. You
can refresh your memory in
`🔪 JAX - The Sharp Bits 🔪 Randomness and PRNG keys <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng>`__.

Begin by splitting the PRNG key using
`jax.random.split() <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__
into three keys, including one for Flax Linen ``Dropout``.

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  root_key = jax.random.PRNGKey(seed=0)
  main_key, params_key = jax.random.split(key=root_key)
  ---
  root_key = jax.random.PRNGKey(seed=0)
  main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3) #!

**Note:** In Flax, you provide *PRNG streams* with *names*, so that you can use them later
in your :meth:`flax.linen.Module`. For example, you pass the stream ``'params'``
for initializing parameters, and ``'dropout'`` for applying
:meth:`flax.linen.Dropout`.

Define your model with ``Dropout``
**********************************

To create a model with dropout:

* Subclass :meth:`flax.linen.Module`, and then use
  :meth:`flax.linen.Dropout` to add a dropout layer. Recall that
  :meth:`flax.linen.Module` is the
  `base class for all neural network Modules <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module>`__,
  and all layers and models are subclassed from it.

* In :meth:`flax.linen.Dropout`, the ``deterministic`` argument is required to
  be passed as a keyword argument, either:

  * When constructing the :meth:`flax.linen.Module`; or
  * When calling :meth:`flax.linen.init()` or :meth:`flax.linen.apply()` on a constructed ``Module``. (Refer to :meth:`flax.linen.module.merge_param` for more details.)

* Because ``deterministic`` is a boolean:

  * If it's set to ``False``, the inputs are masked (that is, set to zero) with
    a probability set by ``rate``. And the remaining inputs are scaled by
    ``1 / (1 - rate)``, which ensures that the means of the inputs are
    preserved.
  * If it's set to ``True``, no mask is applied (the dropout is turned off),
    and the inputs are returned as-is.

A common pattern is to accept a ``training`` (or ``train``) argument (a boolean)
in the parent Flax ``Module``, and use it to enable or disable dropout (as
demonstrated in later sections of this guide). In other machine learning
frameworks, like PyTorch or TensorFlow (Keras), this is specified via a
mutable state or a call flag (for example, in
`torch.nn.Module.eval <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval>`__
or ``tf.keras.Model`` by setting the
`training <https://www.tensorflow.org/api_docs/python/tf/keras/Model#call>`__ flag).

**Note:** Flax provides an implicit way of handling PRNG key streams via Flax
:meth:`flax.linen.Module`'s :meth:`flax.linen.Module.make_rng` method.
This allows you to split off a fresh PRNG key inside Flax Modules (or their
sub-Modules) from the PRNG stream. The ``make_rng`` method guarantees to provide a
unique key each time you call it. Internally, :meth:`flax.linen.Dropout` makes
use of :meth:`flax.linen.Module.make_rng` to create a key for dropout. You can
check out the
`source code <https://github.com/google/flax/blob/5714e57a0dc8146eb58a7a06ed768ed3a17672f9/flax/linen/stochastic.py#L72>`__.
In short, :meth:`flax.linen.Module.make_rng` *guarantees full reproducibility*.

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

Initialize the model
********************

After creating your model:

* Instantiate the model.
* Then, in the :meth:`flax.linen.init()` call, set ``training=False``.
* Finally, extract the ``params`` from the
  `variable dictionary <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.core.variables>`__.

Here, the main difference between the code without Flax ``Dropout``
and with ``Dropout`` is that the ``training`` (or ``train``) argument must be
provided if you need dropout enabled.

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
  # Dropout is disabled with `training=False` (that is, `deterministic=True`). #!
  variables = my_model.init(params_key, x, training=False) #!
  params = variables['params']

Perform the forward pass during training
****************************************

When using :meth:`flax.linen.apply()` to run your model:

* Pass ``training=True`` to :meth:`flax.linen.apply()`.
* Then, to draw PRNG keys during the forward pass (with dropout), provide a PRNG key
  to seed the ``'dropout'`` stream when you call :meth:`flax.linen.apply()`.

.. codediff::
  :title_left: No Dropout
  :title_right: With Dropout
  :sync:

  # No need to pass the `training` and `rngs` flags.
  y = my_model.apply({'params': params}, x)
  ---
  # Dropout is enabled with `training=True` (that is, `deterministic=False`). #!
  y = my_model.apply({'params': params}, x, training=True, rngs={'dropout': dropout_key}) #!

Here, the main difference between the code without Flax ``Dropout``
and with ``Dropout`` is that the ``training`` (or ``train``) and ``rngs``
arguments must be provided if you need dropout enabled.

During evaluation, use the above code with no dropout enabled (this means you do
not have to pass a RNG either).

``TrainState`` and the training step
************************************

This section explains how to amend your code inside the training step function if
you have dropout enabled.

**Note:** Recall that Flax has a common pattern where you create a dataclass
that represents the whole training state, including parameters and the optimizer
state. Then, you can pass a single parameter, ``state: TrainState``, to
the training step function. Refer to the
:meth:`flax.training.train_state.TrainState` API docs to learn more.

* First, add a ``key`` field to a custom :meth:`flax.training.train_state.TrainState` class.
* Then, pass the ``key`` value—in this case, the ``dropout_key``—to the :meth:`train_state.TrainState.create` method.

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

* Next, in the Flax training step function, ``train_step``, generate a new PRNG
  key from the ``dropout_key`` to apply dropout at each step. This can be done with one of the following:

  * `jax.random.split() <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__; or
  * `jax.random.fold_in() <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html>`__

  Using ``jax.random.fold_in()`` is generally faster. When you use
  ``jax.random.split()`` you split off a PRNG key that can be reused
  afterwards. However, using ``jax.random.fold_in()`` makes sure to 1) fold in
  unique data; and 2) can result in longer sequences of PRNG streams.

* Finally, when performing the forward pass, pass the new PRNG key to ``state.apply_fn()``
  as an extra parameter.

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
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step) #!
    def loss_fn(params):
      logits = state.apply_fn(
        {'params': params},
        x=batch['image'],
        training=True, #!
        rngs={'dropout': dropout_train_key} #!
        )
      loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

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
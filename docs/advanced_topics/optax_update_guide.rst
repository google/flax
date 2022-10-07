.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/optax_update_guide.ipynb

Upgrading my Codebase to Optax
==============================

We have proposed to replace :py:mod:`flax.optim` with `Optax
<https://optax.readthedocs.io>`_ in 2021 with `FLIP #1009
<https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md>`_ and
the Flax optimizers have been removed in v0.6.0 - this guide is targeted
towards :py:mod:`flax.optim` users to help them update their code to Optax.

See also Optax's quick start documentation:
https://optax.readthedocs.io/en/latest/optax-101.html

.. testsetup::

  import flax
  import jax
  import jax.numpy as jnp
  import flax.linen as nn
  import optax

  # Note: this is the minimal code required to make below code run. See in the
  # Colab linked above for a more meaningful definition of datasets etc.
  batch = {'image': jnp.ones([1, 28, 28, 1]), 'label': jnp.array([0])}
  ds_train = [batch]
  get_ds_train = lambda: [batch]
  model = nn.Dense(1)
  variables = model.init(jax.random.PRNGKey(0), batch['image'])
  learning_rate, momentum, weight_decay, grad_clip_norm = .1, .9, 1e-3, 1.
  loss = lambda params, batch: jnp.array(0.)

Replacing ``flax.optim`` with ``optax``
---------------------------------------

Optax has drop-in replacements for all of Flax's optimizers. Refer to Optax's
documentation `Common Optimizers <https://optax.readthedocs.io/en/latest/api.html>`_
for API details.

The usage is very similar, with the difference that ``optax`` does not keep a
copy of the ``params``, so they need to be passed around separately. Flax
provides the utility :py:class:`~flax.training.train_state.TrainState` to store
optimizer state, parameters, and other associated data in a single dataclass
(not used in code below).

.. codediff::
  :title_left: flax.optim
  :title_right: optax
  :sync:

  @jax.jit
  def train_step(optimizer, batch):
    grads = jax.grad(loss)(optimizer.target, batch)


    return optimizer.apply_gradient(grads)

  optimizer_def = flax.optim.Momentum(
      learning_rate, momentum)
  optimizer = optimizer_def.create(variables['params'])

  for batch in get_ds_train():
    optimizer = train_step(optimizer, batch)

  ---

  @jax.jit
  def train_step(params, opt_state, batch):
    grads = jax.grad(loss)(params, batch)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

  tx = optax.sgd(learning_rate, momentum)
  params = variables['params']
  opt_state = tx.init(params)

  for batch in ds_train:
    params, opt_state = train_step(params, opt_state, batch)


Composable Gradient Transformations
-----------------------------------

The function |optax.sgd()|_ used in the code snippet above is simply a wrapper
for the sequential application of two gradient transformations. Instead of using
this alias, it is common to use |optax.chain()|_ to combine multiple of these
generic building blocks.

.. |optax.sgd()| replace:: ``optax.sgd()``
.. _optax.sgd(): https://optax.readthedocs.io/en/latest/api.html#optax.sgd
.. |optax.chain()| replace:: ``optax.chain()``
.. _optax.chain(): https://optax.readthedocs.io/en/latest/api.html#chain

.. codediff::
  :title_left: Pre-defined alias
  :title_right: Combining transformations

  # Note that the aliases follow the convention to use positive
  # values for the learning rate by default.
  tx = optax.sgd(learning_rate, momentum)

  ---

  #

  tx = optax.chain(
      # 1. Step: keep a trace of past updates and add to gradients.
      optax.trace(decay=momentum),
      # 2. Step: multiply result from step 1 with negative learning rate.
      # Note that `optax.apply_updates()` simply adds the final updates to the
      # parameters, so we must make sure to flip the sign here for gradient
      # descent.
      optax.scale(-learning_rate),
  )

Weight Decay
------------

Some of Flax's optimizers also include a weight decay. In Optax, some optimizers
also have a weight decay parameter (such as |optax.adamw()|_), and to others the
weight decay can be added as another "gradient transformation"
|optax.add_decayed_weights()|_ that adds an update derived from the parameters.

.. |optax.adamw()| replace:: ``optax.adamw()``
.. _optax.adamw(): https://optax.readthedocs.io/en/latest/api.html#optax.adamw
.. |optax.add_decayed_weights()| replace:: ``optax.add_decayed_weights()``
.. _optax.add_decayed_weights(): https://optax.readthedocs.io/en/latest/api.html#optax.add_decayed_weights

.. codediff::
  :title_left: flax.optim
  :title_right: optax
  :sync:

  optimizer_def = flax.optim.Adam(
      learning_rate, weight_decay=weight_decay)
  optimizer = optimizer_def.create(variables['params'])

  ---

  # (Note that you could also use `optax.adamw()` in this case)
  tx = optax.chain(
      optax.scale_by_adam(),
      optax.add_decayed_weights(weight_decay),
      # params -= learning_rate * (adam(grads) + params * weight_decay)
      optax.scale(-learning_rate),
  )
  # Note that you'll need to specify `params` when computing the udpates:
  # tx.update(grads, opt_state, params)

Gradient Clipping
-----------------

Training can be stabilized by clipping gradients to a global norm (`Pascanu et
al, 2012 <https://arxiv.org/abs/1211.5063>`_). In Flax this is often done by
processing the gradients before passing them to the optimizer. With Optax this
becomes just another gradient transformation |optax.clip_by_global_norm()|_.

.. |optax.clip_by_global_norm()| replace:: ``optax.clip_by_global_norm()``
.. _optax.clip_by_global_norm(): https://optax.readthedocs.io/en/latest/api.html#optax.clip_by_global_norm

.. codediff::
  :title_left: flax.optim
  :title_right: optax
  :sync:

  def train_step(optimizer, batch):
    grads = jax.grad(loss)(optimizer.target, batch)
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
    g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
    grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
    return optimizer.apply_gradient(grads)

  ---

  tx = optax.chain(
      optax.clip_by_global_norm(grad_clip_norm),
      optax.trace(decay=momentum),
      optax.scale(-learning_rate),
  )

Learning Rate Schedules
-----------------------

For learning rate schedules, Flax allows overwriting hyper parameters when
applying the gradients. Optax maintains a step counter and provides this as an
argument to a function for scaling the updates added with
|optax.scale_by_schedule()|_. Optax also allows specifying a functions to
inject arbitrary scalar values for other gradient updates via
|optax.inject_hyperparams()|_.

Read more about learning rate schedules in the :doc:`lr_schedule` guide.

Read more about schedules defined in Optax under `Optimizer Schedules
<https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules>`_. the
standard optimizers (like ``optax.adam()``, ``optax.sgd()`` etc.) also accept a
learning rate schedule as a parameter for ``learning_rate``.


.. |optax.scale_by_schedule()| replace:: ``optax.scale_by_schedule()``
.. _optax.scale_by_schedule(): https://optax.readthedocs.io/en/latest/api.html#optax.scale_by_schedule
.. |optax.inject_hyperparams()| replace:: ``optax.inject_hyperparams()``
.. _optax.inject_hyperparams(): https://optax.readthedocs.io/en/latest/api.html#optax.inject_hyperparams

.. codediff::
  :title_left: flax.optim
  :title_right: optax
  :sync:

  def train_step(step, optimizer, batch):
    grads = jax.grad(loss)(optimizer.target, batch)
    return step + 1, optimizer.apply_gradient(grads, learning_rate=schedule(step))

  ---

  tx = optax.chain(
      optax.trace(decay=momentum),
      # Note that we still want a negative value for scaling the updates!
      optax.scale_by_schedule(lambda step: -schedule(step)),
  )

Multiple Optimizers / Updating a Subset of Parameters
-----------------------------------------------------

In Flax, traversals are used to specify which parameters should be updated by an
optimizer. And you can combine traversals using
:py:class:`flax.optim.MultiOptimizer` to apply different optimizers on different
parameters. The equivalent in Optax is |optax.masked()|_ and |optax.chain()|_.

Note that the example below is using :py:mod:`flax.traverse_util` to create the
boolean masks required by |optax.masked()|_ - alternatively you could also
create them manually, or use |optax.multi_transform()|_ that takes a
multivalent pytree to specify gradient transformations.

Beware that |optax.masked()|_ flattens the pytree internally and the inner
gradient transformations will only be called with that partial flattened view of
the params/gradients. This is not a problem usually, but it makes it hard to
nest multiple levels of masked gradient transformations (because the inner
masks will expect the mask to be defined in terms of the partial flattened view
that is not readily available outside the outer mask).

.. |optax.masked()| replace:: ``optax.masked()``
.. _optax.masked(): https://optax.readthedocs.io/en/latest/api.html#optax.masked
.. |optax.multi_transform()| replace:: ``optax.multi_transform()``
.. _optax.multi_transform(): https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform

.. codediff::
  :title_left: flax.optim
  :title_right: optax
  :sync:

  kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
  biases = flax.traverse_util.ModelParamTraversal(lambda p, _: 'bias' in p)

  kernel_opt = flax.optim.Momentum(learning_rate, momentum)
  bias_opt = flax.optim.Momentum(learning_rate * 0.1, momentum)


  optimizer = flax.optim.MultiOptimizer(
      (kernels, kernel_opt),
      (biases, bias_opt)
  ).create(variables['params'])

  ---

  kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
  biases = flax.traverse_util.ModelParamTraversal(lambda p, _: 'bias' in p)

  all_false = jax.tree_util.tree_map(lambda _: False, params)
  kernels_mask = kernels.update(lambda _: True, all_false)
  biases_mask = biases.update(lambda _: True, all_false)

  tx = optax.chain(
      optax.trace(decay=momentum),
      optax.masked(optax.scale(-learning_rate), kernels_mask),
      optax.masked(optax.scale(-learning_rate * 0.1), biases_mask),
  )

Final Words
-----------

All above patterns can of course also be mixed and Optax makes it possible to
encapsulate all these transformations into a single place outside the main
training loop, which makes testing much easier.

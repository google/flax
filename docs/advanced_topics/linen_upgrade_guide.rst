Upgrading my Codebase to Linen
==============================

As of Flax v0.4.0, ``flax.nn`` no longer exists, and is replaced with the new
Linen API at ``flax.linen``. If your codebase is still using the old API, you
can use this upgrade guide to upgrade it to Linen.

.. testsetup::

  from flax.training import train_state
  from jax import random
  import optax
  import jax
  from flax.linen import initializers

  from jax import lax
  import jax.numpy as jnp
  import numpy as np
  from typing import Any, Callable, Sequence, Tuple

  PRNGKey = Any
  Shape = Tuple[int, ...]
  Dtype = Any
  Array = Any

  default_kernel_init = initializers.lecun_normal()

Defining Simple Modules
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  from flax import nn

  class Dense(base.Module):
    def apply(self,
              inputs,
              features,
              use_bias=True,
              kernel_init=default_kernel_init,
              bias_init=initializers.zeros_init()):

      kernel = self.param('kernel',
        (inputs.shape[-1], features), kernel_init)
      y = jnp.dot(inputs, kernel)
      if use_bias:
        bias = self.param(
          'bias', (features,), bias_init)
        y = y + bias
      return y

    return new_state, metrics
  ---
  from flax import linen as nn  # [1] #!

  class Dense(nn.Module):
    features: int  # [2] #!
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):  # [3] #!
      kernel = self.param('kernel',
        self.kernel_init, (inputs.shape[-1], self.features))  # [4] #!
      y = jnp.dot(inputs, kernel)
      if self.use_bias:
        bias = self.param(
          'bias', self.bias_init, (self.features,))  # [5] #!
        y = y + bias
      return y

1. Replace from ``flax import nn`` with from ``flax import linen as nn``.

2. Move arguments to ``apply`` into dataclass attributes. Add type annotations
   (or use type ``Any`` to bypass).

3. Rename method ``apply`` to ``__call__`` and (optionally) wrap with
   |@compact|_. Methods wrapped in |@compact|_ can define submodules directly
   within the method (like in old Flax). You can only wrap a single method with
   |@compact|_. Alternatively, you can define a ``setup`` method. For more
   details, please see our other HOWTO `Should I use setup or nn.compact?`_.

4. Access dataclass attributes values by ``self.<attr>`` inside methods, e.g.
   ``self.features``.

5. Move shape to the end of the arguments to |self.param|_ (initializer functions
   can take arbitrary argument lists).


Using Modules inside other Modules
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  class Encoder(nn.Module):

    def apply(self, x):
      x = nn.Dense(x, 500)
      x = nn.relu(x)
      z = nn.Dense(x, 500, name="latents")
      return z
  ---
  class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(500)(x)  # [1] #!
      x = nn.relu(x)
      z = nn.Dense(500, name='latents')(x)  # [2] #!
      return z

1. Module constructors no longer return the outputs. Instead, they work like
   normal constructors and return module instances. These instances can be
   shared like in normal Python (instead of using ``.shared()`` in old Flax).
   Since most modules implement ``__call__``, you can retain the conciseness of
   old Flax.

2. Names can be optionally passed to all module constructors.

Sharing submodules and defining multiple methods
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  class AutoEncoder(nn.Module):
    def _create_submodules(self):
      return Decoder.shared(name="encoder")

    def apply(self, x, z_rng, latents=20):
      decoder = self._create_decoder()
      z = Encoder(x, latents, name="encoder")
      return decoder(z)

    @nn.module_method
    def generate(self, z, **unused_kwargs):
      decoder = self._create_decoder()
      return nn.sigmoid(decoder(z))
  ---
  class AutoEncoder(nn.Module):
    latents: int = 20

    def setup(self):  # [1] #!
      self.encoder = Encoder(self.latents)  # [2] #!
      self.decoder = Decoder()

    def __call__(self, x):  # [3] #!
      z = self.encoder(x)
      return self.decoder(z)

    def generate(self, z):  # [4] #!
      return nn.sigmoid(self.decoder(z))


1. Use |setup|_ instead of ``__init__``, which is already defined in
   the dataclasses library. Flax calls setup right after modules are ready to be
   used. (You can do this for all modules if you like instead of using
   |@compact|, but we like how |@compact| co-locates where modules are defined
   and used, especially if you have loops or conditionals).

2. Like regular Python, share submodules by assigning to self during
   initialization. Similar to PyTorch, ``self.encoder`` automatically has the
   name ``"encoder"``.

3. We don't use |@compact|_ here because we're not defining any inline
   submodules (all submodules are defined in setup).

4. Define additional methods just like in regular Python.

``Module.partial`` inside other modules
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  # no import #!

  class ResNet(nn.Module):
    """ResNetV1."""


    def apply(self, x,
              stage_sizes,
              num_filters=64,
              train=True):
      conv = nn.Conv.partial(bias=False)
      norm = nn.BatchNorm.partial(
          use_running_average=not train,
          momentum=0.9, epsilon=1e-5)

      x = conv(x, num_filters, (7, 7), (2, 2),
              padding=[(3, 3), (3, 3)],
              name='conv_init')
      x = norm(x, name='bn_init')

      # [...]
      return x
  ---
  from functools import partial  #!

  class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    num_filters: int = 64
    train: bool = True

    @nn.compact
    def __call__(self, x):
      conv = partial(nn.Conv, use_bias=False) #!
      norm = partial(nn.BatchNorm,  #!
                    use_running_average=not self.train, #!
                    momentum=0.9, epsilon=1e-5) #!

      x = conv(self.num_filters, (7, 7), (2, 2),
              padding=[(3, 3), (3, 3)],
              name='conv_init')(x)
      x = norm(name='bn_init')(x)

      # [...]
      return x

Use normal ``functools.partial`` instead of ``Module.partial``. The rest stays
the same.

Top-level training code patterns
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  def create_model(key):
    _, initial_params = CNN.init_by_shape(
      key, [((1, 28, 28, 1), jnp.float32)])
    model = nn.Model(CNN, initial_params)
    return model

  def create_optimizer(model, learning_rate):
    optimizer_def = optim.Momentum(learning_rate=learning_rate)
    optimizer = optimizer_def.create(model)
    return optimizer

  def cross_entropy_loss(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))

  def loss_fn(model):
    logits = model(batch['image'])
    one_hot = jax.nn.one_hot(batch['label'], num_classes=10)
    loss = -jnp.mean(jnp.sum(one_hot_labels * batch['label'],
                             axis=-1))
    return loss, logits
  ---
  def create_train_state(rng, config):  # [1] #!
    variables = CNN().init(rng, jnp.ones([1, 28, 28, 1]))  # [2] #!
    params = variables['params']  # [3] #!
    tx = optax.sgd(config.learning_rate, config.momentum)  # [4] #!
    return train_state.TrainState.create(
        apply_fn=CNN.apply, params=params, tx=tx)










  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])  # [5] #!
    one_hot = jax.nn.one_hot(batch['label'], 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits,
                                                labels=one_hot))
    return loss, logits


1. We no longer use the ``Model`` abstraction -- instead we pass parameters
   around directly, usually encapsulated in a `Train State`_ object, which can
   directly be passed to JAX transformations.

2. To compute initial parameters, construct a module instance and call |init|_
   or |init_with_output|_. We haven't ported over ``init_by_shape`` because this
   function did some magic we did not like (it evaluated the function by shape.
   but returned real values anyway). Therefore, you should now pass concrete
   values to the initializer functions, and you can optimize the initialization
   by wrapping it with |jax.jit|_, which is highly recommended to avoid running
   a full forward pass.

3. Linen generalizes parameters into variables. Parameters are one
   "collection" of variables. Variables are nested dicts, where the top-level
   keys reflect the different variable collections, of which "param" is one of.
   See the `Variables documentation`_ for more details.

4. We recommend using Optax optimizers. See our separate HOWTO called
   `Upgrading my Codebase to Optax`_ for more details.

5. To make predictions with your model, make an instance at the top level (this
   is free -- just a wrapper around constructor attributes) and call the
   ``apply`` method (which will call ``__call__`` internally).

Non-trainable variables ("state"): Use within Modules
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  class BatchNorm(nn.Module):
    def apply(self, x, ...):
      # [...]
      ra_mean = self.state(
        'mean', (x.shape[-1], ), initializers.zeros_init())
      ra_var = self.state(
        'var', (x.shape[-1], ), initializers.ones_init())
      # [...]
  ---
  class BatchNorm(nn.Module):
    def __call__(self, x):
      # [...]
      ra_mean = self.variable(  #!
        'batch_stats', 'mean', initializers.zeros_init(), (x.shape[-1], ))
      ra_var = self.variable(
        'batch_stats', 'var', initializers.ones_init(), (x.shape[-1], ))
      # [...]

The first argument is the name of the variable collection ("param" is the only
variable collection that's always available). Some colllections may be treated
as mutable, and others as immutable at top-level training code (see next section
for details). Flax also lets you treat each variable collection differently when
using JAX transformations inside modules.

Non-trainable variables ("state"): Top-level training code patterns
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  # initial params and state
  def initial_model(key, init_batch):
    with nn.stateful() as initial_state:
      _, initial_params = ResNet.init(key, init_batch)
    model = nn.Model(ResNet, initial_params)
    return model, init_state


  # updates batch statistics during training
  def loss_fn(model, model_state):
    with nn.stateful(model_state) as new_model_state:
      logits = model(batch['image'])
    # [...]




  # reads immutable batch statistics during evaluation
  def eval_step(model, model_state, batch):
  with nn.stateful(model_state, mutable=False):
      logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])
  ---
  # initial variables ({"param": ..., "batch_stats": ...})
  def initial_variables(key, init_batch):
    return ResNet().init(key, init_batch)  # [1] #!





  # updates batch statistics during training
  def loss_fn(params, batch_stats):
    variables = {'params': params, 'batch_stats': batch_stats}  # [2] #!
    logits, new_variables = ResNet(train=true).apply(
      variables, batch['image'], mutable=['batch_stats'])  # [3] #!
    new_batch_stats = new_variables['batch_stats']
    # [...]


  # reads immutable batch statistics during evaluation
  def eval_step(params, batch_stats, batch):
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = ResNet(train=False).apply(
      variables, batch['image'], mutable=False)  # [4] #!
    return compute_metrics(logits, batch['label'])

1. |init|_ returns a variable dict, e.g. ``{"param": ..., "batch_stats": ...}``
   (see `Variable documentation`_).

2. Combine the different variable collections into a variable dict.

3. During training, the ``batch_stats`` variable collection changes. Since we
   specify that in the mutable argument, the return value from ``module.apply``
   becomes an ordered pair of ``output, new_variables``.

4. During evaluation, we want to raise an error if we're accidentally applying
   Batch Norm in training mode. By passing ``mutable=False`` into
   ``module.apply`` we enforce that. Since no variables are mutated, the return
   value is once again just the output.

Loading pre-Linen checkpoints
--------------------------------

While most Linen modules should be able to use pre-Linen weights without any
modification, there is one catch: In pre-Linen API submodules were numbered
incrementally, independent of the submodule class. With Linen this behavior has
changed to keep separate submodule counts per module class.

In pre-Linen, params have the following structure:

``{'Conv_0': { ... }, 'Dense_1': { ... } }``

In Linen this is instead:

``{'Conv_0': { ... }, 'Dense_0': { ... } }``

TODO: Add an example here how to load a new ``TrainState`` object.

Randomness
--------------------------------

.. codediff::
  :title_left: Old Flax
  :title_right: Linen
  :sync:

  def dropout(inputs, rate, deterministic=False):
    keep_prob = 1. - rate
    if deterministic:
      return inputs
    else:
      mask = random.bernoulli(
      make_rng(), p=keep_prob, shape=inputs.shape)
      return lax.select(
        mask, inputs / keep_prob, jnp.zeros_like(inputs))


  def loss_fn(model, dropout_rng):
    with nn.stochastic(dropout_rng):
      logits = model(inputs)
  ---
  class Dropout(nn.Module):
    rate: float

    @nn.compact
    def __call__(self, inputs, deterministic=False):
      keep_prob = 1. - self.rate
      if deterministic:
        return inputs
      else:
        mask = random.bernoulli(
          self.make_rng('dropout'), p=keep_prob, shape=inputs.shape)  # [1] #!
        return lax.select(
          mask, inputs / keep_prob, jnp.zeros_like(inputs))


  def loss_fn(params, dropout_rng):
    logits = Transformer().apply(
      {'params': params}, inputs, rngs={'dropout': dropout_rng})  # [2] #!

1. RNGs in Linen have "kinds" -- in this case "dropout". Different kinds can be
   treated different in JAX transformations (for example -- do you want the same
   dropout mask for each timestep in a sequence model or a different one?)

2. Instead of using the ``nn.stochastic`` context manager, you pass in RNGs
   explicitly to ``module.apply``. During evaluation you wouldn't pass any RNGs
   -- then if you accidentally use dropout in non-deterministic mode,
   ``self.make_rng('dropout')`` would raise an error.


Lifted Transforms
--------------------------------

In Linen, rather than using JAX transformation directly, we are using
"lifted transforms", which are JAX transformations applied to Flax Modules.

For more information, please see the design note on `Lifted Transformations`_.

TODO: Given an example of ``jax.scan_in_dim`` (pre-Linen) vs. ``nn.scan``
(Linen).

.. _`Should I use setup or nn.compact?`: https://flax.readthedocs.io/en/latest/design_notes/setup_or_nncompact.html
.. _`Variables documentation`: https://flax.readthedocs.io/en/latest/flax.linen.html#module-flax.core.variables
.. _`TrainState`: https://flax.readthedocs.io/en/latest/flax.training.html#train-state
.. _`Upgrading my Codebase to Optax`: https://flax.readthedocs.io/en/latest/howtos/optax_update_guide.html
.. _`Lifted Transformations`: https://flax.readthedocs.io/en/latest/design_notes/lift.html


.. |@compact| replace:: ``@compact``
.. _@compact: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.compact

.. |init| replace:: ``init``
.. _init: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init

.. |init_with_output| replace:: ``init_with_output``
.. _init_with_output: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.init_with_output

.. |jax.jit| replace:: ``jax.jit``
.. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit

.. |self.param| replace:: ``self.param``
.. _self.param: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.param

.. |setup| replace:: ``setup``
.. _setup: https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.setup

.. |@flax.struct.dataclass| replace:: ``@flax.struct.dataclass``
.. _@flax.struct.dataclass: https://flax.readthedocs.io/en/latest/flax.struct.html#flax.struct.dataclass

.. |checkpoints.convert_pre_linen()| replace:: ``checkpoints.convert_pre_linen()``
.. _checkpoints.convert_pre_linen(): https://flax.readthedocs.io/en/latest/flax.training.html#flax.training.checkpoints.convert_pre_linen

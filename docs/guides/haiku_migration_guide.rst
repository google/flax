
Migrating from Haiku to Flax
==========

This guide will walk through the process of migrating Haiku models to Flax,
and highlight the differences between the two libraries.

.. testsetup::

  import jax
  import jax.numpy as jnp
  from jax.random import PRNGKey
  import optax
  import flax.linen as nn

Basic Example
-----------------

To create custom Modules you subclass from a ``Module`` base class in
both Haiku and Flax. However, Haiku classes use a regular ``__init__`` method
whereas Flax classes are ``dataclasses``, meaning you define some class
attributes that are used to automatically generate a constructor. Also,
all Flax Modules accept a ``name`` argument without needing to define it,
whereas in Haiku ``name`` must be explicitly defined in the constructor
signature and passed to the superclass constructor.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  import haiku as hk

  class Block(hk.Module):
    def __init__(self, features: int, name=None):
      super().__init__(name=name)
      self.features = features

    def __call__(self, x, training: bool):
      x = hk.Linear(self.features)(x)
      x = hk.dropout(hk.next_rng_key(), 0.5 if training else 0, x)
      x = jax.nn.relu(x)
      return x

  class Model(hk.Module):
    def __init__(self, dmid: int, dout: int, name=None):
      super().__init__(name=name)
      self.dmid = dmid
      self.dout = dout

    def __call__(self, x, training: bool):
      x = Block(self.dmid)(x, training)
      x = hk.Linear(self.dout)(x)
      return x

  ---

  import flax.linen as nn

  class Block(nn.Module):
    features: int


    @nn.compact
    def __call__(self, x, training: bool):
      x = nn.Dense(self.features)(x)
      x = nn.Dropout(0.5, deterministic=not training)(x)
      x = jax.nn.relu(x)
      return x

  class Model(nn.Module):
    dmid: int
    dout: int


    @nn.compact
    def __call__(self, x, training: bool):
      x = Block(self.dmid)(x, training)
      x = nn.Dense(self.dout)(x)
      return x

The ``__call__`` method looks very similar in both libraries, however, in Flax
you have to use the ``@nn.compact`` decorator in order to be able to define
submodules inline. In Haiku, this is the default behavior.

Now, a place where Haiku and Flax differ substantially is in how you construct
the model. In Haiku, you use ``hk.transform`` over a function
that calls your Module, ``transform`` will return an object with ``init``
and ``apply`` methods. In Flax, you simply instantiate your Module.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform(forward)

  ---

  ...


  model = Model(256, 10)

To get the model parameters in both libraries you use the ``init`` method
with a ``PRNGKey`` plus some inputs to run the model. The main difference here is
that Flax returns a mapping from collection names to nested array dictionaries,
``params`` is just one of these possible collections. In Haiku, you get the ``params``
structure directly.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  sample_x = jax.numpy.ones((1, 784))
  params = model.init(
    PRNGKey(0),
    sample_x, training=False # <== inputs
  )
  ...

  ---

  sample_x = jax.numpy.ones((1, 784))
  variables = model.init(
    PRNGKey(0),
    sample_x, training=False # <== inputs
  )
  params = variables["params"]

One very important thing to note is that in Flax the parameters structure is
hierarchical, with one level per nested module and a final level for the
parameter name.
In Haiku the parameters structure is a python dictionary with a two level hierarchy:
the fully qualified module name mapping to the parameter name. The module name
consists of a ``/`` separated string path of all the nested Modules.

.. tab-set::

  .. tab-item:: Haiku
    :sync: Haiku

    .. code-block:: python

      ...
      {
        'model/block/linear': {
          'b': (256,),
          'w': (784, 256),
        },
        'model/linear': {
          'b': (10,),
          'w': (256, 10),
        }
      }
      ...


  .. tab-item:: Flax
    :sync: Flax

    .. code-block:: python

      FrozenDict({
        Block_0: {
          Dense_0: {
            bias: (256,),
            kernel: (784, 256),
          },
        },
        Dense_0: {
          bias: (10,),
          kernel: (256, 10),
        },
      })

During training in both frameworks you pass the parameters structure to the
``apply`` method to run the forward pass. Since we are using dropout, in
both cases we must provide a ``key`` to ``apply`` in order to generate
the random dropout masks.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  def train_step(key, params, inputs, labels):
    def loss_fn(params):
        logits = model.apply(
          params,
          key,
          inputs, training=True # <== inputs
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(params)
    params = jax.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params

  ---

  def train_step(key, params, inputs, labels):
    def loss_fn(params):
        logits = model.apply(
          {'params': params},
          inputs, training=True, # <== inputs
          rngs={'dropout': key}
        )
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(params)
    params = jax.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params

.. testcode::
  :hide:

  train_step(PRNGKey(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

The most notable differences is that in Flax you have to
pass the parameters inside a dictionary with a ``params`` key, and the
PRNGKey inside a dictionary with a ``dropout`` key. This is because in Flax
you can have many types of model state and random state. In Haiku, you
just pass the parameters and the PRNGKey directly.

Handling State
-----------------

Now let's see how mutable state is handled in both libraries. We will take
the same model as before, but now we will replace Dropout with BatchNorm.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  class Block(hk.Module):
    def __init__(self, features: int, name=None):
      super().__init__(name=name)
      self.features = features

    def __call__(self, x, training: bool):
      x = hk.Linear(self.features)(x)
      x = hk.BatchNorm(
        create_scale=True, create_offset=True, decay_rate=0.99
      )(x, is_training=training)
      x = jax.nn.relu(x)
      return x

  ---

  class Block(nn.Module):
    features: int


    @nn.compact
    def __call__(self, x, training: bool):
      x = nn.Dense(self.features)(x)
      x = nn.BatchNorm(
        momentum=0.99
      )(x, use_running_average=not training)
      x = jax.nn.relu(x)
      return x

The code is very similar in this case as both libraries provide a BatchNorm
layer. The most notable difference is that Haiku uses ``is_training`` to
control whether or not to update the running statistics, whereas Flax uses
``use_running_average`` for the same purpose.

To instantiate a stateful model in Haiku you use ``hk.transform_with_state``,
which changes the signature for ``init`` and ``apply`` to accept and return
state. As before, in Flax you construct the Module directly.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform_with_state(forward)

  ---

  ...


  model = Model(256, 10)


To initialize both the parameters and state you just call the ``init`` method
as before. However, in Haiku you now get ``state`` as a second return value, and
in Flax you get a new ``batch_stats`` collection in the ``variables`` dictionary.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  sample_x = jax.numpy.ones((1, 784))
  params, state = model.init(
    PRNGKey(0),
    sample_x, training=True # <== inputs
  )
  ...

  ---

  sample_x = jax.numpy.ones((1, 784))
  variables = model.init(
    PRNGKey(0),
    sample_x, training=False # <== inputs
  )
  params, batch_stats = variables["params"], variables["batch_stats"]


In general, in Flax you might find other state collections in the ``variables``
dictionary such as ``cache`` for auto-regressive transformers models,
``intermediates`` for intermediate values added using ``Module.sow``, or other
collection names defined by custom layers. Haiku only makes a distinction
between ``params`` (variables which do not change while running ``apply``) and
``state`` (variables which can change while running ``apply``).

Now, training looks very similar in both frameworks as you use the same
``apply`` method to run the forward pass. In Haiku, now pass the ``state``
as the second argument to ``apply``, and get the new state as the second
return value. In Flax, you instead add ``batch_stats`` as a new key to the
input dictionary, and get the ``updates`` variables dictionary as the second
return value.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  def train_step(params, state, inputs, labels):
    def loss_fn(params):
      logits, new_state = model.apply(
        params, state,
        None, # <== rng
        inputs, training=True # <== inputs
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, new_state

    grads, new_state = jax.grad(loss_fn, has_aux=True)(params)
    params = jax.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params, new_state
  ---

  def train_step(params, batch_stats, inputs, labels):
    def loss_fn(params):
      logits, updates = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        inputs, training=True, # <== inputs
        mutable='batch_stats',
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, updates["batch_stats"]

    grads, batch_stats = jax.grad(loss_fn, has_aux=True)(params)
    params = jax.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params, batch_stats

.. testcode::
  :hide:

  train_step(params, batch_stats, sample_x, jnp.ones((1,), dtype=jnp.int32))

One major difference is that in Flax a state collection can be mutable or immutable.
During ``init`` all collections are mutable by default, however, during ``apply``
you have to explicitly specify which collections are mutable. In this example,
we specify that ``batch_stats`` is mutable. Here a single string is passed but a list
can also be given if there are more mutable collections. If this is not done an
error will be raised at runtime when trying to mutate ``batch_stats``.
Also, when ``mutable`` is anything other than ``False``, the ``updates``
dictionary is returned as the second return value of ``apply``, else only the
model output is returned.
Haiku makes the mutable/immutable distinction through having ``params``
(immutable) and ``state`` (mutable) and using either ``hk.transform`` or
``hk.transform_with_state``

Using Multiple Methods
-----------------------

In this section we will take a look at how to use multiple methods in Haiku and Flax.
As an example, we will implement an auto-encoder model with three methods:
``encode``, ``decode``, and ``__call__``.

In Haiku, we can just define the submodules that ``encode`` and ``decode`` need
directly in ``__init__``, in this case each will just use a ``Linear`` layer.
In Flax, we will define an ``encoder`` and a ``decoder`` Module ahead of time
in ``setup``, and use them in the ``encode`` and ``decode`` respectively.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  class AutoEncoder(hk.Module):


    def __init__(self, embed_dim: int, output_dim: int, name=None):
      super().__init__(name=name)
      self.encoder = hk.Linear(embed_dim, name="encoder")
      self.decoder = hk.Linear(output_dim, name="decoder")

    def encode(self, x):
      return self.encoder(x)

    def decode(self, x):
      return self.decoder(x)

    def __call__(self, x):
      x = self.encode(x)
      x = self.decode(x)
      return x

  ---

  class AutoEncoder(nn.Module):
    embed_dim: int
    output_dim: int

    def setup(self):
      self.encoder = nn.Dense(self.embed_dim)
      self.decoder = nn.Dense(self.output_dim)

    def encode(self, x):
      return self.encoder(x)

    def decode(self, x):
      return self.decoder(x)

    def __call__(self, x):
      x = self.encode(x)
      x = self.decode(x)
      return x

Note that in Flax ``setup`` doesn't run after ``__init__``, instead it runs
when ``init`` or ``apply`` are called.

Now, we want to be able to call any method from our ``AutoEncoder`` model. In Haiku we
can define multiple ``apply`` methods for a module through ``hk.multi_transform``. The
function passed to ``multi_transform`` defines how to initialize the module and which
different apply methods to generate.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  def forward():
    module = AutoEncoder(256, 784)
    init = lambda x: module(x)
    return init, (module.encode, module.decode)

  model = hk.multi_transform(forward)

  ---

  ...




  model = AutoEncoder(256, 784)


To initialize the parameters of our model, ``init`` can be used to trigger the
``__call__`` method, which uses both the ``encode`` and ``decode``
method. This will create all the necessary parameters for the model.

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  params = model.init(
    PRNGKey(0),
    x=jax.numpy.ones((1, 784)),
  )
  ...

  ---

  variables = model.init(
    PRNGKey(0),
    x=jax.numpy.ones((1, 784)),
  )
  params = variables["params"]

This generates the following parameter structure.

.. tab-set::

  .. tab-item:: Haiku
    :sync: Haiku

    .. code-block:: python

      {
          'auto_encoder/~/decoder': {
              'b': (784,),
              'w': (256, 784)
          },
          'auto_encoder/~/encoder': {
              'b': (256,),
              'w': (784, 256)
          }
      }

  .. tab-item:: Flax
    :sync: Flax

    .. code-block:: python

      FrozenDict({
          decoder: {
              bias: (784,),
              kernel: (256, 784),
          },
          encoder: {
              bias: (256,),
              kernel: (784, 256),
          },
      })


Finally, let's explore how we can employ the ``apply`` function to invoke the ``encode`` method:

.. codediff::
  :title_left: Haiku
  :title_right: Flax
  :sync:

  encode, decode = model.apply
  z = encode(
    params,
    None, # <== rng
    x=jax.numpy.ones((1, 784)),

  )

  ---

  ...
  z = model.apply(
    {"params": params},

    x=jax.numpy.ones((1, 784)),
    method="encode",
  )

Because the Haiku ``apply`` function is generated through
``hk.multi_transform``, it's a tuple of two functions which we can unpack into
an ``encode`` and ``decode`` function which correspond to the methods on the
``AutoEncoder`` module. In Flax we call the ``encode`` method through passing
the method name as a string.
Another noteworthy distinction here is that in Haiku, ``rng`` needs to be
explicitly passed, even though the module does not use any stochastic
operations during ``apply``. In Flax this is not necessary. The Haiku ``rng``
is set to ``None`` here, but you could also use ``hk.without_apply_rng`` on the
``apply`` function to remove the ``rng`` argument.


Migrating from Haiku/Linen to NNX
=================================

This guide will showcase the differences between Haiku, Flax Linen and Flax NNX.
Both Haiku and Linen enforce a functional paradigm with stateless modules,
while NNX is a new, next-generation API that embraces the python language to
provide a more intuitive development experience.

.. testsetup:: Haiku, Linen, NNX

  import jax
  import jax.numpy as jnp
  from jax import random
  import optax
  import flax.linen as nn
  from typing import Any

  # TODO: double check the params output match the rendered tab-set
  # TODO: change filename to haiku_linen_upgrade.rst and update other .rst file references
  # TODO: make sure code lines are not too long
  # TODO: make sure all code diffs are aligned

Basic Example
-----------------

To create custom Modules you subclass from a ``Module`` base class in
both Haiku and Flax. Modules can be defined inline in Haiku and Flax
Linen (using the ``@nn.compact`` decorator), whereas modules can't be
defined inline in NNX and must be defined in ``__init__``.

Linen requires a ``deterministic`` argument to control whether or
not dropout is used. NNX also uses a ``deterministic`` argument
but the value can be set later using ``.eval()`` and ``.train()`` methods
that will be shown in a later code snippet.

.. codediff::
  :title: Haiku, Linen, NNX
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

  ---

  from flax.experimental import nnx

  class Block(nnx.Module):
    def __init__(self, in_features: int , out_features: int, rngs: nnx.Rngs):
      self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
      self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x):
      x = self.linear(x)
      x = self.dropout(x)
      x = jax.nn.relu(x)
      return x

  class Model(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
      self.block = Block(din, dmid, rngs=rngs)
      self.linear = nnx.Linear(dmid, dout, rngs=rngs)


    def __call__(self, x):
      x = self.block(x)
      x = self.linear(x)
      return x

Since modules are defined inline in Haiku and Linen, the parameters
are lazily initialized, by inferring the shape of a sample input. In Flax
NNX, the module is stateful and is initialized eagerly. This means that the
input shape must be explicitly passed during module instantiation since there
is no shape inference in NNX.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform(forward)

  ---

  ...


  model = Model(256, 10)

  ---

  ...


  model = Model(784, 256, 10, rngs=nnx.Rngs(0))

To get the model parameters in both Haiku and Linen, you use the ``init`` method
with a ``random.key`` plus some inputs to run the model.

In NNX, the model parameters are automatically initialized when the user
instantiates the model because the input shapes are already explicitly passed at
instantiation time.

Since NNX is eager and the module is bound upon instantiation, the user can access
the parameters (and other fields defined in ``__init__`` via dot-access). On the other
hand, Haiku and Linen use lazy initialization and so the parameters can only be accessed
once the module is initialized with a sample input and both frameworks do not support
dot-access of their attributes.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  sample_x = jnp.ones((1, 784))
  params = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )


  assert params['model/linear']['b'].shape == (10,)
  assert params['model/block/linear']['w'].shape == (784, 256)
  ---

  sample_x = jnp.ones((1, 784))
  variables = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )
  params = variables["params"]

  assert params['Dense_0']['bias'].shape == (10,)
  assert params['Block_0']['Dense_0']['kernel'].shape == (784, 256)

  ---

  ...




  # parameters were already initialized during model instantiation

  assert model.linear.bias.value.shape == (10,)
  assert model.block.linear.kernel.value.shape == (784, 256)

Let's take a look at the parameter structure. In Haiku and Linen, we can
simply inspect the ``params`` object returned from ``.init()``.

To see the parameter structure in NNX, the user can call ``nnx.split`` to
generate ``Graphdef`` and ``State`` objects. The ``Graphdef`` is a static pytree
denoting the structure of the model (for example usages, see
`NNX Basics <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html>`__).
``State`` objects contains all the module variables (i.e. any class that sub-classes
``nnx.Variable``). If we filter for ``nnx.Param``, we will generate a ``State`` object
of all the learnable module parameters.

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


  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python

      ...


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


  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      graphdef, params, rngs = nnx.split(model, nnx.Param, nnx.RngState)

      params
      State({
        'block': {
          'linear': {
            'bias': VariableState(type=Param, value=(256,)),
            'kernel': VariableState(type=Param, value=(784, 256))
          }
        },
        'linear': {
          'bias': VariableState(type=Param, value=(10,)),
          'kernel': VariableState(type=Param, value=(256, 10))
        }
      })

During training in Haiku and Linen, you pass the parameters structure to the
``apply`` method to run the forward pass. To use dropout, we must pass in
``training=True`` and provide a ``key`` to ``apply`` in order to generate the
random dropout masks. To use dropout in NNX, we first call ``model.train()``,
which will set the dropout layer's ``deterministic`` attribute to ``False``
(conversely, calling ``model.eval()`` would set ``deterministic`` to ``True``).
Since the stateful NNX module already contains both the parameters and RNG key
(used for dropout), we simply need to call the module to run the forward pass. We
use ``nnx.split`` to extract the learnable parameters (all learnable parameters
subclass the NNX class ``nnx.Param``) and then apply the gradients and statefully
update the model using ``nnx.update``.

To compile ``train_step``, we decorate the function using ``@jax.jit`` for Haiku
and Linen, and ``@nnx.jit`` for NNX. Similar to ``@jax.jit``, ``@nnx.jit`` also
compiles functions, with the additional feature of allowing the user to compile
functions that take in NNX modules as arguments.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  ...

  @jax.jit
  def train_step(key, params, inputs, labels):
    def loss_fn(params):
      logits = model.apply(
        params, key,
        inputs, training=True # <== inputs

      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(params)


    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params

  ---

  ...

  @jax.jit
  def train_step(key, params, inputs, labels):
    def loss_fn(params):
      logits = model.apply(
        {'params': params},
        inputs, training=True, # <== inputs
        rngs={'dropout': key}
      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(params)


    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params

  ---

  model.train() # set deterministic=False

  @nnx.jit
  def train_step(model, inputs, labels):
    def loss_fn(model):
      logits = model(

        inputs, # <== inputs

      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = nnx.grad(loss_fn)(model)
    # we can use Ellipsis to filter out the rest of the variables
    _, params, _ = nnx.split(model, nnx.Param, ...)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    nnx.update(model, params)

.. testcode:: Haiku, Linen
  :hide:

  train_step(random.key(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  sample_x = jnp.ones((1, 784))
  train_step(model, sample_x, jnp.ones((1,), dtype=jnp.int32))

Flax also offers a convenient ``TrainState`` dataclass to bundle the model,
parameters and optimizer, to simplify training and updating the model. In Haiku
and Linen, we simply pass in the ``model.apply`` function, initialized parameters
and optimizer as arguments to the ``TrainState`` constructor.

In NNX, we must first call ``nnx.split`` on the model to get the
separated ``GraphDef`` and ``State`` objects. We can pass in ``nnx.Param`` to filter
all trainable parameters into a single ``State``, and pass in ``...`` for the remaining
variables. We also need to subclass ``TrainState`` to add a field for the other variables.
We can then pass in ``GraphDef.apply`` as the apply function, ``State`` as the parameters
and other variables and an optimizer as arguments to the ``TrainState`` constructor.
One thing to note is that ``GraphDef.apply`` will take in ``State``'s as arguments and
return a callable function. This function can be called on the inputs to output the
model's logits, as well as updated ``GraphDef`` and ``State`` objects. This isn't needed
for our current example with dropout, but in the next section, you will see that using
these updated objects are relevant with layers like batch norm. Notice we also use
``@jax.jit`` since we aren't passing in NNX modules into ``train_step``.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  from flax.training import train_state







  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,

    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(key, state, inputs, labels):
    def loss_fn(params):
      logits = state.apply_fn(
        params, key,
        inputs, training=True # <== inputs

      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(state.params)


    state = state.apply_gradients(grads=grads)

    return state

  ---

  from flax.training import train_state







  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,

    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(key, state, inputs, labels):
    def loss_fn(params):
      logits = state.apply_fn(
        {'params': params},
        inputs, training=True, # <== inputs
        rngs={'dropout': key}
      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(state.params)


    state = state.apply_gradients(grads=grads)

    return state

  ---

  from flax.training import train_state

  model.train() # set deterministic=False
  graphdef, params, other_variables = nnx.split(model, nnx.Param, ...)

  class TrainState(train_state.TrainState):
    other_variables: nnx.State

  state = TrainState.create(
    apply_fn=graphdef.apply,
    params=params,
    other_variables=other_variables,
    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(state, inputs, labels):
    def loss_fn(params, other_variables):
      logits, (graphdef, new_state) = state.apply_fn(
        params,
        other_variables

      )(inputs) # <== inputs
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(state.params, state.other_variables)


    state = state.apply_gradients(grads=grads)

    return state

.. testcode:: Haiku, Linen
  :hide:

  train_step(random.key(0), state, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  train_step(state, sample_x, jnp.ones((1,), dtype=jnp.int32))

Handling State
-----------------

Now let's see how mutable state is handled in all three frameworks. We will take
the same model as before, but now we will replace Dropout with BatchNorm.

.. codediff::
  :title: Haiku, Linen, NNX
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

  ---

  class Block(nnx.Module):
    def __init__(self, in_features: int , out_features: int, rngs: nnx.Rngs):
      self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
      self.batchnorm = nnx.BatchNorm(
        num_features=out_features, momentum=0.99, rngs=rngs
      )

    def __call__(self, x):
      x = self.linear(x)
      x = self.batchnorm(x)


      x = jax.nn.relu(x)
      return x

Haiku requires an ``is_training`` argument and Linen requires a
``use_running_average`` argument to control whether or not to update the
running statistics. NNX also uses a ``use_running_average`` argument
but the value can be set later using ``.eval()`` and ``.train()`` methods
that will be shown in later code snippets.

As before, you need to pass in the input shape to construct the Module
eagerly in NNX.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform_with_state(forward)

  ---

  ...


  model = Model(256, 10)

  ---

  ...


  model = Model(784, 256, 10, rngs=nnx.Rngs(0))


To initialize both the parameters and state in Haiku and Linen, you just
call the ``init`` method as before. However, in Haiku you now get ``batch_stats``
as a second return value, and in Linen you get a new ``batch_stats`` collection
in the ``variables`` dictionary.
Note that since ``hk.BatchNorm`` only initializes batch statistics when
``is_training=True``, we must set ``training=True`` when initializing parameters
of a Haiku model with an ``hk.BatchNorm`` layer. In Linen, we can set
``training=False`` as usual.

In NNX, the parameters and state are already initialized upon module
instantiation. The batch statistics are of class ``nnx.BatchStat`` which
subclasses the ``nnx.Variable`` class (not ``nnx.Param`` since they aren't
learnable parameters). Calling ``nnx.split`` with no additional filter arguments
will return a state containing all ``nnx.Variable``'s by default.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  sample_x = jnp.ones((1, 784))
  params, batch_stats = model.init(
    random.key(0),
    sample_x, training=True # <== inputs
  )
  ...

  ---

  sample_x = jnp.ones((1, 784))
  variables = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )
  params, batch_stats = variables["params"], variables["batch_stats"]

  ---

  ...




  graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)


Now, training looks very similar in Haiku and Linen as you use the same
``apply`` method to run the forward pass. In Haiku, now pass the ``batch_stats``
as the second argument to ``apply``, and get the newly updated ``batch_stats``
as the second return value. In Linen, you instead add ``batch_stats`` as a new
key to the input dictionary, and get the ``updates`` variables dictionary as the
second return value. To update the batch statistics, we must pass in
``training=True`` to ``apply``.

In NNX, the training code is identical to the earlier example as the
batch statistics (which are bounded to the stateful NNX module) are updated
statefully. To update batch statistics in NNX, we first call ``model.train()``,
which will set the batchnorm layer's ``use_running_average`` attribute to ``False``
(conversely, calling ``model.eval()`` would set ``use_running_average`` to ``True``).
Since the stateful NNX module already contains the parameters and batch statistics,
we simply need to call the module to run the forward pass. We use ``nnx.split`` to
extract the learnable parameters (all learnable parameters subclass the NNX class
``nnx.Param``) and then apply the gradients and statefully update the model using
``nnx.update``.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  ...

  @jax.jit
  def train_step(params, batch_stats, inputs, labels):
    def loss_fn(params, batch_stats):
      logits, batch_stats = model.apply(
        params, batch_stats,
        None, # <== rng
        inputs, training=True # <== inputs
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, batch_stats

    grads, batch_stats = jax.grad(loss_fn, has_aux=True)(params, batch_stats)

    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params, batch_stats
  ---

  ...

  @jax.jit
  def train_step(params, batch_stats, inputs, labels):
    def loss_fn(params, batch_stats):
      logits, updates = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        inputs, training=True, # <== inputs
        mutable='batch_stats',
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, updates["batch_stats"]

    grads, batch_stats = jax.grad(loss_fn, has_aux=True)(params, batch_stats)

    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params, batch_stats

  ---

  model.train() # set use_running_average=False

  @nnx.jit
  def train_step(model, inputs, labels):
    def loss_fn(model):
      logits = model(

        inputs, # <== inputs

      ) # batch statistics are updated statefully in this step
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss

    grads = nnx.grad(loss_fn)(model)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    nnx.update(model, params)

.. testcode:: Haiku, Linen
  :hide:

  train_step(params, batch_stats, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  train_step(model, sample_x, jnp.ones((1,), dtype=jnp.int32))

To use ``TrainState``, we subclass to add an additional field that can store
the batch statistics:

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  ...


  class TrainState(train_state.TrainState):
    batch_stats: Any

  state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    batch_stats=batch_stats,
    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(state, inputs, labels):
    def loss_fn(params, batch_stats):
      logits, batch_stats = state.apply_fn(
        params, batch_stats,
        None, # <== rng
        inputs, training=True # <== inputs
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, batch_stats

    grads, batch_stats = jax.grad(
      loss_fn, has_aux=True
    )(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats)

    return state

  ---

  ...


  class TrainState(train_state.TrainState):
    batch_stats: Any

  state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    batch_stats=batch_stats,
    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(state, inputs, labels):
    def loss_fn(params, batch_stats):
      logits, updates = state.apply_fn(
        {'params': params, 'batch_stats': batch_stats},
        inputs, training=True, # <== inputs
        mutable='batch_stats'
      )
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, updates['batch_stats']

    grads, batch_stats = jax.grad(
      loss_fn, has_aux=True
    )(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats)

    return state

  ---

  model.train() # set deterministic=False
  graphdef, params, batch_stats = nnx.split(model, nnx.Param, nnx.BatchStat)

  class TrainState(train_state.TrainState):
    batch_stats: Any

  state = TrainState.create(
    apply_fn=graphdef.apply,
    params=params,
    batch_stats=batch_stats,
    tx=optax.adam(1e-3)
  )

  @jax.jit
  def train_step(state, inputs, labels):
    def loss_fn(params, batch_stats):
      logits, (graphdef, new_state) = state.apply_fn(
        params, batch_stats
      )(inputs) # <== inputs

      _, batch_stats = new_state.split(nnx.Param, nnx.BatchStat)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
      return loss, batch_stats

    grads, batch_stats = jax.grad(
      loss_fn, has_aux=True
    )(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats)

    return state

.. testcode:: Haiku, Linen
  :hide:

  train_step(state, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  train_step(state, sample_x, jnp.ones((1,), dtype=jnp.int32))


Using Multiple Methods
-----------------------

In this section we will take a look at how to use multiple methods in all three
frameworks. As an example, we will implement an auto-encoder model with three methods:
``encode``, ``decode``, and ``__call__``.

As before, we define the encoder and decoder layers without having to pass in the
input shape, since the module parameters will be initialized lazily using shape
inference in Haiku and Linen. In NNX, we must pass in the input shape
since the module parameters will be initialized eagerly without shape inference.

.. codediff::
  :title: Haiku, Linen, NNX
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

  ---

  class AutoEncoder(nnx.Module):



    def __init__(self, in_dim: int, embed_dim: int, output_dim: int, rngs):
      self.encoder = nnx.Linear(in_dim, embed_dim, rngs=rngs)
      self.decoder = nnx.Linear(embed_dim, output_dim, rngs=rngs)

    def encode(self, x):
      return self.encoder(x)

    def decode(self, x):
      return self.decoder(x)

    def __call__(self, x):
      x = self.encode(x)
      x = self.decode(x)
      return x

As before, we pass in the input shape when instantiating the NNX module.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  def forward():
    module = AutoEncoder(256, 784)
    init = lambda x: module(x)
    return init, (module.encode, module.decode)

  model = hk.multi_transform(forward)

  ---

  ...




  model = AutoEncoder(256, 784)

  ---

  ...




  model = AutoEncoder(784, 256, 784, rngs=nnx.Rngs(0))


For Haiku and Linen, ``init`` can be used to trigger the
``__call__`` method to initialize the parameters of our model,
which uses both the ``encode`` and ``decode`` method. This will
create all the necessary parameters for the model. In NNX,
the parameters are already initialized upon module instantiation.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  params = model.init(
    random.key(0),
    x=jnp.ones((1, 784)),
  )

  ---

  params = model.init(
    random.key(0),
    x=jnp.ones((1, 784)),
  )['params']

  ---

  # parameters were already initialized during model instantiation


  ...

The parameter structure is as follows:

.. tab-set::

  .. tab-item:: Haiku
    :sync: Haiku

    .. code-block:: python

      ...


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

  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python

      ...


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

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      _, params, _ = nnx.split(model, nnx.Param, ...)

      params
      State({
        'decoder': {
          'bias': VariableState(type=Param, value=(784,)),
          'kernel': VariableState(type=Param, value=(256, 784))
        },
        'encoder': {
          'bias': VariableState(type=Param, value=(256,)),
          'kernel': VariableState(type=Param, value=(784, 256))
        }
      })


Finally, let's explore how we can employ the forward pass. In Haiku
and Linen, we use the ``apply`` function to invoke the ``encode``
method. In NNX, we simply can simply call the ``encode`` method
directly.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  encode, decode = model.apply
  z = encode(
    params,
    None, # <== rng
    x=jnp.ones((1, 784)),

  )

  ---

  ...
  z = model.apply(
    {"params": params},

    x=jnp.ones((1, 784)),
    method="encode",
  )

  ---

  ...
  z = model.encode(jnp.ones((1, 784)))




  ...


Lifted Transforms
-----------------

Both Flax and Haiku provide a set of transforms, which we will refer to as lifted transforms,
that wrap JAX transformations in such a way that they can be used with Modules and sometimes
provide additional functionality. In this section we will take a look at how to use the
lifted version of ``scan`` in both Flax and Haiku to implement a simple RNN layer.

To begin, we will first define a ``RNNCell`` module that will contain the logic for a single
step of the RNN. We will also define a ``initial_state`` method that will be used to initialize
the state (a.k.a. ``carry``) of the RNN. Like with ``jax.lax.scan``, the ``RNNCell.__call__``
method will be a function that takes the carry and input, and returns the new
carry and output. In this case, the carry and the output are the same.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  class RNNCell(hk.Module):
    def __init__(self, hidden_size: int, name=None):
      super().__init__(name=name)
      self.hidden_size = hidden_size

    def __call__(self, carry, x):
      x = jnp.concatenate([carry, x], axis=-1)
      x = hk.Linear(self.hidden_size)(x)
      x = jax.nn.relu(x)
      return x, x

    def initial_state(self, batch_size: int):
      return jnp.zeros((batch_size, self.hidden_size))

  ---

  class RNNCell(nn.Module):
    hidden_size: int


    @nn.compact
    def __call__(self, carry, x):
      x = jnp.concatenate([carry, x], axis=-1)
      x = nn.Dense(self.hidden_size)(x)
      x = jax.nn.relu(x)
      return x, x

    def initial_state(self, batch_size: int):
      return jnp.zeros((batch_size, self.hidden_size))

  ---

  class RNNCell(nnx.Module):
    def __init__(self, input_size, hidden_size, rngs):
      self.linear = nnx.Linear(hidden_size + input_size, hidden_size, rngs=rngs)
      self.hidden_size = hidden_size

    def __call__(self, carry, x):
      x = jnp.concatenate([carry, x], axis=-1)
      x = self.linear(x)
      x = jax.nn.relu(x)
      return x, x

    def initial_state(self, batch_size: int):
      return jnp.zeros((batch_size, self.hidden_size))

Next, we will define a ``RNN`` Module that will contain the logic for the entire RNN.
In Haiku, we will first initialze the ``RNNCell``, then use it to construct the ``carry``,
and finally use ``hk.scan`` to run the ``RNNCell`` over the input sequence.

In Linen, we will use ``nn.scan`` to define a new temporary type that wraps
``RNNCell``. During this process we will also specify instruct ``nn.scan`` to broadcast
the ``params`` collection (all steps share the same parameters) and to not split the
``params`` rng stream (so all steps intialize with the same parameters), and finally
we will specify that we want scan to run over the second axis of the input and stack
the outputs along the second axis as well. We will then use this temporary type immediately
to create an instance of the lifted ``RNNCell`` and use it to create the ``carry`` and
the run the ``__call__`` method which will ``scan`` over the sequence.

In NNX, we define a scan function ``scan_fn`` that will use the ``RNNCell`` defined
in ``__init__`` to scan over the sequence.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  class RNN(hk.Module):
    def __init__(self, hidden_size: int, name=None):
      super().__init__(name=name)
      self.hidden_size = hidden_size

    def __call__(self, x):
      cell = RNNCell(self.hidden_size)
      carry = cell.initial_state(x.shape[0])
      carry, y = hk.scan(
        cell, carry,
        jnp.swapaxes(x, 1, 0)
      )
      y = jnp.swapaxes(y, 0, 1)
      return y

  ---

  class RNN(nn.Module):
    hidden_size: int


    @nn.compact
    def __call__(self, x):
      rnn = nn.scan(
        RNNCell, variable_broadcast='params',
        split_rngs={'params': False}, in_axes=1, out_axes=1
      )(self.hidden_size)
      carry = rnn.initial_state(x.shape[0])
      carry, y = rnn(carry, x)

      return y

  ---

  class RNN(nnx.Module):
    def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs):
      self.hidden_size = hidden_size
      self.cell = RNNCell(input_size, self.hidden_size, rngs=rngs)

    def __call__(self, x):
      scan_fn = lambda carry, cell, x: cell(carry, x)
      carry = self.cell.initial_state(x.shape[0])
      carry, y = nnx.scan(
        scan_fn, state_axes={},
        in_axes=1, out_axes=1
      )(carry, self.cell, x)

      return y

In general, the main difference between lifted transforms between Flax and Haiku is that
in Haiku the lifted transforms don't operate over the state, that is, Haiku will handle the
``params`` and ``state`` in such a way that it keeps the same shape inside and outside of the
transform. In Flax, the lifted transforms can operate over both variable collections and rng
streams, the user must define how different collections are treated by each transform
according to the transform's semantics.

As before, the parameters must be initialized via ``.init()`` and passed into ``.apply()``
to conduct a forward pass in Haiku and Linen. In NNX, the parameters are already
eagerly initialized and bound to the stateful module, and the module can be simply called
on the input to conduct a forward pass.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  x = jnp.ones((3, 12, 32))

  def forward(x):
    return RNN(64)(x)

  model = hk.without_apply_rng(hk.transform(forward))

  params = model.init(
    random.key(0),
    x=jnp.ones((3, 12, 32)),
  )

  y = model.apply(
    params,
    x=jnp.ones((3, 12, 32)),
  )

  ---

  x = jnp.ones((3, 12, 32))




  model = RNN(64)

  params = model.init(
    random.key(0),
    x=jnp.ones((3, 12, 32)),
  )['params']

  y = model.apply(
    {'params': params},
    x=jnp.ones((3, 12, 32)),
  )

  ---

  x = jnp.ones((3, 12, 32))




  model = RNN(x.shape[2], 64, rngs=nnx.Rngs(0))






  y = model(x)


  ...

The only notable change with respect to the examples in the previous sections is that
this time around we used ``hk.without_apply_rng`` in Haiku so we didn't have to
pass the ``rng`` argument as ``None`` to the ``apply`` method.

Scan over layers
----------------
One very important application of ``scan`` is apply a sequence of layers iteratively
over an input, passing the output of each layer as the input to the next layer. This
is very useful to reduce compilation time for big models. As an example we will create
a simple ``Block`` Module, and then use it inside an ``MLP`` Module that will apply
the ``Block`` Module ``num_layers`` times.

In Haiku, we define the ``Block`` Module as usual, and then inside ``MLP`` we will
use ``hk.experimental.layer_stack`` over a ``stack_block`` function to create a stack
of ``Block`` Modules.

In Linen, the definition of ``Block`` is a little different,
``__call__`` will accept and return a second dummy input/output that in both cases will
be ``None``. In ``MLP``, we will use ``nn.scan`` as in the previous example, but
by setting ``split_rngs={'params': True}`` and ``variable_axes={'params': 0}``
we are telling ``nn.scan`` create different parameters for each step and slice the
``params`` collection along the first axis, effectively implementing a stack of
``Block`` Modules as in Haiku.

In NNX, we use ``nnx.Scan.constructor()`` to define a stack of ``Block`` modules.
We can then simply call the stack of ``Block``'s, ``self.blocks``, on the input and
carry to get the forward pass output.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  class Block(hk.Module):
    def __init__(self, features: int, name=None):
      super().__init__(name=name)
      self.features = features

    def __call__(self, x, training: bool):
      x = hk.Linear(self.features)(x)
      x = hk.dropout(hk.next_rng_key(), 0.5 if training else 0, x)
      x = jax.nn.relu(x)
      return x

  class MLP(hk.Module):
    def __init__(self, features: int, num_layers: int, name=None):
        super().__init__(name=name)
        self.features = features
        self.num_layers = num_layers



    def __call__(self, x, training: bool):
      @hk.experimental.layer_stack(self.num_layers)
      def stack_block(x):
        return Block(self.features)(x, training)

      stack = hk.experimental.layer_stack(self.num_layers)
      return stack_block(x)

  ---

  class Block(nn.Module):
    features: int
    training: bool

    @nn.compact
    def __call__(self, x, _):
      x = nn.Dense(self.features)(x)
      x = nn.Dropout(0.5)(x, deterministic=not self.training)
      x = jax.nn.relu(x)
      return x, None

  class MLP(nn.Module):
    features: int
    num_layers: int




    @nn.compact
    def __call__(self, x, training: bool):
      ScanBlock = nn.scan(
        Block, variable_axes={'params': 0}, split_rngs={'params': True},
        length=self.num_layers)

      y, _ = ScanBlock(self.features, training)(x, None)
      return y

  ---

  class Block(nnx.Module):
    def __init__(self, input_dim, features, rngs):
      self.linear = nnx.Linear(input_dim, features, rngs=rngs)
      self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array, _):
      x = self.linear(x)
      x = self.dropout(x)
      x = jax.nn.relu(x)
      return x, None

  class MLP(nnx.Module):
    def __init__(self, input_dim, features, num_layers, rngs):
      self.blocks = nnx.Scan.constructor(
        Block, length=num_layers
      )(input_dim, features, rngs=rngs)



    def __call__(self, x):




      y, _ = self.blocks(x, None)
      return y

Notice how in Flax we pass ``None`` as the second argument to ``ScanBlock`` and ignore
its second output. These represent the inputs/outputs per-step but they are ``None``
because in this case we don't have any.

Initializing each model is the same as in previous examples. In this case,
we will be specifying that we want to use ``5`` layers each with ``64`` features.
As before, we also pass in the input shape for NNX.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  def forward(x, training: bool):
    return MLP(64, num_layers=5)(x, training)

  model = hk.transform(forward)

  sample_x = jnp.ones((1, 64))
  params = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )

  ---

  ...


  model = MLP(64, num_layers=5)

  sample_x = jnp.ones((1, 64))
  params = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )['params']

  ---

  ...


  model = MLP(64, 64, num_layers=5, rngs=nnx.Rngs(0))





  ...

When using scan over layers the one thing you should notice is that all layers
are fused into a single layer whose parameters have an extra "layer" dimension on
the first axis. In this case, the shape of all parameters will start with ``(5, ...)``
as we are using ``5`` layers.

.. tab-set::

  .. tab-item:: Haiku
    :sync: Haiku

    .. code-block:: python

      ...


      {
          'mlp/__layer_stack_no_per_layer/block/linear': {
              'b': (5, 64),
              'w': (5, 64, 64)
          }
      }



      ...

  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python

      ...


      FrozenDict({
          ScanBlock_0: {
              Dense_0: {
                  bias: (5, 64),
                  kernel: (5, 64, 64),
              },
          },
      })

      ...

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      _, params, _ = nnx.split(model, nnx.Param, ...)

      params
      State({
        'blocks': {
          'scan_module': {
            'linear': {
              'bias': VariableState(type=Param, value=(5, 64)),
              'kernel': VariableState(type=Param, value=(5, 64, 64))
            }
          }
        }
      })

Top-level Haiku functions vs top-level Flax modules
-----------------------------------

In Haiku, it is possible to write the entire model as a single function by using
the raw ``hk.{get,set}_{parameter,state}`` to define/access model parameters and
states. It very common to write the top-level "Module" as a function instead.

The Flax team recommends a more Module-centric approach that uses ``__call__`` to
define the forward function. In Linen, the corresponding accessor will be
``Module.param`` and ``Module.variable`` (go to `Handling State <#handling-state>`__
for an explanation on collections). In NNX, the parameters and variables can
be set and accessed as normal using regular Python class semantics.

.. codediff::
  :title: Haiku, Linen, NNX
  :sync:

  ...


  def forward(x):


    counter = hk.get_state('counter', shape=[], dtype=jnp.int32, init=jnp.ones)
    multiplier = hk.get_parameter(
      'multiplier', shape=[1,], dtype=x.dtype, init=jnp.ones
    )

    output = x + multiplier * counter

    hk.set_state("counter", counter + 1)
    return output

  model = hk.transform_with_state(forward)

  params, state = model.init(random.key(0), jnp.ones((1, 64)))

  ---

  ...


  class FooModule(nn.Module):
    @nn.compact
    def __call__(self, x):
      counter = self.variable('counter', 'count', lambda: jnp.ones((), jnp.int32))
      multiplier = self.param(
        'multiplier', nn.initializers.ones_init(), [1,], x.dtype
      )

      output = x + multiplier * counter.value
      if not self.is_initializing():  # otherwise model.init() also increases it
        counter.value += 1
      return output

  model = FooModule()
  variables = model.init(random.key(0), jnp.ones((1, 64)))
  params, counter = variables['params'], variables['counter']

  ---

  class Counter(nnx.Variable):
    pass

  class FooModule(nnx.Module):

    def __init__(self, rngs):
      self.counter = Counter(jnp.ones((), jnp.int32))
      self.multiplier = nnx.Param(
        nnx.initializers.ones(rngs.params(), [1,], jnp.float32)
      )
    def __call__(self, x):
      output = x + self.multiplier * self.counter.value

      self.counter.value += 1
      return output

  model = FooModule(rngs=nnx.Rngs(0))

  _, params, counter = nnx.split(model, nnx.Param, Counter)
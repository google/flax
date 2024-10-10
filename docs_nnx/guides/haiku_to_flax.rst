Migrating from Haiku to Flax
============================

This guide compares and contrasts Haiku with Flax Linen and Flax NNX, which should help you migrate to Flax from Haiku. You will learn the differences between the frameworks through various examples, such as:

* Simple model creation with ``Module`` and dropout, model instantiation and parameter initialization, and setting up the ``train_step`` for training.
* Handling mutable states (using ``BatchNorm`` instead of dropout from model creation to training).
* Using multiple methods (using an auto-encoder model).
* Lifted transformations (using ``scan`` and a recurrent neural network).
* ``Scan`` over layers.
* Top-level Haiku functions vs top-level Flax ``Module``s.

Both Haiku and Flax Linen enforce a functional paradigm with stateless ``Module``, while Flax NNX embraces the Python language to provide a more intuitive development experience.

First, some necessary imports:

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

Basic example
-------------

Let’s begin with a basic example of a one-layer network with dropout and a ReLU activation. To create a custom ``Module``, in Haiku and Flax (Linen and NNX) you subclass the ``Module`` base class.

Note that:

* In Haiku and Flax Linen, ``Module``s can be defined inline using the ``@nn.compact`` decorator).
* In Flax NNX, ``nnx.Module``s can't be defined inline in NNX but instead must be defined in ``__init__``.

In addition:

* Flax Linen requires a ``deterministic`` argument to control whether or not dropout is used.
* Flax NNX also uses a ``deterministic`` argument but the value can be set later using ``.eval()`` and ``.train()`` methods that will be shown in a later code snippet.

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

Initializing the parameters:

* In Haiku and Flax Linen, since ``Module``s are defined inline, the parameters are lazily initialized by inferring the shape of a sample input.
* In Flax NNX, the ``nnx.Module`` is stateful and is initialized eagerly. This means that the input shape must be explicitly passed during ``nnx.Module`` instantiation since there is no shape inference in NNX.

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

To get the model parameters:

* In both Haiku and Flax Linen, use the ``init`` method with a ``random.key`` plus some inputs to run the model.
* In Flax NNX, the model parameters are automatically initialized when you instantiate the model because the input shapes are already explicitly passed at instantiation time.

Also:

* Since Flax NNX is eager and the ``Module`` is bound upon instantiation, you can access the parameters (and other fields defined in ``__init__`` via dot-access).
* On the other hand, Haiku and Flax Linen use lazy initialization. Therefore the parameters can only be accessed once the ``Module`` is initialized with a sample input, and both frameworks do not support dot-access of their attributes.

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




  # Parameters were already initialized during model instantiation.

  assert model.linear.bias.value.shape == (10,)
  assert model.block.linear.kernel.value.shape == (784, 256)

Let's review the parameter structure:

* In Haiku and Flax Linen, you would simply inspect the ``params`` object returned from ``.init()``.
* In Flax NNX, to view the parameter structure, you can call ``nnx.split`` to generate ``nnx.Graphdef`` and ``nnx.State`` objects. The ``nnx.Graphdef`` is a static pytree denoting the structure of the model (for examples, check out the `Flax basics<https://flax.readthedocs.io/en/latest/nnx_basics.html>`__). ``nnx.State`` objects contain all ``Module`` variables (i.e. any class that subclasses ``nnx.Variable``). If you filter for ``nnx.Param``, you will generate a ``nnx.State`` object of all the learnable ``Module`` parameters (you can learn more in `Using Filters<https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__.

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

During training:

* In Haiku and Flax Linen, you pass the parameters structure to the ``apply`` method to run the forward pass. To use dropout, you must pass in ``training=True`` and provide a ``key`` to ``apply`` to generate the random dropout masks. To use dropout in NNX, you first call ``model.train()``, which will set the dropout layer's ``deterministic`` attribute to ``False`` (conversely, calling ``model.eval()`` would set ``deterministic`` to ``True``).
* In Flax NNX, since the stateful `nnx.Module` already contains both the parameters and the PRNG key (used for dropout), you simply need to call the ``nnx.Module`` to run the forward pass. Use ``nnx.split`` to extract the learnable parameters (all learnable parameters subclass the ``nnx.Param`` class), and then apply the gradients and statefully update the model using ``nnx.update``.

To compile the ``train_step``:

* Haiku and Flax Linen, you decorate the function using ``@jax.jit``.
* In Flax NNX, you decorate it with ``@nnx.jit``. Similar to ``@jax.jit``, ``@nnx.jit`` also compiles functions, with the additional feature of allowing the user to compile functions that take in NNX modules as arguments.

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
    # You can use Ellipsis to filter out the rest of the variables.
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

In addition, Flax NNX offers a convenient ``nnx.TrainState`` dataclass to bundle the model, parameters and the optimizer to simplify training and updating the model (you can learn more in `Using TrainState in NNX<https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html#using-trainstate-in-nnx>`__.

* In Haiku and Flax Linen, you simply pass in the ``model.apply`` function, initialized parameters and the optimizer as arguments to the ``TrainState`` constructor.
* In Flax NNX, you must first call ``nnx.split`` on the model to get the separated ``nnx.GraphDef`` and ``nnx.State`` objects.You can pass in ``nnx.Param`` to filter
all trainable parameters into a single ``nnx.State``, and pass in ``...`` for the remaining variables. You also need to subclass ``nnx.TrainState`` to add a field for the other variables. Then, you can pass in ``nnx.GraphDef.apply`` as the apply function, ``nnx.State`` as the parameters and other variables and an optimizer as arguments to the ``nnx.TrainState`` constructor.

**Note:** ``nnx.GraphDef.apply`` will take in ``nnx.State``s as arguments and return a callable function. This function can be called on the inputs to output the model's logits, as well as updated ``nnx.GraphDef`` and ``nnx.State`` objects. This isn't needed for our current example with dropout, but in the next section, you will see that using these updated objects are relevant with layers like batch normalization. Notice the use of ``@jax.jit`` since you aren't passing in NNX modules into ``train_step``.

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

Handling ``State``
------------------

Now let's review how mutable state is handled in all three frameworks. You will use the same model as before, but this time you will replace ``Dropout`` with ``BatchNorm``:

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

To control whether or not to update the running statistics:

* Haiku requires the ``is_training`` argument, while Flax Linen requires the ``use_running_average`` argument.
* Flax NNX also uses the ``use_running_average`` argument but the value can be set later using ``.eval()`` and ``.train()`` methods that will be shown in later code snippets. As before, you need to pass in the input shape to construct the ``nnx.Module`` eagerly.

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


To initialize both the parameters and state:

* In Haiku and Flax Linen, you just call the ``init`` method as before. However:
  * In Haiku you now get ``batch_stats`` as a second return value.
  * in Linen you get a new ``batch_stats`` collection in the ``variables`` dictionary.
  * **Note:** In Haiku, since ``hk.BatchNorm`` only initializes batch statistics when ``is_training=True``, you must set ``training=True`` when initializing parameters of a Haiku model with an ``hk.BatchNorm`` layer. And in Linen, you can set ``training=False`` as usual.
* In Flax NNX, the parameters and state are already initialized upon ```nnx.Module`` instantiation. The batch statistics are of class ``nnx.BatchStat`` which subclasses the ``nnx.Variable`` class (not ``nnx.Param`` since they aren't learnable parameters). Calling ``nnx.split`` with no additional filter arguments will return a state containing all ``nnx.Variable``'s by default.

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


Now, for training:

* In Haiku and Flax Linen, training is very similar, as you use the same ``apply`` method to run the forward pass.
  * In Haiku, you pass the ``batch_stats`` as the second argument to ``apply``, and get the newly updated ``batch_stats`` as the second return value.
  * In Flax Linen, you add ``batch_stats`` as a new key to the input dictionary, and get the ``updates`` variable dictionary as the second return value.
  * To update the batch statistics, you must pass in ``training=True`` to ``apply``.
* In Flax NNX, the training code is identical to the earlier example as the batch statistics (which are bound to the stateful ``nnx.Module``) are updated statefully. To update batch statistics in NNX, you first call ``model.train()``, which will set the batchnorm layer's ``use_running_average`` attribute to ``False`` (conversely, calling ``model.eval()`` would set ``use_running_average`` to ``True``). Since the stateful ``nnx.Module`` already contains the parameters and batch statistics, you simply need to call the ``nnx.Module`` to run the forward pass. Use ``nnx.split`` to extract the learnable parameters (all learnable parameters subclass the Flax ``nnx.Param`` class), and then apply the gradients and statefully update the model using ``nnx.update``.

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

To use ``TrainState``, you subclass to add an additional field that can store the batch statistics:

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


Using multiple methods
----------------------

This section examines how to use multiple methods in all three frameworks using an implementation of an auto-encoder model with three methods - ``encode``, ``decode``, and ``__call__`` - as an example.

* In Haiku and Flax Linen, as before, you define the encoder and decoder layers without having to pass in the input shape, since the ``Module`` parameters will be initialized lazily using shape inference.
* In Flax NNX, you must pass in the input shape since the ``nnx.Module`` parameters will be initialized eagerly without shape inference.

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

As before, in Flax NNX you pass in the input shape when instantiating the ``nnx.Module``.

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


Initializing the parameters:

* For Haiku and Flax Linen, ``init`` can be used to trigger the ``__call__`` method to initialize the parameters of your model, which uses both the ``encode`` and ``decode`` method. This will create all necessary parameters for the model.
* In Flax NNX, the parameters are already initialized upon model instantiation.

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

  # Parameters were already initialized during model instantiation.


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


Finally, let's explore how to employ the forward pass:

* In Haiku and Flax Linen, use the ``apply`` function to invoke the ``encode`` method.
* In Flax NNX, you can simply call the ``encode`` method directly.

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


Lifted transformations
----------------------

Flax `Linen <https://flax-linen.readthedocs.io/en/latest/developer_notes/lift.html>`__, Flax `NNX <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__) and `Haiku <https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html#A-first-example-with-hk.transform>`__ provide a set of transforms, which are referred to as lifted transformations. These transforms wrap `JAX transformations <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ in such a way that they can be used with ``Module``s, and sometimes provide additional functionality.

This section examines how to use the lifted version of ``scan`` in Flax (Linen and NNX) and Haiku to implement a simple recurrent neural network (RNN) layer.

Start with defining the ``RNNCell`` ``Module`` that will contain the logic for a single step of the RNN. In addition, define the ``initial_state`` method that will be used to initialize the state (a.k.a. ``carry``) of the RNN. Similar to the ``jax.lax.scan`` `API <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__, the ``RNNCell.__call__`` method will be a function that takes the carry and input, and returns the new carry and output. In this case, the carry and the output are the same.

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

Next, define the ``RNN`` ``Module`` that will contain the logic for the entire RNN:

* In Haiku, you first initialize the ``RNNCell``, then use it to construct the ``carry``, and then use ``hk.scan`` to run the ``RNNCell`` over the input sequence.
* In Flax Linen, you use ``flax.linen.scan`` (``nn.scan``) to define a new temporary type that wraps ``RNNCell``. During this process, specify the instruct ``nn.scan`` to broadcast the ``params`` collection (all steps share the same parameters) and to not split the ``params`` PRNG stream (so that all steps initialize with the same parameters); and, finally, specify that you want ``scan`` to run over the second axis of the input and stack the outputs along the second axis as well. You will then use this temporary type immediately to create an instance of the lifted ``RNNCell``, and use it to create the ``carry`` and the run the ``__call__`` method, which will ``scan`` over the sequence.
* In Flax NNX, you define a custom scan function ``scan_fn`` that will use the ``RNNCell`` defined in ``__init__`` to scan over the sequence.

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
        scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)
      )(carry, self.cell, x)

      return y

In general, the main difference between lifted transforms between Flax (Linen and NNX) and Haiku is as follows:

- In Haiku, the lifted transforms don't operate over the state. Tat is, Haiku will handle the ``params`` and ``state`` in such a way that it keeps the same shape inside and outside of the transform.
- In Flax Linen and NNX, the lifted transforms can operate over both variable collections and rng streams, the user must define how different collections are treated by each transform according to the transform's semantics.

Initializing the parameters:

* In Haiku and Flax Linen, as before, the parameters must be initialized via ``.init()`` and passed into ``.apply()`` to conduct a forward pass in Haiku and Flax Linen.
* In Flax NNX, the parameters are already eagerly initialized and bound to the stateful ``Module``, and the ``Module`` can be simply called on the input to conduct a forward pass.

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

The Haiku example contains the only notable change compared with the examples in the previous sections:

* Here you used ``hk.without_apply_rng``, so that you didn't have to pass the ``rng`` argument as ``None`` to the ``apply`` method.

Scan over layers
----------------

One very important application of ``scan`` is to apply a sequence of layers iteratively over an input, passing the output of each layer as the input to the next layer. This is very useful to reduce compilation time for large models.

As an example, let’s create a simple ``Block`` ``Module``, and then use it inside an ``MLP`` ``Module`` that will apply the ``Block`` ``Module`` ``num_layers`` times:

* In Haiku, define the ``Block`` ``Module`` as usual, and then inside the ``MLP`` you use ``hk.experimental.layer_stack`` over a ``stack_block`` function to create a stack of ``Block`` Modules.
* In Flax Linen, the definition of ``Block`` is a little different. ``__call__`` will accept and return a second dummy input/output that in both cases will be ``None``. In he ``MLP``, use ``nn.scan`` similar to the previous example, but by setting ``split_rngs={'params': True}`` and ``variable_axes={'params': 0}`` you instruct ``nn.scan`` to create different parameters for each step, and slice the ``params`` collection along the first axis effectively implementing a stack of ``Block`` ``Module``s, similar to Haiku.
* In Flax NNX, use ``nnx.Scan.constructor()`` to define a stack of ``Block`` ``Module``s. You can then simply call the stack of ``Block``'s, ``self.blocks``, on the input and carry to get the forward pass output.

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

**Note:** Notice how in Flax Linen and NNX, you pass ``None`` as the second argument to ``ScanBlock`` and ignore its second output. These normally represent the inputs/outputs per-step but here they are ``None`` because in this case you don't have any.

Next, initializing each model is the same as in previous examples. In this example, you will specify that you want to use ``5`` layers each with ``64`` features:
- In Flax NNX, as before, you also pass in the input shape.

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

When using ``scan`` over layers, one thing you should notice is that all layers are fused into a single layer whose parameters have an extra "layer" dimension on the first axis. In this case, the shape of all parameters will start with ``(5, ...)`` since you are using ``5`` layers:

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


Top-level Haiku functions vs top-level Flax ``Module``s
-------------------------------------------------------

In Haiku, it is possible to write the entire model as a single function by using the raw ``hk.{get,set}_{parameter,state}`` to define/access model parameters and states. It is very common to write the top-level ``Module`` as a function instead.

The Flax team recommends a more ``Module``-centric approach that uses ``__call__`` to define the forward function:

* In Flax Linen, the corresponding accessor will be ``Module.param`` and ``Module.variable`` (go to `Handling State <#handling-state>`__ for an explanation on collections).
* In Flax NNX, the parameters and variables can be set and accessed as normal using regular Python class semantics.

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
      if not self.is_initializing():  # Otherwise model.init() also increases it
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

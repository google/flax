
Migrating from Haiku to Flax
============================

This guide will walk through the process of migrating Haiku models to Flax,
and highlight the differences between the two libraries.

.. testsetup:: Haiku, Flax

  import jax
  import jax.numpy as jnp
  from jax import random
  import optax
  import flax.linen as nn
  import haiku as hk

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
  :title: Haiku, Flax
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
  :title: Haiku, Flax
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform(forward)

  ---

  ...


  model = Model(256, 10)

To get the model parameters in both libraries you use the ``init`` method
with a ``random.key`` plus some inputs to run the model. The main difference here is
that Flax returns a mapping from collection names to nested array dictionaries,
``params`` is just one of these possible collections. In Haiku, you get the ``params``
structure directly.

.. codediff::
  :title: Haiku, Flax
  :sync:

  sample_x = jax.numpy.ones((1, 784))
  params = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )
  ...

  ---

  sample_x = jax.numpy.ones((1, 784))
  variables = model.init(
    random.key(0),
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
  :title: Haiku, Flax
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
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

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
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params

.. testcode:: Haiku, Flax
  :hide:

  train_step(random.key(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

The most notable differences is that in Flax you have to
pass the parameters inside a dictionary with a ``params`` key, and the
key inside a dictionary with a ``dropout`` key. This is because in Flax
you can have many types of model state and random state. In Haiku, you
just pass the parameters and the key directly.

Handling State
-----------------

Now let's see how mutable state is handled in both libraries. We will take
the same model as before, but now we will replace Dropout with BatchNorm.

.. codediff::
  :title: Haiku, Flax
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
  :title: Haiku, Flax
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
Note that since ``hk.BatchNorm`` only initializes batch statistics when
``is_training=True``, we must set ``training=True`` when initializing parameters
of a Haiku model with an ``hk.BatchNorm`` layer. In Flax, we can set
``training=False`` as usual.

.. codediff::
  :title: Haiku, Flax
  :sync:

  sample_x = jax.numpy.ones((1, 784))
  params, state = model.init(
    random.key(0),
    sample_x, training=True # <== inputs #!
  )
  ...

  ---

  sample_x = jax.numpy.ones((1, 784))
  variables = model.init(
    random.key(0), #!
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
  :title: Haiku, Flax
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
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

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
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    return params, batch_stats

.. testcode:: Flax
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
  :title: Haiku, Flax
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
  :title: Haiku, Flax
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
  :title: Haiku, Flax
  :sync:

  params = model.init(
    random.key(0),
    x=jax.numpy.ones((1, 784)),
  )
  ...

  ---

  variables = model.init(
    random.key(0),
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
  :title: Haiku, Flax
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
operations during ``apply``. In Flax this is not necessary (check out
`Randomness and PRNGs in Flax <https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/rng_guide.html>`_).
The Haiku ``rng`` is set to ``None`` here, but you could also use
``hk.without_apply_rng`` on the ``apply`` function to remove the ``rng`` argument.


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
  :title: Haiku, Flax
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

Next, we will define a ``RNN`` Module that will contain the logic for the entire RNN.
In Haiku, we will first initialze the ``RNNCell``, then use it to construct the ``carry``,
and finally use ``hk.scan`` to run the ``RNNCell`` over the input sequence. In Flax its
done a bit differently, we will use ``nn.scan`` to define a new temporary type that wraps
``RNNCell``. During this process we will also specify instruct ``nn.scan`` to broadcast
the ``params`` collection (all steps share the same parameters) and to not split the
``params`` rng stream (so all steps intialize with the same parameters), and finally
we will specify that we want scan to run over the second axis of the input and stack
the outputs along the second axis as well. We will then use this temporary type immediately
to create an instance of the lifted ``RNNCell`` and use it to create the ``carry`` and
the run the ``__call__`` method which will ``scan`` over the sequence.

.. codediff::
  :title: Haiku, Flax
  :sync:

  class RNN(hk.Module):
    def __init__(self, hidden_size: int, name=None):
      super().__init__(name=name)
      self.hidden_size = hidden_size

    def __call__(self, x):
      cell = RNNCell(self.hidden_size)
      carry = cell.initial_state(x.shape[0])
      carry, y = hk.scan(cell, carry, jnp.swapaxes(x, 1, 0))
      y = jnp.swapaxes(y, 0, 1)
      return y

  ---

  class RNN(nn.Module):
    hidden_size: int


    @nn.compact
    def __call__(self, x):
      rnn = nn.scan(RNNCell, variable_broadcast='params', split_rngs={'params': False},
                    in_axes=1, out_axes=1)(self.hidden_size)
      carry = rnn.initial_state(x.shape[0])
      carry, y = rnn(carry, x)
      return y

In general, the main difference between lifted transforms between Flax and Haiku is that
in Haiku the lifted transforms don't operate over the state, that is, Haiku will handle the
``params`` and ``state`` in such a way that it keeps the same shape inside and outside of the
transform. In Flax, the lifted transforms can operate over both variable collections and rng
streams, the user must define how different collections are treated by each transform
according to the transform's semantics.

Finally, let's quickly view how the ``RNN`` Module would be used in both Haiku and Flax.

.. codediff::
  :title: Haiku, Flax
  :sync:

  def forward(x):
    return RNN(64)(x)

  model = hk.without_apply_rng(hk.transform(forward))

  params = model.init(
    random.key(0),
    x=jax.numpy.ones((3, 12, 32)),
  )

  y = model.apply(
    params,
    x=jax.numpy.ones((3, 12, 32)),
  )

  ---

  ...


  model = RNN(64)

  variables = model.init(
    random.key(0),
    x=jax.numpy.ones((3, 12, 32)),
  )
  params = variables['params']
  y = model.apply(
    {'params': params},
    x=jax.numpy.ones((3, 12, 32)),
  )

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
of ``Block`` Modules. In Flax, the definition of ``Block`` is a little different,
``__call__`` will accept and return a second dummy input/output that in both cases will
be ``None``. In ``MLP``, we will use ``nn.scan`` as in the previous example, but
by setting ``split_rngs={'params': True}`` and ``variable_axes={'params': 0}``
we are telling ``nn.scan`` create different parameters for each step and slice the
``params`` collection along the first axis, effectively implementing a stack of
``Block`` Modules as in Haiku.


.. codediff::
  :title: Haiku, Flax
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

Notice how in Flax we pass ``None`` as the second argument to ``ScanBlock`` and ignore
its second output. These represent the inputs/outputs per-step but they are ``None``
because in this case we don't have any.

Initializing each model is the same as in previous examples. In this case,
we will be specifying that we want to use ``5`` layers each with ``64`` features.

.. codediff::
  :title: Haiku, Flax
  :sync:

  def forward(x, training: bool):
    return MLP(64, num_layers=5)(x, training)

  model = hk.transform(forward)

  sample_x = jax.numpy.ones((1, 64))
  params = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )
  ...

  ---

  ...


  model = MLP(64, num_layers=5)

  sample_x = jax.numpy.ones((1, 64))
  variables = model.init(
    random.key(0),
    sample_x, training=False # <== inputs
  )
  params = variables['params']

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

  .. tab-item:: Flax
    :sync: Flax

    .. code-block:: python

      FrozenDict({
          ScanBlock_0: {
              Dense_0: {
                  bias: (5, 64),
                  kernel: (5, 64, 64),
              },
          },
      })

Top-level Haiku functions vs top-level Flax modules
-----------------------------------

In Haiku, it is possible to write the entire model as a single function by using the raw ``hk.{get,set}_{parameter,state}`` to define/access model parameters and states. It very common to write the top-level "Module" as a function instead:

The Flax team recommends a more Module-centric approach that uses `__call__` to define the forward function. The corresponding accessor will be `nn.module.param` and `nn.module.variable` (go to `Handling State <#handling-state>`__ for an explanaion on collections).

.. codediff::
  :title: Haiku, Flax
  :sync:

  def forward(x):


    counter = hk.get_state('counter', shape=[], dtype=jnp.int32, init=jnp.ones)
    multiplier = hk.get_parameter('multiplier', shape=[1,], dtype=x.dtype, init=jnp.ones)
    output = x + multiplier * counter
    hk.set_state("counter", counter + 1)

    return output

  model = hk.transform_with_state(forward)

  params, state = model.init(random.key(0), jax.numpy.ones((1, 64)))

  ---

  class FooModule(nn.Module):
    @nn.compact
    def __call__(self, x):
      counter = self.variable('counter', 'count', lambda: jnp.ones((), jnp.int32))
      multiplier = self.param('multiplier', nn.initializers.ones_init(), [1,], x.dtype)
      output = x + multiplier * counter.value
      if not self.is_initializing():  # otherwise model.init() also increases it
        counter.value += 1
      return output

  model = FooModule()
  variables = model.init(random.key(0), jax.numpy.ones((1, 64)))
  params, counter = variables['params'], variables['counter']
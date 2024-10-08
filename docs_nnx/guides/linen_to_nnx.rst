Evolution from Linen to NNX
##########

This guide will walk you through the differences between Flax Linen and Flax NNX
models, and side-by-side comparisions to help you migrate your code from the Linen API to NNX.

Before this guide, it's highly recommended to read through `The Basics of Flax NNX <https://flax.readthedocs.io/en/latest/nnx_basics.html>`__ to learn about the core concepts and code examples of Flax NNX.

This guide mainly covers converting arbitratry Linen code to NNX. If you want to play it safe and convert your codebase iteratively, check out the guide that allows you to `use NNX and Linen code together <https://flax.readthedocs.io/en/latest/guides/bridge_guide.html>`__


.. testsetup:: Linen, NNX

  import jax
  import jax.numpy as jnp
  import optax
  import flax.linen as nn
  from typing import Any

Basic Module Definition
==========

Both Linen and NNX uses the ``Module`` as the default way to express a neural
library layer.  There are two fundamental difference between Linen and NNX
modules:

* **Stateless vs. stateful**: Linen module instances are stateless: variables are returned from a purely functional ``.init()`` call and managed separately. NNX modules, however, owns its variables as attributes of this Python object.

* **Lazy vs. eager**: Linen modules only allocate space to create variables when they actually see their input. Whereas NNX module instances create their variables the moment they are instantiated, without seeing a sample input.

  * Linen can use the ``@nn.compact`` decorator to define the model in a single method and use shape inference from the input sample, whereas NNX modules generally requests additional shape information to create all parameters during ``__init__``  and separately define the computation in ``__call__``.

.. codediff::
  :title: Linen, NNX
  :sync:

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

  from flax import nnx

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


Variable Creation
==========

To generate the model parameters for a Linen model, you call the ``init`` method with a ``jax.random.key`` plus some sample inputs that the model shall take. The result is a nested dictionary of JAX arrays to be carried around and maintained separately.

In NNX, the model parameters are automatically initialized when the user instantiates the model, and the variables are stored inside the module (or its submodule) as attributes. You still need to give it an RNG key, but the key will be wrapped inside a ``nnx.Rngs`` class and will be stored inside, generating more RNG keys when needed.

If you want to access NNX model parameters in the stateless, dictionary-like fashion for checkpoint saving or model surgery, check out the `NNX split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__.

.. codediff::
  :title: Linen, NNX
  :sync:

  model = Model(256, 10)
  sample_x = jnp.ones((1, 784))
  variables = model.init(jax.random.key(0), sample_x, training=False)
  params = variables["params"]

  assert params['Dense_0']['bias'].shape == (10,)
  assert params['Block_0']['Dense_0']['kernel'].shape == (784, 256)

  ---

  model = Model(784, 256, 10, rngs=nnx.Rngs(0))


  # parameters were already initialized during model instantiation

  assert model.linear.bias.value.shape == (10,)
  assert model.block.linear.kernel.value.shape == (784, 256)


Training Step and Compilation
==========

Now we write a training step and compile it using JAX just-in-time compilation. Note a few differences here:

* Linen uses ``@jax.jit`` to compile the training step, whereas NNX uses ``@nnx.jit``.  ``jax.jit`` only accepts pure stateless arguments, but ``nnx.jit`` allows the arguments to be stateful NNX modules. This greatly reduced the number of lines needed for a train step.

* Similarly, Linen uses ``jax.grad()`` to return a raw dictionary of gradients, wheras NNX can use ``nnx.grad`` to return the gradients of Modules as NNX ``State`` dictionaries. To use regular ``jax.grad`` with NNX you need to use the `NNX split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__.

  * If you are already using Optax optimizers like ``optax.adamw`` (instead of the raw ``jax.tree.map`` computation shown here), check out `nnx.Optimizer example <https://flax.readthedocs.io/en/latest/nnx_basics.html#transforms>`__ for a much more concise way of training and updating your model.

* The Linen train step needs to return a tree of parameters, as the input of the next step. On the other hand, NNX's step doesn't need to return anything, because the ``model`` was already in-place-updated within ``nnx.jit``.

* NNX modules are stateful and automatically tracks a few things within, such as RNG keys and BatchNorm stats. That's why you don't need to explicitly pass an RNG key in on every step. Note that you can use `nnx.reseed <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/rnglib.html#flax.nnx.reseed>`__ to reset its underlying RNG state.

* In Linen, you need to explicitly define and pass in an argument ``training`` to control the behavior of ``nn.Dropout`` (namely, its ``deterministic`` flag, which means random dropout only happens if ``training=True``). In NNX, you can call ``model.train()`` to automatically switch ``nnx.Dropout`` to training mode. Conversely, call ``model.eval()`` to turn off training mode. You can learn more about what this API does at its `API reference <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module.train>`__.


.. codediff::
  :title: Linen, NNX
  :sync:

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

    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    return params

  ---

  model.train() # Sets ``deterministic=False` under the hood for nnx.Dropout

  @nnx.jit
  def train_step(model, inputs, labels):
    def loss_fn(model):
      logits = model(inputs)




      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = nnx.grad(loss_fn)(model)
    _, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    nnx.update(model, nnx.GraphState.merge(params, rest))

.. testcode:: Linen
  :hide:

  train_step(jax.random.key(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  sample_x = jnp.ones((1, 784))
  train_step(model, sample_x, jnp.ones((1,), dtype=jnp.int32))


Collections and Variable Types
==========

One key difference between Linen and NNX APIs is how we group variables into categories. In Linen, we use different collections; in NNX, since all variables shall be top-level Python attributes, you use different variable types.

You can freely create your own variable types as subclasses of ``nnx.Variable``.

For all the built-in Flax Linen layers and collections, NNX already created the corresponding layers and variable type. For example:

 * ``nn.Dense`` creates ``params`` -> ``nnx.Linear`` creates ``nnx.Param``.

 * ``nn.BatchNorm`` creates ``batch_stats`` -> ``nnx.BatchNorm`` creates ``nnx.BatchStats``.

 * ``linen.Module.sow()`` creates ``intermediates`` -> ``nnx.Module.sow()`` creates ``nnx.Intermediates``.

   * You can also simply get the intermediates by assigning it to a module attribute, like ``self.sowed = nnx.Intermediates(x)``. This will be similar to Linen's ``self.variable('intermediates' 'sowed', lambda: x)``.

.. codediff::
  :title: Linen, NNX
  :sync:

  class Block(nn.Module):
    features: int
    def setup(self):
      self.dense = nn.Dense(self.features)
      self.batchnorm = nn.BatchNorm(momentum=0.99)
      self.count = self.variable('counter', 'count',
                                  lambda: jnp.zeros((), jnp.int32))


    @nn.compact
    def __call__(self, x, training: bool):
      x = self.dense(x)
      x = self.batchnorm(x, use_running_average=not training)
      self.count.value += 1
      x = jax.nn.relu(x)
      return x

  x = jax.random.normal(jax.random.key(0), (2, 4))
  model = Block(4)
  variables = model.init(jax.random.key(0), x, training=True)
  variables['params']['dense']['kernel'].shape         # (4, 4)
  variables['batch_stats']['batchnorm']['mean'].shape  # (4, )
  variables['counter']['count']                        # 1

  ---

  class Counter(nnx.Variable): pass

  class Block(nnx.Module):
    def __init__(self, in_features: int , out_features: int, rngs: nnx.Rngs):
      self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
      self.batchnorm = nnx.BatchNorm(
        num_features=out_features, momentum=0.99, rngs=rngs
      )
      self.count = Counter(jnp.array(0))

    def __call__(self, x):
      x = self.linear(x)
      x = self.batchnorm(x)
      self.count += 1
      x = jax.nn.relu(x)
      return x



  model = Block(4, 4, rngs=nnx.Rngs(0))

  model.linear.kernel   # Param(value=...)
  model.batchnorm.mean  # BatchStat(value=...)
  model.count           # Counter(value=...)

If you want to extract certain arrays from the tree of variables, you can access the specific dictionary path in Linen, or use ``nnx.split`` to distinguish the types apart in NNX. The code below is an easier example, and check out `Filter API Guide <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__ for more sophisticated filtering expressions.

.. codediff::
  :title: Linen, NNX
  :sync:

  params, batch_stats, counter = (
    variables['params'], variables['batch_stats'], variables['counter'])
  params.keys()       # ['dense', 'batchnorm']
  batch_stats.keys()  # ['batchnorm']
  counter.keys()      # ['count']

  # ... make arbitrary modifications ...
  # Merge back with raw dict to carry on:
  variables = {'params': params, 'batch_stats': batch_stats, 'counter': counter}

  ---

  graphdef, params, batch_stats, count = nnx.split(
    model, nnx.Param, nnx.BatchStat, Counter)
  params.keys()       # ['batchnorm', 'linear']
  batch_stats.keys()  # ['batchnorm']
  count.keys()        # ['count']

  # ... make arbitrary modifications ...
  # Merge back with ``nnx.merge`` to carry on:
  model = nnx.merge(graphdef, params, batch_stats, count)



Using Multiple Methods
==========

In this section we will take a look at how to use multiple methods in both
frameworks. As an example, we will implement an auto-encoder model with three methods:
``encode``, ``decode``, and ``__call__``.

As before, we define the encoder and decoder layers without having to pass in the
input shape, since the module parameters will be initialized lazily using shape
inference in Linen. In NNX, we must pass in the input shape
since the module parameters will be initialized eagerly without shape inference.

.. codediff::
  :title: Linen, NNX
  :sync:

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

  model = AutoEncoder(256, 784)
  variables = model.init(jax.random.key(0), x=jnp.ones((1, 784)))

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

  model = AutoEncoder(784, 256, 784, rngs=nnx.Rngs(0))


The variable structure is as follows:

.. tab-set::

  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python


      # variables['params']
      {
        decoder: {
            bias: (784,),
            kernel: (256, 784),
        },
        encoder: {
            bias: (256,),
            kernel: (784, 256),
        },
      }

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      # _, params, _ = nnx.split(model, nnx.Param, ...)
      # params
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

To call methods other than ``__call__``, in Linen you still need to use the ``apply`` API, wheras in NNX you can simply call the method directly.

.. codediff::
  :title: Linen, NNX
  :sync:

  z = model.apply(variables, x=jnp.ones((1, 784)), method="encode")

  ---

  z = model.encode(jnp.ones((1, 784)))



Lifted Transforms
==========

Flax APIs provide a set of transforms, which we will refer to as lifted transforms, that wrap JAX transforms in such a way that they can be used with Modules.

Most of the transforms in Linen doesn't change much in NNX. See the next section (Scan over Layers) for a case in which the code differs a lot more.

To begin, we will first define a ``RNNCell`` module that will contain the logic for a single
step of the RNN. We will also define a ``initial_state`` method that will be used to initialize
the state (a.k.a. ``carry``) of the RNN. Like with ``jax.lax.scan``, the ``RNNCell.__call__``
method will be a function that takes the carry and input, and returns the new
carry and output. In this case, the carry and the output are the same.

.. codediff::
  :title: Linen, NNX
  :sync:

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

In Linen, we will use ``nn.scan`` to define a new temporary type that wraps
``RNNCell``. During this process we will also specify instruct ``nn.scan`` to broadcast
the ``params`` collection (all steps share the same parameters) and to not split the
``params`` rng stream (so all steps intialize with the same parameters), and finally
we will specify that we want scan to run over the second axis of the input and stack
the outputs along the second axis as well. We will then use this temporary type immediately
to create an instance of the lifted ``RNNCell`` and use it to create the ``carry`` and
the run the ``__call__`` method which will ``scan`` over the sequence.

In NNX, we define a scan function ``scan_fn`` that will use the ``RNNCell`` defined
in ``__init__`` to scan over the sequence, and explicitly set ``in_axes=(nnx.Carry, None, 1)``,
``Carry`` means that the ``carry`` argument will be the carry, ``None`` means that ``cell`` will
be broadcasted to all steps, and ``1`` means ``x`` will be scanned across axis 1.

.. codediff::
  :title: Linen, NNX
  :sync:

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

  x = jnp.ones((3, 12, 32))
  model = RNN(64)
  variables = model.init(jax.random.key(0), x=jnp.ones((3, 12, 32)))
  y = model.apply(variables, x=jnp.ones((3, 12, 32)))

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

  x = jnp.ones((3, 12, 32))
  model = RNN(x.shape[2], 64, rngs=nnx.Rngs(0))

  y = model(x)



Scan over Layers
==========

In general, lifted transforms of Linen and NNX should look the same. However, NNX lifted transforms are designed to be closer to their lower level JAX counterparts, and thus we throw away some assumptions in certain Linen lifted transforms. This scan-over-layers use case will be a good example to showcase it.

Scan-over-layers is a technique in which, we want run an input through a sequence of N repeated layers, passing the output of each layer as the input to the next layer. This pattern can significantly reduce compilation time for big models. In this example, we will repeat the module ``Block`` for 5 times in a top-level module ``MLP``.

In Linen, we apply a ``nn.scan`` upon the module ``Block`` to create a larger module ``ScanBlock`` that contains 5 ``Block``. It will automatically create a large parameter of shape ``(5, 64, 64)`` at initialization time, and at call time iterate over every ``(64, 64)`` slice for a total of 5 times, like a ``jax.lax.scan`` would.

But if you think closely, there actually isn't any need for ``jax.lax.scan`` operation at initialization time. What happened there is more like a ``jax.vmap`` operation - you are given a ``Block`` that accepts ``(in_dim, out_dim)``, and you "vmap" it over ``num_layers`` of times to create a larger array.

In NNX we take advantage of the fact that model initialization and running code are completely decoupled, and instead use ``nnx.vmap`` to initialize the underlying blocks, and ``nnx.scan`` to run the model input through them.

For more information on NNX transforms, check out the `Transforms Guide <https://flax.readthedocs.build/en/guides/transforms.html>`__.

.. codediff::
  :title: Linen, NNX
  :sync:

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

  model = MLP(64, num_layers=5)

  ---

  class Block(nnx.Module):
    def __init__(self, input_dim, features, rngs):
      self.linear = nnx.Linear(input_dim, features, rngs=rngs)
      self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array):  # No need to require a second input!
      x = self.linear(x)
      x = self.dropout(x)
      x = jax.nn.relu(x)
      return x   # No need to return a second output!

  class MLP(nnx.Module):
    def __init__(self, features, num_layers, rngs):
      @nnx.split_rngs(splits=num_layers)
      @nnx.vmap(in_axes=(0,), out_axes=0)
      def create_block(rngs: nnx.Rngs):
        return Block(features, features, rngs=rngs)

      self.blocks = create_block(rngs)
      self.num_layers = num_layers

    def __call__(self, x):
      @nnx.split_rngs(splits=self.num_layers)
      @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
      def forward(x, model):
        x = model(x)
        return x

      return forward(x, self.blocks)

  model = MLP(64, num_layers=5, rngs=nnx.Rngs(0))


There are a few other details to explain in this example:

* **What is that `nnx.split_rngs` decorator?** NNX transforms are completely agnostic of RNG state, which makes them behave more like JAX transforms but diverge from the Linen transforms that handle RNG state. To regain this functionality, the ``nnx.split_rngs`` decorator allows you to split the ``Rngs`` before passing them to the decorated function and 'lower' them afterwards so they can be used outside.

  * Here we split the RNG keys because ``jax.vmap`` and ``jax.lax.scan`` requires a list of RNG keys if each of its internal operations needs its own key. So for the 5 layers inside ``MLP``, we split and provide 5 different RNG keys from its arguments before going down to the JAX transform.

  * Note that actually ``create_block()`` knows it needs to create 5 layers *precisely because* it sees 5 RNG keys, because ``in_axes=(0,)`` means ``vmap`` will look into the first argument's first dimension to know the size it will map over.

  * Same goes for ``forward()``, which looks at the variables inside the first argument (aka. ``model``) to find out how many times it needs to scan. ``nnx.split_rngs`` here actually splits the RNG state inside the ``model``. (If ``Block`` doesn't have dropout, you don't need the ``nnx.split_rngs`` line because it would not consume any RNG key anyway.)

* **Why the `Block` in NNX doesn't need to take and return that extra dummy value?** This is a requirement from `jax.lax.scan <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__. NNX simplifies this so that now you can choose to ignore the second input/output if you set ``out_axes=nnx.Carry`` instead of the default ``(nnx.Carry, 0)``.

  * This is one of the rare cases in which NNX transforms diverge from JAX transforms API.

This is more lines of code, but it expresses what happened at each time more precisely. Since NNX lifted transforms become way closer to JAX APIs, it's recommended to have a good understanding of the underlying JAX transform before using their NNX versions.

Now take a look at the variable tree on both sides:

.. tab-set::

  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python

      # variables = model.init(key, x=jnp.ones((1, 64)), training=True)
      # variables['params']
      {
        ScanBlock_0: {
          Dense_0: {
            bias: (5, 64),
            kernel: (5, 64, 64),
          },
        },
      }

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      # _, params, _ = nnx.split(model, nnx.Param, ...)
      # params
      State({
        'blocks': {
          'linear': {
            'bias': VariableState(type=Param, value=(5, 64)),
            'kernel': VariableState(type=Param, value=(5, 64, 64))
          }
        }
      })


Using ``TrainState`` in NNX
==========

Flax offered a convenient ``TrainState`` dataclass to bundle the model,
parameters and optimizer. This is not really necessary in NNX era, but this section we would show how to construct your NNX code around it, for any backward compatibility needs.

In NNX, we must first call ``nnx.split`` on the model to get the
separated ``GraphDef`` and ``State`` objects. We can pass in ``nnx.Param`` to filter
all trainable parameters into a single ``State``, and pass in ``...`` for the remaining
variables. We also need to subclass ``TrainState`` to add a field for the other variables.
We can then pass in ``GraphDef.apply`` as the apply function, ``State`` as the parameters
and other variables and an optimizer as arguments to the ``TrainState`` constructor.
One thing to note is that ``GraphDef.apply`` will take in ``State``'s as arguments and
return a callable function. This function can be called on the inputs to output the
model's logits, as well as updated ``GraphDef`` and ``State`` objects. Notice we also use
``@jax.jit`` since we aren't passing in NNX modules into ``train_step``.

.. codediff::
  :title: Linen, NNX
  :sync:

  from flax.training import train_state

  sample_x = jnp.ones((1, 784))
  model = nn.Dense(features=10)
  params = model.init(jax.random.key(0), sample_x)['params']




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
        inputs, # <== inputs
        rngs={'dropout': key}
      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn)(state.params)


    state = state.apply_gradients(grads=grads)

    return state

  ---

  from flax.training import train_state

  model = nnx.Linear(784, 10, rngs=nnx.Rngs(0))
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

.. testcode:: Linen
  :hide:

  train_step(jax.random.key(0), state, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  sample_x = jnp.ones((1, 784))
  train_step(state, sample_x, jnp.ones((1,), dtype=jnp.int32))
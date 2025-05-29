Evolution from Flax Linen to NNX
################################

This guide demonstrates the differences between Flax Linen and Flax NNX models, providing side-by-side example code to help you migrate to the Flax NNX API from Flax Linen.

This document mainly teaches how to convert arbitrary Flax Linen code to Flax NNX. If you want to play it “safe” and convert your codebase iteratively, check out the `Use Flax NNX and Linen together via nnx.bridge <https://flax.readthedocs.io/en/latest/guides/bridge_guide.html>`__ guide.

To get the most out of this guide, it is highly recommended to get go through `Flax NNX basics <https://flax.readthedocs.io/en/latest/nnx_basics.html>`__ document, which covers the :class:`nnx.Module<flax.nnx.Module>` system, `Flax transformations <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__, and the `Functional API <https://flax.readthedocs.io/en/latest/nnx_basics.html#the-flax-functional-api>`__ with examples.

.. testsetup:: Linen, NNX

  import jax
  import jax.numpy as jnp
  import optax
  import flax.linen as nn
  from typing import Any

Basic ``Module`` definition
===========================

Both Flax Linen and Flax NNX use the ``Module`` class as the default unit to express a neural network library layer. In the example below, you first create a ``Block`` (by subclassing ``Module``) composed of one linear layer with dropout and a ReLU activation function; then you use it as a sub-``Module`` when creating a ``Model`` (also by subclassing ``Module``), which is made up of ``Block`` and a linear layer.

There are two fundamental differences between Flax Linen and Flax NNX ``Module`` objects:

* **Stateless vs. stateful**: A ``flax.linen.Module`` (``nn.Module``) instance is stateless - the variables are returned from a purely functional ``Module.init()`` call and managed separately. A :class:`flax.nnx.Module`, however, owns its variables as attributes of this Python object.

* **Lazy vs. eager**: A ``flax.linen.Module`` only allocates space to create variables when they actually see their input (lazy). A :class:`flax.nnx.Module` instance creates variables the moment they are instantiated before seeing a sample input (eager).

* Flax Linen can use the ``@nn.compact`` decorator to define the model in a single method, and use shape inference from the input sample. A Flax NNX ``Module`` generally requests additional shape information to create all parameters during ``__init__`` , and separately defines the computation in the ``__call__`` method.

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


Variable creation
=================

Next, let’s discuss instantiating the model and initializing its parameters:

* To generate model parameters for a Flax Linen model, you call the ``flax.linen.Module.init`` (``nn.Module.init``) method with a ``jax.random.key`` (`doc <https://jax.readthedocs.io/en/latest/random-numbers.html>`__) plus some sample inputs that the model shall take. This results in a nested dictionary of `JAX Arrays <https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array>`__ (``jax.Array`` data types) to be carried around and maintained separately.

* In Flax NNX, the model parameters are automatically initialized when you instantiate the model, and the variables (:class:`nnx.Variable<flax.nnx.Variable>` objects) are stored inside the :class:`nnx.Module<flax.nnx.Module>` (or its sub-``Module``) as attributes. You still need to provide it with a `pseudorandom number generator (PRNG) <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ key, but that key will be wrapped inside an :class:`nnx.Rngs<flax.nnx.Rngs>` class and stored inside, generating more PRNG keys when needed.

If you want to access Flax NNX model parameters in the stateless, dictionary-like fashion for checkpoint saving or model surgery, check out the `Flax NNX split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__ (:func:`nnx.split<flax.nnx.split>` / :func:`nnx.merge<flax.nnx.merge>`).

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


  # Parameters were already initialized during model instantiation.

  assert model.linear.bias.value.shape == (10,)
  assert model.block.linear.kernel.value.shape == (784, 256)


Training step and compilation
=============================

Now, let’s proceed to writing a training step and compiling it using `JAX just-in-time compilation <https://jax.readthedocs.io/en/latest/jit-compilation.html>`__. Below are certain differences between Flax Linen and Flax NNX approaches.

Compiling the training step:

* Flax Linen uses ``@jax.jit`` - a `JAX transform <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ - to compile the training step.
* Flax NNX uses :meth:`@nnx.jit<flax.nnx.jit>` - a `Flax NNX transform <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__ (one of several transform APIs that behave similarly to JAX transforms, but also `work well with Flax NNX objects <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__). So, while ``jax.jit`` only accepts functions pure stateless arguments, ``nnx.jit`` allows the arguments to be stateful NNX Modules. This greatly reduced the number of lines needed for a train step.

Taking gradients:

* Similarly, Flax Linen uses ``jax.grad`` (a JAX transform for `automatic differentiation <https://jax.readthedocs.io/en/latest/automatic-differentiation.html#taking-gradients-with-jax-grad>`__) to return a raw dictionary of gradients.
* Flax NNX uses :meth:`nnx.grad<flax.nnx.grad>` (a Flax NNX transform) to return the gradients of NNX Modules as :class:`nnx.State<flax.nnx.State>` dictionaries. If you want to use regular ``jax.grad`` with Flax NNX you need to use the `Flax NNX split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__.

Optimizers:

* If you are already using `Optax <https://optax.readthedocs.io/>`__ optimizers like ``optax.adamw`` (instead of the raw ``jax.tree.map`` computation shown here) with Flax Linen, check out the :class:`nnx.Optimizer<flax.nnx.Optimizer>` example in the `Flax NNX basics <https://flax.readthedocs.io/en/latest/nnx_basics.html#transforms>`__ guide for a much more concise way of training and updating your model.

Model updates during each training step:

* The Flax Linen training step needs to return a `pytree <https://jax.readthedocs.io/en/latest/working-with-pytrees.html>`__ of parameters as the input of the next step.
* The Flax NNX training step doesn't need to return anything, because the ``model`` was already updated in-place within :meth:`nnx.jit<flax.nnx.jit>`.
* In addition, :class:`nnx.Module<flax.nnx.Module>` objects are stateful, and ``Module`` automatically tracks several things within it, such as PRNG keys and ``BatchNorm`` stats. That is why you don't need to explicitly pass an PRNG key in on every step. Also note that you can use :meth:`nnx.reseed<flax.nnx.reseed>` to reset its underlying PRNG state.

Dropout behavior:

* In Flax Linen, you need to explicitly define and pass in the ``training`` argument to control the behavior of ``flax.linen.Dropout`` (``nn.Dropout``), namely, its ``deterministic`` flag, which means random dropout only happens if ``training=True``.
* In Flax NNX, you can call ``model.train()`` (:meth:`flax.nnx.Module.train`) to automatically switch :class:`nnx.Dropout<flax.nnx.Dropout>` to the training mode. Conversely, you can call ``model.eval()`` (:meth:`flax.nnx.Module.eval`) to turn off the training mode. You can learn more about what ``nnx.Module.train`` does in its `API reference <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module.train>`__.


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
    nnx.update(model, nnx.merge_state(params, rest))

.. testcode:: Linen
  :hide:

  train_step(jax.random.key(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: NNX
  :hide:

  sample_x = jnp.ones((1, 784))
  train_step(model, sample_x, jnp.ones((1,), dtype=jnp.int32))


Collections and variable types
==============================

One key difference between Flax Linen and NNX APIs is how they group variables into categories. Flax Linen uses different collections, while Flax NNX, since all variables shall be top-level Python attributes, you use different variable types.

In Flax NNX, you can freely create your own variable types as subclasses of ``nnx.Variable``.

For all the built-in Flax Linen layers and collections, Flax NNX already creates the corresponding layers and variable types. For example:

* ``flax.linen.Dense`` (``nn.Dense``) creates ``params`` -> :class:`nnx.Linear<flax.nnx.Linear>` creates :class:nnx.Param<flax.nnx.Param>`.
* ``flax.linen.BatchNorm`` (``nn.BatchNorm``) creates ``batch_stats`` -> :class:`nnx.BatchNorm<flax.nnx.BatchNorm>` creates :class:`nnx.BatchStats<flax.nnx.BatchStats>`.
* ``flax.linen.Module.sow()`` creates ``intermediates`` -> :class:`nnx.Module.sow()<flax.nnx.Module.sow>` creates :class:`nnx.Intermediaries<flax.nnx.Intermediates>`.
* In Flax NNX, you can also simply obtain the intermediates by assigning it to an ``nnx.Module`` attribute - for example, ``self.sowed = nnx.Intermediates(x)``. This will be similar to Flax Linen's ``self.variable('intermediates' 'sowed', lambda: x)``.

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

If you want to extract certain arrays from the pytree of variables:

* In Flax Linen, you can access the specific dictionary path.
* In Flax NNX, you can use :func:`nnx.split<flax.nnx.split>` to distinguish the types apart in Flax NNX. The code below is a simple example that splits up the variables by their types - check out the `Flax NNX Filters <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__ guide for more sophisticated filtering expressions.

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



Using multiple methods
======================

In this section you will learn how to use multiple methods in both Flax Linen and Flax NNX. As an example, you will implement an auto-encoder model with three methods: ``encode``, ``decode``, and ``__call__``.

Defining the encoder and decoder layers:

* In Flax Linen, as before, define the layers without having to pass in the input shape, since the ``flax.linen.Module`` parameters will be initialized lazily using shape inference.
* In Flax NNX, you must pass in the input shape since the :class:`nnx.Module<flax.nnx.Module>` parameters will be initialized eagerly without shape inference.

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
      {
        'decoder': {
          'bias': VariableState(type=Param, value=(784,)),
          'kernel': VariableState(type=Param, value=(256, 784))
        },
        'encoder': {
          'bias': VariableState(type=Param, value=(256,)),
          'kernel': VariableState(type=Param, value=(784, 256))
        }
      }

To call methods other than ``__call__``:

* In Flax Linen, you still need to use the ``apply`` API.
* In Flax NNX, you can simply call the method directly.

.. codediff::
  :title: Linen, NNX
  :sync:

  z = model.apply(variables, x=jnp.ones((1, 784)), method="encode")

  ---

  z = model.encode(jnp.ones((1, 784)))



Transformations
===============

Both Flax Linen and `Flax NNX transformations <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__ provide their own set of transforms that wrap `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ in a way that they can be used with ``Module`` objects.

Most of the transforms in Flax Linen, such as ``grad`` or ``jit``, don't change much in Flax NNX. But, for example, if you try to do ``scan`` over layers, as described in the next section, the code differs by a lot.

Let’s start with an example:

* First, define an ``RNNCell`` ``Module`` that will contain the logic for a single step of the RNN.
* Define a ``initial_state`` method that will be used to initialize the state (a.k.a. ``carry``) of the RNN. Like with ``jax.lax.scan`` (`API doc <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__), the ``RNNCell.__call__`` method will be a function that takes the carry and input, and returns the new carry and output. In this case, the carry and the output are the same.

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

Next, define an ``RNN`` ``Module`` that will contain the logic for the entire RNN.

In Flax Linen:

* You will use ``flax.linen.scan`` (``nn.scan``) to define a new temporary type that wraps ``RNNCell``. During this process you will also: 1) instruct ``nn.scan`` to broadcast the ``params`` collection (all steps share the same parameters) and to not split the ``params`` PRNG stream (so that all steps initialize with the same parameters); and, finally, 2) specify that you want scan to run over the second axis of the input and stack outputs along the second axis as well.
* You will then use this temporary type immediately to create an instance of the “lifted” ``RNNCell`` and use it to create the ``carry``, and the run the ``__call__`` method, which will ``scan`` over the sequence.

In Flax NNX:

* You will create a ``scan`` function (``scan_fn``) that will use the ``RNNCell`` defined in ``__init__`` to scan over the sequence, and explicitly set ``in_axes=(nnx.Carry, None, 1)``. ``nnx.Carry`` means that the ``carry`` argument will be the carry, ``None`` means that ``cell`` will be broadcasted to all steps, and ``1`` means ``x`` will be scanned across axis `1`.

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



Scan over layers
================

In general, transforms of Flax Linen and Flax NNX should look the same. However, `Flax NNX transforms <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__ are designed to be closer to their lower-level `JAX counterparts <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__, and thus we throw away some assumptions in certain Linen lifted transforms. This scan-over-layers use case will be a good example to showcase it.

Scan-over-layers is a technique where you run an input through a sequence of N repeated layers, passing the output of each layer as the input to the next layer. This pattern can significantly reduce compilation time for large models. In the example below, you will repeat the ``Block`` ``Module`` 5 times in the top-level ``MLP`` ``Module``.

* In Flax Linen, you apply the ``flax.linen.scan`` (``nn.scan``) transforms upon the ``Block`` ``nn.Module`` to create a larger ``ScanBlock`` ``nn.Module`` that contains 5 ``Block`` ``nn.Module`` objects. It will automatically create a large parameter of shape ``(5, 64, 64)`` at initialization time, and iterate over at call time every ``(64, 64)`` slice for a total of 5 times, like a ``jax.lax.scan`` (`API doc <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__) would.
* Up close, in the logic of this model there actually is no need for the ``jax.lax.scan`` operation at initialization time. What happens there is more like a ``jax.vmap`` operation - you are given a ``Block`` sub-``Module`` that accepts ``(in_dim, out_dim)``, and you "vmap" it over ``num_layers`` of times to create a larger array.
* In Flax NNX, you take advantage of the fact that model initialization and running code are completely decoupled, and instead use the :func:`nnx.vmap<flax.nnx.vmap>` transform to initialize the underlying ``Block`` parameters, and the :func:`nnx.scan<flax.nnx.scan>` transform to run the model input through them.

For more information on Flax NNX transforms, check out the `Transforms guide <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__.

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


There are a few other details to explain in the Flax NNX example above:

* **The `@nnx.split_rngs` decorator:** Flax NNX transforms are completely agnostic of PRNG state, which makes them behave more like JAX transforms but diverge from the Flax Linen transforms that handle PRNG state. To regain this functionality, the ``nnx.split_rngs`` decorator allows you to split the ``nnx.Rngs`` before passing them to the decorated function and 'lower' them afterwards, so they can be used outside.

  * Here, you split the PRNG keys because ``jax.vmap`` and ``jax.lax.scan`` require a list of PRNG keys if each of its internal operations needs its own key. So for the 5 layers inside the ``MLP``, you split and provide 5 different PRNG keys from its arguments before going down to the JAX transform.

  * Note that actually ``create_block()`` knows it needs to create 5 layers *precisely because* it sees 5 PRNG keys, because ``in_axes=(0,)`` indicates that ``vmap`` will look into the first argument's first dimension to know the size it will map over.

  * Same goes for ``forward()``, which looks at the variables inside the first argument (aka. ``model``) to find out how many times it needs to scan. ``nnx.split_rngs`` here actually splits the PRNG state inside the ``model``. (If the ``Block`` ``Module`` doesn't have dropout, you don't need the :meth:`nnx.split_rngs<flax.nnx.split_rngs>` line as it would not consume any PRNG key anyway.)

* **Why the Block Module in Flax NNX doesn't need to take and return that extra dummy value:** This is a requirement from ``jax.lax.scan`` `(API doc <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__. Flax NNX simplifies this, so that you can now choose to ignore the second output if you set ``out_axes=nnx.Carry`` instead of the default ``(nnx.Carry, 0)``.

  * This is one of the rare cases where Flax NNX transforms diverge from the `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ APIs.

There are more lines of code in the Flax NNX example above, but they express what happens at each time more precisely. Since Flax NNX transforms become way closer to the JAX transform APIs, it is recommended to have a good understanding of the underlying `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ before using their `Flax NNX equivalents <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__

Now inspect the variable pytree on both sides:

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
      {
        'blocks': {
          'linear': {
            'bias': VariableState(type=Param, value=(5, 64)),
            'kernel': VariableState(type=Param, value=(5, 64, 64))
          }
        }
      }


Using ``TrainState`` in Flax NNX
================================

Flax Linen has a convenient ``TrainState`` data class to bundle the model,
parameters and optimizer. In Flax NNX, this is not really necessary. In this section,
you will learn how to construct your Flax NNX code around ``TrainState`` for any backward
compatibility needs.

In Flax NNX:

* You must first call :meth:`nnx.split<flax.linen.split>` on the model to get the
  separate :class:`nnx.GraphDef<flax.nnx.GraphDef>` and :class:`nnx.State<flax.nnx.State>`
  objects.
* You can pass in :class:`nnx.Param<flax.nnx.Param>` to filter all trainable parameters
  into a single :class:`nnx.State<flax.nnx.State>`, and pass in ``...`` for the remaining
  variables.
* You also need to subclass ``TrainState`` to add a field for the other variables.
* Then, you can pass in :meth:`nnx.GraphDef.apply<flax.nnx.GraphDef.apply>` as the ``apply`` function,
  :class:`nnx.State<flax.nnx.State>` as the parameters and other variables, and an optimizer as arguments to the
  ``TrainState`` constructor.

Note that :class:`nnx.GraphDef.apply<flax.nnx.GraphDef.apply>` will take in :class:`nnx.State<flax.nnx.State>` objects as arguments and
return a callable function. This function can be called on the inputs to output the
model's logits, as well as the updated :class:`nnx.GraphDef<flax.nnx.GraphDef>` and :class:`nnx.State<flax.nnx.State>` objects.
Notice below the use of ``@jax.jit`` since you aren't passing in Flax NNX Modules into
the ``train_step``.

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



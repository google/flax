Migrating from Haiku to Flax
############################

This guide demonstrates the differences between Haiku and Flax NNX models, providing side-by-side example code to help you migrate to the Flax NNX API from Haiku.

If you are new to Flax NNX, make sure you become familiarized with `Flax NNX basics <https://flax.readthedocs.io/en/latest/nnx_basics.html>`__, which covers the :class:`nnx.Module<flax.nnx.Module>` system, `Flax transformations <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__, and the `Functional API <https://flax.readthedocs.io/en/latest/nnx_basics.html#the-flax-functional-api>`__ with examples.

Let’s start with some imports.

.. testsetup:: Haiku, Flax NNX

  import jax
  import jax.numpy as jnp
  import optax
  from typing import Any


Basic Module definition
=======================

Both Haiku and Flax use the ``Module`` class as the default unit to express a neural network library layer. For example, to create a one-layer network with dropout and a ReLU activation function, you:

* First, create a ``Block`` (by subclassing ``Module``) composed of one linear layer with dropout and a ReLU activation function.
* Then, use ``Block`` as a sub-``Module`` when creating a ``Model`` (also by subclassing ``Module``), which is made up of ``Block`` and a linear layer.

There are two fundamental differences between Haiku and Flax ``Module`` objects:

* **Stateless vs. stateful**:

  * A ``haiku.Module`` instance is stateless. This means, the variables are returned from a purely functional ``Module.init()`` call and managed separately.
  * A :class:`flax.nnx.Module`, however, owns its variables as attributes of this Python object.

* **Lazy vs. eager**:

  * A ``haiku.Module`` only allocates space to create variables when they actually see the input when the user calls the model (lazy).
  * A ``flax.nnx.Module`` instance creates variables the moment they are instantiated, before seeing a sample input (eager).


.. codediff::
  :title: Haiku, Flax NNX
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

This section is about instantiating a model and initializing its parameters.

* To generate model parameters for a Haiku model, you need to put it inside a forward function and use ``haiku.transform`` to make it purely functional. This results in a nested dictionary of `JAX Arrays <https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array>`__ (``jax.Array`` data types) to be carried around and maintained separately.

* In Flax NNX, the model parameters are automatically initialized when you instantiate the model, and the variables (:class:`nnx.Variable<flax.nnx.Variable>` objects) are stored inside the :class:`nnx.Module<flax.nnx.Module>` (or its sub-Module) as attributes. You still need to provide it with a `pseudorandom number generator (PRNG) <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ key, but that key will be wrapped inside an :class:`nnx.Rngs<flax.nnx.Rngs>` class and stored inside, generating more PRNG keys when needed.

If you want to access Flax model parameters in the stateless, dictionary-like fashion for checkpoint saving or model surgery, check out the `Flax NNX split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__ (:func:`nnx.split<flax.nnx.split>` / :func:`nnx.merge<flax.nnx.merge>`).


.. codediff::
  :title: Haiku, Flax NNX
  :sync:

  def forward(x, training: bool):
    return Model(256, 10)(x, training)

  model = hk.transform(forward)
  sample_x = jnp.ones((1, 784))
  params = model.init(jax.random.key(0), sample_x, training=False)


  assert params['model/linear']['b'].shape == (10,)
  assert params['model/block/linear']['w'].shape == (784, 256)

  ---

  ...


  model = Model(784, 256, 10, rngs=nnx.Rngs(0))


  # Parameters were already initialized during model instantiation.

  assert model.linear.bias.value.shape == (10,)
  assert model.block.linear.kernel.value.shape == (784, 256)

Training step and compilation
=============================

This section covers writing a training step and compiling it using the `JAX just-in-time compilation <https://jax.readthedocs.io/en/latest/jit-compilation.html>`__.

When compiling the training step:

* Haiku uses ``@jax.jit`` - a `JAX transformation <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ - to compile a purely functional training step.
* Flax NNX uses :meth:`@nnx.jit<flax.nnx.jit>` - a `Flax NNX transformation <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__ (one of several transform APIs that behave similarly to JAX transforms, but also `work well with Flax objects <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__). While ``jax.jit`` only accepts functions with pure stateless arguments, ``flax.nnx.jit`` allows the arguments to be stateful Modules. This greatly reduces the number of lines needed for a train step.

When taking gradients:

* Similarly, Haiku uses ``jax.grad`` (a JAX transformation for `automatic differentiation <https://jax.readthedocs.io/en/latest/automatic-differentiation.html#taking-gradients-with-jax-grad>`__) to return a raw dictionary of gradients.
* Meanwhile, Flax NNX uses :meth:`flax.nnx.grad<flax.nnx.grad>` (a Flax NNX transformation) to return the gradients of Flax NNX Modules as :class:`flax.nnx.State<flax.nnx.State>` dictionaries. If you want to use regular ``jax.grad`` with Flax NNX, you need to use the `split/merge API <https://flax.readthedocs.io/en/latest/nnx_basics.html#state-and-graphdef>`__.

For optimizers:

* If you are already using `Optax <https://optax.readthedocs.io/>`__ optimizers like ``optax.adamw`` (instead of the raw ``jax.tree.map`` computation shown here) with Haiku, check out the :class:`flax.nnx.Optimizer<flax.nnx.Optimizer>` example in the `Flax basics <https://flax.readthedocs.io/en/latest/nnx_basics.html#transforms>`__ guide for a much more concise way of training and updating your model.

Model updates during each training step:

* The Haiku training step needs to return a `JAX pytree <https://jax.readthedocs.io/en/latest/working-with-pytrees.html>`__ of parameters as the input of the next step.
* The Flax NNX training step does not need to return anything, because the ``model`` was already updated in-place within :meth:`nnx.jit<flax.nnx.jit>`.
* In addition, :class:`nnx.Module<flax.nnx.Module>` objects are stateful, and ``Module`` automatically tracks several things within it, such as PRNG keys and ``flax.nnx.BatchNorm`` stats. That is why you don't need to explicitly pass a PRNG key in at every step. Also note that you can use :meth:`flax.nnx.reseed<flax.nnx.reseed>` to reset its underlying PRNG state.

The dropout behavior:

* In Haiku, you need to explicitly define and pass in the ``training`` argument to toggle ``haiku.dropout`` and make sure that random dropout only happens if ``training=True``.
* In Flax NNX, you can call ``model.train()`` (:meth:`flax.nnx.Module.train`) to automatically switch :class:`flax.nnx.Dropout<flax.nnx.Dropout>` to the training mode. Conversely, you can call ``model.eval()`` (:meth:`flax.nnx.Module.eval`) to turn off the training mode. You can learn more about what ``flax.nnx.Module.train`` does in its `API reference <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module.train>`__.

.. codediff::
  :title: Haiku, Flax NNX
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

  model.train() # set deterministic=False

  @nnx.jit
  def train_step(model, inputs, labels):
    def loss_fn(model):
      logits = model(

        inputs, # <== inputs

      )
      return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = nnx.grad(loss_fn)(model)
    _, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    nnx.update(model, nnx.merge_state(params, rest))

.. testcode:: Haiku
  :hide:

  train_step(jax.random.key(0), params, sample_x, jnp.ones((1,), dtype=jnp.int32))

.. testcode:: Flax NNX
  :hide:

  sample_x = jnp.ones((1, 784))
  train_step(model, sample_x, jnp.ones((1,), dtype=jnp.int32))



Handling non-parameter states
=============================

Haiku makes a distinction between trainable parameters and all other data ("states") that the model tracks. For example, the batch stats used in batch norm is considered a state. Models with states needs to be transformed with ``hk.transform_with_state`` so that their ``.init()`` returns both params and states.

In Flax, there isn't such a strong distinction - they are all subclasses of ``nnx.Variable`` and seen by a module as its attributes. Parameters are instances of a subclass called ``nnx.Param``, and batch stats can be of another subclass called ``nnx.BatchStat``. You can use :func:`nnx.split<flax.nnx.split>` to quickly extract all data of a certain variable type.

Let's see an example of this by taking the ``Block`` definition above but replace dropout with ``BatchNorm``.

.. codediff::
  :title: Haiku, Flax NNX
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

  def forward(x, training: bool):
    return Model(256, 10)(x, training)
  model = hk.transform_with_state(forward)

  sample_x = jnp.ones((1, 784))
  params, batch_stats = model.init(jax.random.key(0), sample_x, training=True)

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



  model = Block(4, 4, rngs=nnx.Rngs(0))

  model.linear.kernel   # Param(value=...)
  model.batchnorm.mean  # BatchStat(value=...)


Flax takes the difference of trainable params and other data into account. ``nnx.grad`` will only take gradients on the ``nnx.Param`` variables, thus skipping the ``batchnorm`` arrays automatically. Therefore, the training step will look the same for Flax NNX with this model.


Using multiple methods
======================

In this section you will learn how to use multiple methods in Haiku and Flax. As an example, you will implement an auto-encoder model with three methods: ``encode``, ``decode``, and ``__call__``.

In Haiku, you need to use ``hk.multi_transform`` to explicitly define how the model shall be initialized and what methods (``encode`` and ``decode`` here) it can call. Note that you still need to define a ``__call__`` that activates both layers for the lazy initialization of all model parameters.

In Flax, it's simpler as you initialized parameters in ``__init__`` and the :class:`nnx.Module<flax.nnx.Module>` methods ``encode`` and ``decode`` can be used directly.

.. codediff::
  :title: Haiku, Flax NNX
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

  def forward():
    module = AutoEncoder(256, 784)
    init = lambda x: module(x)
    return init, (module.encode, module.decode)

  model = hk.multi_transform(forward)
  params = model.init(jax.random.key(0), x=jnp.ones((1, 784)))

  ---

  class AutoEncoder(nnx.Module):

    def __init__(self, in_dim: int, embed_dim: int, output_dim: int, rngs):

      self.encoder = nnx.Linear(in_dim, embed_dim, rngs=rngs)
      self.decoder = nnx.Linear(embed_dim, output_dim, rngs=rngs)

    def encode(self, x):
      return self.encoder(x)

    def decode(self, x):
      return self.decoder(x)











  model = AutoEncoder(784, 256, 784, rngs=nnx.Rngs(0))
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

  .. tab-item:: Flax NNX
    :sync: Flax NNX

    .. code-block:: python

      _, params, _ = nnx.split(model, nnx.Param, ...)

      params
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


To call those custom methods:

* In Haiku, you need to decouple the `.apply` function to extract your method before calling it.
* In Flax, you can simply call the method directly.

.. codediff::
  :title: Haiku, Flax NNX
  :sync:

  encode, decode = model.apply
  z = encode(params, None, x=jnp.ones((1, 784)))

  ---

  ...
  z = model.encode(jnp.ones((1, 784)))



Transformations
=======================

Both Haiku and `Flax transformations <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__ provide their own set of transforms that wrap `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ in a way that they can be used with ``Module`` objects.

For more information on Flax transforms, check out the `Transforms guide <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__.

Let's start with an example:

* First, define an ``RNNCell`` ``Module`` that will contain the logic for a single step of the RNN.
* Define a ``initial_state`` method that will be used to initialize the state (a.k.a. ``carry``) of the RNN. Like with ``jax.lax.scan`` (`API doc <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__), the ``RNNCell.__call__`` method will be a function that takes the carry and input, and returns the new carry and output. In this case, the carry and the output are the same.


.. codediff::
  :title: Haiku, Flax NNX
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

Next, we will define a ``RNN`` Module that will contain the logic for the entire RNN. In both cases, we use the library's ``scan`` call to run the ``RNNCell`` over the input sequence.

The only difference is that Flax ``nnx.scan`` allows you to specify which axis to repeat over in arguments ``in_axes`` and ``out_axes``, which will be forwarded to the underlying `jax.lax.scan<https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__, whereas in Haiku you need to transpose the input and output explicitly.

.. codediff::
  :title: Haiku, Flax NNX
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


Scan over layers
=======================

Most Haiku transforms should look similar with Flax, since they all wraps their JAX counterparts, but the scan-over-layers use case is an exception.

Scan-over-layers is a technique where you run an input through a sequence of N repeated layers, passing the output of each layer as the input to the next layer. This pattern can significantly reduce compilation time for large models. In the example below, you will repeat the ``Block`` ``Module`` 5 times in the top-level ``MLP`` ``Module``.

In Haiku, we define the ``Block`` Module as usual, and then inside ``MLP`` we will
use ``hk.experimental.layer_stack`` over a ``stack_block`` function to create a stack
of ``Block`` Modules. The same code will create 5 layers of parameters in initialization time, and run the input through them in call time.

In Flax, model initialization and calling code are completely decoupled, so we use the :func:`nnx.vmap<flax.nnx.vmap>` transform to initialize the underlying ``Block`` parameters, and the :func:`nnx.scan<flax.nnx.scan>` transform to run the model input through them.

.. codediff::
  :title: Haiku, Flax NNX
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

  def forward(x, training: bool):
    return MLP(64, num_layers=5)(x, training)
  model = hk.transform(forward)

  sample_x = jnp.ones((1, 64))
  params = model.init(jax.random.key(0), sample_x, training=False)

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


There are a few other details to explain in the Flax example above:

* **The `@nnx.split_rngs` decorator:** Flax transforms, like their JAX counterparts, are completely agnostic of the PRNG state and rely on input for PRNG keys. The ``nnx.split_rngs`` decorator allows you to split the ``nnx.Rngs`` before passing them to the decorated function and 'lower' them afterwards, so they can be used outside.

  * Here, you split the PRNG keys because ``jax.vmap`` and ``jax.lax.scan`` require a list of PRNG keys if each of its internal operations needs its own key. So for the 5 layers inside the ``MLP``, you split and provide 5 different PRNG keys from its arguments before going down to the JAX transform.

  * Note that actually ``create_block()`` knows it needs to create 5 layers *precisely because* it sees 5 PRNG keys, because ``in_axes=(0,)`` indicates that ``vmap`` will look into the first argument's first dimension to know the size it will map over.

  * Same goes for ``forward()``, which looks at the variables inside the first argument (aka. ``model``) to find out how many times it needs to scan. ``nnx.split_rngs`` here actually splits the PRNG state inside the ``model``. (If the ``Block`` ``Module`` doesn't have dropout, you don't need the :meth:`nnx.split_rngs<flax.nnx.split_rngs>` line as it would not consume any PRNG key anyway.)

* **Why the Block Module in Flax doesn't need to take and return that extra dummy value:** ``jax.lax.scan`` `(API doc <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html>`__ requires its function to return two inputs - the carry and the stacked output. In this case, we didn't use the latter. Flax simplifies this, so that you can now choose to ignore the second output if you set ``out_axes=nnx.Carry`` instead of the default ``(nnx.Carry, 0)``.

  * This is one of the rare cases where Flax NNX transforms diverge from the `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ APIs.

There are more lines of code in the Flax example above, but they express what happens at each time more precisely. Since Flax transforms become way closer to the JAX transform APIs, it is recommended to have a good understanding of the underlying `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ before using their `Flax NNX equivalents <https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html>`__

Now inspect the variable pytree on both sides:

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

  .. tab-item:: Flax NNX
    :sync: Flax NNX

    .. code-block:: python

      _, params, _ = nnx.split(model, nnx.Param, ...)

      params
      {
        'blocks': {
          'linear': {
            'bias': VariableState(type=Param, value=(5, 64)),
            'kernel': VariableState(type=Param, value=(5, 64, 64))
          }
        }
      }


Top-level Haiku functions vs top-level Flax modules
=======================

In Haiku, it is possible to write the entire model as a single function by using
the raw ``hk.{get,set}_{parameter,state}`` to define/access model parameters and
states. It is very common to write the top-level "Module" as a function instead.

The Flax team recommends a more Module-centric approach that uses ``__call__`` to
define the forward function. In Flax modules, the parameters and variables can
be set and accessed as normal using regular Python class semantics.

.. codediff::
  :title: Haiku, Flax NNX
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

  params, state = model.init(jax.random.key(0), jnp.ones((1, 64)))

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





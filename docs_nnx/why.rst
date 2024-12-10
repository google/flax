Why Flax NNX?
=============

In 2020, the Flax team released the Flax Linen API to support modeling research on JAX, with a focus on scaling
and performance. We have learned a lot from users since then. The team introduced certain ideas that have proven to be beneficial to users, such as:

* Organizing variables into `collections <https://flax.readthedocs.io/en/latest/glossary.html#term-Variable-collections>`_.
* Automatic and efficient `pseudorandom number generator (PRNG) management <https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences>`_.
* `Variable metadata <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/spmd.html#flax.linen.with_partitioning>`_
  for `Single Program Multi Data (SPMD) <https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD>`_ annotations, optimizer metadata, and other use cases.

One of the choices the Flax team made was to use functional (``compact``) semantics for neural network programming via lazy initialization of parameters.
This made for concise implementation code and aligned the Flax Linen API with Haiku.

However, this also meant that the semantics of Modules and variables in Flax were non-Pythonic and often surprising. It also led to implementation
complexity and obscured the core ideas of `transformations (transforms) <https://jax.readthedocs.io/en/latest/glossary.html#term-transformation>`_ on neural networks.

.. testsetup:: Linen, NNX

    import jax
    from jax import random, numpy as jnp
    from flax import nnx
    import flax.linen as nn

Introducing Flax NNX
--------------------

Fast forward to 2024, the Flax team developed Flax NNX - an attempt to retain the features that made Flax Linen useful for users, while introducing some new principles.
The central idea behind Flax NNX is to introduce reference semantics into JAX. The following are its main features:

- **NNX is Pythonic**: Regular Python semantics for Modules, including support for mutability and shared references.
- **NNX is simple**: Many of the complex APIs in Flax Linen are either simplified using Python idioms or completely removed.
- **Better JAX integration**: Custom NNX transforms adopt the same APIs as the JAX transforms. And with NNX
  it is easier to use `JAX transforms (higher-order functions) <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`_ directly.

Here is an example of a simple Flax NNX program that illustrates many of the points from above:

.. testcode:: NNX

  from flax import nnx
  import optax


  class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
      self.linear = nnx.Linear(din, dmid, rngs=rngs)
      self.bn = nnx.BatchNorm(dmid, rngs=rngs)
      self.dropout = nnx.Dropout(0.2, rngs=rngs)
      self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
      x = nnx.relu(self.dropout(self.bn(self.linear(x))))
      return self.linear_out(x)

  model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # Eager initialization
  optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # Reference sharing.

  @nnx.jit  # Automatic state management for JAX transforms.
  def train_step(model, optimizer, x, y):
    def loss_fn(model):
      y_pred = model(x)  # call methods directly
      return ((y_pred - y) ** 2).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # in-place updates

    return loss

Flax NNX's improvements on Linen
--------------------------------

The rest of this document uses various examples that demonstrate how Flax NNX improves on Flax Linen.

Inspection
^^^^^^^^^^

The first improvement is that Flax NNX Modules are regular Python objects. This means that you can easily
construct and inspect ``Module`` objects.

On the other hand, Flax Linen Modules are not easy to inspect and debug because they are lazy, which means some attributes are not available upon construction and are only accessible at runtime.

.. codediff::
  :title: Linen, NNX
  :sync:

  class Block(nn.Module):
    def setup(self):
      self.linear = nn.Dense(10)

  block = Block()

  try:
    block.linear  # AttributeError: "Block" object has no attribute "linear".
  except AttributeError as e:
    pass





  ...

  ---

  class Block(nnx.Module):
    def __init__(self, rngs):
      self.linear = nnx.Linear(5, 10, rngs=rngs)

  block = Block(nnx.Rngs(0))


  block.linear
  # Linear(
  #   kernel=Param(
  #     value=Array(shape=(5, 10), dtype=float32)
  #   ),
  #   bias=Param(
  #     value=Array(shape=(10,), dtype=float32)
  #   ),
  #   ...

Notice that in the Flax NNX example above, there is no shape inference - both the input and output shapes must be provided
to the ``Linear`` ``nnx.Module``. This is a tradeoff that allows for more explicit and predictable behavior.

Running computation
^^^^^^^^^^^^^^^^^^^

In Flax Linen, all top-level computation must be done through the ``flax.linen.Module.init`` or ``flax.linen.Module.apply`` methods, and the
parameters or any other type of state are handled as a separate structure. This creates an asymmetry between: 1) code that runs inside
``apply`` that can run methods and other ``Module`` objects directly; and 2) code that runs outside of ``apply`` that must use the ``apply`` method.

In Flax NNX, there's no special context because parameters are held as attributes and methods can be called directly. That means your NNX Module's ``__init__`` and ``__call__`` methods are not treated differently from other class methods, whereas Flax Linen Module's ``setup()`` and ``__call__`` methods are special.

.. codediff::
  :title: Linen, NNX
  :sync:

  Encoder = lambda: nn.Dense(10)
  Decoder = lambda: nn.Dense(2)

  class AutoEncoder(nn.Module):
    def setup(self):
      self.encoder = Encoder()
      self.decoder = Decoder()

    def __call__(self, x) -> jax.Array:
      return self.decoder(self.encoder(x))

    def encode(self, x) -> jax.Array:
      return self.encoder(x)

  x = jnp.ones((1, 2))
  model = AutoEncoder()
  params = model.init(random.key(0), x)['params']

  y = model.apply({'params': params}, x)
  z = model.apply({'params': params}, x, method='encode')
  y = Decoder().apply({'params': params['decoder']}, z)

  ---

  Encoder = lambda rngs: nnx.Linear(2, 10, rngs=rngs)
  Decoder = lambda rngs: nnx.Linear(10, 2, rngs=rngs)

  class AutoEncoder(nnx.Module):
    def __init__(self, rngs):
      self.encoder = Encoder(rngs)
      self.decoder = Decoder(rngs)

    def __call__(self, x) -> jax.Array:
      return self.decoder(self.encoder(x))

    def encode(self, x) -> jax.Array:
      return self.encoder(x)

  x = jnp.ones((1, 2))
  model = AutoEncoder(nnx.Rngs(0))


  y = model(x)
  z = model.encode(x)
  y = model.decoder(z)

In Flax Linen, calling sub-Modules directly is not possible because they are not initialized.
Therefore, what you must do is construct a new instance and then provide a proper parameter structure.

But in Flax NNX you can call sub-Modules directly without any issues.

State handling
^^^^^^^^^^^^^^

One of the areas where Flax Linen is notoriously complex is in state handling. When you use either a
`Dropout` layer, a `BatchNorm` layer, or both, you suddenly have to handle the new state and use it to
configure the ``flax.linen.Module.apply`` method.

In Flax NNX, state is kept inside an ``nnx.Module`` and is mutable, which means it can just be called directly.

.. codediff::
  :title: Linen, NNX
  :sync:

  class Block(nn.Module):
    train: bool

    def setup(self):
      self.linear = nn.Dense(10)
      self.bn = nn.BatchNorm(use_running_average=not self.train)
      self.dropout = nn.Dropout(0.1, deterministic=not self.train)

    def __call__(self, x):
      return nn.relu(self.dropout(self.bn(self.linear(x))))

  x = jnp.ones((1, 5))
  model = Block(train=True)
  vs = model.init(random.key(0), x)
  params, batch_stats = vs['params'], vs['batch_stats']

  y, updates = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x,
    rngs={'dropout': random.key(1)},
    mutable=['batch_stats'],
  )
  batch_stats = updates['batch_stats']

  ---

  class Block(nnx.Module):


    def __init__(self, rngs):
      self.linear = nnx.Linear(5, 10, rngs=rngs)
      self.bn = nnx.BatchNorm(10, rngs=rngs)
      self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
      return nnx.relu(self.dropout(self.bn(self.linear(x))))

  x = jnp.ones((1, 5))
  model = Block(nnx.Rngs(0))



  y = model(x)





  ...

The main benefit of Flax NNX's state handling is that you don't have to change the training code when you add a new stateful layer.

In addition, in Flax NNX, layers that handle state are also very easy to implement. Below
is a simplified version of a ``BatchNorm`` layer that updates the mean and variance every time it is called.

.. testcode:: NNX

  class BatchNorm(nnx.Module):
    def __init__(self, features: int, mu: float = 0.95):
      # Variables
      self.scale = nnx.Param(jax.numpy.ones((features,)))
      self.bias = nnx.Param(jax.numpy.zeros((features,)))
      self.mean = nnx.BatchStat(jax.numpy.zeros((features,)))
      self.var = nnx.BatchStat(jax.numpy.ones((features,)))
      self.mu = mu  # Static

  def __call__(self, x):
    mean = jax.numpy.mean(x, axis=-1)
    var = jax.numpy.var(x, axis=-1)
    # ema updates
    self.mean.value = self.mu * self.mean + (1 - self.mu) * mean
    self.var.value = self.mu * self.var + (1 - self.mu) * var
    # normalize and scale
    x = (x - mean) / jax.numpy.sqrt(var + 1e-5)
    return x * self.scale + self.bias


Model surgery
^^^^^^^^^^^^^

In Flax Linen, `model surgery <https://flax.readthedocs.io/en/latest/guides/surgery.html>`_ has historically been challenging because of two reasons:

1. Due to lazy initialization, it is not guaranteed that you can replace a sub-``Module`` with a new one.
2. The parameter structure is separated from the ``flax.linen.Module`` structure, which means you have to manually keep them in sync.

In Flax NNX, you can replace sub-Modules directly as per the Python semantics. Since parameters are
part of the ``nnx.Module`` structure, they are never out of sync. Below is an example of how you can
implement a LoRA layer, and then use it to replace a ``Linear`` layer in an existing model.

.. codediff::
  :title: Linen, NNX
  :sync:

  class LoraLinear(nn.Module):
    linear: nn.Dense
    rank: int

    @nn.compact
    def __call__(self, x: jax.Array):
      A = self.param(random.normal, (x.shape[-1], self.rank))
      B = self.param(random.normal, (self.rank, self.linear.features))

      return self.linear(x) + x @ A @ B

  try:
    model = Block(train=True)
    model.linear = LoraLinear(model.linear, rank=5) # <-- ERROR

    lora_params = model.linear.init(random.key(1), x)
    lora_params['linear'] = params['linear']
    params['linear'] = lora_params

  except AttributeError as e:
    pass

  ---

  class LoraParam(nnx.Param): pass

  class LoraLinear(nnx.Module):
    def __init__(self, linear, rank, rngs):
      self.linear = linear
      self.A = LoraParam(random.normal(rngs(), (linear.in_features, rank)))
      self.B = LoraParam(random.normal(rngs(), (rank, linear.out_features)))

    def __call__(self, x: jax.Array):
      return self.linear(x) + x @ self.A @ self.B

  rngs = nnx.Rngs(0)
  model = Block(rngs)
  model.linear = LoraLinear(model.linear, rank=5, rngs=rngs)






  ...

As shown above, in Flax Linen this doesn't really work in this case because the ``linear`` sub-``Module``
is not available. However, the rest of the code provides an idea of how the ``params`` structure must be manually updated.

Performing arbitrary model surgery is not easy in Flax Linen, and currently the
`intercept_methods <https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.intercept_methods>`_
API is the only way to do generic patching of methods. But this API is not very ergonomic.

In Flax NNX, to do generic model surgery you can just use ``nnx.iter_graph``, which is much simpler and easier than in Linen. Below is an example of replacing all ``nnx.Linear`` layers in a model with custom-made ``LoraLinear`` NNX layers.

.. testcode:: NNX

  rngs = nnx.Rngs(0)
  model = Block(rngs)

  for path, module in nnx.iter_graph(model):
    if isinstance(module, nnx.Module):
      for name, value in vars(module).items():
        if isinstance(value, nnx.Linear):
          setattr(module, name, LoraLinear(value, rank=5, rngs=rngs))

Transforms
^^^^^^^^^^

Flax Linen transforms are very powerful in that they enable fine-grained control over the model's state.
However, Flax Linen transforms have drawbacks, such as:

1. They expose additional APIs that are not part of JAX, making their behavior confusing and sometimes divergent from their JAX counterparts. This also constrains your ways to interact with `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`_ and keep up with JAX API changes.
2. They work on functions with very specific signatures, namely:
  - A ``flax.linen.Module`` must be the first argument.
  - They accept other ``Module`` objects as arguments but not as return values.
3. They can only be used inside ``flax.linen.Module.apply``.

On the other hand, `Flax NNX transforms <https://flax.readthedocs.io/en/latest/guides/transforms.html>`_
are intented to be equivalent to their corresponding `JAX transforms <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`_
with an exception - they can be used on Flax NNX Modules. This means that Flax transforms:

1) Have the same API as JAX transforms.
2) Can accept Flax NNX Modules on any argument, and ``nnx.Module`` objects can be returned from it/them.
3) Can be used anywhere including the training loop.

Below is an example of using ``vmap`` with Flax NNX to both create a stack of weights by transforming the
``create_weights`` function, which returns some ``Weights``, and to apply that stack of weights to a batch
of inputs individually by transforming the ``vector_dot`` function, which takes ``Weights`` as the first
argument and a batch of inputs as the second argument.

.. testcode:: NNX

  class Weights(nnx.Module):
    def __init__(self, kernel: jax.Array, bias: jax.Array):
      self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)

  def create_weights(seed: jax.Array):
    return Weights(
      kernel=random.uniform(random.key(seed), (2, 3)),
      bias=jnp.zeros((3,)),
    )

  def vector_dot(weights: Weights, x: jax.Array):
    assert weights.kernel.ndim == 2, 'Batch dimensions not allowed'
    assert x.ndim == 1, 'Batch dimensions not allowed'
    return x @ weights.kernel + weights.bias

  seeds = jnp.arange(10)
  weights = nnx.vmap(create_weights, in_axes=0, out_axes=0)(seeds)

  x = jax.random.normal(random.key(1), (10, 2))
  y = nnx.vmap(vector_dot, in_axes=(0, 0), out_axes=1)(weights, x)

Contrary to Flax Linen transforms, the ``in_axes`` argument and other APIs do affect how the ``nnx.Module`` state is transformed.

In addition, Flax NNX transforms can be used as method decorators, because ``nnx.Module`` methods are simply
functions that take a ``Module`` as the first argument. This means that the previous example can be
rewritten as follows:

.. testcode:: NNX

  class WeightStack(nnx.Module):
    @nnx.vmap(in_axes=(0, 0), out_axes=0)
    def __init__(self, seed: jax.Array):
      self.kernel = nnx.Param(random.uniform(random.key(seed), (2, 3)))
      self.bias = nnx.Param(jnp.zeros((3,)))

    @nnx.vmap(in_axes=(0, 0), out_axes=1)
    def __call__(self, x: jax.Array):
      assert self.kernel.ndim == 2, 'Batch dimensions not allowed'
      assert x.ndim == 1, 'Batch dimensions not allowed'
      return x @ self.kernel + self.bias

  weights = WeightStack(jnp.arange(10))

  x = jax.random.normal(random.key(1), (10, 2))
  y = weights(x)



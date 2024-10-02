Why NNX?
========

Years ago we developed the Flax "Linen" API to support modeling research on JAX, with a focus on scaling scaling
and performance.  We've learned a lot from our users over these years. We introduced some ideas that have proven to be good:

* Organizing variables into `collections <https://flax.readthedocs.io/en/latest/glossary.html#term-Variable-collections>`_.
* Automatic and efficient `PRNG management <https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences>`_.
* `Variable Metadata <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/spmd.html#flax.linen.with_partitioning>`_
  for SPMD annotations, optimizer metadata, etc.

One choice we made was to use functional (``compact``) semantics for NN programming via the lazy initialization of parameters,
this made for concise implementation code and aligned our API with Haiku. However, this also meant that the semantics of
modules and variables in Flax were non-pythonic and often surprising. It also led to implementation complexity and obscured
the core ideas of transformations on neural nets.

.. testsetup:: Linen, NNX

    import jax
    from jax import random, numpy as jnp
    from flax import nnx
    import flax.linen as nn

Introducing Flax NNX
--------------------
Flax NNX is an attempt to keep the features that made Linen useful while introducing some new principles.
The central idea behind Flax NNX is to introduce reference semantics into JAX. These are its main features:

- **Pythonic**: supports regular Python semantics for Modules, including for mutability and shared references.
- **Simple**: many of the complex APIs in Flax Linen are either simplified using Python idioms or removed entirely.
- **Better JAX integration**: both by making custom transforms adopt the same APIs as JAX transforms, and by making
  it easier to use JAX transforms directly.

Here's an example of a simple Flax NNX program that illustrates many of the points above:

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

  model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
  optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

  @nnx.jit  # automatic state management for JAX transforms
  def train_step(model, optimizer, x, y):
    def loss_fn(model):
      y_pred = model(x)  # call methods directly
      return ((y_pred - y) ** 2).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # in-place updates

    return loss

Improvements
------------
Through the rest of this document, we'll key examples of how Flax NNX improves on Flax Linen.

Inspection
^^^^^^^^^^
The first improvement is that Flax NNX modules are regular Python objects, so you can easily
construct and inspect them. Because Flax Linen Modules are lazy, some attributes are not available
upon construction and are only accesible at runtime. This makes it hard to inspect and debug.

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

Notice that in Flax NNX there is no shape inference so both the input and output shapes must be provided
to the Linear module. This is a tradeoff that allows for more explicit and predictable behavior.

Running Computation
^^^^^^^^^^^^^^^^^^^
In Flax Linen, all top-level computation must be done through the ``init`` or ``apply`` methods and the
parameters or any other type of state is handled as a separate structure. This creates an asymmetry
between code that runs inside ``apply`` that can run methods and other Modules directly, and code
outside of ``apply`` that must use the ``apply`` method. In Flax NNX, there's no special context
as parameters are held as attributes and methods can be called directly.

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

Note that in Linen, calling submodules directly is not possible as they are not initialized.
So you must construct a new instance and provide proper parameter structure. In NNX
you can call submodules directly without any issues.

State Handling
^^^^^^^^^^^^^^
One of the areas where Flax Linen is notoriously complex is in handling state. When you either use a
Dropout layer or a BatchNorm layer, or both, you suddenly have to handle the new state and use it to
configure the ``apply`` method. In Flax NNX, state is kept inside the Module and is mutable, so it can
just be called directly.

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

The main benefit is that this usually means you don't have to change the training code when you add
a new stateful layers. Layers that handle state are also very easy to implement in Flax NNX, below
is a simplified version of a BatchNorm layer that updates the mean and variance every time it's called.

.. testcode:: NNX

  class BatchNorm(nnx.Module):
    def __init__(self, features: int, mu: float = 0.95):
      # Variables
      self.scale = nnx.Param(jax.numpy.ones((features,)))
      self.bias = nnx.Param(jax.numpy.zeros((features,)))
      self.mean = nnx.BatchStat(jax.numpy.zeros((features,)))
      self.var = nnx.BatchStat(jax.numpy.ones((features,)))
      self.mu = mu  # static

  def __call__(self, x):
    mean = jax.numpy.mean(x, axis=-1)
    var = jax.numpy.var(x, axis=-1)
    # ema updates
    self.mean.value = self.mu * self.mean + (1 - self.mu) * mean
    self.var.value = self.mu * self.var + (1 - self.mu) * var
    # normalize and scale
    x = (x - mean) / jax.numpy.sqrt(var + 1e-5)
    return x * self.scale + self.bias


Surgery
^^^^^^^
Model surgery historically has been a difficult problem in Flax Linen because of two reasons:
1. Due to lazy initialization, its not guaranteed you can replace a submodule with new one.
2. The parameter structure is separate from the module structure, so you manually have to keep
  them in sync.

In Flax NNX, you can replace submodules directly per Python semantics. Since the parameters are
part of the Module structre, they are never out of sync. Below is an example of how you can
implement a LoRA layer and replace a Linear layer of an existing model with it.

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

As should above, in Linen this doesn't really work in this case because the ``.linear`` submodule
is not available, however the rest of the code gives an idea how the ``params`` structure must be
manually updated.

Performing arbitrary model surgery is not very easy in Flax Linen, currently the
`intercept_methods <https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.intercept_methods>`_
API is the only was to do generic patching of methods but it's not very ergonomic. In NNX, using ``iter_graph`` its very easy
to do generic model surgery, below is an example of replacing all Linear layers in a model with LoRA layers.

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
Flax Linen transforms are very powerful in that they allow fine-grained control over the model's state,
however Linen transforms have the following drawbacks:
1. They expose additional APIs that are not part of JAX.
2. They work on functions with very specific signatures:
  * A Module must be the first argument.
  * They accepts other Modules as arguments but not as return values.
3. They can only be used inside ``apply``.

`Flax NNX transforms <https://flax-nnx.readthedocs.io/en/latest/guides/transforms.html>`_ on the other hand
are intented to be equivalent to JAX transforms with the exception that they can be used on Modules. This
means they have the same API as JAX transforms, can accepts Modules on any argument and Modules can be
returned from them, and they can be used anywhere including the training loop.

Here is an example of using ``vmap`` with Flax NNX to both create a stack of weights by transforming the
``create_weights`` function which returns some ``Weights``, and to apply the stack of weights to a batch
of inputs individually by transforming the ``vector_dot`` function which takes a ``Weights`` as the first
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

Contrary to Linen transforms, the arguments ``in_axes`` and other APIs do affect how the Module state is transformed.

Flax NNX transforms can also be used as method decorators, as Module methods are simply
functions that take a Module as the first argument. This means that the previous example can be
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
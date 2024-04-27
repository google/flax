NNX vs JAX Transformations
==========================

In this guide, you will learn the differences using NNX and JAX transformations, and how to
seamlessly switch between them or use them together. We will be focusing on the ``jit`` and
``grad`` function transformations in this guide.

First, let's set up imports and generate some dummy data:

.. testcode::

  from flax.experimental import nnx
  import jax

  x = jax.random.normal(jax.random.key(0), (1, 2))
  y = jax.random.normal(jax.random.key(1), (1, 3))

Differences between NNX and JAX transformations
***********************************************

The primary difference between NNX and JAX transformations is that NNX transformations allow you to
transform functions that take in NNX graph objects as arguments (`Module`, `Rngs`, `Optimizer`, etc),
even those whose state will be mutated, whereas they aren't recognized in JAX transformations.
Therefore NNX transformations can transform functions that are not pure and make mutations and
side-effects.

NNX's `Functional API <https://flax.readthedocs.io/en/latest/experimental/nnx/nnx_basics.html#the-functional-api>`_
provides a way to convert graph structures to pytrees and back, by doing this at every function
boundary you can effectively use graph structures with any JAX transform and propagate state updates
in a way consistent with functional purity. NNX custom transforms such as ``nnx.jit`` and ``nnx.grad``
simply remove the boilerplate, as a result the code looks stateful.

Below is an example of using the ``nnx.jit`` and ``nnx.grad`` transformations compared to using the
``jax.jit`` and ``jax.grad`` transformations. Notice the function signature of NNX-transformed
functions can accept the ``nnx.Linear`` module directly and can make stateful updates to the module,
whereas the function signature of JAX-transformed functions can only accept the pytree-registered
``State`` and ``GraphDef`` objects and must return an updated copy of them to maintain the purity of
the transformed function.

.. codediff::
  :title_left: NNX transforms
  :title_right: JAX transforms
  :sync:

  @nnx.jit
  def train_step(model, x, y):
    def loss_fn(model):
      return ((model(x) - y) ** 2).mean()
    grads = nnx.grad(loss_fn)(model)
    params = nnx.state(model, nnx.Param)
    params = jax.tree_util.tree_map(
      lambda p, g: p - 0.1 * g, params, grads
    )
    nnx.update(model, params)

  model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
  train_step(model, x, y)

  ---
  @jax.jit #!
  def train_step(graphdef, state, x, y): #!
    def loss_fn(graphdef, state): #!
      model = nnx.merge(graphdef, state) #!
      return ((model(x) - y) ** 2).mean()
    grads = jax.grad(loss_fn, argnums=1)(graphdef, state) #!

    model = nnx.merge(graphdef, state) #!
    params = nnx.state(model, nnx.Param)
    params = jax.tree_util.tree_map(
      lambda p, g: p - 0.1 * g, params, grads
    )
    nnx.update(model, params)
    return nnx.split(model) #!

  graphdef, state = nnx.split(nnx.Linear(2, 3, rngs=nnx.Rngs(0))) #!
  graphdef, state = train_step(graphdef, state, x, y) #!


Mixing NNX and JAX transformations
**********************************

NNX and JAX transformations can be mixed together, so long as the JAX-transformed function is
pure and has valid argument types that are recognized by JAX.

.. codediff::
  :title_left: Using ``nnx.jit`` with ``jax.grad``
  :title_right: Using ``jax.jit`` with ``nnx.grad``
  :sync:

  @nnx.jit
  def train_step(model, x, y):
    def loss_fn(graphdef, state): #!
      model = nnx.merge(graphdef, state)
      return ((model(x) - y) ** 2).mean()
    grads = jax.grad(loss_fn, 1)(*nnx.split(model)) #!
    params = nnx.state(model, nnx.Param)
    params = jax.tree_util.tree_map(
      lambda p, g: p - 0.1 * g, params, grads
    )
    nnx.update(model, params)

  model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
  train_step(model, x, y)

  ---
  @jax.jit #!
  def train_step(graphdef, state, x, y): #!
    model = nnx.merge(graphdef, state)
    def loss_fn(model):
      return ((model(x) - y) ** 2).mean()
    grads = nnx.grad(loss_fn)(model)
    params = nnx.state(model, nnx.Param)
    params = jax.tree_util.tree_map(
      lambda p, g: p - 0.1 * g, params, grads
    )
    nnx.update(model, params)
    return nnx.split(model)

  graphdef, state = nnx.split(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
  graphdef, state = train_step(graphdef, state, x, y)



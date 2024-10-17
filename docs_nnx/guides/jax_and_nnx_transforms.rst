Flax NNX vs JAX transformations
===============================

This guide describes the differences between
`Flax NNX transformations <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__
and `JAX transformations <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__,
and how to seamlessly switch between them or use them side-by-side. The examples here will focus on
``nnx.jit``, ``jax.jit``, ``nnx.grad`` and ``jax.grad`` function transformations (transforms).

First, let's set up imports and generate some dummy data:

.. testcode:: Flax NNX, JAX

  from flax import nnx
  import jax

  x = jax.random.normal(jax.random.key(0), (1, 2))
  y = jax.random.normal(jax.random.key(1), (1, 3))

Differences
***********

Flax NNX transformations can transform functions that are not pure and make mutations and
side-effects:
- Flax NNX transforms enable you to transform functions that take in Flax NNX graph objects as
arguments - such as ``nnx.Module``, ``nnx.Rngs``, ``nnx.Optimizer``, and so on - even those whose state
will be mutated.
- In comparison, this kind of functions aren't recognized in JAX transformations.

The Flax NNX `Functional API <https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#the-functional-api>`_
provides a way to convert graph structures to `pytrees <https://jax.readthedocs.io/en/latest/working-with-pytrees.html>`__
and back. By doing this at every function boundary you can effectively use graph structures with any
JAX transform, and propagate state updates in a way consistent with functional purity.

Flax NNX custom transforms, such as ``nnx.jit`` and ``nnx.grad``, simply remove the boilerplate, and
as a result the code looks stateful.

Below is an example of using the ``nnx.jit`` and ``nnx.grad`` transforms compared to the
the code that uses ``jax.jit`` and ``jax.grad`` transforms.

Notice that:

- The function signature of Flax NNX-transformed functions can accept the ``nnx.Linear``
  ``nnx.Module`` directly and can make stateful updates to the ``Module``.
- The function signature of JAX-transformed functions can only accept the pytree-registered
  ``nnx.State`` and ``nnx.GraphDef`` objects, and must return an updated copy of them to maintain the
  purity of the transformed function.

.. codediff::
  :title: Flax NNX transforms, JAX transforms
  :groups: Flax NNX, JAX
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


Mixing Flax NNX and JAX transforms
**********************************

Both Flax NNX transforms and JAX transforms can be mixed together, so long as the JAX-transformed function
in your code is pure and has valid argument types that are recognized by JAX.

.. codediff::
  :title: Using ``nnx.jit`` with ``jax.grad``, Using ``jax.jit`` with ``nnx.grad``
  :groups: Flax NNX, JAX
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

NNX 0.10 to NNX 0.11
#########################################

In this guide we present the code changes required when we update Flax NNX code from Flax version
``0.10.x`` to ``0.11.x``.


Using Rngs in NNX Transforms
====================================

NNX layers that use RNGs like Dropout or MultiHeadAttention now hold a ``fork``-ed copy of the ``Rngs``
object given at construction time instead of a shared reference to the original ``Rngs`` object. This has
two consequences:
* It changes the checkpoint structure, as each layer will have unique RNG state.
* It changes how ``nnx.split_rngs`` interacts with transforms like ``nnx.vmap`` and ``nnx.scan``,
  as the resulting RNG state will now not be stored in scalar form.

Here is how a "scan over layers" looks like in the new version:

.. tab-set::

  .. tab-item:: v0.11
    :sync: v0.11

    .. code-block:: python

      import flax.nnx as nnx

      class MLP(nnx.Module):
        @nnx.split_rngs(splits=5)
        @nnx.vmap(in_axes=(0, 0))
        def __init__(self, rngs: nnx.Rngs):
          self.linear = nnx.Linear(3, 3, rngs=rngs)
          self.bn = nnx.BatchNorm(3, rngs=rngs)
          self.dropout = nnx.Dropout(0.5, rngs=rngs)
          self.node = nnx.Param(jnp.ones((2,)))


        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def __call__(self, x: jax.Array):
          return nnx.gelu(self.dropout(self.bn(self.linear(x))))

  .. tab-item:: v0.10
    :sync: v0.10

    .. code-block:: python
      :emphasize-lines: 12

      import flax.nnx as nnx

      class MLP(nnx.Module):
        @nnx.split_rngs(splits=5)
        @nnx.vmap(in_axes=(0, 0))
        def __init__(self, rngs: nnx.Rngs):
          self.linear = nnx.Linear(3, 3, rngs=rngs)
          self.bn = nnx.BatchNorm(3, rngs=rngs)
          self.dropout = nnx.Dropout(0.5, rngs=rngs)
          self.node = nnx.Param(jnp.ones((2,)))

        @nnx.split_rngs(splits=5)
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def __call__(self, x: jax.Array):
          return nnx.gelu(self.dropout(self.bn(self.linear(x))))


The main thing to note is that the ``nnx.split_rngs`` over ``scan`` is not needed anymore, as the RNGs produced
by ``__init__`` are no longer in scalar form (they keep the additional dimension) and thus can be used directly
in ``scan`` without the need to split them again. Alternatively, can even remove the ``nnx.split_rngs`` decorator
from the ``__init__`` method and use ``Rngs.fork`` directly before passing the RNGs to the module.

.. code-block:: python

  class MLP(nnx.Module):
    @nnx.vmap(in_axes=(0, 0))
    def __init__(self, rngs: nnx.Rngs):
      self.linear = nnx.Linear(3, 3, rngs=rngs)
      self.bn = nnx.BatchNorm(3, rngs=rngs)
      self.dropout = nnx.Dropout(0.5, rngs=rngs)
      self.node = nnx.Param(jnp.ones((2,)))

    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
    def __call__(self, x: jax.Array):
      return nnx.gelu(self.dropout(self.bn(self.linear(x))))

  rngs = nnx.Rngs(0)
  mlp = MLP(rngs=rngs.fork(splits=5))

Loading Checkpoints with RNGs
==================================================

When loading checkpoints in the new version, you need to drop the old RNGs structure and
partially reinitialize the model with new RNGs. To do this, you can use ``nnx.jit`` to

1. Remove the RNGs from the checkpoint.
2. Perform partial initialization of the model with new RNGs.

.. code-block:: python

  # load checkpoint
  checkpointer = ocp.StandardCheckpointer()
  checkpoint = checkpointer.restore(path / "state")

  @jax.jit
  def fix_checkpoint(checkpoint, rngs: nnx.Rngs):
    # drop rngs keys
    flat_paths = nnx.traversals.flatten_mapping(checkpoint)
    flat_paths = {
        path[:-1] if path[-1] == "value" else path: value  # remove "value" suffix
        for path, value in flat_paths.items()
        if "rngs" not in path  # remove rngs paths
    }
    checkpoint = nnx.traversals.unflatten_mapping(flat_paths)

    # initialize new model with given rngs
    model = MyModel(rngs=rngs)
    # overwrite model parameters with checkpoint
    nnx.update(model, checkpoint)
    # get full checkpoint with new rngs
    new_checkpoint = nnx.state(model)

    return new_checkpoint

  checkpoint = fix_checkpoint(checkpoint, rngs=nnx.Rngs(params=0, dropout=1))
  checkpointer.save(path.with_name(path.name + "_new"), checkpoint)

The previous code is efficient because ``jit`` performs dead code elimination (DCE) so it will not
actually initialize the existing model parameters in memory.

Optimizer Updates
====================================

Optimizer has been updated to not hold a reference to the model anymore. Instead, it now
takes the model and gradients as arguments in the ``update`` method. Concretely, these are the
the new changes:

1. The ``wrt`` constructor argument is now required.
2. The ``model`` attribute has been removed.
3. The ``update`` method now takes ``(model, grads)`` instead of only ``(grads)``.

.. tab-set::

  .. tab-item:: v0.11
    :sync: v0.11

    .. code-block:: python
      :emphasize-lines: 17, 26

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

      model = Model(2, 64, 3, rngs=nnx.Rngs(0))
      optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

      @nnx.jit
      def train_step(model, optimizer, x, y):
        def loss_fn(model):
          y_pred = model(x)
          return ((y_pred - y) ** 2).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        return loss

  .. tab-item:: v0.10
    :sync: v0.10

    .. code-block:: python
      :emphasize-lines: 17, 26

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

      model = Model(2, 64, 3, rngs=nnx.Rngs(0))
      optimizer = nnx.Optimizer(model, optax.adam(1e-3))

      @nnx.jit
      def train_step(model, optimizer, x, y):
        def loss_fn(model):
          y_pred = model(x)
          return ((y_pred - y) ** 2).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

Pytrees containing NNX Objects
====================================

In the new version, NNX modules are now Pytrees. This means that you can use them with JAX transforms
like ``jax.vmap`` and ``jax.jit`` directly (more documentation on this will be available soon). However,
this also means that code using ``jax.tree.*`` functions on structures that contain NNX modules will
need to take this into account to maintain the current behavior. In these cases, the solution is to
use the ``is_leaf`` argument to specify that NNX modules and other NNX objects should be treated as leaves.


.. code-block:: python

  modules = [nnx.Linear(3, 3, rngs=nnx.Rngs(0)), nnx.BatchNorm(3, rngs=nnx.Rngs(1))]

  type_names = jax.tree.map(
      lambda x: type(x).__name__,
      modules,
      is_leaf=lambda x: isinstance(x, nnx.Pytree)  # <-- specify that NNX objects are leaves
  )

Hijax (experimental)
====================



----

Basic usage
^^^^^^^^^^^^

.. testsetup::

  import jax
  import jax.numpy as jnp

  current_mode = nnx.using_hijax()

.. testcode::

  from flax import nnx
  import optax

  nnx.use_hijax(True)

  class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
      self.linear = nnx.Linear(din, dmid, rngs=rngs)
      self.bn = nnx.BatchNorm(dmid, rngs=rngs)
      self.dropout = nnx.Dropout(0.2)
      self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x, rngs):
      x = nnx.relu(self.dropout(self.bn(self.linear(x)), rngs=rngs))
      return self.linear_out(x)

  model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

  @jax.jit
  def train_step(model, optimizer, rngs, x, y):
    graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)
    def loss_fn(params):
      model = nnx.merge(graphdef, params, nondiff)
      return ((model(x, rngs) - y) ** 2).mean()
    loss, grads = jax.value_and_grad(loss_fn)(nnx.as_immutable_vars(params))
    optimizer.update(model, grads)  # in-place updates
    return loss

  nnx.use_hijax(current_mode)  # clean up for CI tests


----

.. toctree::
   :hidden:
   :maxdepth: 2

   hijax

Managing Parameters and State
=============================

We will show you how to...

* manage the variables from initialization to updates.
* split and re-assemble parameters and state.

.. code-block:: python

  class BiasAdderWithRunningMean(nn.Module):
    momentum: float = 0.9

    @nn.compact
    def __call__(self, x):
      is_initialized = self.has_variable('batch_stats', 'mean')
      mean = self.variable('batch_stats', 'mean', jnp.zeros, x.shape[1:])
      bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
      if is_initialized:
        mean.value = (self.momentum * mean.value +
                      (1.0 - self.momentum) * jnp.mean(x, axis=0, keepdims=True))
      return mean.value + bias

This example model is a minimal example that contains both parameters (declared with :code:`self.param`) 
and state variables (declared with :code:`self.variable`).

The tricky part with initialization here is that we need to split the state variables and the parameters
we're going to optimize for.

.. code-block:: python

  model = BiasAdderWithRunningMean()
  variables = model.init(random.PRNGKey(0), x) # x is a dummy input
  state, params = variables.pop('params') # Split state and params to optimize for
  del variables # Delete variables to avoid wasting resources
  optimizer = optim.sgd.GradientDescent(learning_rate=0.02).create(params)

  for _ in range(10):
    optimizer, state = update_step(model.apply, x, optimizer, state)

With the update_step function defined as follow (with dummy loss that should be replaced for yours):

.. code-block:: python

  def update_step(apply_fun, x, optimizer, state):

    def loss(params):
      y, updated_state = apply_fun({'params': params, **state},
                                  x, mutable=list(state.keys()))
      l = ((x - y) ** 2).sum() # Replace with your loss here.
      return l, updated_state

    (l, updated_state), grads = jax.value_and_grad(
        loss, has_aux=True)(optimizer.target)
    optimizer = optimizer.apply_gradient(grads)
    return optimizer, updated_state
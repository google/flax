Ensembling on multiple devices
=============================

We show you how to rewrite code for training a single mode to training an 
ensemble of models, one on each device.

Example of the new directive:

.. codediff::
  :title_left: Without ensembling
  :title_right: With ensembling
  
  # Jitting #!
  @jax.jit #!
  def get_initial_params(key):
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_val)['params']
    return initial_params
  ---
  # Pmapping #!
  @jax.pmap #!
  def get_initial_params(key):
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = CNN().init(key, init_val)['params']
    return initial_params
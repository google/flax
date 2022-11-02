Working with Random Number Generation
=============================

Introduction
------------

Often in Flax we don't need to explicitly refer to the random seed. In
``Module`` implementations, randomness is often only needed for
initialization. So, it only needs to be exposed as a ``Callable``
passed during parameter initialization.

But for some models, randomness occurs when the model is being used.
For example, In the Dropout model, a random subnetwork is used during
training. This randomness needs to be managed and can't be hidden
within the parameters.

In JAX, we would simply thread an rng seed around.

.. code:: python

   from jax import random
   key = random.PRNGKey(seed)
   key, rng = jax.random.split(key)
   random.bernoulli(rng, p=0.5)

But these seeds are threaded in a slightly different way in
Flax. Using Dropout as an example, we can see how it's handled in flax.

.. code:: python

  class Dropout(Module):
    rate: float
    deterministic: Optional[bool] = None
    rng_collection: str = 'dropout'

    @compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
      keep_prob = 1. - self.rate
      rng = self.make_rng(self.rng_collection)
      ...
      mask = random.bernoulli(rng, p=keep_prob)
      mask = jnp.broadcast_to(mask, inputs.shape)
      return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))

As can be seen in this example, the logic for randomness needs to be moved
to the ``__call__`` method.

What occurs here is within each Module we maintain a dictionary of rng collections.
These are threaded for us by Flax. We then can use the `make_rng` method to
split off seeds as we need them.

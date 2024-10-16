*****************
Flax NNX glossary
*****************

For additional terms, refer to the `JAX glossary <https://jax.readthedocs.io/en/latest/glossary.html>`__.

.. glossary::

    Filter
      A way to extract only certain :term:`nnx.Variable<Variable>` objects out of a Flax NNX :term:`Module<Module>` (``nnx.Module``). This is usually done by calling :meth:`nnx.split <flax.nnx.split>` upon the :class:`nnx.Module<flax.nnx.Module>`. Refer to the `Filter guide <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__ to learn more.

    Folding in
      In Flax, `folding in <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html>`__ means generating a new `JAX pseudorandom number generator (PRNG) <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ key, given an input PRNG key and integer. This is typically used when you want to generate a new key but still be able to use the original PRNG key afterwards. You can also do this in JAX with `jax.random.split <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__, but this method will effectively create two PRNG keys, which is slower. Learn how Flax generates new PRNG keys automatically in the `Randomness/PRNG guide <https://flax.readthedocs.io/en/latest/guides/randomness.html>`__.

    GraphDef
      :class:`nnx.GraphDef<flax.nnx.GraphDef>` is a class that represents all the static, stateless, and Pythonic parts of a Flax :term:`Module<Module>` (:class:`nnx.Module<flax.nnx.Module>`).

    Merge
      Refer to :term:`Split and merge<Split and merge>`.

    Module
      :class:`nnx.Module <flax.nnx.Module>` is a dataclass that enables defining and initializing parameters in a referentially-transparent form. It is responsible for storing and updating :term:`Variable<Variable> objects and parameters within itself.

    Params / parameters
       :class:`nnx.Param <flax.nnx.Param>` is a particular subclass of :class:`nnx.Variable <flax.nnx.Variable>` that generally contains the trainable weights.

    PRNG states
      A Flax :class:`nnx.Module <flax.nnx.Module>` can keep a reference of a `pseudorandom number generator (PRNG) <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ state object :class:`nnx.Rngs <flax.nnx.Rngs>` that can generate new `JAX PRNG <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ keys. These keys are used to generate random JAX arrays through `JAX's functional PRNGs <https://jax.readthedocs.io/en/latest/random-numbers.html>`__.
      You can use a PRNG state with different seeds to add more fine-grained control to your model (for example, to have independent random numbers for parameters and dropout masks).
      Refer to the Flax `Randomness/PRNG guide <https://flax.readthedocs.io/en/latest/guides/randomness.html>`__
      for more details.

    Split and merge
      :meth:`nnx.split <flax.nnx.split>` is a way to represent an :class:`nnx.Module <flax.nnx.Module>` by two parts: 1) a static Flax NNX :term:`GraphDef <GraphDef>` that captures its Pythonic static information; and 2) one or more :term:`Variable state(s)<Variable state>` that capture its `JAX arrays <https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array>`__ (``jax.Array``) in the form of `JAX pytrees <https://jax.readthedocs.io/en/latest/working-with-pytrees.html>`__. They can be merged back to the original ``nnx.Module`` using :meth:`nnx.merge <flax.nnx.merge>`.

    Transformation
      A Flax NNX transformation (transform) is a wrapped version of a `JAX transformation <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__ that allows the function that is being transformed to take the Flax NNX :term:`Module<Module>` (``nnx.Module``) as input or output. For example, a "lifted" version of `jax.jit <https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit>`__ is :meth:`nnx.jit <flax.nnx.jit>`. Check out the `Flax NNX transforms guide <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__ to learn more.

    Variable
      The weights / parameters / data / array :class:`nnx.Variable <flax.nnx.Variable>` residing in a Flax :term:`Module<Module>`. Variables are defined inside modules as :class:`nnx.Variable <flax.nnx.Variable>` or its subclasses.

    Variable state
      :class:`nnx.VariableState <flax.nnx.VariableState>` is a purely functional `JAX pytree <https://jax.readthedocs.io/en/latest/working-with-pytrees.html>`__ of all the :term:`Variables<Variable>` inside a :term:`Module<Module>`. Since it is pure, it can be an input or output of a `JAX transformation <https://jax.readthedocs.io/en/latest/key-concepts.html#transformations>`__ function. ``nnx.VariableState`` is obtained by using :meth:`nnx.split <flax.nnx.split>` on the :class:`nnx.Module <flax.nnx.Module>`. (Refer to  :term:`splitting<Split and merge>` and :term:`Module<Module>` to learn more.)

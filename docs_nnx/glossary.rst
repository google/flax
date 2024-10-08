*********
Glossary
*********

For additional terms, refer to the `JAX glossary <https://jax.readthedocs.io/en/latest/glossary.html>`__.

.. glossary::

    Filter
      A way to extract only certain :term:`Variables<Variable>` out of a :term:`Module<Module>`. Usually done via calling :meth:`nnx.split <flax.nnx.split>` upon the module. See the `Filter guide <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__ to learn more.

    `Folding in <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html>`__
      Generating a new PRNG key given an input PRNG key and integer. Typically used when you want to
      generate a new key but still be able to use the original rng key afterwards. You can also do this with
      `jax.random.split <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__
      but this will effectively create two RNG keys, which is slower. See how Flax generates new PRNG keys
      automatically in our
      `RNG guide <https://flax.readthedocs.io/en/latest/guides/randomness.html>`__.

    GraphDef
      :class:`nnx.GraphDef<flax.nnx.GraphDef>`, a class that represents all the static, stateless, Pythonic part of an :class:`nnx.Module<flax.nnx.Module>` definition.

    Lifted transformation
      A wrapped version of the `JAX transformations <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__ that allows the transformed function to take Flax :term:`Modules<Module>` as input or output. For example, a lifted version of `jax.jit <https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit>`__ will be :meth:`flax.nnx.jit <flax.nnx.jit>`. See the `lifted transforms guide <https://flax.readthedocs.io/en/latest/guides/transforms.html>`__.

    Merge
      See :term:`Split and merge<Split and merge>`.

    Module
      :class:`nnx.Module <flax.nnx.Module>`, a dataclass allowing the definition and initialization of parameters in a
      referentially-transparent form. This is responsible for storing and updating variables
      and parameters within itself.

    Params / parameters
       :class:`nnx.Param <flax.nnx.Param>`, a particular subclass of :class:`nnx.Variable <flax.nnx.Variable>` that generally contains the trainable weights.

    RNG states
      A Flax :class:`module <flax.nnx.Module>` can keep a reference of an :class:`RNG state object <flax.nnx.Rngs>` that can generate new JAX `PRNG <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`__ keys. They keys are used to generate random JAX arrays through `JAX's functional random number generators <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.
      You can use an RNG state with different seeds to make more fine-grained control on your model (e.g., independent random numbers for parameters and dropout masks).
      See the `RNG guide <https://flax.readthedocs.io/en/latest/guides/randomness.html>`__
      for more details.

    Split and merge
      :meth:`nnx.split <flax.nnx.split>`, a way to represent an `nnx.Module` by two parts - a static :term:`GraphDef <GraphDef>` that captures its Pythonic, static information, and one or more :term:`Variable state(s)<Variable state>` that captures its JAX arrays in the form of pytrees. They can be merged back to the original module with :meth:`nnx.merge <flax.nnx.merge>`.

    Variable
      The `weights / parameters / data / arrays <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/variable.html#flax.linen.Variable>`__ residing in a Flax :term:`Module<Module>`. Variables are defined inside modules as :class:`nnx.Variable <flax.nnx.Variable>` or its subclasses.

    Variable state
      :class:`nnx.VariableState <flax.nnx.VariableState>`, a purely functional pytree of all the :term:`Variables<Variable>` inside a :term:`Module<Module>`. Since it's pure, it can be an input or output of a JAX transformation function. Obtained by using :term:`splitting<Split and merge>` the module.
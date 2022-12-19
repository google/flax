*********
Glossary
*********

For additional terms, refer to the `Jax glossary <https://jax.readthedocs.io/en/latest/glossary.html>`__.

.. glossary::

    Bound Module
      When a `Module <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module>`__
      is created in a JAX program, it is in an *unbound* state. This means that only
      dataclass attributes are set, and no variables are bound to the module. When the pure
      functions `Module.init() <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.init>`
      or `Module.apply() <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.apply>`__
      are called, Flax binds the variables to the module, and the module's method code is
      executed in a bound state, allowing things like calling submodules directly without
      providing variables. For more details, refer to the
      `module lifecycle <https://flax.readthedocs.io/en/latest/advanced_topics/module_lifecycle.html>`__.

    Compact/Non-compact Module
      Modules with a single method are able to declare submodules and variables inline by
      using the  `@nn.compact` decorator.  These are referred to as “compact-style modules”,
      whereas modules defining a `setup()` method (usually but not always with multiple
      callable methods) are referred to as “setup-style modules”.

    `Folding in <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html>`__
      Generating a new PRNG key given an input PRNG key and integer. Generally faster than
      `jax.random.split <https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html>`__.

    `FrozenDict <https://flax.readthedocs.io/en/latest/api_reference/flax.core.frozen_dict.html#flax.core.frozen_dict.FrozenDict>`__
      An immutable dictionary which can be “`unfrozen <https://flax.readthedocs.io/en/latest/api_reference/flax.core.frozen_dict.html#flax.core.frozen_dict.unfreeze>`__”
      to a regular, mutable dictionary. Internally, Flax uses FrozenDIcts to ensure variable dicts
      aren't accidentally mutated. Note: We are considering using regular dicts everywhere
      (see `#1223 <https://github.com/google/flax/issues/1223>`__).

    Functional core
      The flax core library implements the simple container Scope API for threading
      variables and PRNGs though a model, as well as the lifting machinery needed to
      transform functions passing Scope objects., The python class-based module API
      is built on top of this core library.

    Lazy initialization
      Variables in Flax are initialized late, only when needed. That is, during normal
      execution of a module, if a requested variable name isn’t found in the provided
      variable collection data, we call the initializer function to create it.  This
      allows us to treat initialization and application under the same code-paths,
      simplifying the use of JAX transforms with layers.

    Lifted transformation
      Refer to the `Flax docs <https://flax.readthedocs.io/en/latest/advanced_topics/lift.html>`__.

    Module
      A dataclass allowing the definition and initialization of parameters in a
      referentially-transparent form. This is responsible for storing and updating variables
      and parameters within itself. Modules can be readily transformed into functions,
      allowing them to be trivially used with JAX transformations like `vmap` and `scan`.

    Params / parameters
      "params" is the canonical variable collection in the variable dictionary (dict).
      The “params” collection generally contains the trainable weights.

    RNG sequences
      Inside Flax `Module`s, you can obtain a new `PRNG <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`__
      key through `Module.make_rng() <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.make_rng>`__.
      These keys can be used to generate random numbers through
      `JAX's functional random number generators <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.
      Having different RNG sequences (e.g. for "params" and "dropout") allows fine-grained
      control in a multi-host setup (e.g. initializing parameters identically on different
      hosts, but have different dropout masks) and treating these sequences differently when
      `lifting transformations <https://flax.readthedocs.io/en/latest/advanced_topics/lift.html>`__.

    Scope
      A container class for holding the variables and PRNG keys for each layer.

    Shape inference
      Modules do not need to specify the shape of the input array in their definitions.
      Flax upon initialization inspects the input array, and infers the correct shapes
      for parameters in the model.

    TrainState
      Refer to `flax.training.train_state.TrainState <https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state>`__.

    Variable
      The `weights / parameters / data / arrays <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.core.variables.Variable>`__
      residing in the leaves of :term:`variable collections<Variable collections>`. Variables are defined inside modules
      using `Module.variable() <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.variable>`__.
      A variable of collection "params" is simply called a param and can be set using
      `Module.param() <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.param>`__.

    Variable collections
      Entries in the variable dict, containing weights / parameters / data / arrays that
      are used by the model. “params” is the canonical collection in the variable dict.
      They are typically differentiable, updated by an outer SGD-like loop / optimizer,
      rather than modified directly by forward-pass code.

    `Variable dictionary <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.core.variables>`__
      A dictionary containing :term:`variable collections<Variable collections>`. Each variable collection is a mapping
      from a string name (e.g., ":term:`params<Params / parameters>`" or "batch_stats") to a (possibly nested)
      dictionary with :term:`Variables<Variable>` as leaves, matching the submodule tree structure.
      Read more about pytrees and leaves in the `Jax docs <https://jax.readthedocs.io/en/latest/pytrees.html>`__.
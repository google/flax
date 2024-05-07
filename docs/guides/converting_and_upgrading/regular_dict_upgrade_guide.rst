Migrate to regular dicts
========================

Flax will migrate from returning ``FrozenDicts`` to regular dicts when calling
:meth:`.init <flax.linen.Module.init>`, :meth:`.init_with_output <flax.linen.Module.init_with_output>` and
:meth:`.apply <flax.linen.Module.apply>` ``Module`` methods.

The original issue is outlined `here <https://github.com/google/flax/issues/1223>`__.

This guide shows some common upgrade patterns.


Utility functions
-----------------

``FrozenDicts`` are immutable dictionaries that implement an additional 4 methods:

* :meth:`copy <flax.core.frozen_dict.FrozenDict.copy>`
* :meth:`pop <flax.core.frozen_dict.FrozenDict.pop>`
* :meth:`pretty_repr <flax.core.frozen_dict.FrozenDict.pretty_repr>`
* :meth:`unfreeze <flax.core.frozen_dict.FrozenDict.unfreeze>`

To accommodate the regular dict change, replace usage of ``FrozenDict`` methods with their utility function equivalent from ``flax.core.frozen_dict``.
These utility functions mimic the behavior of their corresponding ``FrozenDict`` method, and can be called on either ``FrozenDicts`` or regular dicts.
The following are the utility functions and example upgrade patterns:

.. testsetup:: default, Only ``FrozenDict``, Both ``FrozenDict`` and regular dict

  import flax
  import flax.linen as nn
  import jax
  import jax.numpy as jnp

  x = jnp.empty((1,3))
  variables = flax.core.freeze(nn.Dense(5).init(jax.random.key(0), x))

  other_variables = jnp.array([1, 1, 1, 1, 1], dtype=jnp.float32)

:meth:`copy <flax.core.frozen_dict.copy>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codediff::
  :title: Only ``FrozenDict``, Both ``FrozenDict`` and regular dict
  :sync:

  variables = variables.copy(add_or_replace={'other_variables': other_variables})

  ---

  variables = flax.core.copy(variables, add_or_replace={'other_variables': other_variables})

:meth:`pop <flax.core.frozen_dict.pop>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codediff::
  :title: Only ``FrozenDict``, Both ``FrozenDict`` and regular dict
  :sync:

  state, params = variables.pop('params')

  ---

  state, params = flax.core.pop(variables, 'params')

:meth:`pretty_repr <flax.core.frozen_dict.pretty_repr>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codediff::
  :title: Only ``FrozenDict``, Both ``FrozenDict`` and regular dict
  :sync:

  str_repr = variables.pretty_repr()

  ---

  str_repr = flax.core.pretty_repr(variables)

:meth:`unfreeze <flax.core.frozen_dict.unfreeze>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codediff::
  :title: Only ``FrozenDict``, Both ``FrozenDict`` and regular dict
  :sync:

  variables = variables.unfreeze()

  ---

  variables = flax.core.unfreeze(variables)


Modifying config values
-----------------------

A temporary feature flag ``flax_return_frozendict`` is set up to help with the migration.
To toggle behavior between returning FrozenDict and regular dict variables at runtime,
run ``flax.config.update('flax_return_frozendict', <BOOLEAN_VALUE>)`` in your code.

For example:

.. testcode::

  x = jnp.empty((1,3))

  flax.config.update('flax_return_frozendict', True) # set Flax to return FrozenDicts
  variables = nn.Dense(5).init(jax.random.key(0), x)

  assert isinstance(variables, flax.core.FrozenDict)

  flax.config.update('flax_return_frozendict', False) # set Flax to return regular dicts
  variables = nn.Dense(5).init(jax.random.key(0), x)

  assert isinstance(variables, dict)

Alternatively, the environment variable ``flax_return_frozendict``
(found `here <https://github.com/google/flax/blob/main/flax/configurations.py>`__) can be directly modified in the Flax source code.


Migration status
--------------

As of July 19th, 2023, ``flax_return_frozendict`` is set to ``False`` (see
`#3193 <https://github.com/google/flax/pull/3193>`__), meaning Flax will default to
returning regular dicts from version `0.7.1 <https://github.com/google/flax/releases/tag/v0.7.1>`__
onward. This flag can be flipped to ``True`` temporarily to have Flax return
``Frozendicts``. However this feature flag will eventually be removed in the future.
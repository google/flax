graph
------------------------

.. automodule:: flax.nnx
.. currentmodule:: flax.nnx

State
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: split
.. autofunction:: merge
.. autofunction:: flatten
.. autofunction:: unflatten

.. autoclass:: GraphDef
  :members:

.. autodata:: GraphState

.. autodata:: GraphFlatState


State Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphdef
.. autofunction:: state
.. autofunction:: update
.. autofunction:: pop
.. autofunction:: clone
.. autofunction:: pure

.. autodata:: PureState


Graph Traversal
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: iter_graph
.. autofunction:: recursive_map
.. autofunction:: find_duplicates


Function Call Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: call


Context Management
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UpdateContext
  :members:

.. autofunction:: update_context
.. autofunction:: current_update_context

.. autoclass:: SplitContext
  :members:

.. autofunction:: split_context

.. autoclass:: MergeContext
  :members:

.. autofunction:: merge_context


Metadata and Caching
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: set_metadata
.. autofunction:: cached_partial

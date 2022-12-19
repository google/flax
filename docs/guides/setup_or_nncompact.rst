``setup`` vs ``compact``
=========================================

In Flax's module system (named `Linen`_), submodules and variables (parameters or others)
can be defined in two ways:

1. **Explicitly** (using ``setup``):

   Assign submodules or variables to ``self.<attr>`` inside a
   :meth:`setup <flax.linen.Module.setup>` method. Then use the submodules
   and variables assigned to ``self.<attr>`` in ``setup`` from
   any "forward pass" method defined on the class.
   This resembles how modules are defined in PyTorch.

2. **In-line** (using ``nn.compact``):

   Write your network's logic directly within a single "forward pass" method annotated
   with :meth:`nn.compact <flax.linen.compact>`. This allows you to define your whole module
   in a single method, and "co-locate" submodules and variables next to
   where they are used.

**Both of these approaches are perfectly valid, behave the same way, and interoperate with all of Flax**.

Here is a short example of a module defined in both ways, with exactly
the same functionality.

.. testsetup::

  import flax.linen as nn

.. codediff::
  :title_left: Using ``setup``
  :title_right: Using ``nn.compact``

  class MLP(nn.Module):
    def setup(self):
      # Submodule names are derived by the attributes you assign to. In this
      # case, "dense1" and "dense2". This follows the logic in PyTorch.
      self.dense1 = nn.Dense(32)
      self.dense2 = nn.Dense(32)

    def __call__(self, x):
      x = self.dense1(x)
      x = nn.relu(x)
      x = self.dense2(x)
      return x
  ---
  class MLP(nn.Module):





    @nn.compact #!
    def __call__(self, x):
      x = nn.Dense(32, name="dense1")(x) #!
      x = nn.relu(x)
      x = nn.Dense(32, name="dense2")(x) #!
      return x

So, how would you decide which style to use? It can be a matter of taste, but here are some pros and cons:

Reasons to prefer using ``nn.compact``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Allows defining submodules, parameters and other variables next to where they are used: less
   scrolling up/down to see how everything is defined.
2. Reduces code duplication when there are conditionals or for loops that conditionally define
   submodules, parameters or variables.
3. Code typically looks more like mathematical notation: ``y = self.param('W', ...) @ x + self.param('b', ...)``
   looks similar to :math:`y=Wx+b``)
4. If you are using shape inference, i.e. using parameters whose shape/value depend on shapes of
   the inputs (which are unknown at initialization), this is not possible using ``setup``.

Reasons to prefer using ``setup``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Closer to the PyTorch convention, thus easier when porting models
   from PyTorch
2. Some people find it more natural to explicitly separate the definition
   of submodules and variables from where they are used
3. Allows defining more than one "forward pass" method
   (see :class:`MultipleMethodsCompactError <flax.errors.MultipleMethodsCompactError>`)




.. _`Linen`: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#JIT-mechanics:-tracing-and-static-variables

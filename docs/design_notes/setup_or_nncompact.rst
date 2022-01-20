Should I use ``setup`` or ``nn.compact``?
=========================================

In Flax's module system (named `Linen`_), submodules, parameters and/or other variables
can be defined in two ways:

1. **Explicitly** (using ``setup``):

   Assign to ``self.<attr>`` inside a :meth:`setup <flax.linen.Module.setup>`
   method. Then read ``self.<attr>`` from any "forward pass" method defined on the class.
   This resembles how most objects work in Python, and how modules are defined in PyTorch.
2. **In-line** (using ``nn.compact``):
  
   Declare directly within a single "forward pass" method annotated
   with :meth:`nn.compact <flax.linen.compact>`. This allows you to define your whole module
   in a single method, and "co-locate" submodules, parameters and variables next to 
   where they are used.

**Both of these approaches are perfectly valid, behave the same way, and interoperate with all of Flax**.

Here is a short example of a module defined in both ways. 

.. testsetup::

  import flax.linen as nn

.. codediff:: 
  :title_left: Using ``setup``
  :title_right: Using ``nn.compact``
  
  class MLP(nn.Module):
    def setup(self):
      self.dense1 = nn.Dense(32)
      self.dense2 = nn.Dense(32)

    def __call__(self, x):
      x = self.dense1(x)
      x = nn.relu(x)
      x = self.dense2(x)
      return x
  ---
  class MLP(nn.Module):



    @nn.compact
    def __call__(self, x):
      x = nn.Dense(32, name="dense1")(x)
      x = nn.relu(x)
      x = nn.Dense(32, name="dense2")(x)
      return x

So, how would you decide which style to use? It can be a matter of taste, but here are some pros and cons:

Reasons to prefer using ``nn.compact``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Allows defining submodules, parameters and other variables next to where they are used: less    
   scrolling up/down to see how everything is defined.
2. Reduces code duplication when there are conditionals or for loops that conditionally define
   submodules, parameters or variables.
3. Code typically looks more like mathematical notation.
4. Generally, shorter code.

Reasons to prefer using ``setup``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Closer to normal Python object semantics
2. Easier when porting models from PyTorch
3. Allows defining more than one "forward pass" method
   (see :class:`MultipleMethodsCompactError <flax.errors.MultipleMethodsCompactError>`)






.. _`Linen`: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#JIT-mechanics:-tracing-and-static-variables

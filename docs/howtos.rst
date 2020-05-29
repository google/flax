Flax HOWTOs
===========

Flax aims to be a thin library of composable primitives. You compose
these primites yourself into a training loop that you write and is fully under
your control. You have full freedom to modify the behavior of your training loop,
and you should generally not need special library support to implement
the modifications you want.

To help you get started, we show some sample diffs, which
we call "HOWTOs". These HOWTOs show common modifications to training loops. For instance,
the HOWTO for ensembling learning demonstrates what changes should be made to
the standard MNIST example in order to train an ensemble of models on
multiple devices.

Note that these HOWTOs do not require special library support, they just
demonstate how assembling the JAX and Flax primitives in different ways
allow you to make various training loop modifications.

There are currently two ways to access a HOWTO: 1) check out a branch with the HOWTO diff applied to the master branch or 2) apply the HOWTO diff yourself:

.. code-block:: bash
   # Clone repository
   git clone https://github.com/google/flax
   cd flax

   # Method 1: Check out HOWTO (e.g., distributed-training):
   git checkout howto/distributed-training

   # Method 2: Apply HOWTO diff
   git apply --3way howtos/diffs/distributed-training.diff

Currently the following HOWTOs are available:

Multi-device data-parallel training
-----------------------------------

⟶ `View as a side-by-side diff <https://github.com/google/flax/compare/master..howto/distributed-training?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/distributed-training.diff.html


Ensembling on multiple devices
------------------------------

⟶ `View as a side-by-side diff <https://github.com/google/flax/compare/master..howto/ensembling?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/ensembling.diff.html


Polyak averaging
----------------

⟶ `View as a side-by-side diff <https://github.com/google/flax/compare/master..howto/polyak-averaging?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/polyak-averaging.diff.html

Scheduled Sampling
----------------

⟶ `View as a side-by-side diff <https://github.com/google/flax/compare/master..howto/scheduled-sampling?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/scheduled-sampling.diff.html


How do HOWTOs work?
-------------------

Read the `HOWTOs HOWTO <howtos-howto.md>`_ to learn how we maintain HOWTOs.


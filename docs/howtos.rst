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

Currently the following HOWTOs are available:

Multi-device data-parallel training
-----------------------------------

⟶ `View as a side-by-side diff <https://github.com/google-research/flax/compare/prerelease..howto-distributed-training?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/howto-distributed-training.diff.html


Ensembling on multiple devices
------------------------------

`View as a side-by-side diff <https://github.com/google-research/flax/compare/prerelease..howto-ensembling?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/howto-ensembling.diff.html


Polyak averaging
----------------

`View as a side-by-side diff <https://github.com/google-research/flax/compare/prerelease..howto-polyak-averaging?diff=split>`_

.. raw:: html
   :file: _formatted_howtos/howto-polyak-averaging.diff.html


How do HOWTOs work?
-------------------

Read the `HOWTOs HOWTO <howtos-howto.md>`_ to learn how we maintain HOWTOs.


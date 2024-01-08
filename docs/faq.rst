Frequently Asked Questions (FAQ)
================================

This is a collection of answers to frequently asked questions (FAQ). You can contribute to the Flax FAQ by starting a new topic in `GitHub Discussions <https://github.com/google/flax/discussions>`__.

Where to search for an answer to a Flax-related question?
*********************************************************

There are a number of official Flax resources to search for information:

- `Flax Documentation on ReadTheDocs <https://flax.readthedocs.io/en/latest/>`__ (this site): Use the `search bar <https://flax.readthedocs.io/en/search.html>`__ or the table of contents on the left-hand side.
- `google/flax GitHub Discussions <https://github.com/google/flax/discussions>`__: Search for an existing topic or start a new one. If you can't find what you're looking for, feel free to ask the Flax team or community a question.
- `google/flax GitHub Issues <https://github.com/google/flax/issues>`__: Use the search bar to look for an existing issue or a feature request, or start a new one.

How to take the derivative with respect to an intermediate value (using :code:`Module.perturb`)?
************************************************************************************************

To take the derivative(s) or gradient(s) of the output with respect to a hidden/intermediate activation inside a model layer, you can use :meth:`flax.linen.Module.perturb`. You define a zero-value :class:`flax.linen.Module` "perturbation" parameter – :code:`perturb(...)` – in the forward pass with the same shape as the intermediate activation, define the loss function with :code:`'perturbations'` as an added standalone argument, perform a JAX derivative operation with :code:`jax.grad` on the perturbation argument.

For full examples and detailed documentation, go to:

- The :meth:`flax.linen.Module.perturb` API docs
- The `Extracting gradients of intermediate values <https://flax.readthedocs.io/en/latest/guides/model_inspection/extracting_intermediates.html#extracting-gradients-of-intermediate-values>`_ guide
- `Flax GitHub Discussions #1152 <https://github.com/google/flax/discussions/1152>`__

Is Flax Linen :code:`remat_scan()` the same as :code:`scan(remat(...))`?
************************************************************************

Flax :code:`remat_scan()` (:meth:`flax.linen.remat_scan()`) and :code:`scan(remat(...))` (:meth:`flax.linen.scan` over :meth:`flax.linen.remat`) are not the same, and :code:`remat_scan()` is limited in cases it supports. Namely, :code:`remat_scan()` treats the inputs and outputs as carries (hidden states that are carried through the training loop). You are recommended to use :code:`scan(remat(...))`, as typically you would need the extra parameters, such as ``in_axes`` (for input array axes) or ``out_axes`` (output array axes), which :meth:`flax.linen.remat_scan` does not expose.

What are the recommended training loop libraries?
*************************************************

Consider using CLU (Common Loop Utils) `google/CommonLoopUtils <https://github.com/google/CommonLoopUtils>`__. To get started, go to this `CLU Synopsis Colab <https://colab.research.google.com/github/google/CommonLoopUtils/blob/main/clu_synopsis.ipynb>`__. You can find answers to common questions about CLU with Flax on `google/flax GitHub Discussions <https://github.com/google/flax/discussions?discussions_q=clu>`__.

Check out the official `google/flax Examples <https://github.com/google/flax/tree/main/examples>`__ for examples of using the training loop with  (CLU) metrics. For example, this is `Flax ImageNet's train.py <https://github.com/google/flax/blob/main/examples/imagenet/train.py>`__.

For computer vision research, consider `google-research/scenic <https://github.com/google-research/scenic>`__. Scenic is a set of shared light-weight libraries solving commonly encountered tasks when training large-scale vision models (with examples of several projects). Scenic is developed in JAX with Flax. To get started, go to the `README page on GitHub <https://github.com/google-research/scenic#getting-started>`__.
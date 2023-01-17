Community examples
==================

In addition to the `curated list of official Flax examples on GitHub <https://github.com/google/flax/tree/main/examples>`__,
there is a growing community of people using Flax to build new types of machine
learning models. We are happy to showcase any example built by the community here!

If you want to submit your own Flax example, you can start by forking
one of the `official Flax examples on GitHub <https://github.com/google/flax/tree/main/examples>`__.

Models
******
.. list-table::
    :header-rows: 1

    * - Link
      - Author
      - Task type
      - Reference
    * - `matthias-wright/flaxmodels <https://github.com/matthias-wright/flaxmodels>`__
      - `@matthias-wright <https://github.com/matthias-wright>`__
      - Various
      - GPT-2, ResNet, StyleGAN-2, VGG, ...
    * - `DarshanDeshpande/jax-models <https://github.com/DarshanDeshpande/jax-models>`__
      - `@DarshanDeshpande <https://github.com/DarshanDeshpande>`__
      - Various
      - Segformer, Swin Transformer, ... also some stand-alone layers
    * - `google/vision_transformer <https://github.com/google-research/vision_transformer>`__
      - `@andsteing <https://github.com/andsteing>`__
      - Image classification, image/text
      - https://arxiv.org/abs/2010.11929, https://arxiv.org/abs/2105.01601, https://arxiv.org/abs/2111.07991, ...
    * - `jax-resnet <https://github.com/n2cholas/jax-resnet>`__
      - `@n2cholas <https://github.com/n2cholas>`__
      - Various resnet implementations
      - `torch.hub <https://pytorch.org/docs/stable/hub.html>`__
    * - `Wav2Vec2 finetuning <https://github.com/vasudevgupta7/speech-jax>`__
      - `@vasudevgupta7 <https://github.com/vasudevgupta7>`__
      - Automatic Speech Recognition
      - https://arxiv.org/abs/2006.11477

Examples
********

.. list-table::
    :header-rows: 1

    * - Link
      - Author
      - Task type
      - Reference
    * - `JAX-RL <https://github.com/henry-prior/jax-rl>`__
      - `@henry-prior <https://github.com/henry-prior>`__
      - Reinforcement learning
      - N/A
    * - `BigBird Fine-tuning <https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/big_bird>`__
      - `@vasudevgupta7 <https://github.com/vasudevgupta7>`__
      - Question-Answering
      - https://arxiv.org/abs/2007.14062
    * - `Bayesian Networks with BlackJAX <https://blackjax-devs.github.io/blackjax/examples/SGMCMC.html>`__
      - `@rlouf <https://github.com/rlouf>`__
      - Bayesian Inference, SGMCMC
      - https://arxiv.org/abs/1402.4102
    * - `DCGAN <https://github.com/bkkaggle/jax-dcgan>`__
      - `@bkkaggle <https://github.com/bkkaggle>`__
      - Image Synthesis
      - https://arxiv.org/abs/1511.06434
    * - `denoising-diffusion-flax <https://github.com/yiyixuxu/denoising-diffusion-flax>`__
      - `@yiyixuxu <https://github.com/yiyixuxu>`__
      - Image generation
      - https://arxiv.org/abs/2006.11239

Tutorials
*********

.. currently left empty as a placeholder for tutorials
.. list-table::
    :header-rows: 1

    * - Link
      - Author
      - Task type
      - Reference
    * -
      -
      -
      -

Contributing policy
*******************

If you are interested in adding a project to the Community Examples section, take the following
into consideration:

* **Code examples**: Examples for must contain a README that is helpful, clear, and explains
  how to run the code. The code itself should be easy to follow.
* **Tutorials**: These docs should preferrably be a Jupyter Notebook format
  (refer to `Contributing <https://flax.readthedocs.io/en/latest/contributing.html>`__
  to learn how to convert a Jupyter Notebook into a Markdown file with `jupytext`).
  Your tutorial should be well-written, and discuss/decsribe an interesting topic/task.
  To avoid duplication, the content of these docs must be different from
  `existing docs on the Flax documentation site <https://flax.readthedocs.io/>`__
  or other community examples mentioned in this document.
* **Models**: repositories with models ported to Flax must provide at least one of the following:

  * Metrics that are comparable to the original work when the model is trained to completion. Having
    available plots of the metric's history during training is highly encouraged.
  * Tests to verify numerical equivalence against a well known implementation (same inputs
    + weights = same outputs) preferably using pretrained weights.

In all cases mentioned above, the code must work with the latest stable versions of the
following packages: ``jax``, ``flax``, and ``optax``, and make substantial use of Flax.
Note that both ``jax`` and ``optax`` are `required packages <https://github.com/google/flax/blob/main/setup.py>`__
of ``flax`` (refer to the `installation instructions <https://github.com/google/flax/blob/main/README.md#quick-install>`__
for more details).

*********
Examples
*********

Core examples
##############


Core examples are hosted on the Flax repo in the `examples <https://github.com/google/flax/tree/main/examples>`__
directory.

Each example is designed to be **self-contained and easily forkable**, while
reproducing relevant results in different areas of machine learning.

As discussed in `#231 <https://github.com/google/flax/issues/231>`__, we decided
to go for a standard pattern for all examples including the simplest ones (like MNIST).
This makes every example a bit more verbose, but once you know one example, you
know the structure of all of them. Having unit tests and integration tests is also
very useful when you fork these examples.

Some of the examples below have a link "Interactive🕹" that lets you run them
directly in Colab.

Image classification
********************

- :octicon:`mark-github;0.9em` `MNIST <https://github.com/google/flax/tree/main/examples/mnist/>`__ -
  `Interactive🕹 <https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb>`__:
  Convolutional neural network for MNIST classification (featuring simple
  code).

- :octicon:`mark-github;0.9em` `ImageNet <https://github.com/google/flax/tree/main/examples/imagenet/>`__ -
  `Interactive🕹 <https://colab.research.google.com/github/google/flax/blob/main/examples/imagenet/imagenet.ipynb>`__:
  Resnet-50 on ImageNet with weight decay (featuring multi host SPMD, custom
  preprocessing, checkpointing, dynamic scaling, mixed precision).

Reinforcement learning
**********************

- :octicon:`mark-github;0.9em` `Proximal Policy Optimization <https://github.com/google/flax/tree/main/examples/ppo/>`__:
  Learning to play Atari games (featuring single host SPMD, RL setup).

Natural language processing
***************************

-  :octicon:`mark-github;0.9em` `Sequence to sequence for number
   addition <https://github.com/google/flax/tree/main/examples/seq2seq/>`__:
   (featuring simple code, LSTM state handling, on the fly data generation).
-  :octicon:`mark-github;0.9em` `Parts-of-speech
   tagging <https://github.com/google/flax/tree/main/examples/nlp_seq/>`__: Simple
   transformer encoder model using the universal dependency dataset.
-  :octicon:`mark-github;0.9em` `Sentiment
   classification <https://github.com/google/flax/tree/main/examples/sst2/>`__:
   with a LSTM model.
-  :octicon:`mark-github;0.9em` `Transformer encoder/decoder model trained on
   WMT <https://github.com/google/flax/tree/main/examples/wmt/>`__:
   Translating English/German (featuring multihost SPMD, dynamic bucketing,
   attention cache, packed sequences, recipe for TPU training on GCP).
-  :octicon:`mark-github;0.9em` `Transformer encoder trained on one billion word
   benchmark <https://github.com/google/flax/tree/main/examples/lm1b/>`__:
   for autoregressive language modeling, based on the WMT example above.

Generative models
*****************

-  :octicon:`mark-github;0.9em` `Variational
   auto-encoder <https://github.com/google/flax/tree/main/examples/vae/>`__:
   Trained on binarized MNIST (featuring simple code, vmap).

Graph modeling
**************

- :octicon:`mark-github;0.9em` `Graph Neural Networks <https://github.com/google/flax/tree/main/examples/ogbg_molpcba/>`__:
  Molecular predictions on ogbg-molpcba from the Open Graph Benchmark.

Contributing Examples
*********************

Most of the core examples follow a structure that we found to work
well with Flax projects, and we strive to make the examples easy to explore and
easy to fork. In particular (taken from `#231 <https://github.com/google/flax/issues/231>`__)

- README: contains links to paper, command line, `TensorBoard <https://tensorboard.dev/>`__ metrics
- Focus: an example is about a single model/dataset
- Configs: we use ``ml_collections.ConfigDict`` stored under ``configs/``
- Tests: executable ``main.py`` loads ``train.py`` which has ``train_test.py``
- Data: is read from `TensorFlow Datasets <https://www.tensorflow.org/datasets>`__
- Standalone: every directory is self-contained
- Requirements: versions are pinned in ``requirements.txt``
- Boilerplate: is reduced by using `clu <https://pypi.org/project/clu/>`__
- Interactive: the example can be explored with a `Colab <https://colab.research.google.com/>`__

Repositories Using Flax
#######################

The following code bases use Flax and provide training frameworks and a wealth
of examples, in many cases with pre-trained weights:

- `🤗 Hugging Face <https://huggingface.co/flax-community>`__ is a
  very popular library for building, training, and deploying state of the art
  machine learning models.
  These models can be applied on text, images, and audio. After organizing the
  `JAX/Flax community week <https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md>`__,
  they have now over 5,000
  `Flax/JAX models <https://huggingface.co/models?library=jax&sort=downloads>`__ in
  their repository.

- `🥑 DALLE Mini <https://huggingface.co/dalle-mini>`__ is a Transformer-based
  text-to-image model implemented in JAX/Flax that follows the ideas from the
  original `DALLE <https://openai.com/blog/dall-e/>`__ paper by OpenAI.

- `Scenic <https://github.com/google-research/scenic>`__ is a codebase/library
  for computer vision research and beyond. Scenic's main focus is around
  attention-based models. Scenic has been successfully used to develop
  classification, segmentation, and detection models for multiple modalities
  including images, video, audio, and multimodal combinations of them.

- `Big Vision <https://github.com/google-research/big_vision/>`__ is a codebase
  designed for training large-scale vision models using Cloud TPU VMs or GPU
  machines. It is based on Jax/Flax libraries, and uses tf.data and TensorFlow
  Datasets for scalable and reproducible input pipelines. This is the original
  codebase of ViT, MLP-Mixer, LiT, UViM, and many more models.

- `T5X <https://github.com/google-research/t5x>`__  is a modular, composable,
  research-friendly framework for high-performance, configurable, self-service
  training, evaluation, and inference of sequence models (starting with
  language) at many scales.

Community Examples
###################

In addition to the curated list of official Flax examples, there is a growing
community of people using Flax to build new types of machine learning models. We
are happy to showcase any example built by the community here! If you want to
submit your own example, we suggest that you start by forking one of the
official Flax example, and start from there.

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
      - ``torch.hub``

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
********

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

Contributing Policy
********************

If you are interested in adding a project to the Community Examples section, take the following
into consideration:

* **Examples**: examples should contain a README that is helpful, clear, and makes it easy to run
  the code. The code itself should be easy to follow.
* **Tutorials**: tutorials must preferably be runnable notebooks, be well written, and discuss
  an interesting topic. Also, the tutorial's content must be different from the existing
  guides in the Flax documentation and other community examples to be considered for inclusion.
* **Models**: repositories with models ported to Flax must provide at least one of the following:

  * Metrics that are comparable to the original work when the model is trained to completion. Having
    available plots of the metric's history during training is highly encouraged.
  * Tests to verify numerical equivalence against a well known implementation (same inputs
    + weights = same outputs) preferably using pretrained weights.

On all cases above, code should work with the latest stable version of packages like ``jax``,
``flax``, and ``optax``, and make substantial use of Flax.
